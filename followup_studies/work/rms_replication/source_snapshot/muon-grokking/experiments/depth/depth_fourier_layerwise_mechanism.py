from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from depth_fourier_mode_identification import (
    MODE_FAMILIES,
    accuracy,
    build_model,
    get_device,
    load_checkpoint,
    mode_family_masks,
    ordered_full_grid,
    release_accelerator_cache,
)


NON_DC_FAMILIES = (
    "addition",
    "subtraction",
    "a_only",
    "b_only",
    "generic_interaction",
)


@dataclass(frozen=True)
class Selection:
    study: str
    run: str
    depth: int
    regime: str
    role: str
    step: int
    checkpoint: Path


@dataclass
class Stage:
    name: str
    kind: str
    layer: int
    order: int
    values: Tensor
    continue_fn: Callable[[Tensor], Tensor]


@dataclass
class State:
    selection: Selection
    model: object
    targets: Tensor
    baseline_accuracy: float
    stages: list[Stage]


@dataclass(frozen=True)
class ModePair:
    representative: tuple[int, int]
    conjugate: tuple[int, int]
    power: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track and causally intervene on Fourier circuits at every "
            "attention, MLP, and residual stage, including matched "
            "control/freeze activation patching."
        )
    )
    parser.add_argument(
        "--model-summary",
        type=Path,
        default=Path(
            "runs/depth_fourier_mode_model_summary_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--matched-paired-summary",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_mode_paired_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--random-replicates",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--flow-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_layerwise_flow_v2_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--causal-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_layerwise_causal_v2_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--matched-flow-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_layerwise_flow_v2_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--matched-patching-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_layerwise_patching_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--landmark-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_layerwise_landmarks_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_layerwise_selection_v2_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_layerwise_mechanism_seed_0.html"
        ),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main_panel(frame: pd.DataFrame) -> list[Selection]:
    selections: list[Selection] = []

    for (run, depth, regime), group in frame.groupby(
        ["run", "depth", "regime"],
        sort=False,
    ):
        group = group.sort_values("step")
        roles: list[tuple[str, pd.Series]] = []

        generalized = group[group["full_accuracy"] >= 0.95]
        if not generalized.empty:
            first = generalized.iloc[0]
            roles.append(("first_generalized", first))
            post = group[group["step"] >= first["step"]]
            roles.append(
                (
                    "minimum_after_first_generalized",
                    post.loc[post["full_accuracy"].idxmin()],
                )
            )
        else:
            best = group.loc[group["full_accuracy"].idxmax()]
            roles.append(("best_available", best))
            post = group[group["step"] >= best["step"]]
            roles.append(
                (
                    "minimum_after_best",
                    post.loc[post["full_accuracy"].idxmin()],
                )
            )

        roles.append(
            (
                "maximum_accuracy",
                group.loc[group["full_accuracy"].idxmax()],
            )
        )
        roles.append(("final", group.iloc[-1]))

        seen: set[int] = set()
        for role, row in roles:
            step = int(row["step"])
            if step in seen:
                continue
            seen.add(step)
            selections.append(
                Selection(
                    study="main",
                    run=str(run),
                    depth=int(depth),
                    regime=str(regime),
                    role=role,
                    step=step,
                    checkpoint=Path(str(row["checkpoint"])),
                )
            )

    return selections


def matched_panel(frame: pd.DataFrame) -> list[int]:
    frame = frame.sort_values("step")
    branch = int(frame.iloc[0]["step"])
    final = int(frame.iloc[-1]["step"])

    severe = frame[frame["control_full_accuracy"] < 0.10]
    first_severe = (
        int(severe.iloc[0]["step"])
        if not severe.empty
        else int(
            frame.loc[
                frame["control_full_accuracy"].idxmin(),
                "step",
            ]
        )
    )
    minimum = int(
        frame.loc[
            frame["control_full_accuracy"].idxmin(),
            "step",
        ]
    )
    later = frame[frame["step"] > minimum]
    recovery = (
        int(
            later.loc[
                later["control_full_accuracy"].idxmax(),
                "step",
            ]
        )
        if not later.empty
        else final
    )

    return list(
        dict.fromkeys(
            [branch, first_severe, minimum, recovery, final]
        )
    )


def matched_role(step: int, steps: list[int]) -> str:
    mapping = {
        steps[0]: "branch",
        steps[-1]: "final",
    }
    if len(steps) > 1:
        mapping[steps[1]] = "first_control_below_10pct"
    if len(steps) > 2:
        mapping[steps[2]] = "minimum_control_accuracy"
    if len(steps) > 3:
        mapping[steps[3]] = "post_collapse_recovery"
    return mapping.get(step, "matched_panel")


def run_block(block, hidden: Tensor) -> Tensor:
    hidden = hidden + block.attention(hidden)
    mlp_activation = F.relu(
        block.mlp.input_projection(hidden)
    )
    hidden = hidden + block.mlp.output_projection(
        mlp_activation
    )
    return hidden


def continue_from_block_input(
    model,
    hidden: Tensor,
    start_layer: int,
) -> Tensor:
    for block in model.transformer_blocks[start_layer:]:
        hidden = run_block(block, hidden)
    return hidden


def build_state(
    selection: Selection,
    device: torch.device,
) -> State:
    checkpoint = load_checkpoint(selection.checkpoint)
    model = build_model(
        checkpoint,
        expected_depth=selection.depth,
        device=device,
    )
    modulus = int(checkpoint["model_config"]["modulus"])
    inputs, targets = ordered_full_grid(modulus, device)

    stages: list[Stage] = []

    with torch.no_grad():
        positions = torch.arange(
            inputs.shape[1],
            device=device,
        )
        hidden = (
            model.token_embedding(inputs)
            + model.position_embedding(positions)
        )

        input_values = hidden.detach().float().cpu()
        stages.append(
            Stage(
                name="input_residual",
                kind="residual_state",
                layer=0,
                order=0,
                values=input_values,
                continue_fn=lambda replacement, m=model: (
                    continue_from_block_input(
                        m,
                        replacement,
                        0,
                    )
                ),
            )
        )

        order = 1

        for layer_index, block in enumerate(
            model.transformer_blocks,
            start=1,
        ):
            block_zero_index = layer_index - 1
            before_attention = hidden
            attention_output = block.attention(hidden)
            post_attention = hidden + attention_output
            mlp_activation = F.relu(
                block.mlp.input_projection(
                    post_attention
                )
            )
            mlp_output = block.mlp.output_projection(
                mlp_activation
            )
            post_block = post_attention + mlp_output

            before_attention_cpu = (
                before_attention.detach().float().cpu()
            )
            attention_output_cpu = (
                attention_output.detach().float().cpu()
            )
            post_attention_cpu = (
                post_attention.detach().float().cpu()
            )
            mlp_activation_cpu = (
                mlp_activation.detach().float().cpu()
            )
            mlp_output_cpu = (
                mlp_output.detach().float().cpu()
            )
            post_block_cpu = (
                post_block.detach().float().cpu()
            )

            def continue_attention_write(
                replacement: Tensor,
                *,
                m=model,
                idx=block_zero_index,
                before=before_attention_cpu,
            ) -> Tensor:
                current = before.to(device) + replacement
                current_block = m.transformer_blocks[idx]
                activation = F.relu(
                    current_block.mlp.input_projection(
                        current
                    )
                )
                current = (
                    current
                    + current_block.mlp.output_projection(
                        activation
                    )
                )
                return continue_from_block_input(
                    m,
                    current,
                    idx + 1,
                )

            def continue_post_attention(
                replacement: Tensor,
                *,
                m=model,
                idx=block_zero_index,
            ) -> Tensor:
                current_block = m.transformer_blocks[idx]
                activation = F.relu(
                    current_block.mlp.input_projection(
                        replacement
                    )
                )
                current = (
                    replacement
                    + current_block.mlp.output_projection(
                        activation
                    )
                )
                return continue_from_block_input(
                    m,
                    current,
                    idx + 1,
                )

            def continue_mlp_activation(
                replacement: Tensor,
                *,
                m=model,
                idx=block_zero_index,
                post=post_attention_cpu,
            ) -> Tensor:
                current_block = m.transformer_blocks[idx]
                current = (
                    post.to(device)
                    + current_block.mlp.output_projection(
                        replacement
                    )
                )
                return continue_from_block_input(
                    m,
                    current,
                    idx + 1,
                )

            def continue_mlp_write(
                replacement: Tensor,
                *,
                m=model,
                idx=block_zero_index,
                post=post_attention_cpu,
            ) -> Tensor:
                current = post.to(device) + replacement
                return continue_from_block_input(
                    m,
                    current,
                    idx + 1,
                )

            def continue_post_block(
                replacement: Tensor,
                *,
                m=model,
                idx=block_zero_index,
            ) -> Tensor:
                return continue_from_block_input(
                    m,
                    replacement,
                    idx + 1,
                )

            stages.extend(
                [
                    Stage(
                        name=(
                            f"layer_{layer_index}_attention_output"
                        ),
                        kind="attention_write",
                        layer=layer_index,
                        order=order,
                        values=attention_output_cpu,
                        continue_fn=continue_attention_write,
                    ),
                    Stage(
                        name=(
                            f"layer_{layer_index}_post_attention"
                        ),
                        kind="residual_state",
                        layer=layer_index,
                        order=order + 1,
                        values=post_attention_cpu,
                        continue_fn=continue_post_attention,
                    ),
                    Stage(
                        name=(
                            f"layer_{layer_index}_mlp_activation"
                        ),
                        kind="mlp_activation",
                        layer=layer_index,
                        order=order + 2,
                        values=mlp_activation_cpu,
                        continue_fn=continue_mlp_activation,
                    ),
                    Stage(
                        name=(
                            f"layer_{layer_index}_mlp_output"
                        ),
                        kind="mlp_write",
                        layer=layer_index,
                        order=order + 3,
                        values=mlp_output_cpu,
                        continue_fn=continue_mlp_write,
                    ),
                    Stage(
                        name=(
                            f"layer_{layer_index}_post_block"
                        ),
                        kind="residual_state",
                        layer=layer_index,
                        order=order + 4,
                        values=post_block_cpu,
                        continue_fn=continue_post_block,
                    ),
                ]
            )

            hidden = post_block
            order += 5

        logits = model.unembedding(hidden[:, -1, :])
        baseline = accuracy(logits, targets)

    return State(
        selection=selection,
        model=model,
        targets=targets,
        baseline_accuracy=baseline,
        stages=stages,
    )


def modulus_from_state(state: State) -> int:
    return int(math.isqrt(state.targets.shape[0]))


def fft_values(
    values: Tensor,
    modulus: int,
) -> tuple[Tensor, Tensor]:
    grid = values.reshape(
        modulus,
        modulus,
        values.shape[1],
        values.shape[2],
    )
    spectrum = torch.fft.fft2(
        grid,
        dim=(0, 1),
        norm="ortho",
    )
    power = spectrum.abs().square().sum(dim=(-1, -2))
    return spectrum, power


def final_token_fft(
    values: Tensor,
    modulus: int,
) -> tuple[Tensor, Tensor]:
    grid = values[:, -1, :].reshape(
        modulus,
        modulus,
        -1,
    )
    spectrum = torch.fft.fft2(
        grid,
        dim=(0, 1),
        norm="ortho",
    )
    power = spectrum.abs().square().sum(dim=-1)
    return spectrum, power


def inverse_values(
    spectrum: Tensor,
    original_shape: torch.Size,
) -> Tensor:
    reconstructed = torch.fft.ifft2(
        spectrum,
        dim=(0, 1),
        norm="ortho",
    ).real
    return reconstructed.reshape(original_shape)


def family_statistics(
    power: Tensor,
    masks: dict[str, Tensor],
) -> dict[str, float]:
    total = power.sum()
    dc_power = power[masks["dc"]].sum()
    non_dc = total - dc_power
    result = {
        "total_power": float(total.item()),
        "non_dc_power": float(non_dc.item()),
    }

    for family in MODE_FAMILIES:
        family_power = power[masks[family]].sum()
        result[f"{family}_power_fraction_total"] = (
            float((family_power / total).item())
            if float(total.item()) > 0.0
            else 0.0
        )
        result[f"{family}_power_fraction_non_dc"] = (
            float((family_power / non_dc).item())
            if family != "dc"
            and float(non_dc.item()) > 1e-20
            else float("nan")
        )

    return result


def canonical_pair(
    k: int,
    l: int,
    modulus: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    original = (k, l)
    conjugate = ((-k) % modulus, (-l) % modulus)
    return min(original, conjugate), max(original, conjugate)


def addition_pairs(power: Tensor) -> list[ModePair]:
    modulus = int(power.shape[0])
    seen: set[tuple[int, int]] = set()
    pairs = []

    for k in range(1, modulus):
        representative, conjugate = canonical_pair(
            k,
            k,
            modulus,
        )
        if representative in seen:
            continue
        seen.add(representative)
        pair_power = float(
            (
                power[
                    representative[0],
                    representative[1],
                ]
                + power[
                    conjugate[0],
                    conjugate[1],
                ]
            ).item()
        )
        pairs.append(
            ModePair(
                representative=representative,
                conjugate=conjugate,
                power=pair_power,
            )
        )

    pairs.sort(key=lambda pair: pair.power, reverse=True)
    return pairs


def top_pairs_all(
    power: Tensor,
    count: int = 10,
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    modulus = int(power.shape[0])
    seen: set[tuple[int, int]] = set()
    rows = []

    for k in range(modulus):
        for l in range(modulus):
            representative, conjugate = canonical_pair(
                k,
                l,
                modulus,
            )
            if representative in seen:
                continue
            seen.add(representative)
            if representative == (0, 0):
                continue
            if representative == conjugate:
                pair_power = float(
                    power[
                        representative[0],
                        representative[1],
                    ].item()
                )
            else:
                pair_power = float(
                    (
                        power[
                            representative[0],
                            representative[1],
                        ]
                        + power[
                            conjugate[0],
                            conjugate[1],
                        ]
                    ).item()
                )
            rows.append(
                (representative, conjugate, pair_power)
            )

    rows.sort(key=lambda row: row[2], reverse=True)
    return rows[:count]


def pair_mask(
    pairs: Iterable[ModePair],
    modulus: int,
) -> Tensor:
    mask = torch.zeros(
        modulus,
        modulus,
        dtype=torch.bool,
    )
    for pair in pairs:
        mask[
            pair.representative[0],
            pair.representative[1],
        ] = True
        mask[
            pair.conjugate[0],
            pair.conjugate[1],
        ] = True
    return mask


def cumulative_pairs(
    pairs: list[ModePair],
    threshold: float,
) -> list[ModePair]:
    total = sum(pair.power for pair in pairs)
    if total <= 0.0:
        return []

    selected = []
    accumulated = 0.0
    for pair in pairs:
        selected.append(pair)
        accumulated += pair.power
        if accumulated / total >= threshold:
            break
    return selected


def pair_text(
    pairs: Iterable[
        ModePair
        | tuple[
            tuple[int, int],
            tuple[int, int],
            float,
        ]
    ],
) -> str:
    output = []
    for pair in pairs:
        if isinstance(pair, ModePair):
            first = pair.representative
            second = pair.conjugate
        else:
            first, second, _ = pair
        output.append(
            f"({first[0]},{first[1]})/"
            f"({second[0]},{second[1]})"
        )
    return "|".join(output)


def complex_cosine(
    first: Tensor,
    second: Tensor,
) -> float:
    if first.shape != second.shape:
        return float("nan")
    first_vector = torch.view_as_real(first).reshape(-1).float()
    second_vector = torch.view_as_real(second).reshape(-1).float()
    denominator = (
        torch.linalg.vector_norm(first_vector)
        * torch.linalg.vector_norm(second_vector)
    )
    if float(denominator.item()) <= 0.0:
        return 0.0
    return float(
        (
            torch.dot(first_vector, second_vector)
            / denominator
        ).item()
    )


def real_cosine(first: Tensor, second: Tensor) -> float:
    first = first.reshape(-1).float()
    second = second.reshape(-1).float()
    denominator = (
        torch.linalg.vector_norm(first)
        * torch.linalg.vector_norm(second)
    )
    if float(denominator.item()) <= 0.0:
        return 0.0
    return float(
        (torch.dot(first, second) / denominator).item()
    )


def keep_mask_intervention(
    values: Tensor,
    modulus: int,
    keep_mask: Tensor,
) -> Tensor:
    spectrum, _ = fft_values(values, modulus)
    filtered = torch.where(
        keep_mask[:, :, None, None],
        spectrum,
        torch.zeros_like(spectrum),
    )
    return inverse_values(filtered, values.shape)


def ablate_mask_intervention(
    values: Tensor,
    modulus: int,
    remove_mask: Tensor,
) -> Tensor:
    spectrum, _ = fft_values(values, modulus)
    filtered = torch.where(
        remove_mask[:, :, None, None],
        torch.zeros_like(spectrum),
        spectrum,
    )
    return inverse_values(filtered, values.shape)


def phase_scramble_addition(
    values: Tensor,
    modulus: int,
    rng: random.Random,
    channelwise: bool,
) -> Tensor:
    spectrum, _ = fft_values(values, modulus)
    output = spectrum.clone()
    pairs = addition_pairs(
        spectrum.abs().square().sum(dim=(-1, -2))
    )

    for pair in pairs:
        coefficient = spectrum[
            pair.representative[0],
            pair.representative[1],
            :,
            :,
        ]

        if channelwise:
            angles = torch.tensor(
                [
                    [
                        rng.uniform(0.0, 2.0 * math.pi)
                        for _ in range(coefficient.shape[1])
                    ]
                    for _ in range(coefficient.shape[0])
                ],
                dtype=coefficient.real.dtype,
            )
        else:
            angle = rng.uniform(0.0, 2.0 * math.pi)
            angles = torch.full(
                coefficient.shape,
                angle,
                dtype=coefficient.real.dtype,
            )

        phase = torch.polar(
            torch.ones_like(angles),
            angles,
        )
        changed = coefficient * phase

        output[
            pair.representative[0],
            pair.representative[1],
            :,
            :,
        ] = changed
        output[
            pair.conjugate[0],
            pair.conjugate[1],
            :,
            :,
        ] = changed.conj()

    return inverse_values(output, values.shape)


def deranged_addition_relocation(
    values: Tensor,
    modulus: int,
    rng: random.Random,
) -> Tensor:
    spectrum, power = fft_values(values, modulus)
    pairs = addition_pairs(power)
    indices = list(range(len(pairs)))

    if len(indices) > 1:
        shift = rng.randrange(1, len(indices))
        permutation = [
            (index + shift) % len(indices)
            for index in indices
        ]
    else:
        permutation = indices

    output = spectrum.clone()
    addition_mask = mode_family_masks(modulus)["addition"]
    output[
        addition_mask[:, :, None, None].expand_as(output)
    ] = 0.0

    for source_index, target_index in enumerate(permutation):
        source = pairs[source_index]
        target = pairs[target_index]
        coefficient = spectrum[
            source.representative[0],
            source.representative[1],
            :,
            :,
        ]
        output[
            target.representative[0],
            target.representative[1],
            :,
            :,
        ] = coefficient
        output[
            target.conjugate[0],
            target.conjugate[1],
            :,
            :,
        ] = coefficient.conj()

    return inverse_values(output, values.shape)


def stage_logits(
    state: State,
    stage: Stage,
    replacement: Tensor,
    device: torch.device,
    readout_model=None,
) -> Tensor:
    final_hidden = stage.continue_fn(
        replacement.to(device)
    )
    model_for_readout = (
        state.model
        if readout_model is None
        else readout_model
    )
    return model_for_readout.unembedding(
        final_hidden[:, -1, :]
    )


def flow_rows(
    state: State,
    reference_pairs: list[ModePair],
) -> list[dict[str, object]]:
    modulus = modulus_from_state(state)
    masks = mode_family_masks(modulus)
    final_stage = state.stages[-1]
    final_spectrum, final_power = final_token_fft(
        final_stage.values,
        modulus,
    )
    final_top = top_pairs_all(final_power, count=10)
    reference_mask = pair_mask(reference_pairs, modulus)
    dc_mask = masks["dc"]

    rows = []

    for stage in state.stages:
        all_spectrum, all_power = fft_values(
            stage.values,
            modulus,
        )
        final_token_spectrum, final_token_power = (
            final_token_fft(
                stage.values,
                modulus,
            )
        )
        all_stats = family_statistics(all_power, masks)
        final_stats = family_statistics(
            final_token_power,
            masks,
        )

        direct = float("nan")
        addition_only = float("nan")
        nonaddition_only = float("nan")
        reference_set_only = float("nan")

        if stage.kind == "residual_state":
            device = state.model.unembedding.weight.device
            values = stage.values.to(device)
            direct = accuracy(
                state.model.unembedding(values[:, -1, :]),
                state.targets,
            )

            addition_values = keep_mask_intervention(
                stage.values,
                modulus,
                dc_mask | masks["addition"],
            ).to(device)
            addition_only = accuracy(
                state.model.unembedding(
                    addition_values[:, -1, :]
                ),
                state.targets,
            )

            nonaddition_values = keep_mask_intervention(
                stage.values,
                modulus,
                ~masks["addition"],
            ).to(device)
            nonaddition_only = accuracy(
                state.model.unembedding(
                    nonaddition_values[:, -1, :]
                ),
                state.targets,
            )

            reference_values = keep_mask_intervention(
                stage.values,
                modulus,
                dc_mask | reference_mask,
            ).to(device)
            reference_set_only = accuracy(
                state.model.unembedding(
                    reference_values[:, -1, :]
                ),
                state.targets,
            )

        rows.append(
            {
                "study": state.selection.study,
                "run": state.selection.run,
                "depth": state.selection.depth,
                "regime": state.selection.regime,
                "role": state.selection.role,
                "step": state.selection.step,
                "checkpoint": str(state.selection.checkpoint),
                "baseline_full_accuracy": state.baseline_accuracy,
                "stage_name": stage.name,
                "stage_kind": stage.kind,
                "layer": stage.layer,
                "stage_order": stage.order,
                "activation_width": stage.values.shape[-1],
                "direct_readout_accuracy": direct,
                "addition_only_direct_readout_accuracy": (
                    addition_only
                ),
                "nonaddition_only_direct_readout_accuracy": (
                    nonaddition_only
                ),
                "reference_frequency_set_direct_readout_accuracy": (
                    reference_set_only
                ),
                "top_final_token_modes": pair_text(
                    top_pairs_all(final_token_power, count=10)
                ),
                "top_mode_overlap_with_final": (
                    len(
                        {
                            pair[0]
                            for pair in top_pairs_all(
                                final_token_power,
                                count=10,
                            )
                        }
                        & {
                            pair[0]
                            for pair in final_top
                        }
                    )
                    / 10.0
                ),
                "final_token_power_cosine_with_final": (
                    real_cosine(
                        final_token_power,
                        final_power,
                    )
                ),
                "final_token_complex_cosine_with_final": (
                    complex_cosine(
                        final_token_spectrum,
                        final_spectrum,
                    )
                ),
                **{
                    f"all_token_{key}": value
                    for key, value in all_stats.items()
                },
                **{
                    f"final_token_{key}": value
                    for key, value in final_stats.items()
                },
            }
        )

    return rows


def causal_rows(
    state: State,
    reference_pairs: list[ModePair],
    device: torch.device,
    random_replicates: int,
    base_seed: int,
) -> list[dict[str, object]]:
    modulus = modulus_from_state(state)
    masks = mode_family_masks(modulus)
    dc = masks["dc"]
    addition = masks["addition"]
    reference_mask = pair_mask(reference_pairs, modulus)

    rows = []

    for stage in state.stages:
        _, local_power = fft_values(
            stage.values,
            modulus,
        )
        local_pairs = addition_pairs(local_power)
        local_top90 = cumulative_pairs(
            local_pairs,
            0.90,
        )
        local_top90_mask = pair_mask(
            local_top90,
            modulus,
        )

        deterministic: list[
            tuple[str, Tensor]
        ] = [
            (
                "identity",
                stage.values,
            ),
            (
                "addition_only",
                keep_mask_intervention(
                    stage.values,
                    modulus,
                    dc | addition,
                ),
            ),
            (
                "addition_ablation",
                ablate_mask_intervention(
                    stage.values,
                    modulus,
                    addition,
                ),
            ),
            (
                "local_top90_addition_only",
                keep_mask_intervention(
                    stage.values,
                    modulus,
                    dc | local_top90_mask,
                ),
            ),
            (
                "local_top90_addition_ablation",
                ablate_mask_intervention(
                    stage.values,
                    modulus,
                    local_top90_mask,
                ),
            ),
            (
                "reference_top90_addition_only",
                keep_mask_intervention(
                    stage.values,
                    modulus,
                    dc | reference_mask,
                ),
            ),
            (
                "reference_top90_addition_ablation",
                ablate_mask_intervention(
                    stage.values,
                    modulus,
                    reference_mask,
                ),
            ),
            (
                "nonaddition_only",
                keep_mask_intervention(
                    stage.values,
                    modulus,
                    ~addition,
                ),
            ),
            (
                "remove_all_nonaddition",
                keep_mask_intervention(
                    stage.values,
                    modulus,
                    dc | addition,
                ),
            ),
        ]

        if stage.kind in {
            "attention_write",
            "mlp_activation",
            "mlp_write",
        }:
            deterministic.append(
                (
                    "zero_entire_component",
                    torch.zeros_like(stage.values),
                )
            )

        for family in (
            "a_only",
            "b_only",
            "subtraction",
            "generic_interaction",
        ):
            deterministic.append(
                (
                    f"remove_{family}",
                    ablate_mask_intervention(
                        stage.values,
                        modulus,
                        masks[family],
                    ),
                )
            )

        for intervention, replacement in deterministic:
            logits = stage_logits(
                state,
                stage,
                replacement,
                device,
            )
            rows.append(
                {
                    "study": state.selection.study,
                    "run": state.selection.run,
                    "depth": state.selection.depth,
                    "regime": state.selection.regime,
                    "role": state.selection.role,
                    "step": state.selection.step,
                    "checkpoint": str(
                        state.selection.checkpoint
                    ),
                    "baseline_full_accuracy": (
                        state.baseline_accuracy
                    ),
                    "stage_name": stage.name,
                    "stage_kind": stage.kind,
                    "layer": stage.layer,
                    "stage_order": stage.order,
                    "intervention": intervention,
                    "replicate": -1,
                    "local_top90_pair_count": len(
                        local_top90
                    ),
                    "local_top90_pairs": pair_text(
                        local_top90
                    ),
                    "reference_top90_pair_count": len(
                        reference_pairs
                    ),
                    "reference_top90_pairs": pair_text(
                        reference_pairs
                    ),
                    "intervention_full_accuracy": accuracy(
                        logits,
                        state.targets,
                    ),
                }
            )

        for replicate in range(random_replicates):
            rng_seed = (
                base_seed
                + 1_000_003 * state.selection.depth
                + 10_007 * state.selection.step
                + 101 * stage.order
                + replicate
            )
            rng = random.Random(rng_seed)

            for phase_type, channelwise in (
                ("pair_global", False),
                ("channelwise", True),
            ):
                replacement = phase_scramble_addition(
                    stage.values,
                    modulus,
                    rng,
                    channelwise=channelwise,
                )
                logits = stage_logits(
                    state,
                    stage,
                    replacement,
                    device,
                )
                rows.append(
                    {
                        "study": state.selection.study,
                        "run": state.selection.run,
                        "depth": state.selection.depth,
                        "regime": state.selection.regime,
                        "role": state.selection.role,
                        "step": state.selection.step,
                        "checkpoint": str(
                            state.selection.checkpoint
                        ),
                        "baseline_full_accuracy": (
                            state.baseline_accuracy
                        ),
                        "stage_name": stage.name,
                        "stage_kind": stage.kind,
                        "layer": stage.layer,
                        "stage_order": stage.order,
                        "intervention": (
                            f"phase_scramble_{phase_type}"
                        ),
                        "replicate": replicate,
                        "local_top90_pair_count": len(
                            local_top90
                        ),
                        "local_top90_pairs": pair_text(
                            local_top90
                        ),
                        "reference_top90_pair_count": len(
                            reference_pairs
                        ),
                        "reference_top90_pairs": pair_text(
                            reference_pairs
                        ),
                        "intervention_full_accuracy": (
                            accuracy(
                                logits,
                                state.targets,
                            )
                        ),
                    }
                )

            relocated = deranged_addition_relocation(
                stage.values,
                modulus,
                rng,
            )
            logits = stage_logits(
                state,
                stage,
                relocated,
                device,
            )
            rows.append(
                {
                    "study": state.selection.study,
                    "run": state.selection.run,
                    "depth": state.selection.depth,
                    "regime": state.selection.regime,
                    "role": state.selection.role,
                    "step": state.selection.step,
                    "checkpoint": str(
                        state.selection.checkpoint
                    ),
                    "baseline_full_accuracy": (
                        state.baseline_accuracy
                    ),
                    "stage_name": stage.name,
                    "stage_kind": stage.kind,
                    "layer": stage.layer,
                    "stage_order": stage.order,
                    "intervention": (
                        "addition_frequency_relocation"
                    ),
                    "replicate": replicate,
                    "local_top90_pair_count": len(
                        local_top90
                    ),
                    "local_top90_pairs": pair_text(
                        local_top90
                    ),
                    "reference_top90_pair_count": len(
                        reference_pairs
                    ),
                    "reference_top90_pairs": pair_text(
                        reference_pairs
                    ),
                    "intervention_full_accuracy": accuracy(
                        logits,
                        state.targets,
                    ),
                }
            )

    return rows


def replace_masked_coefficients(
    target_values: Tensor,
    source_values: Tensor,
    modulus: int,
    replace_mask: Tensor,
) -> Tensor:
    target_spectrum, _ = fft_values(
        target_values,
        modulus,
    )
    source_spectrum, _ = fft_values(
        source_values,
        modulus,
    )
    output = torch.where(
        replace_mask[:, :, None, None],
        source_spectrum,
        target_spectrum,
    )
    return inverse_values(
        output,
        target_values.shape,
    )


def phase_only_patch(
    target_values: Tensor,
    source_values: Tensor,
    modulus: int,
    addition_mask: Tensor,
) -> Tensor:
    target_spectrum, _ = fft_values(
        target_values,
        modulus,
    )
    source_spectrum, _ = fft_values(
        source_values,
        modulus,
    )
    output = target_spectrum.clone()

    target_coefficients = target_spectrum[
        addition_mask[:, :, None, None].expand_as(
            target_spectrum
        )
    ]
    source_coefficients = source_spectrum[
        addition_mask[:, :, None, None].expand_as(
            source_spectrum
        )
    ]

    source_magnitude = source_coefficients.abs()
    source_unit = torch.where(
        source_magnitude > 1e-20,
        source_coefficients / source_magnitude,
        torch.ones_like(source_coefficients),
    )
    changed = target_coefficients.abs() * source_unit
    output[
        addition_mask[:, :, None, None].expand_as(output)
    ] = changed

    return inverse_values(
        output,
        target_values.shape,
    )


def source_basis_scaled_patch(
    target_values: Tensor,
    source_values: Tensor,
    modulus: int,
    addition_mask: Tensor,
) -> tuple[Tensor, list[ModePair], float]:
    target_spectrum, target_power = fft_values(
        target_values,
        modulus,
    )
    source_spectrum, source_power = fft_values(
        source_values,
        modulus,
    )

    source_pairs = cumulative_pairs(
        addition_pairs(source_power),
        0.90,
    )
    source_mask = pair_mask(
        source_pairs,
        modulus,
    )

    target_addition_power = float(
        target_power[addition_mask].sum().item()
    )
    source_selected_power = float(
        source_power[source_mask].sum().item()
    )
    scale = (
        math.sqrt(
            target_addition_power
            / source_selected_power
        )
        if source_selected_power > 0.0
        else 0.0
    )

    output = target_spectrum.clone()
    output[
        addition_mask[:, :, None, None].expand_as(output)
    ] = 0.0
    output[
        source_mask[:, :, None, None].expand_as(output)
    ] = (
        source_spectrum[
            source_mask[:, :, None, None].expand_as(
                source_spectrum
            )
        ]
        * scale
    )

    return (
        inverse_values(
            output,
            target_values.shape,
        ),
        source_pairs,
        scale,
    )


def matched_patch_rows(
    target: State,
    source: State,
    direction: str,
    device: torch.device,
) -> list[dict[str, object]]:
    if len(target.stages) != len(source.stages):
        raise ValueError(
            "Matched states have different stage counts."
        )

    modulus = modulus_from_state(target)
    masks = mode_family_masks(modulus)
    addition = masks["addition"]
    nonaddition = (
        ~(masks["addition"] | masks["dc"])
    )

    rows = []

    for target_stage, source_stage in zip(
        target.stages,
        source.stages,
        strict=True,
    ):
        if (
            target_stage.name != source_stage.name
            or target_stage.values.shape
            != source_stage.values.shape
        ):
            raise ValueError(
                "Matched stages do not align: "
                f"{target_stage.name} vs {source_stage.name}"
            )

        full_patch = source_stage.values
        addition_patch = replace_masked_coefficients(
            target_stage.values,
            source_stage.values,
            modulus,
            addition,
        )
        nonaddition_patch = replace_masked_coefficients(
            target_stage.values,
            source_stage.values,
            modulus,
            nonaddition,
        )
        phase_patch = phase_only_patch(
            target_stage.values,
            source_stage.values,
            modulus,
            addition,
        )
        (
            basis_patch,
            source_basis_pairs,
            basis_scale,
        ) = source_basis_scaled_patch(
            target_stage.values,
            source_stage.values,
            modulus,
            addition,
        )

        patches = [
            (
                "identity",
                target_stage.values,
                "",
                1.0,
            ),
            (
                "full_activation_patch",
                full_patch,
                "",
                1.0,
            ),
            (
                "addition_component_patch",
                addition_patch,
                "",
                1.0,
            ),
            (
                "nonaddition_component_patch",
                nonaddition_patch,
                "",
                1.0,
            ),
            (
                "addition_phase_only_patch",
                phase_patch,
                "",
                1.0,
            ),
            (
                "source_top90_basis_scaled_patch",
                basis_patch,
                pair_text(source_basis_pairs),
                basis_scale,
            ),
        ]

        for patch_name, replacement, basis_pairs, scale in patches:
            final_hidden = target_stage.continue_fn(
                replacement.to(device)
            )

            for readout_name, readout_model in (
                ("target_readout", target.model),
                ("source_readout", source.model),
            ):
                logits = readout_model.unembedding(
                    final_hidden[:, -1, :]
                )
                rows.append(
                    {
                        "direction": direction,
                        "target_run": target.selection.run,
                        "source_run": source.selection.run,
                        "role": target.selection.role,
                        "step": target.selection.step,
                        "target_checkpoint": str(
                            target.selection.checkpoint
                        ),
                        "source_checkpoint": str(
                            source.selection.checkpoint
                        ),
                        "target_baseline_accuracy": (
                            target.baseline_accuracy
                        ),
                        "source_baseline_accuracy": (
                            source.baseline_accuracy
                        ),
                        "stage_name": target_stage.name,
                        "stage_kind": target_stage.kind,
                        "layer": target_stage.layer,
                        "stage_order": target_stage.order,
                        "patch": patch_name,
                        "readout": readout_name,
                        "source_basis_pairs": basis_pairs,
                        "basis_scale": scale,
                        "intervention_full_accuracy": accuracy(
                            logits,
                            target.targets,
                        ),
                    }
                )

    return rows


def reference_pairs_for_run(
    frame: pd.DataFrame,
    run: str,
    device: torch.device,
) -> tuple[list[ModePair], State]:
    group = frame[frame["run"] == run]
    row = group.loc[group["full_accuracy"].idxmax()]
    selection = Selection(
        study="reference",
        run=run,
        depth=int(row["depth"]),
        regime=str(row["regime"]),
        role="maximum_accuracy_reference",
        step=int(row["step"]),
        checkpoint=Path(str(row["checkpoint"])),
    )
    state = build_state(selection, device)
    modulus = modulus_from_state(state)
    _, power = final_token_fft(
        state.stages[-1].values,
        modulus,
    )
    pairs = cumulative_pairs(
        addition_pairs(power),
        0.90,
    )
    return pairs, state


def landmark_rows(
    flow_frame: pd.DataFrame,
    causal_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    rows = []

    keys = [
        "study",
        "run",
        "depth",
        "regime",
        "role",
        "step",
    ]

    for key_values, flow_group in flow_frame.groupby(keys):
        key_dict = dict(zip(keys, key_values))
        causal_group = causal_frame
        for key, value in key_dict.items():
            causal_group = causal_group[
                causal_group[key] == value
            ]

        residual_flow = flow_group[
            flow_group["stage_kind"] == "residual_state"
        ].sort_values("stage_order")
        residual_causal = causal_group[
            causal_group["stage_kind"] == "residual_state"
        ]

        emergence_candidates = residual_causal[
            (
                residual_causal["intervention"]
                == "addition_only"
            )
            & (
                residual_causal["intervention_full_accuracy"]
                >= 0.95
            )
        ].sort_values("stage_order")

        commitment_candidates = residual_causal[
            (
                residual_causal["intervention"]
                == "addition_ablation"
            )
            & (
                residual_causal["intervention_full_accuracy"]
                < 0.10
            )
        ].sort_values("stage_order")

        readout_candidates = residual_flow[
            residual_flow["direct_readout_accuracy"] >= 0.95
        ].sort_values("stage_order")

        def first_stage(frame: pd.DataFrame) -> str:
            return (
                str(frame.iloc[0]["stage_name"])
                if not frame.empty
                else ""
            )

        def first_order(frame: pd.DataFrame) -> float:
            return (
                float(frame.iloc[0]["stage_order"])
                if not frame.empty
                else float("nan")
            )

        rows.append(
            {
                **key_dict,
                "emergence_stage": first_stage(
                    emergence_candidates
                ),
                "emergence_stage_order": first_order(
                    emergence_candidates
                ),
                "commitment_stage": first_stage(
                    commitment_candidates
                ),
                "commitment_stage_order": first_order(
                    commitment_candidates
                ),
                "readout_ready_stage": first_stage(
                    readout_candidates
                ),
                "readout_ready_stage_order": first_order(
                    readout_candidates
                ),
            }
        )

    return rows


def write_html(
    path: Path,
    flow: pd.DataFrame,
    causal: pd.DataFrame,
    matched_flow: pd.DataFrame,
    patching: pd.DataFrame,
    landmarks: pd.DataFrame,
) -> None:
    try:
        import plotly.express as px
    except ImportError as error:
        raise RuntimeError(
            "Plotly is required for the HTML report."
        ) from error

    figures = []

    residual_flow = flow[
        flow["stage_kind"] == "residual_state"
    ]
    figures.append(
        px.line(
            residual_flow,
            x="stage_order",
            y=(
                "final_token_addition_"
                "power_fraction_non_dc"
            ),
            color="run",
            facet_col="role",
            facet_col_wrap=2,
            markers=True,
            hover_data=["stage_name", "depth"],
            title=(
                "Addition-family structure through "
                "the residual stream"
            ),
        )
    )

    figures.append(
        px.line(
            residual_flow,
            x="stage_order",
            y="direct_readout_accuracy",
            color="run",
            facet_col="role",
            facet_col_wrap=2,
            markers=True,
            hover_data=["stage_name", "depth"],
            title="Direct readout through the model",
        )
    )

    core_causal = causal[
        causal["intervention"].isin(
            [
                "addition_only",
                "addition_ablation",
                "remove_all_nonaddition",
                "nonaddition_only",
            ]
        )
    ]
    figures.append(
        px.line(
            core_causal,
            x="stage_order",
            y="intervention_full_accuracy",
            color="intervention",
            facet_row="run",
            facet_col="role",
            markers=True,
            hover_data=["stage_name", "stage_kind"],
            title=(
                "Layerwise causal continuation after "
                "Fourier filtering"
            ),
        )
    )

    write_causal = causal[
        (
            causal["stage_kind"].isin(
                [
                    "attention_write",
                    "mlp_activation",
                    "mlp_write",
                ]
            )
        )
        & (
            causal["intervention"].isin(
                [
                    "zero_entire_component",
                    "addition_only",
                    "addition_ablation",
                    "remove_all_nonaddition",
                ]
            )
        )
    ]
    figures.append(
        px.line(
            write_causal,
            x="layer",
            y="intervention_full_accuracy",
            color="intervention",
            facet_row="run",
            facet_col="stage_kind",
            markers=True,
            hover_data=["stage_name", "role"],
            title=(
                "Which attention and MLP components "
                "write the functional circuit"
            ),
        )
    )

    matched_selected = patching[
        patching["readout"] == "target_readout"
    ]
    figures.append(
        px.line(
            matched_selected,
            x="stage_order",
            y="intervention_full_accuracy",
            color="patch",
            facet_row="direction",
            facet_col="role",
            markers=True,
            hover_data=["stage_name", "step"],
            title=(
                "Matched control/freeze internal "
                "activation patching"
            ),
        )
    )

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Layerwise Fourier mechanism</title>",
        "</head><body>",
        "<h1>Layerwise Fourier mechanism</h1>",
        (
            "<p>This study tracks exact Fourier families, modes, "
            "complex coefficients, direct readout, internal causal "
            "filtering, phase/frequency interventions, and matched "
            "control/freeze activation patching.</p>"
        ),
    ]

    for index, figure in enumerate(figures):
        html_parts.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=(
                    "inline" if index == 0 else False
                ),
            )
        )

    for title, frame in (
        ("Main flow", flow),
        ("Main causal interventions", causal),
        ("Matched flow", matched_flow),
        ("Matched patching", patching),
        ("Circuit landmarks", landmarks),
    ):
        html_parts.append(f"<h2>{title}</h2>")
        html_parts.append(frame.to_html(index=False))

    html_parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(html_parts),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_arguments()

    if args.random_replicates < 1:
        raise ValueError(
            "--random-replicates must be positive."
        )
    if not args.model_summary.is_file():
        raise FileNotFoundError(
            f"Missing model summary: {args.model_summary}"
        )
    if not args.matched_paired_summary.is_file():
        raise FileNotFoundError(
            "Missing matched paired summary: "
            f"{args.matched_paired_summary}"
        )

    device = get_device(args.device)
    model_frame = pd.read_csv(args.model_summary)
    matched_frame = pd.read_csv(
        args.matched_paired_summary
    )

    main_selections = main_panel(model_frame)
    matched_steps = matched_panel(matched_frame)

    selection_rows = [
        {
            "study": selection.study,
            "run": selection.run,
            "depth": selection.depth,
            "regime": selection.regime,
            "role": selection.role,
            "step": selection.step,
            "checkpoint": str(selection.checkpoint),
        }
        for selection in main_selections
    ]

    for step in matched_steps:
        row = matched_frame[
            matched_frame["step"] == step
        ].iloc[0]
        role = matched_role(step, matched_steps)
        for branch in ("control", "freeze"):
            selection_rows.append(
                {
                    "study": "matched",
                    "run": f"depth4_matched_{branch}",
                    "depth": 4,
                    "regime": branch,
                    "role": role,
                    "step": step,
                    "checkpoint": str(
                        row[f"{branch}_checkpoint"]
                    ),
                }
            )

    write_csv(args.selection_output, selection_rows)

    print(f"Device: {device}")
    print(
        f"Main selections: {len(main_selections)}"
    )
    print(
        f"Matched steps: {len(matched_steps)}"
    )

    flow_rows_main = []
    causal_rows_main = []

    selections_by_run: dict[str, list[Selection]] = {}
    for selection in main_selections:
        selections_by_run.setdefault(
            selection.run,
            [],
        ).append(selection)

    for run, selections in selections_by_run.items():
        print()
        print("=" * 80)
        print(f"Reference circuit for {run}")
        print("=" * 80)

        reference_pairs, reference_state = (
            reference_pairs_for_run(
                model_frame,
                run,
                device,
            )
        )

        for selection in selections:
            if (
                selection.step
                == reference_state.selection.step
                and selection.checkpoint
                == reference_state.selection.checkpoint
            ):
                state = reference_state
                state.selection = selection
            else:
                state = build_state(selection, device)

            print(
                f"Processing {selection.run} "
                f"{selection.role} step {selection.step}"
            )

            flow_rows_main.extend(
                flow_rows(
                    state,
                    reference_pairs,
                )
            )
            causal_rows_main.extend(
                causal_rows(
                    state,
                    reference_pairs,
                    device,
                    random_replicates=(
                        args.random_replicates
                    ),
                    base_seed=args.seed,
                )
            )

            if state is not reference_state:
                del state.model
                release_accelerator_cache(device)

        del reference_state.model
        release_accelerator_cache(device)

    write_csv(args.flow_output, flow_rows_main)
    write_csv(args.causal_output, causal_rows_main)

    matched_flow_rows = []
    patch_rows = []

    for step in matched_steps:
        row = matched_frame[
            matched_frame["step"] == step
        ].iloc[0]
        role = matched_role(step, matched_steps)

        control_selection = Selection(
            study="matched",
            run="depth4_matched_control",
            depth=4,
            regime="control",
            role=role,
            step=step,
            checkpoint=Path(
                str(row["control_checkpoint"])
            ),
        )
        freeze_selection = Selection(
            study="matched",
            run="depth4_matched_freeze",
            depth=4,
            regime="freeze",
            role=role,
            step=step,
            checkpoint=Path(
                str(row["freeze_checkpoint"])
            ),
        )

        control = build_state(
            control_selection,
            device,
        )
        freeze = build_state(
            freeze_selection,
            device,
        )

        control_modulus = modulus_from_state(control)
        _, control_final_power = final_token_fft(
            control.stages[-1].values,
            control_modulus,
        )
        control_pairs = cumulative_pairs(
            addition_pairs(control_final_power),
            0.90,
        )

        freeze_modulus = modulus_from_state(freeze)
        _, freeze_final_power = final_token_fft(
            freeze.stages[-1].values,
            freeze_modulus,
        )
        freeze_pairs = cumulative_pairs(
            addition_pairs(freeze_final_power),
            0.90,
        )

        matched_flow_rows.extend(
            flow_rows(control, control_pairs)
        )
        matched_flow_rows.extend(
            flow_rows(freeze, freeze_pairs)
        )

        patch_rows.extend(
            matched_patch_rows(
                target=control,
                source=freeze,
                direction="freeze_into_control",
                device=device,
            )
        )
        patch_rows.extend(
            matched_patch_rows(
                target=freeze,
                source=control,
                direction="control_into_freeze",
                device=device,
            )
        )

        del control.model
        del freeze.model
        release_accelerator_cache(device)

    write_csv(
        args.matched_flow_output,
        matched_flow_rows,
    )
    write_csv(
        args.matched_patching_output,
        patch_rows,
    )

    flow_frame = pd.DataFrame(
        flow_rows_main
    ).sort_values(
        ["depth", "run", "step", "stage_order"]
    )
    causal_frame = pd.DataFrame(
        causal_rows_main
    ).sort_values(
        [
            "depth",
            "run",
            "step",
            "stage_order",
            "intervention",
            "replicate",
        ]
    )
    matched_flow_frame = pd.DataFrame(
        matched_flow_rows
    ).sort_values(
        ["run", "step", "stage_order"]
    )
    patch_frame = pd.DataFrame(
        patch_rows
    ).sort_values(
        [
            "step",
            "direction",
            "stage_order",
            "patch",
            "readout",
        ]
    )

    landmarks = landmark_rows(
        flow_frame,
        causal_frame,
    )
    write_csv(
        args.landmark_output,
        landmarks,
    )
    landmark_frame = pd.DataFrame(
        landmarks
    ).sort_values(
        ["depth", "run", "step"]
    )

    write_html(
        args.html_output,
        flow_frame,
        causal_frame,
        matched_flow_frame,
        patch_frame,
        landmark_frame,
    )

    print()
    print(
        "Layerwise Fourier mechanism study complete."
    )
    print(f"Saved flow: {args.flow_output}")
    print(f"Saved causal results: {args.causal_output}")
    print(
        f"Saved matched flow: "
        f"{args.matched_flow_output}"
    )
    print(
        f"Saved matched patching: "
        f"{args.matched_patching_output}"
    )
    print(
        f"Saved landmarks: "
        f"{args.landmark_output}"
    )
    print(
        f"Saved selection: "
        f"{args.selection_output}"
    )
    print(
        f"Saved HTML: "
        f"{args.html_output}"
    )


if __name__ == "__main__":
    main()
