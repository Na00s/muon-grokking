from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

try:
    from depth_fourier_mode_identification import (
        MODE_FAMILIES,
        accuracy,
        build_model,
        forward_with_layer_cache,
        get_device,
        load_checkpoint,
        mode_family,
        mode_family_masks,
        ordered_full_grid,
        release_accelerator_cache,
        two_dimensional_spectrum,
    )
except ImportError as error:
    raise ImportError(
        "This script requires "
        "experiments/depth/depth_fourier_mode_identification.py "
        "from the preceding identification phase."
    ) from error


NON_DC_FAMILIES = (
    "addition",
    "subtraction",
    "a_only",
    "b_only",
    "generic_interaction",
)

DEFAULT_THRESHOLDS = (
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
)


@dataclass(frozen=True)
class MainCheckpoint:
    run: str
    depth: int
    regime: str
    role: str
    step: int
    checkpoint: Path


@dataclass
class CheckpointState:
    run: str
    depth: int
    regime: str
    role: str
    step: int
    checkpoint: Path
    model: object
    targets: Tensor
    hidden_grid: Tensor
    spectrum: Tensor
    power: Tensor
    baseline_accuracy: float


@dataclass(frozen=True)
class ModePair:
    representative: tuple[int, int]
    conjugate: tuple[int, int]
    family: str
    power: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test Fourier-circuit sufficiency, necessity, frequency "
            "identity, phase sensitivity, and hidden/readout alignment "
            "after the observational mode-identification phase."
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
        "--operation",
        choices=["addition", "subtraction"],
        default="addition",
        help=(
            "Modular operation the checkpoints were trained on. "
            "This selects the labels the interventions are scored "
            "against; it does not affect the mode-family partition."
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
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--family-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_family_interventions_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--threshold-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_addition_thresholds_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--frequency-control-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_frequency_controls_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--phase-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_phase_controls_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--cross-readout-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_cross_readout_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--matched-family-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_family_interventions_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--matched-threshold-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_addition_thresholds_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--matched-frequency-control-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_frequency_controls_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--matched-phase-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_phase_controls_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--matched-cross-readout-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_cross_readout_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_hypothesis_test_selection_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_hypothesis_tests_seed_0.html"
        ),
    )
    return parser.parse_args()


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def canonical_pair(
    k: int,
    l: int,
    modulus: int,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
]:
    original = (k, l)
    conjugate = (
        (-k) % modulus,
        (-l) % modulus,
    )
    return (
        min(original, conjugate),
        max(original, conjugate),
    )


def all_mode_pairs(
    power: Tensor,
) -> list[ModePair]:
    modulus = int(power.shape[0])
    seen: set[tuple[int, int]] = set()
    pairs = []

    for k in range(modulus):
        for l in range(modulus):
            representative, conjugate = (
                canonical_pair(
                    k,
                    l,
                    modulus,
                )
            )
            if representative in seen:
                continue
            seen.add(representative)

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

            pairs.append(
                ModePair(
                    representative=representative,
                    conjugate=conjugate,
                    family=mode_family(
                        representative[0],
                        representative[1],
                        modulus,
                    ),
                    power=pair_power,
                )
            )

    pairs.sort(
        key=lambda pair: pair.power,
        reverse=True,
    )
    return pairs


def family_pairs(
    pairs: list[ModePair],
    family: str,
) -> list[ModePair]:
    return [
        pair
        for pair in pairs
        if pair.family == family
    ]


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


def dc_mask(modulus: int) -> Tensor:
    mask = torch.zeros(
        modulus,
        modulus,
        dtype=torch.bool,
    )
    mask[0, 0] = True
    return mask


def reconstruct(
    spectrum: Tensor,
    keep_mask: Tensor,
) -> Tensor:
    filtered = torch.where(
        keep_mask.unsqueeze(-1),
        spectrum,
        torch.zeros_like(spectrum),
    )
    return torch.fft.ifft2(
        filtered,
        dim=(0, 1),
        norm="ortho",
    ).real


def readout_logits(
    hidden_grid: Tensor,
    readout_model,
) -> Tensor:
    device = (
        readout_model.unembedding.weight.device
    )
    flat = hidden_grid.reshape(
        hidden_grid.shape[0]
        * hidden_grid.shape[1],
        -1,
    ).to(device)
    return F.linear(
        flat,
        readout_model.unembedding.weight,
    )


def intervention_accuracy(
    hidden_grid: Tensor,
    targets: Tensor,
    readout_model,
) -> float:
    return accuracy(
        readout_logits(
            hidden_grid,
            readout_model,
        ),
        targets,
    )


def spectrum_accuracy(
    spectrum: Tensor,
    keep_mask: Tensor,
    targets: Tensor,
    readout_model,
) -> float:
    return intervention_accuracy(
        reconstruct(
            spectrum,
            keep_mask,
        ),
        targets,
        readout_model,
    )


def pair_list_text(
    pairs: Iterable[ModePair],
) -> str:
    return "|".join(
        (
            f"({pair.representative[0]},"
            f"{pair.representative[1]})/"
            f"({pair.conjugate[0]},"
            f"{pair.conjugate[1]})"
        )
        for pair in pairs
    )


def selected_by_cumulative_power(
    pairs: list[ModePair],
    threshold: float,
) -> tuple[list[ModePair], float]:
    if not pairs:
        return [], 0.0

    total = sum(pair.power for pair in pairs)
    if total <= 0.0:
        return [], 0.0

    selected = []
    accumulated = 0.0

    for pair in pairs:
        selected.append(pair)
        accumulated += pair.power
        if (
            accumulated / total >= threshold
            or len(selected) == len(pairs)
        ):
            break

    return selected, accumulated / total


def scaled_spectrum_for_pairs(
    spectrum: Tensor,
    selected_pairs: list[ModePair],
    target_power: float,
    include_dc: bool,
) -> tuple[Tensor, float]:
    modulus = int(spectrum.shape[0])
    selected_mask = pair_mask(
        selected_pairs,
        modulus,
    )
    output = torch.zeros_like(spectrum)

    if include_dc:
        output[0, 0, :] = spectrum[0, 0, :]

    source_power = sum(
        pair.power
        for pair in selected_pairs
    )
    if source_power <= 0.0:
        return output, 0.0

    scale = math.sqrt(
        target_power / source_power
    )
    output[selected_mask] = (
        spectrum[selected_mask] * scale
    )
    return output, scale


def random_derangement(
    size: int,
    rng: random.Random,
) -> list[int]:
    if size <= 1:
        return list(range(size))

    indices = list(range(size))
    for _ in range(1_000):
        permutation = indices.copy()
        rng.shuffle(permutation)
        if all(
            index != permutation[index]
            for index in indices
        ):
            return permutation

    shift = rng.randrange(1, size)
    return [
        (index + shift) % size
        for index in indices
    ]


def relocate_pair_coefficients(
    spectrum: Tensor,
    source_pairs: list[ModePair],
    all_target_pairs: list[ModePair],
    rng: random.Random,
    context: str,
) -> tuple[Tensor, list[ModePair]]:
    if context not in {
        "addition_only",
        "full_hidden",
    }:
        raise ValueError(
            f"Unknown relocation context: {context}"
        )

    if context == "full_hidden":
        output = spectrum.clone()
        source_mask = pair_mask(
            source_pairs,
            spectrum.shape[0],
        )
        output[source_mask] = 0.0
    else:
        output = torch.zeros_like(spectrum)
        output[0, 0, :] = spectrum[0, 0, :]

    target_count = len(all_target_pairs)
    permutation = random_derangement(
        target_count,
        rng,
    )
    source_index = {
        pair.representative: index
        for index, pair in enumerate(
            all_target_pairs
        )
    }

    mapped_targets = []

    for source in source_pairs:
        index = source_index[
            source.representative
        ]
        target = all_target_pairs[
            permutation[index]
        ]
        mapped_targets.append(target)

        coefficient = spectrum[
            source.representative[0],
            source.representative[1],
            :,
        ]

        output[
            target.representative[0],
            target.representative[1],
            :,
        ] = coefficient
        output[
            target.conjugate[0],
            target.conjugate[1],
            :,
        ] = coefficient.conj()

    return output, mapped_targets


def phase_scramble(
    spectrum: Tensor,
    selected_pairs: list[ModePair],
    rng: random.Random,
    context: str,
    phase_type: str,
) -> Tensor:
    if context not in {
        "addition_only",
        "full_hidden",
    }:
        raise ValueError(
            f"Unknown phase context: {context}"
        )
    if phase_type not in {
        "pair_global",
        "channelwise",
    }:
        raise ValueError(
            f"Unknown phase type: {phase_type}"
        )

    if context == "full_hidden":
        output = spectrum.clone()
    else:
        output = torch.zeros_like(spectrum)
        output[0, 0, :] = spectrum[0, 0, :]

    width = int(spectrum.shape[-1])

    for pair in selected_pairs:
        coefficient = spectrum[
            pair.representative[0],
            pair.representative[1],
            :,
        ]

        if phase_type == "pair_global":
            angle = rng.uniform(
                0.0,
                2.0 * math.pi,
            )
            phase = torch.polar(
                torch.tensor(
                    1.0,
                    dtype=coefficient.real.dtype,
                ),
                torch.tensor(
                    angle,
                    dtype=coefficient.real.dtype,
                ),
            )
        else:
            angles = torch.tensor(
                [
                    rng.uniform(
                        0.0,
                        2.0 * math.pi,
                    )
                    for _ in range(width)
                ],
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
        ] = changed
        output[
            pair.conjugate[0],
            pair.conjugate[1],
            :,
        ] = changed.conj()

    return output


def full_grid_from_checkpoint(
    run: str,
    depth: int,
    regime: str,
    role: str,
    step: int,
    checkpoint_path: Path,
    device: torch.device,
    operation: str = "addition",
) -> CheckpointState:
    checkpoint = load_checkpoint(
        checkpoint_path
    )
    model = build_model(
        checkpoint,
        expected_depth=depth,
        device=device,
    )
    modulus = int(
        checkpoint["model_config"]["modulus"]
    )
    inputs, targets = ordered_full_grid(
        modulus,
        device,
        operation=operation,
    )

    with torch.no_grad():
        logits, stages = forward_with_layer_cache(
            model,
            inputs,
        )

    final_stage_name = (
        f"layer_{depth}_post_block_residual"
    )
    final_stage = next(
        stage
        for stage in stages
        if stage.stage_name == final_stage_name
    )

    hidden_grid = (
        final_stage.values.reshape(
            modulus,
            modulus,
            -1,
        )
        .detach()
        .float()
        .cpu()
    )
    spectrum, power = two_dimensional_spectrum(
        hidden_grid
    )

    state = CheckpointState(
        run=run,
        depth=depth,
        regime=regime,
        role=role,
        step=step,
        checkpoint=checkpoint_path,
        model=model,
        targets=targets,
        hidden_grid=hidden_grid,
        spectrum=spectrum,
        power=power,
        baseline_accuracy=accuracy(
            logits,
            targets,
        ),
    )

    del logits
    del stages
    return state


def main_panel(
    model_frame: pd.DataFrame,
) -> list[MainCheckpoint]:
    checkpoints: list[MainCheckpoint] = []

    for (
        run,
        depth,
        regime,
    ), group in model_frame.groupby(
        ["run", "depth", "regime"],
        sort=False,
    ):
        group = group.sort_values("step")
        roles: list[
            tuple[str, pd.Series]
        ] = []

        generalized = group[
            group["full_accuracy"] >= 0.95
        ]

        if not generalized.empty:
            first = generalized.iloc[0]
            roles.append(
                ("first_generalized", first)
            )

            after_first = group[
                group["step"] >= first["step"]
            ]
            minimum = after_first.loc[
                after_first[
                    "full_accuracy"
                ].idxmin()
            ]
            roles.append(
                (
                    "minimum_after_first_generalized",
                    minimum,
                )
            )
        else:
            best = group.loc[
                group["full_accuracy"].idxmax()
            ]
            after_best = group[
                group["step"] >= best["step"]
            ]
            minimum = after_best.loc[
                after_best[
                    "full_accuracy"
                ].idxmin()
            ]
            roles.append(
                ("best_available", best)
            )
            roles.append(
                ("minimum_after_best", minimum)
            )

        best = group.loc[
            group["full_accuracy"].idxmax()
        ]
        final = group.iloc[-1]

        roles.append(("maximum_accuracy", best))
        roles.append(("final", final))

        seen_steps: set[int] = set()
        for role, row in roles:
            step = int(row["step"])
            if step in seen_steps:
                continue
            seen_steps.add(step)
            checkpoints.append(
                MainCheckpoint(
                    run=str(run),
                    depth=int(depth),
                    regime=str(regime),
                    role=role,
                    step=step,
                    checkpoint=Path(
                        str(row["checkpoint"])
                    ),
                )
            )

    return checkpoints


def matched_panel(
    paired_frame: pd.DataFrame,
) -> list[int]:
    paired_frame = paired_frame.sort_values(
        "step"
    )

    branch_step = int(
        paired_frame.iloc[0]["step"]
    )
    final_step = int(
        paired_frame.iloc[-1]["step"]
    )

    below_ten = paired_frame[
        paired_frame[
            "control_full_accuracy"
        ] < 0.10
    ]
    first_below_ten = (
        int(below_ten.iloc[0]["step"])
        if not below_ten.empty
        else int(
            paired_frame.loc[
                paired_frame[
                    "control_full_accuracy"
                ].idxmin(),
                "step",
            ]
        )
    )

    minimum_step = int(
        paired_frame.loc[
            paired_frame[
                "control_full_accuracy"
            ].idxmin(),
            "step",
        ]
    )

    after_minimum = paired_frame[
        paired_frame["step"] > minimum_step
    ]
    if after_minimum.empty:
        recovery_step = final_step
    else:
        recovery_step = int(
            after_minimum.loc[
                after_minimum[
                    "control_full_accuracy"
                ].idxmax(),
                "step",
            ]
        )

    return list(
        dict.fromkeys(
            [
                branch_step,
                first_below_ten,
                minimum_step,
                recovery_step,
                final_step,
            ]
        )
    )


def state_metadata(
    state: CheckpointState,
) -> dict[str, object]:
    return {
        "run": state.run,
        "depth": state.depth,
        "regime": state.regime,
        "role": state.role,
        "step": state.step,
        "checkpoint": str(
            state.checkpoint
        ),
        "baseline_full_accuracy": (
            state.baseline_accuracy
        ),
    }


def family_interventions(
    state: CheckpointState,
) -> list[dict[str, object]]:
    modulus = int(state.spectrum.shape[0])
    masks = mode_family_masks(modulus)
    dc = dc_mask(modulus)
    full = torch.ones(
        modulus,
        modulus,
        dtype=torch.bool,
    )
    rows = []

    for family in NON_DC_FAMILIES:
        family_mask = masks[family]

        sufficiency_mask = (
            dc | family_mask
        )
        ablation_mask = (
            full & ~family_mask
        )

        rows.extend(
            [
                {
                    **state_metadata(state),
                    "family": family,
                    "intervention": "sufficiency",
                    "retained_pair_count": int(
                        family_mask.sum().item()
                        // 2
                    ),
                    "intervention_accuracy": (
                        spectrum_accuracy(
                            state.spectrum,
                            sufficiency_mask,
                            state.targets,
                            state.model,
                        )
                    ),
                },
                {
                    **state_metadata(state),
                    "family": family,
                    "intervention": "ablation",
                    "retained_pair_count": int(
                        (
                            ablation_mask.sum()
                            - 1
                        ).item()
                        // 2
                    ),
                    "intervention_accuracy": (
                        spectrum_accuracy(
                            state.spectrum,
                            ablation_mask,
                            state.targets,
                            state.model,
                        )
                    ),
                },
            ]
        )

    non_addition_mask = (
        full & ~masks["addition"]
    )
    rows.append(
        {
            **state_metadata(state),
            "family": "all_non_addition",
            "intervention": "sufficiency",
            "retained_pair_count": int(
                (
                    non_addition_mask.sum()
                    - 1
                ).item()
                // 2
            ),
            "intervention_accuracy": (
                spectrum_accuracy(
                    state.spectrum,
                    non_addition_mask,
                    state.targets,
                    state.model,
                )
            ),
        }
    )

    rows.append(
        {
            **state_metadata(state),
            "family": "dc_only",
            "intervention": "sufficiency",
            "retained_pair_count": 0,
            "intervention_accuracy": (
                spectrum_accuracy(
                    state.spectrum,
                    dc,
                    state.targets,
                    state.model,
                )
            ),
        }
    )

    return rows


def addition_threshold_interventions(
    state: CheckpointState,
) -> tuple[
    list[dict[str, object]],
    dict[float, list[ModePair]],
]:
    pairs = all_mode_pairs(
        state.power
    )
    addition_pairs = family_pairs(
        pairs,
        "addition",
    )
    modulus = int(state.spectrum.shape[0])
    dc = dc_mask(modulus)
    full = torch.ones(
        modulus,
        modulus,
        dtype=torch.bool,
    )
    addition_mask = pair_mask(
        addition_pairs,
        modulus,
    )

    rows = []
    selected_by_threshold: dict[
        float,
        list[ModePair],
    ] = {}

    for threshold in DEFAULT_THRESHOLDS:
        selected, achieved = (
            selected_by_cumulative_power(
                addition_pairs,
                threshold,
            )
        )
        selected_by_threshold[
            threshold
        ] = selected

        selected_mask = pair_mask(
            selected,
            modulus,
        )
        top_sufficiency = (
            dc | selected_mask
        )
        top_ablation = (
            full & ~selected_mask
        )
        tail_sufficiency = (
            dc
            | (
                addition_mask
                & ~selected_mask
            )
        )

        common = {
            **state_metadata(state),
            "requested_addition_power_threshold": (
                threshold
            ),
            "achieved_addition_power_fraction": (
                achieved
            ),
            "selected_pair_count": len(
                selected
            ),
            "selected_pairs": pair_list_text(
                selected
            ),
        }

        rows.extend(
            [
                {
                    **common,
                    "intervention": (
                        "top_addition_sufficiency"
                    ),
                    "intervention_accuracy": (
                        spectrum_accuracy(
                            state.spectrum,
                            top_sufficiency,
                            state.targets,
                            state.model,
                        )
                    ),
                },
                {
                    **common,
                    "intervention": (
                        "top_addition_ablation"
                    ),
                    "intervention_accuracy": (
                        spectrum_accuracy(
                            state.spectrum,
                            top_ablation,
                            state.targets,
                            state.model,
                        )
                    ),
                },
                {
                    **common,
                    "intervention": (
                        "remaining_addition_sufficiency"
                    ),
                    "intervention_accuracy": (
                        spectrum_accuracy(
                            state.spectrum,
                            tail_sufficiency,
                            state.targets,
                            state.model,
                        )
                    ),
                },
            ]
        )

    return rows, selected_by_threshold


def frequency_identity_controls(
    state: CheckpointState,
    selected_by_threshold: dict[
        float,
        list[ModePair],
    ],
    replicates: int,
    base_seed: int,
) -> list[dict[str, object]]:
    pairs = all_mode_pairs(
        state.power
    )
    addition_pairs = family_pairs(
        pairs,
        "addition",
    )
    modulus = int(state.spectrum.shape[0])
    dc = dc_mask(modulus)
    rows = []

    # Native random subsets and equal-power relocation are applied
    # at partial cumulative-power thresholds. Relocation is evaluated
    # in isolation so that overwriting an unrelated native coefficient
    # cannot change the stated power control.
    for threshold in (
        0.50,
        0.75,
        0.90,
    ):
        selected = selected_by_threshold[
            threshold
        ]
        target_power = sum(
            pair.power
            for pair in selected
        )
        pair_count = len(selected)

        for replicate in range(replicates):
            rng = random.Random(
                (
                    base_seed
                    + 1_000_003
                    * state.depth
                    + 10_007
                    * state.step
                    + 101
                    * replicate
                    + int(
                        10_000 * threshold
                    )
                )
            )

            sampled_native = rng.sample(
                addition_pairs,
                pair_count,
            )
            native_mask = (
                dc
                | pair_mask(
                    sampled_native,
                    modulus,
                )
            )
            sampled_power = sum(
                pair.power
                for pair in sampled_native
            )

            rows.append(
                {
                    **state_metadata(state),
                    "requested_addition_power_threshold": (
                        threshold
                    ),
                    "control": (
                        "random_native_addition_subset"
                    ),
                    "replicate": replicate,
                    "selected_pair_count": (
                        pair_count
                    ),
                    "source_pairs": (
                        pair_list_text(selected)
                    ),
                    "control_pairs": (
                        pair_list_text(
                            sampled_native
                        )
                    ),
                    "target_selected_power": (
                        target_power
                    ),
                    "control_native_power": (
                        sampled_power
                    ),
                    "scale_factor": 1.0,
                    "context": "addition_only",
                    "intervention_accuracy": (
                        spectrum_accuracy(
                            state.spectrum,
                            native_mask,
                            state.targets,
                            state.model,
                        )
                    ),
                }
            )

            relocated, targets = (
                relocate_pair_coefficients(
                    state.spectrum,
                    selected,
                    addition_pairs,
                    rng,
                    context="addition_only",
                )
            )
            rows.append(
                {
                    **state_metadata(state),
                    "requested_addition_power_threshold": (
                        threshold
                    ),
                    "control": (
                        "equal_power_frequency_relocation"
                    ),
                    "replicate": replicate,
                    "selected_pair_count": (
                        pair_count
                    ),
                    "source_pairs": (
                        pair_list_text(
                            selected
                        )
                    ),
                    "control_pairs": (
                        pair_list_text(
                            targets
                        )
                    ),
                    "target_selected_power": (
                        target_power
                    ),
                    "control_native_power": (
                        target_power
                    ),
                    "scale_factor": 1.0,
                    "context": "addition_only",
                    "intervention_accuracy": (
                        intervention_accuracy(
                            torch.fft.ifft2(
                                relocated,
                                dim=(0, 1),
                                norm="ortho",
                            ).real,
                            state.targets,
                            state.model,
                        )
                    ),
                }
            )

        # The 90%-power top addition set receives equal-power,
        # equal-count controls from every alternative mode family.
        if threshold == 0.90:
            for comparison_family in (
                "a_only",
                "b_only",
                "subtraction",
                "generic_interaction",
            ):
                candidates = family_pairs(
                    pairs,
                    comparison_family,
                )[:pair_count]
                scaled, scale = (
                    scaled_spectrum_for_pairs(
                        state.spectrum,
                        candidates,
                        target_power=target_power,
                        include_dc=True,
                    )
                )
                rows.append(
                    {
                        **state_metadata(state),
                        "requested_addition_power_threshold": (
                            threshold
                        ),
                        "control": (
                            "equal_power_other_family"
                        ),
                        "replicate": -1,
                        "selected_pair_count": (
                            pair_count
                        ),
                        "source_pairs": (
                            pair_list_text(selected)
                        ),
                        "control_pairs": (
                            pair_list_text(
                                candidates
                            )
                        ),
                        "target_selected_power": (
                            target_power
                        ),
                        "control_native_power": (
                            sum(
                                pair.power
                                for pair in candidates
                            )
                        ),
                        "scale_factor": scale,
                        "context": (
                            f"{comparison_family}_only"
                        ),
                        "intervention_accuracy": (
                            intervention_accuracy(
                                torch.fft.ifft2(
                                    scaled,
                                    dim=(0, 1),
                                    norm="ortho",
                                ).real,
                                state.targets,
                                state.model,
                            )
                        ),
                    }
                )

    # A derangement of the complete addition family preserves every
    # addition coefficient vector and total addition power exactly.
    # In the full-hidden context it also preserves every non-addition
    # coefficient exactly, isolating frequency identity.
    selected = selected_by_threshold[
        1.00
    ]
    target_power = sum(
        pair.power
        for pair in selected
    )

    for replicate in range(replicates):
        rng = random.Random(
            (
                base_seed
                + 3_000_017
                * state.depth
                + 30_013
                * state.step
                + 307
                * replicate
            )
        )

        for context in (
            "addition_only",
            "full_hidden",
        ):
            relocated, targets = (
                relocate_pair_coefficients(
                    state.spectrum,
                    selected,
                    addition_pairs,
                    rng,
                    context=context,
                )
            )
            rows.append(
                {
                    **state_metadata(state),
                    "requested_addition_power_threshold": (
                        1.00
                    ),
                    "control": (
                        "complete_addition_frequency_derangement"
                    ),
                    "replicate": replicate,
                    "selected_pair_count": len(
                        selected
                    ),
                    "source_pairs": (
                        pair_list_text(selected)
                    ),
                    "control_pairs": (
                        pair_list_text(targets)
                    ),
                    "target_selected_power": (
                        target_power
                    ),
                    "control_native_power": (
                        target_power
                    ),
                    "scale_factor": 1.0,
                    "context": context,
                    "intervention_accuracy": (
                        intervention_accuracy(
                            torch.fft.ifft2(
                                relocated,
                                dim=(0, 1),
                                norm="ortho",
                            ).real,
                            state.targets,
                            state.model,
                        )
                    ),
                }
            )

    return rows

def phase_controls(
    state: CheckpointState,
    selected_by_threshold: dict[
        float,
        list[ModePair],
    ],
    replicates: int,
    base_seed: int,
) -> list[dict[str, object]]:
    rows = []

    for threshold in (
        0.90,
        1.00,
    ):
        selected = selected_by_threshold[
            threshold
        ]

        for phase_type in (
            "pair_global",
            "channelwise",
        ):
            for context in (
                "addition_only",
                "full_hidden",
            ):
                for replicate in range(
                    replicates
                ):
                    rng = random.Random(
                        (
                            base_seed
                            + 2_000_003
                            * state.depth
                            + 20_011
                            * state.step
                            + 211
                            * replicate
                            + int(
                                10_000
                                * threshold
                            )
                            + (
                                1
                                if phase_type
                                == "channelwise"
                                else 0
                            )
                            + (
                                17
                                if context
                                == "full_hidden"
                                else 0
                            )
                        )
                    )
                    changed = phase_scramble(
                        state.spectrum,
                        selected,
                        rng,
                        context=context,
                        phase_type=phase_type,
                    )
                    hidden = torch.fft.ifft2(
                        changed,
                        dim=(0, 1),
                        norm="ortho",
                    ).real
                    rows.append(
                        {
                            **state_metadata(state),
                            "requested_addition_power_threshold": (
                                threshold
                            ),
                            "phase_type": (
                                phase_type
                            ),
                            "context": context,
                            "replicate": (
                                replicate
                            ),
                            "selected_pair_count": (
                                len(selected)
                            ),
                            "selected_pairs": (
                                pair_list_text(
                                    selected
                                )
                            ),
                            "intervention_accuracy": (
                                intervention_accuracy(
                                    hidden,
                                    state.targets,
                                    state.model,
                                )
                            ),
                        }
                    )

    return rows


def addition_only_hidden(
    state: CheckpointState,
) -> Tensor:
    modulus = int(
        state.spectrum.shape[0]
    )
    masks = mode_family_masks(
        modulus
    )
    keep = (
        dc_mask(modulus)
        | masks["addition"]
    )
    return reconstruct(
        state.spectrum,
        keep,
    )


def cosine_similarity(
    first: Tensor,
    second: Tensor,
) -> float:
    first = first.reshape(-1).float()
    second = second.reshape(-1).float()
    denominator = (
        torch.linalg.vector_norm(first)
        * torch.linalg.vector_norm(second)
    )
    if float(
        denominator.item()
    ) <= 0.0:
        return 0.0
    return float(
        (
            torch.dot(first, second)
            / denominator
        ).item()
    )


def cross_readout_rows(
    target: CheckpointState,
    reference: CheckpointState,
    comparison_type: str,
) -> list[dict[str, object]]:
    target_addition = (
        addition_only_hidden(target)
    )
    reference_addition = (
        addition_only_hidden(reference)
    )

    hidden_sources = {
        "target_full": target.hidden_grid,
        "reference_full": (
            reference.hidden_grid
        ),
        "target_addition_only": (
            target_addition
        ),
        "reference_addition_only": (
            reference_addition
        ),
    }
    readouts = {
        "target_readout": target.model,
        "reference_readout": (
            reference.model
        ),
    }

    rows = []

    for hidden_name, hidden in (
        hidden_sources.items()
    ):
        for readout_name, readout in (
            readouts.items()
        ):
            rows.append(
                {
                    "comparison_type": (
                        comparison_type
                    ),
                    "run": target.run,
                    "depth": target.depth,
                    "regime": target.regime,
                    "target_role": (
                        target.role
                    ),
                    "target_step": (
                        target.step
                    ),
                    "target_checkpoint": str(
                        target.checkpoint
                    ),
                    "reference_role": (
                        reference.role
                    ),
                    "reference_step": (
                        reference.step
                    ),
                    "reference_checkpoint": str(
                        reference.checkpoint
                    ),
                    "hidden_source": (
                        hidden_name
                    ),
                    "readout_source": (
                        readout_name
                    ),
                    "target_baseline_accuracy": (
                        target.baseline_accuracy
                    ),
                    "reference_baseline_accuracy": (
                        reference.baseline_accuracy
                    ),
                    "readout_weight_cosine": (
                        cosine_similarity(
                            target.model
                            .unembedding
                            .weight
                            .detach()
                            .cpu(),
                            reference.model
                            .unembedding
                            .weight
                            .detach()
                            .cpu(),
                        )
                    ),
                    "full_hidden_spectral_power_cosine": (
                        cosine_similarity(
                            target.power,
                            reference.power,
                        )
                    ),
                    "intervention_accuracy": (
                        intervention_accuracy(
                            hidden,
                            target.targets,
                            readout,
                        )
                    ),
                }
            )

    return rows


def process_state_interventions(
    state: CheckpointState,
    random_replicates: int,
    seed: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    family_rows = family_interventions(
        state
    )
    (
        threshold_rows,
        selected_by_threshold,
    ) = addition_threshold_interventions(
        state
    )
    frequency_rows = (
        frequency_identity_controls(
            state,
            selected_by_threshold,
            replicates=random_replicates,
            base_seed=seed,
        )
    )
    phase_rows = phase_controls(
        state,
        selected_by_threshold,
        replicates=random_replicates,
        base_seed=seed,
    )

    return (
        family_rows,
        threshold_rows,
        frequency_rows,
        phase_rows,
    )


def role_for_matched_step(
    step: int,
    panel_steps: list[int],
) -> str:
    labels = {
        panel_steps[0]: "branch",
        panel_steps[-1]: "final",
    }
    if len(panel_steps) >= 2:
        labels[panel_steps[1]] = (
            "first_control_below_10pct"
        )
    if len(panel_steps) >= 3:
        labels[panel_steps[2]] = (
            "minimum_control_accuracy"
        )
    if len(panel_steps) >= 4:
        labels[panel_steps[3]] = (
            "post_collapse_recovery"
        )
    return labels.get(
        step,
        "matched_panel",
    )


def write_html(
    path: Path,
    frames: dict[str, pd.DataFrame],
) -> None:
    try:
        import plotly.express as px
    except ImportError as error:
        raise RuntimeError(
            "Plotly is required for the HTML report."
        ) from error

    figures = []

    family = frames["family"]
    addition_family = family[
        family["family"] == "addition"
    ]
    figures.append(
        px.bar(
            addition_family,
            x="run",
            y="intervention_accuracy",
            color="intervention",
            facet_col="role",
            facet_col_wrap=2,
            barmode="group",
            title=(
                "Addition-family sufficiency and ablation"
            ),
        )
    )

    thresholds = frames["threshold"]
    top_sufficiency = thresholds[
        thresholds["intervention"]
        == "top_addition_sufficiency"
    ]
    figures.append(
        px.line(
            top_sufficiency,
            x=(
                "achieved_addition_power_fraction"
            ),
            y="intervention_accuracy",
            color="run",
            facet_row="depth",
            markers=True,
            title=(
                "Accuracy from progressively larger "
                "dominant-addition subsets"
            ),
        )
    )

    frequency = frames["frequency"]
    frequency_summary = (
        frequency.groupby(
            [
                "run",
                "role",
                "requested_addition_power_threshold",
                "control",
                "context",
            ],
            dropna=False,
        )["intervention_accuracy"]
        .mean()
        .reset_index()
    )
    figures.append(
        px.bar(
            frequency_summary[
                frequency_summary[
                    "requested_addition_power_threshold"
                ] == 0.90
            ],
            x="run",
            y="intervention_accuracy",
            color="control",
            facet_col="role",
            facet_col_wrap=2,
            barmode="group",
            title=(
                "Frequency-identity controls at the "
                "90% addition-power threshold"
            ),
        )
    )

    phase = frames["phase"]
    phase_summary = (
        phase.groupby(
            [
                "run",
                "role",
                "requested_addition_power_threshold",
                "phase_type",
                "context",
            ]
        )["intervention_accuracy"]
        .mean()
        .reset_index()
    )
    figures.append(
        px.bar(
            phase_summary[
                phase_summary[
                    "requested_addition_power_threshold"
                ] == 1.0
            ],
            x="run",
            y="intervention_accuracy",
            color="phase_type",
            facet_col="context",
            barmode="group",
            title=(
                "Phase sensitivity of the full "
                "addition family"
            ),
        )
    )

    cross = frames["cross"]
    target_hidden = cross[
        cross["hidden_source"].isin(
            [
                "target_full",
                "target_addition_only",
            ]
        )
    ]
    figures.append(
        px.bar(
            target_hidden,
            x="run",
            y="intervention_accuracy",
            color="readout_source",
            facet_col="hidden_source",
            barmode="group",
            title=(
                "Within-run hidden/readout compatibility"
            ),
        )
    )

    matched_cross = frames[
        "matched_cross"
    ]
    if not matched_cross.empty:
        figures.append(
            px.line(
                matched_cross[
                    matched_cross[
                        "hidden_source"
                    ].isin(
                        [
                            "target_full",
                            "target_addition_only",
                        ]
                    )
                ],
                x="target_step",
                y="intervention_accuracy",
                color="readout_source",
                facet_row="hidden_source",
                markers=True,
                title=(
                    "Matched control/freeze "
                    "cross-readout compatibility"
                ),
            )
        )

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Depth Fourier hypothesis tests</title>",
        "</head><body>",
        "<h1>Depth Fourier hypothesis tests</h1>",
        (
            "<p>This report tests family sufficiency and necessity, "
            "dominant-mode concentration, frequency identity, phase "
            "sensitivity, and hidden/readout compatibility.</p>"
        ),
    ]

    for index, figure in enumerate(
        figures
    ):
        html_parts.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=(
                    "inline"
                    if index == 0
                    else False
                ),
            )
        )

    for name, frame in frames.items():
        html_parts.append(
            f"<h2>{name}</h2>"
        )
        html_parts.append(
            frame.to_html(index=False)
        )

    html_parts.append(
        "</body></html>"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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

    device = get_device(
        args.device
    )
    model_frame = pd.read_csv(
        args.model_summary
    )
    paired_frame = pd.read_csv(
        args.matched_paired_summary
    )

    panel = main_panel(
        model_frame
    )
    matched_steps = matched_panel(
        paired_frame
    )

    selection_rows = []

    for item in panel:
        selection_rows.append(
            {
                "study": "main",
                "run": item.run,
                "depth": item.depth,
                "regime": item.regime,
                "role": item.role,
                "step": item.step,
                "checkpoint": str(
                    item.checkpoint
                ),
            }
        )

    for step in matched_steps:
        row = paired_frame[
            paired_frame["step"] == step
        ].iloc[0]
        for branch in (
            "control",
            "freeze",
        ):
            selection_rows.append(
                {
                    "study": "matched",
                    "run": (
                        f"depth4_matched_{branch}"
                    ),
                    "depth": 4,
                    "regime": branch,
                    "role": role_for_matched_step(
                        step,
                        matched_steps,
                    ),
                    "step": step,
                    "checkpoint": str(
                        row[
                            f"{branch}_checkpoint"
                        ]
                    ),
                }
            )

    write_csv(
        args.selection_output,
        selection_rows,
    )

    family_rows = []
    threshold_rows = []
    frequency_rows = []
    phase_rows = []
    cross_rows = []

    references: dict[
        str,
        CheckpointState,
    ] = {}

    print(f"Device: {device}")
    print(
        f"Main intervention checkpoints: {len(panel)}"
    )
    print(
        f"Random replicates per stochastic control: "
        f"{args.random_replicates}"
    )

    grouped_panel: dict[
        str,
        list[MainCheckpoint],
    ] = {}
    for item in panel:
        grouped_panel.setdefault(
            item.run,
            [],
        ).append(item)

    for run, items in grouped_panel.items():
        corresponding = model_frame[
            model_frame["run"] == run
        ]
        reference_row = corresponding.loc[
            corresponding[
                "full_accuracy"
            ].idxmax()
        ]
        reference_item = MainCheckpoint(
            run=run,
            depth=int(
                reference_row["depth"]
            ),
            regime=str(
                reference_row["regime"]
            ),
            role="maximum_accuracy_reference",
            step=int(
                reference_row["step"]
            ),
            checkpoint=Path(
                str(
                    reference_row[
                        "checkpoint"
                    ]
                )
            ),
        )

        print()
        print("=" * 80)
        print(
            f"Loading reference for {run}: "
            f"step {reference_item.step}"
        )
        print("=" * 80)

        reference_state = (
            full_grid_from_checkpoint(
                run=reference_item.run,
                depth=reference_item.depth,
                regime=reference_item.regime,
                role=reference_item.role,
                step=reference_item.step,
                checkpoint_path=(
                    reference_item.checkpoint
                ),
                device=device,
                operation=args.operation,
            )
        )
        references[run] = reference_state

        for item in items:
            if (
                item.step
                == reference_item.step
            ):
                state = reference_state
            else:
                print(
                    f"Loading {run} "
                    f"{item.role} step {item.step}"
                )
                state = (
                    full_grid_from_checkpoint(
                        run=item.run,
                        depth=item.depth,
                        regime=item.regime,
                        role=item.role,
                        step=item.step,
                        checkpoint_path=(
                            item.checkpoint
                        ),
                        device=device,
                        operation=args.operation,
                    )
                )

            (
                state_family,
                state_threshold,
                state_frequency,
                state_phase,
            ) = process_state_interventions(
                state,
                random_replicates=(
                    args.random_replicates
                ),
                seed=args.seed,
            )
            family_rows.extend(
                state_family
            )
            threshold_rows.extend(
                state_threshold
            )
            frequency_rows.extend(
                state_frequency
            )
            phase_rows.extend(
                state_phase
            )
            cross_rows.extend(
                cross_readout_rows(
                    target=state,
                    reference=reference_state,
                    comparison_type=(
                        "within_run_reference"
                    ),
                )
            )

            if state is not reference_state:
                del state.model
                release_accelerator_cache(
                    device
                )

        # Keep only data, not the model, after this run.
        del reference_state.model
        release_accelerator_cache(
            device
        )

    write_csv(
        args.family_output,
        family_rows,
    )
    write_csv(
        args.threshold_output,
        threshold_rows,
    )
    write_csv(
        args.frequency_control_output,
        frequency_rows,
    )
    write_csv(
        args.phase_output,
        phase_rows,
    )
    write_csv(
        args.cross_readout_output,
        cross_rows,
    )

    matched_family_rows = []
    matched_threshold_rows = []
    matched_frequency_rows = []
    matched_phase_rows = []
    matched_cross_rows = []

    print()
    print("=" * 80)
    print(
        "Processing matched depth-four control/freeze experiment"
    )
    print("=" * 80)

    # All paired steps receive the exact 2x2 full/addition-only
    # hidden-readout compatibility test.
    for row in paired_frame.sort_values(
        "step"
    ).itertuples(index=False):
        step = int(row.step)

        control_state = (
            full_grid_from_checkpoint(
                run="depth4_matched_control",
                depth=4,
                regime="control",
                role="paired_trajectory",
                step=step,
                checkpoint_path=Path(
                    str(row.control_checkpoint)
                ),
                device=device,
                operation=args.operation,
            )
        )
        freeze_state = (
            full_grid_from_checkpoint(
                run="depth4_matched_freeze",
                depth=4,
                regime="freeze",
                role="paired_trajectory",
                step=step,
                checkpoint_path=Path(
                    str(row.freeze_checkpoint)
                ),
                device=device,
                operation=args.operation,
            )
        )

        matched_cross_rows.extend(
            cross_readout_rows(
                target=control_state,
                reference=freeze_state,
                comparison_type=(
                    "matched_control_target_"
                    "freeze_reference"
                ),
            )
        )
        matched_cross_rows.extend(
            cross_readout_rows(
                target=freeze_state,
                reference=control_state,
                comparison_type=(
                    "matched_freeze_target_"
                    "control_reference"
                ),
            )
        )

        if step in matched_steps:
            role = role_for_matched_step(
                step,
                matched_steps,
            )
            control_state.role = role
            freeze_state.role = role

            for state in (
                control_state,
                freeze_state,
            ):
                (
                    state_family,
                    state_threshold,
                    state_frequency,
                    state_phase,
                ) = process_state_interventions(
                    state,
                    random_replicates=(
                        args.random_replicates
                    ),
                    seed=args.seed,
                )
                matched_family_rows.extend(
                    state_family
                )
                matched_threshold_rows.extend(
                    state_threshold
                )
                matched_frequency_rows.extend(
                    state_frequency
                )
                matched_phase_rows.extend(
                    state_phase
                )

        del control_state.model
        del freeze_state.model
        release_accelerator_cache(
            device
        )

    write_csv(
        args.matched_family_output,
        matched_family_rows,
    )
    write_csv(
        args.matched_threshold_output,
        matched_threshold_rows,
    )
    write_csv(
        args.matched_frequency_control_output,
        matched_frequency_rows,
    )
    write_csv(
        args.matched_phase_output,
        matched_phase_rows,
    )
    write_csv(
        args.matched_cross_readout_output,
        matched_cross_rows,
    )

    frames = {
        "family": pd.DataFrame(
            family_rows
        ),
        "threshold": pd.DataFrame(
            threshold_rows
        ),
        "frequency": pd.DataFrame(
            frequency_rows
        ),
        "phase": pd.DataFrame(
            phase_rows
        ),
        "cross": pd.DataFrame(
            cross_rows
        ),
        "matched_family": pd.DataFrame(
            matched_family_rows
        ),
        "matched_threshold": pd.DataFrame(
            matched_threshold_rows
        ),
        "matched_frequency": pd.DataFrame(
            matched_frequency_rows
        ),
        "matched_phase": pd.DataFrame(
            matched_phase_rows
        ),
        "matched_cross": pd.DataFrame(
            matched_cross_rows
        ),
    }

    write_html(
        args.html_output,
        frames,
    )

    print()
    print(
        "All Fourier hypotheses have been tested."
    )
    print(
        f"Saved selection: "
        f"{args.selection_output}"
    )
    print(
        f"Saved family interventions: "
        f"{args.family_output}"
    )
    print(
        f"Saved addition thresholds: "
        f"{args.threshold_output}"
    )
    print(
        f"Saved frequency controls: "
        f"{args.frequency_control_output}"
    )
    print(
        f"Saved phase controls: "
        f"{args.phase_output}"
    )
    print(
        f"Saved cross-readout tests: "
        f"{args.cross_readout_output}"
    )
    print(
        f"Saved matched outputs under: "
        "runs/depth4_matched_fourier_*"
    )
    print(
        f"Saved HTML report: "
        f"{args.html_output}"
    )


if __name__ == "__main__":
    main()
