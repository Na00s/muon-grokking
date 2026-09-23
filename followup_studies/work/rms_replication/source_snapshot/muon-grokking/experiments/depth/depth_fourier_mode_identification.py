from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

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

from data import generate_modular_addition_data
from train_depth_variant import DepthModularAdditionTransformer


CHECKPOINT_PATTERN = re.compile(r"step_(\d+)\.pt$")
RESIDUAL_STAGE_KINDS = {
    "input_residual",
    "post_attention_residual",
    "post_block_residual",
}
MODE_FAMILIES = (
    "dc",
    "addition",
    "subtraction",
    "a_only",
    "b_only",
    "generic_interaction",
)


@dataclass(frozen=True)
class RunSpecification:
    label: str
    depth: int
    regime: str
    csv_path: Path
    checkpoint_directories: tuple[Path, ...]


@dataclass
class StageActivation:
    stage_name: str
    stage_kind: str
    layer: int
    order: int
    values: Tensor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify the complete 2D Fourier mode structure across "
            "controlled depth-two and depth-four modular-addition runs. "
            "This phase is observational only: no family sufficiency or "
            "ablation interventions are performed."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--operation",
        choices=["addition", "subtraction"],
        default="addition",
        help=(
            "Modular operation the checkpoints were trained on. "
            "This selects the labels accuracy is measured against; "
            "it does not affect the mode-family partition."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--checkpoint-stride",
        type=int,
        default=25_000,
        help=(
            "Regular trajectory sampling interval. Checkpoints nearest "
            "grokking, first 99%, freezing, collapse, maximum accuracy, "
            "and final training are added automatically."
        ),
    )
    parser.add_argument(
        "--top-mode-pairs",
        type=int,
        default=20,
        help=(
            "Number of globally strongest conjugate 2D Fourier-mode "
            "pairs to retain per layer and checkpoint."
        ),
    )
    parser.add_argument(
        "--include-all-checkpoints",
        action="store_true",
    )
    parser.add_argument(
        "--model-summary-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_mode_model_summary_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--layer-summary-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_mode_layer_summary_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--mode-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_mode_inventory_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_mode_selected_checkpoints_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path(
            "runs/depth_fourier_mode_identification_seed_0.html"
        ),
    )
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    raise RuntimeError(
        "No CUDA or MPS accelerator is available. "
        "This study intentionally has no CPU inference fallback."
    )


def load_manifest(path: Path) -> list[RunSpecification]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Fourier manifest: {path}")

    frame = pd.read_csv(path)
    required = {
        "label",
        "depth",
        "regime",
        "csv_path",
        "checkpoint_directories",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            "Fourier manifest is missing columns: "
            + ", ".join(sorted(missing))
        )

    specifications = []
    for row in frame.itertuples(index=False):
        directories = tuple(
            Path(part)
            for part in str(row.checkpoint_directories).split("|")
            if part
        )
        if not directories:
            raise ValueError(f"No checkpoint directory for {row.label}.")

        specification = RunSpecification(
            label=str(row.label),
            depth=int(row.depth),
            regime=str(row.regime),
            csv_path=Path(str(row.csv_path)),
            checkpoint_directories=directories,
        )

        if specification.depth not in {1, 2, 4}:
            raise ValueError(
                "This study is restricted to depths 1, 2 and 4."
            )
        if not specification.csv_path.is_file():
            raise FileNotFoundError(
                f"Missing trajectory CSV: {specification.csv_path}"
            )
        for directory in directories:
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Missing checkpoint directory: {directory}"
                )

        specifications.append(specification)

    return specifications


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Unrecognized checkpoint filename: {path.name}")
    return int(match.group(1))


def discover_checkpoints(
    directories: tuple[Path, ...],
) -> dict[int, Path]:
    by_step: dict[int, Path] = {}

    for directory in directories:
        for path in directory.glob("step_*.pt"):
            try:
                step = checkpoint_step(path)
            except ValueError:
                continue
            # Later directories replace duplicate steps from the source
            # segment. This merges the 100k run and its 300k continuation
            # into one trajectory rather than treating them as separate runs.
            by_step[step] = path

    if not by_step:
        raise FileNotFoundError(
            "No step_*.pt checkpoints were found in: "
            + ", ".join(str(path) for path in directories)
        )

    return dict(sorted(by_step.items()))


def first_sustained_step(
    frame: pd.DataFrame,
    field: str,
    threshold: float,
    consecutive: int = 5,
) -> int | None:
    values = frame[field].to_numpy()
    steps = frame["step"].to_numpy()

    for index in range(len(values) - consecutive + 1):
        if (
            values[index:index + consecutive] >= threshold
        ).all():
            return int(steps[index])

    return None


def available_at_or_before(
    available_steps: list[int],
    target: int,
) -> int:
    candidates = [step for step in available_steps if step <= target]
    return max(candidates) if candidates else min(available_steps)


def available_at_or_after(
    available_steps: list[int],
    target: int,
) -> int:
    candidates = [step for step in available_steps if step >= target]
    return min(candidates) if candidates else max(available_steps)


def nearest_available(
    available_steps: list[int],
    target: int,
) -> int:
    return min(
        available_steps,
        key=lambda step: (abs(step - target), step),
    )


def critical_checkpoint_selection(
    specification: RunSpecification,
    checkpoint_map: dict[int, Path],
    stride: int,
    include_all: bool,
) -> tuple[list[int], list[dict[str, object]]]:
    frame = pd.read_csv(specification.csv_path).sort_values("step")

    required = {"step", "test_accuracy"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            f"{specification.csv_path} is missing: "
            + ", ".join(sorted(missing))
        )

    available_steps = sorted(checkpoint_map)
    selected: set[int] = set()
    rows: list[dict[str, object]] = []

    def add(event: str, target_step: int, selection_mode: str) -> None:
        if selection_mode == "before":
            chosen = available_at_or_before(available_steps, target_step)
        elif selection_mode == "after":
            chosen = available_at_or_after(available_steps, target_step)
        elif selection_mode == "nearest":
            chosen = nearest_available(available_steps, target_step)
        else:
            raise ValueError(f"Unknown selection mode: {selection_mode}")

        selected.add(chosen)
        rows.append(
            {
                "run": specification.label,
                "depth": specification.depth,
                "regime": specification.regime,
                "event": event,
                "target_step": target_step,
                "selected_checkpoint_step": chosen,
                "checkpoint": str(checkpoint_map[chosen]),
            }
        )

    if include_all:
        for step in available_steps:
            add("all_checkpoints", step, "nearest")
    else:
        for step in available_steps:
            if step % stride == 0:
                add("regular_stride", step, "nearest")

    add("initial", available_steps[0], "nearest")
    add("final", available_steps[-1], "nearest")

    grokking_step = first_sustained_step(
        frame,
        "test_accuracy",
        0.95,
        consecutive=5,
    )

    first_99_rows = frame[frame["test_accuracy"] >= 0.99]
    first_99_step = (
        int(first_99_rows.iloc[0]["step"])
        if not first_99_rows.empty
        else None
    )

    maximum_index = frame["test_accuracy"].idxmax()
    maximum_step = int(frame.loc[maximum_index, "step"])
    add("maximum_test_accuracy", maximum_step, "nearest")

    if grokking_step is not None:
        add("grokking_before", grokking_step, "before")
        add("grokking_after", grokking_step, "after")

        post = frame[frame["step"] >= grokking_step]
        minimum_index = post["test_accuracy"].idxmin()
        minimum_step = int(post.loc[minimum_index, "step"])
        add(
            "minimum_post_grokking_accuracy",
            minimum_step,
            "nearest",
        )

        below_90 = post[post["test_accuracy"] < 0.90]
        if not below_90.empty:
            first_collapse_step = int(below_90.iloc[0]["step"])
            add(
                "first_post_grokking_below_90_before",
                first_collapse_step,
                "before",
            )
            add(
                "first_post_grokking_below_90_after",
                first_collapse_step,
                "after",
            )

    if first_99_step is not None:
        add("first_99_before", first_99_step, "before")
        add("first_99_after", first_99_step, "after")

    if "auxiliary_frozen" in frame.columns:
        frozen = frame[
            frame["auxiliary_frozen"].astype(float) >= 0.5
        ]
        if not frozen.empty:
            freeze_step = int(frozen.iloc[0]["step"])
            add("freeze_before", freeze_step, "before")
            add("freeze_after", freeze_step, "after")

    return sorted(selected), rows


def load_checkpoint(path: Path) -> dict:
    try:
        loaded = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        loaded = torch.load(path, map_location="cpu")

    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint is not a dictionary: {path}")

    required = {"model_config", "model_state_dict"}
    missing = required - loaded.keys()
    if missing:
        raise KeyError(
            f"{path} is missing fields: "
            + ", ".join(sorted(missing))
        )

    return loaded


def build_model(
    checkpoint: dict,
    expected_depth: int,
    device: torch.device,
) -> DepthModularAdditionTransformer:
    configuration = dict(checkpoint["model_config"])

    found_depth = int(configuration["num_layers"])
    if found_depth != expected_depth:
        raise ValueError(
            "Manifest/checkpoint depth mismatch: "
            f"expected {expected_depth}, found {found_depth}."
        )

    model = DepthModularAdditionTransformer(**configuration)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def ordered_full_grid(
    modulus: int,
    device: torch.device,
    operation: str = "addition",
) -> tuple[Tensor, Tensor]:
    values = torch.arange(
        modulus,
        dtype=torch.long,
        device=device,
    )
    a, b = torch.meshgrid(values, values, indexing="ij")
    equals = torch.full_like(a, modulus)
    inputs = torch.stack(
        [
            a.reshape(-1),
            b.reshape(-1),
            equals.reshape(-1),
        ],
        dim=1,
    )
    if operation == "addition":
        targets = (a + b).remainder(modulus).reshape(-1)
    elif operation == "subtraction":
        targets = (a - b).remainder(modulus).reshape(-1)
    else:
        raise ValueError(f"unknown operation: {operation}")

    return inputs, targets


def split_indices(
    modulus: int,
    train_fraction: float,
    seed: int,
    device: torch.device,
    operation: str = "addition",
) -> tuple[Tensor, Tensor]:
    (
        train_inputs,
        _,
        test_inputs,
        _,
    ) = generate_modular_addition_data(
        modulus=modulus,
        train_fraction=train_fraction,
        seed=seed,
        operation=operation,
    )

    train_indices = (
        train_inputs[:, 0] * modulus + train_inputs[:, 1]
    ).to(device)
    test_indices = (
        test_inputs[:, 0] * modulus + test_inputs[:, 1]
    ).to(device)
    return train_indices, test_indices


@torch.no_grad()
def forward_with_layer_cache(
    model: DepthModularAdditionTransformer,
    token_ids: Tensor,
) -> tuple[Tensor, list[StageActivation]]:
    _, sequence_length = token_ids.shape
    positions = torch.arange(
        sequence_length,
        device=token_ids.device,
    )

    hidden = (
        model.token_embedding(token_ids)
        + model.position_embedding(positions)
    )

    stages = [
        StageActivation(
            stage_name="input_residual",
            stage_kind="input_residual",
            layer=0,
            order=0,
            values=hidden[:, -1, :],
        )
    ]

    stage_order = 1

    for layer_index, block in enumerate(
        model.transformer_blocks,
        start=1,
    ):
        attention_output = block.attention(hidden)
        post_attention = hidden + attention_output

        mlp_pre_activation = block.mlp.input_projection(
            post_attention
        )
        mlp_activation = F.relu(mlp_pre_activation)
        mlp_output = block.mlp.output_projection(
            mlp_activation
        )
        post_block = post_attention + mlp_output

        stages.extend(
            [
                StageActivation(
                    stage_name=(
                        f"layer_{layer_index}_attention_output"
                    ),
                    stage_kind="attention_output",
                    layer=layer_index,
                    order=stage_order,
                    values=attention_output[:, -1, :],
                ),
                StageActivation(
                    stage_name=(
                        f"layer_{layer_index}_post_attention_residual"
                    ),
                    stage_kind="post_attention_residual",
                    layer=layer_index,
                    order=stage_order + 1,
                    values=post_attention[:, -1, :],
                ),
                StageActivation(
                    stage_name=(
                        f"layer_{layer_index}_mlp_activation"
                    ),
                    stage_kind="mlp_activation",
                    layer=layer_index,
                    order=stage_order + 2,
                    values=mlp_activation[:, -1, :],
                ),
                StageActivation(
                    stage_name=f"layer_{layer_index}_mlp_output",
                    stage_kind="mlp_output",
                    layer=layer_index,
                    order=stage_order + 3,
                    values=mlp_output[:, -1, :],
                ),
                StageActivation(
                    stage_name=(
                        f"layer_{layer_index}_post_block_residual"
                    ),
                    stage_kind="post_block_residual",
                    layer=layer_index,
                    order=stage_order + 4,
                    values=post_block[:, -1, :],
                ),
            ]
        )

        hidden = post_block
        stage_order += 5

    logits = model.unembedding(hidden[:, -1, :])
    return logits, stages


def accuracy(
    logits: Tensor,
    targets: Tensor,
    indices: Tensor | None = None,
) -> float:
    if indices is not None:
        logits = logits.index_select(0, indices)
        targets = targets.index_select(0, indices)

    return float(
        (
            logits.argmax(dim=-1) == targets
        ).float().mean().item()
    )


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    indices: Tensor | None = None,
) -> float:
    if indices is not None:
        logits = logits.index_select(0, indices)
        targets = targets.index_select(0, indices)

    return float(F.cross_entropy(logits, targets).item())


def two_dimensional_spectrum(
    grid: Tensor,
) -> tuple[Tensor, Tensor]:
    # Prime-length FFTs are not consistently supported on MPS.
    # Inference remains on the accelerator; FFTs run on CPU.
    cpu_grid = grid.detach().float().cpu()
    spectrum = torch.fft.fft2(
        cpu_grid,
        dim=(0, 1),
        norm="ortho",
    )
    power = spectrum.abs().square().sum(dim=-1)
    return spectrum, power


def mode_family(
    k: int,
    l: int,
    modulus: int,
) -> str:
    if k == 0 and l == 0:
        return "dc"
    if k == l:
        return "addition"
    if l == (-k) % modulus:
        return "subtraction"
    if l == 0:
        return "a_only"
    if k == 0:
        return "b_only"
    return "generic_interaction"


def mode_family_masks(
    modulus: int,
) -> dict[str, Tensor]:
    masks = {
        family: torch.zeros(
            modulus,
            modulus,
            dtype=torch.bool,
        )
        for family in MODE_FAMILIES
    }

    for k in range(modulus):
        for l in range(modulus):
            masks[
                mode_family(k, l, modulus)
            ][k, l] = True

    combined = torch.zeros(
        modulus,
        modulus,
        dtype=torch.int64,
    )
    for mask in masks.values():
        combined += mask.to(torch.int64)

    if not torch.all(combined == 1):
        raise RuntimeError(
            "Mode-family masks must partition the full 2D spectrum."
        )

    return masks


def safe_ratio(
    numerator: Tensor | float,
    denominator: Tensor | float,
) -> float:
    numerator_value = float(
        numerator.item()
        if isinstance(numerator, Tensor)
        else numerator
    )
    denominator_value = float(
        denominator.item()
        if isinstance(denominator, Tensor)
        else denominator
    )
    if denominator_value <= 0.0:
        return 0.0
    return numerator_value / denominator_value


def sum_centroid_grid(grid: Tensor) -> Tensor:
    modulus = grid.shape[0]
    device = grid.device
    a = torch.arange(
        modulus,
        device=device,
    ).view(modulus, 1)
    b = torch.arange(
        modulus,
        device=device,
    ).view(1, modulus)
    sums = (a + b).remainder(modulus)

    flat = grid.reshape(modulus * modulus, -1)
    flat_sums = sums.reshape(-1)

    centroids = torch.zeros(
        modulus,
        grid.shape[-1],
        dtype=grid.dtype,
        device=device,
    )
    centroids.index_add_(0, flat_sums, flat)
    centroids = centroids / modulus

    reconstructed = centroids.index_select(0, flat_sums)
    return reconstructed.reshape_as(grid)


def sum_explained_variance(grid: Tensor) -> float:
    global_mean = grid.mean(
        dim=(0, 1),
        keepdim=True,
    )
    total = (grid - global_mean).square().sum()
    centroid = sum_centroid_grid(grid)
    residual = (grid - centroid).square().sum()

    if float(total.item()) <= 0.0:
        return 0.0

    return float((1.0 - residual / total).item())


def canonical_mode_pair(
    k: int,
    l: int,
    modulus: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    conjugate = ((-k) % modulus, (-l) % modulus)
    original = (k, l)
    representative = min(original, conjugate)
    partner = max(original, conjugate)
    return representative, partner


def conjugate_mode_pairs(
    power: Tensor,
) -> list[dict[str, object]]:
    modulus = power.shape[0]
    seen: set[tuple[int, int]] = set()
    rows = []

    for k in range(modulus):
        for l in range(modulus):
            representative, partner = canonical_mode_pair(
                k,
                l,
                modulus,
            )
            if representative in seen:
                continue
            seen.add(representative)

            if representative == partner:
                pair_power = power[
                    representative[0],
                    representative[1],
                ]
            else:
                pair_power = (
                    power[
                        representative[0],
                        representative[1],
                    ]
                    + power[
                        partner[0],
                        partner[1],
                    ]
                )

            rows.append(
                {
                    "representative_k": representative[0],
                    "representative_l": representative[1],
                    "conjugate_k": partner[0],
                    "conjugate_l": partner[1],
                    "family": mode_family(
                        representative[0],
                        representative[1],
                        modulus,
                    ),
                    "pair_power": float(pair_power.item()),
                }
            )

    rows.sort(
        key=lambda row: row["pair_power"],
        reverse=True,
    )
    return rows


def normalized_entropy(
    values: list[float],
) -> tuple[float, float]:
    positive = torch.tensor(
        [value for value in values if value > 0.0],
        dtype=torch.float64,
    )
    if positive.numel() <= 1:
        return 0.0, float(positive.numel())

    probabilities = positive / positive.sum()
    entropy = -(
        probabilities * probabilities.log()
    ).sum()
    normalized = entropy / math.log(
        probabilities.numel()
    )
    effective_count = torch.exp(entropy)

    return (
        float(normalized.item()),
        float(effective_count.item()),
    )


def family_statistics(
    power: Tensor,
    masks: dict[str, Tensor],
) -> dict[str, object]:
    total_power = power.sum()
    dc_power = power[masks["dc"]].sum()
    non_dc_power = total_power - dc_power

    statistics: dict[str, object] = {
        "total_spectral_power": float(total_power.item()),
        "non_dc_spectral_power": float(non_dc_power.item()),
    }

    non_dc_family_powers = {}

    for family in MODE_FAMILIES:
        family_power = power[
            masks[family]
        ].sum()
        statistics[
            f"{family}_power_fraction_total"
        ] = safe_ratio(family_power, total_power)

        if family == "dc":
            statistics[
                f"{family}_power_fraction_non_dc"
            ] = float("nan")
        else:
            fraction = safe_ratio(
                family_power,
                non_dc_power,
            )
            statistics[
                f"{family}_power_fraction_non_dc"
            ] = fraction
            non_dc_family_powers[family] = fraction

    dominant_non_dc_family = max(
        non_dc_family_powers,
        key=non_dc_family_powers.get,
    )
    statistics["dominant_non_dc_family"] = dominant_non_dc_family
    statistics[
        "dominant_non_dc_family_fraction"
    ] = non_dc_family_powers[
        dominant_non_dc_family
    ]

    return statistics


def top_pair_statistics(
    pair_rows: list[dict[str, object]],
    total_power: float,
    non_dc_power: float,
    top_count: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    non_dc_pairs = [
        row
        for row in pair_rows
        if row["family"] != "dc"
    ]

    fractions = [
        float(row["pair_power"]) / non_dc_power
        if non_dc_power > 0.0
        else 0.0
        for row in non_dc_pairs
    ]
    entropy, effective_count = normalized_entropy(
        [
            float(row["pair_power"])
            for row in non_dc_pairs
        ]
    )

    def concentration(count: int) -> float:
        return sum(fractions[:count])

    summary = {
        "top_1_non_dc_pair_power_fraction": concentration(1),
        "top_5_non_dc_pair_power_fraction": concentration(5),
        "top_10_non_dc_pair_power_fraction": concentration(10),
        "top_20_non_dc_pair_power_fraction": concentration(20),
        "normalized_non_dc_pair_entropy": entropy,
        "effective_non_dc_pair_count": effective_count,
        "top_non_dc_mode_pairs": "|".join(
            (
                f"({row['representative_k']},"
                f"{row['representative_l']})/"
                f"({row['conjugate_k']},"
                f"{row['conjugate_l']}):"
                f"{row['family']}"
            )
            for row in non_dc_pairs[:top_count]
        ),
    }

    detailed = []
    for rank, row in enumerate(
        non_dc_pairs[:top_count],
        start=1,
    ):
        detailed.append(
            {
                **row,
                "rank": rank,
                "pair_power_fraction_total": (
                    float(row["pair_power"]) / total_power
                    if total_power > 0.0
                    else 0.0
                ),
                "pair_power_fraction_non_dc": (
                    float(row["pair_power"]) / non_dc_power
                    if non_dc_power > 0.0
                    else 0.0
                ),
            }
        )

    return summary, detailed


def release_accelerator_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def analyze_checkpoint(
    specification: RunSpecification,
    checkpoint_path: Path,
    device: torch.device,
    top_pair_count: int,
    operation: str = "addition",
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model(
        checkpoint,
        specification.depth,
        device,
    )

    configuration = dict(checkpoint["model_config"])
    arguments = dict(checkpoint.get("arguments", {}))
    modulus = int(configuration["modulus"])
    train_fraction = float(
        arguments.get("train_fraction", 0.3)
    )
    seed = int(
        checkpoint.get(
            "seed",
            arguments.get("seed", 0),
        )
    )

    full_inputs, full_targets = ordered_full_grid(
        modulus,
        device,
        operation=operation,
    )
    train_indices, test_indices = split_indices(
        modulus,
        train_fraction,
        seed,
        device,
        operation=operation,
    )

    with torch.no_grad():
        logits, stages = forward_with_layer_cache(
            model,
            full_inputs,
        )

    step = checkpoint_step(checkpoint_path)
    masks = mode_family_masks(modulus)
    layer_rows = []
    mode_rows = []

    final_stage_name = (
        f"layer_{specification.depth}_post_block_residual"
    )

    for stage in stages:
        grid = stage.values.reshape(
            modulus,
            modulus,
            -1,
        )
        _, power = two_dimensional_spectrum(grid)
        family_stats = family_statistics(
            power,
            masks,
        )
        pairs = conjugate_mode_pairs(power)
        top_summary, top_details = top_pair_statistics(
            pairs,
            total_power=float(
                family_stats["total_spectral_power"]
            ),
            non_dc_power=float(
                family_stats["non_dc_spectral_power"]
            ),
            top_count=top_pair_count,
        )

        row: dict[str, object] = {
            "run": specification.label,
            "depth": specification.depth,
            "regime": specification.regime,
            "step": step,
            "checkpoint": str(checkpoint_path),
            "stage_name": stage.stage_name,
            "stage_kind": stage.stage_kind,
            "layer": stage.layer,
            "stage_order": stage.order,
            "activation_width": grid.shape[-1],
            "sum_explained_variance": (
                sum_explained_variance(grid)
            ),
            "direct_readout_train_accuracy": float("nan"),
            "direct_readout_test_accuracy": float("nan"),
            "direct_readout_full_accuracy": float("nan"),
            **family_stats,
            **top_summary,
        }

        if stage.stage_kind in RESIDUAL_STAGE_KINDS:
            direct_logits = model.unembedding(
                grid.reshape(
                    modulus * modulus,
                    -1,
                ).to(device)
            )
            row.update(
                {
                    "direct_readout_train_accuracy": accuracy(
                        direct_logits,
                        full_targets,
                        train_indices,
                    ),
                    "direct_readout_test_accuracy": accuracy(
                        direct_logits,
                        full_targets,
                        test_indices,
                    ),
                    "direct_readout_full_accuracy": accuracy(
                        direct_logits,
                        full_targets,
                    ),
                }
            )

        layer_rows.append(row)

        for detail in top_details:
            mode_rows.append(
                {
                    "run": specification.label,
                    "depth": specification.depth,
                    "regime": specification.regime,
                    "step": step,
                    "checkpoint": str(checkpoint_path),
                    "stage_name": stage.stage_name,
                    "stage_kind": stage.stage_kind,
                    "layer": stage.layer,
                    "stage_order": stage.order,
                    **detail,
                }
            )

    final_row = next(
        row
        for row in layer_rows
        if row["stage_name"] == final_stage_name
    )

    post_block_rows = [
        row
        for row in layer_rows
        if row["stage_kind"] == "post_block_residual"
    ]
    readable_layers = [
        int(row["layer"])
        for row in post_block_rows
        if (
            not math.isnan(
                float(row["direct_readout_full_accuracy"])
            )
            and float(
                row["direct_readout_full_accuracy"]
            ) >= 0.95
        )
    ]

    model_row = {
        "run": specification.label,
        "depth": specification.depth,
        "regime": specification.regime,
        "step": step,
        "checkpoint": str(checkpoint_path),
        "train_loss": cross_entropy(
            logits,
            full_targets,
            train_indices,
        ),
        "train_accuracy": accuracy(
            logits,
            full_targets,
            train_indices,
        ),
        "test_loss": cross_entropy(
            logits,
            full_targets,
            test_indices,
        ),
        "test_accuracy": accuracy(
            logits,
            full_targets,
            test_indices,
        ),
        "full_accuracy": accuracy(
            logits,
            full_targets,
        ),
        "first_layer_with_95pct_direct_readout": (
            min(readable_layers)
            if readable_layers
            else float("nan")
        ),
        **{
            f"final_{key}": value
            for key, value in final_row.items()
            if (
                key.endswith("_power_fraction_total")
                or key.endswith("_power_fraction_non_dc")
                or key
                in {
                    "sum_explained_variance",
                    "dominant_non_dc_family",
                    "dominant_non_dc_family_fraction",
                    "top_1_non_dc_pair_power_fraction",
                    "top_5_non_dc_pair_power_fraction",
                    "top_10_non_dc_pair_power_fraction",
                    "top_20_non_dc_pair_power_fraction",
                    "normalized_non_dc_pair_entropy",
                    "effective_non_dc_pair_count",
                    "top_non_dc_mode_pairs",
                }
            )
        },
    }

    del model
    del logits
    del stages
    release_accelerator_cache(device)

    return model_row, layer_rows, mode_rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_html(
    path: Path,
    model_frame: pd.DataFrame,
    layer_frame: pd.DataFrame,
    mode_frame: pd.DataFrame,
    selection_frame: pd.DataFrame,
) -> None:
    try:
        import plotly.express as px
    except ImportError as error:
        raise RuntimeError(
            "Plotly is required to create the HTML report."
        ) from error

    family_columns = [
        f"{family}_power_fraction_non_dc"
        for family in MODE_FAMILIES
        if family != "dc"
    ]

    final_family_long = model_frame.melt(
        id_vars=[
            "run",
            "depth",
            "regime",
            "step",
            "full_accuracy",
        ],
        value_vars=[
            f"final_{column}"
            for column in family_columns
        ],
        var_name="mode_family",
        value_name="non_dc_power_fraction",
    )
    final_family_long["mode_family"] = (
        final_family_long["mode_family"]
        .str.removeprefix("final_")
        .str.removesuffix("_power_fraction_non_dc")
    )

    final_steps = (
        model_frame.groupby("run")["step"]
        .max()
        .rename("final_step")
        .reset_index()
    )
    final_layer = layer_frame.merge(
        final_steps,
        on="run",
        how="inner",
    )
    final_layer = final_layer[
        final_layer["step"] == final_layer["final_step"]
    ]

    layer_family_long = final_layer.melt(
        id_vars=[
            "run",
            "depth",
            "regime",
            "stage_name",
            "stage_kind",
            "layer",
            "stage_order",
        ],
        value_vars=family_columns,
        var_name="mode_family",
        value_name="non_dc_power_fraction",
    )
    layer_family_long["mode_family"] = (
        layer_family_long["mode_family"]
        .str.removesuffix("_power_fraction_non_dc")
    )

    figures = [
        px.line(
            model_frame,
            x="step",
            y="test_accuracy",
            color="run",
            facet_row="depth",
            markers=True,
            title="Test accuracy at selected Fourier checkpoints",
        ),
        px.line(
            model_frame,
            x="step",
            y="final_addition_power_fraction_non_dc",
            color="run",
            facet_row="depth",
            markers=True,
            title=(
                "Addition-family share of final-residual "
                "non-DC Fourier power"
            ),
        ),
        px.line(
            model_frame,
            x="step",
            y="final_generic_interaction_power_fraction_non_dc",
            color="run",
            facet_row="depth",
            markers=True,
            title=(
                "Generic-interaction share of final-residual "
                "non-DC Fourier power"
            ),
        ),
        px.line(
            model_frame,
            x="step",
            y="final_effective_non_dc_pair_count",
            color="run",
            facet_row="depth",
            markers=True,
            title=(
                "Effective number of active non-DC Fourier-mode pairs"
            ),
        ),
        px.bar(
            final_family_long,
            x="mode_family",
            y="non_dc_power_fraction",
            color="run",
            facet_row="depth",
            barmode="group",
            title=(
                "Final-checkpoint non-DC power by complete mode family"
            ),
        ),
        px.line(
            layer_family_long,
            x="stage_order",
            y="non_dc_power_fraction",
            color="mode_family",
            facet_row="run",
            markers=True,
            hover_data=["stage_name", "layer"],
            title=(
                "Layerwise complete mode-family decomposition "
                "at each run's final checkpoint"
            ),
        ),
    ]

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Depth Fourier mode identification</title>",
        "</head><body>",
        "<h1>Depth-2 and depth-4 Fourier mode identification</h1>",
        (
            "<p>The full two-dimensional spectrum is partitioned "
            "exactly into DC, addition (k,k), subtraction (k,-k), "
            "a-only (k,0), b-only (0,l), and generic-interaction "
            "modes. No sufficiency or ablation intervention is "
            "performed in this phase.</p>"
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

    html_parts.extend(
        [
            "<h2>Selected checkpoint events</h2>",
            selection_frame.to_html(index=False),
            "<h2>Model-level summary</h2>",
            model_frame.to_html(index=False),
            "<h2>Layer-level final-checkpoint summary</h2>",
            final_layer.to_html(index=False),
            "<h2>Top global conjugate mode pairs</h2>",
            mode_frame.to_html(index=False),
            "</body></html>",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(html_parts),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_arguments()

    if args.checkpoint_stride < 1:
        raise ValueError(
            "--checkpoint-stride must be positive."
        )
    if args.top_mode_pairs < 1:
        raise ValueError(
            "--top-mode-pairs must be positive."
        )

    device = get_device(args.device)
    specifications = load_manifest(args.manifest)

    model_rows = []
    layer_rows = []
    mode_rows = []
    selection_rows = []

    print(f"Device: {device}")

    for specification in specifications:
        checkpoint_map = discover_checkpoints(
            specification.checkpoint_directories
        )
        selected_steps, run_selection = (
            critical_checkpoint_selection(
                specification,
                checkpoint_map,
                args.checkpoint_stride,
                args.include_all_checkpoints,
            )
        )
        selection_rows.extend(run_selection)

        print()
        print("=" * 80)
        print(
            f"Mode-identification run={specification.label} | "
            f"depth={specification.depth} | "
            f"regime={specification.regime} | "
            f"selected_checkpoints={len(selected_steps)}"
        )
        print("=" * 80)

        for step in selected_steps:
            checkpoint_path = checkpoint_map[step]
            print(
                f"Analyzing step {step}: {checkpoint_path}"
            )
            (
                model_row,
                checkpoint_layer_rows,
                checkpoint_mode_rows,
            ) = analyze_checkpoint(
                specification=specification,
                checkpoint_path=checkpoint_path,
                device=device,
                top_pair_count=args.top_mode_pairs,
                operation=args.operation,
            )
            model_rows.append(model_row)
            layer_rows.extend(checkpoint_layer_rows)
            mode_rows.extend(checkpoint_mode_rows)

    write_csv(
        args.model_summary_output,
        model_rows,
    )
    write_csv(
        args.layer_summary_output,
        layer_rows,
    )
    write_csv(
        args.mode_output,
        mode_rows,
    )
    write_csv(
        args.selection_output,
        selection_rows,
    )

    model_frame = pd.DataFrame(model_rows).sort_values(
        ["depth", "run", "step"]
    )
    layer_frame = pd.DataFrame(layer_rows).sort_values(
        [
            "depth",
            "run",
            "step",
            "stage_order",
        ]
    )
    mode_frame = pd.DataFrame(mode_rows).sort_values(
        [
            "depth",
            "run",
            "step",
            "stage_order",
            "rank",
        ]
    )
    selection_frame = (
        pd.DataFrame(selection_rows)
        .drop_duplicates()
        .sort_values(
            [
                "depth",
                "run",
                "selected_checkpoint_step",
                "event",
            ]
        )
    )

    write_html(
        args.html_output,
        model_frame,
        layer_frame,
        mode_frame,
        selection_frame,
    )

    print()
    print("Depth Fourier mode-identification study complete.")
    print(
        f"Saved model summary: {args.model_summary_output}"
    )
    print(
        f"Saved layer summary: {args.layer_summary_output}"
    )
    print(
        f"Saved mode inventory: {args.mode_output}"
    )
    print(
        f"Saved checkpoint selection: {args.selection_output}"
    )
    print(f"Saved HTML report: {args.html_output}")


if __name__ == "__main__":
    main()
