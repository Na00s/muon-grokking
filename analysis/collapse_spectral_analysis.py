from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data import generate_modular_addition_data
from model import ModularAdditionTransformer


MODULUS = 113
TRAIN_FRACTION = 0.3
SEQUENCE_LENGTH = 3
D_MODEL = 128
NUMBER_OF_HEADS = 4
D_MLP = 512

CHECKPOINT_PATTERN = re.compile(r"step_(\d+)\.pt$")


@dataclass
class CheckpointRepresentation:
    step: int
    hidden: Tensor
    diagonal_hidden: Tensor
    off_diagonal_hidden: Tensor
    unembedding: Tensor
    diagonal_pair_power: Tensor
    top_frequencies: tuple[int, ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze dense checkpoints around a Muon collapse. "
            "The script distinguishes destruction of the "
            "sum-dependent Fourier representation, readout "
            "misalignment, spectral rotation, and growth of "
            "off-diagonal interference."
        )
    )

    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )

    parser.add_argument(
        "--reference-step",
        type=int,
        default=None,
        help=(
            "Pre-collapse reference checkpoint. When omitted, "
            "the script selects the last checkpoint above 99% "
            "test accuracy before the first later checkpoint "
            "below 90%."
        ),
    )

    parser.add_argument(
        "--top-frequency-pairs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "runs/collapse_spectral_summary.csv"
        ),
    )

    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path(
            "runs/collapse_spectral_analysis.html"
        ),
    )

    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is unavailable."
            )
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.search(path.name)

    if match is None:
        raise ValueError(
            f"Unrecognized checkpoint filename: {path.name}"
        )

    return int(match.group(1))


def discover_checkpoints(
    directory: Path,
) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {directory}"
        )

    paths = []

    for path in directory.glob("step_*.pt"):
        try:
            checkpoint_step(path)
        except ValueError:
            continue

        paths.append(path)

    paths.sort(key=checkpoint_step)

    if not paths:
        raise FileNotFoundError(
            f"No checkpoints found in {directory}."
        )

    return paths


def load_state_dict(
    checkpoint_path: Path,
) -> dict[str, Tensor]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must contain a dictionary."
        )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    if not isinstance(state, dict):
        raise TypeError(
            "model_state_dict must be a dictionary."
        )

    return state


def build_model(
    checkpoint_path: Path,
    device: torch.device,
) -> ModularAdditionTransformer:
    model = ModularAdditionTransformer(
        modulus=MODULUS,
        sequence_length=SEQUENCE_LENGTH,
        d_model=D_MODEL,
        num_heads=NUMBER_OF_HEADS,
        d_mlp=D_MLP,
    )

    model.load_state_dict(
        load_state_dict(checkpoint_path)
    )

    model.to(device)
    model.eval()

    return model


def ordered_full_grid(
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    values = torch.arange(
        MODULUS,
        dtype=torch.long,
        device=device,
    )

    a, b = torch.meshgrid(
        values,
        values,
        indexing="ij",
    )

    equals = torch.full_like(
        a,
        MODULUS,
    )

    inputs = torch.stack(
        [
            a.reshape(-1),
            b.reshape(-1),
            equals.reshape(-1),
        ],
        dim=1,
    )

    targets = (
        a + b
    ).remainder(MODULUS).reshape(-1)

    return inputs, targets


def split_indices() -> tuple[Tensor, Tensor]:
    (
        train_inputs,
        _,
        test_inputs,
        _,
    ) = generate_modular_addition_data(
        modulus=MODULUS,
        train_fraction=TRAIN_FRACTION,
        seed=0,
    )

    train_indices = (
        train_inputs[:, 0] * MODULUS
        + train_inputs[:, 1]
    )

    test_indices = (
        test_inputs[:, 0] * MODULUS
        + test_inputs[:, 1]
    )

    return train_indices, test_indices


@torch.no_grad()
def final_hidden_and_logits(
    model: ModularAdditionTransformer,
    inputs: Tensor,
) -> tuple[Tensor, Tensor]:
    batch_size, sequence_length = inputs.shape

    positions = torch.arange(
        sequence_length,
        device=inputs.device,
    )

    x = (
        model.token_embedding(inputs)
        + model.position_embedding(positions)
    )

    x = (
        x
        + model.transformer_block.attention(x)
    )

    x = (
        x
        + model.transformer_block.mlp(x)
    )

    hidden = x[:, -1, :]
    logits = model.unembedding(hidden)

    return (
        hidden.detach().float().cpu(),
        logits.detach().float().cpu(),
    )


def diagonal_projection(
    hidden: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    hidden_grid = hidden.reshape(
        MODULUS,
        MODULUS,
        -1,
    )

    spectrum = torch.fft.fft2(
        hidden_grid,
        dim=(0, 1),
        norm="ortho",
    )

    diagonal_mask = torch.zeros(
        MODULUS,
        MODULUS,
        dtype=torch.bool,
    )

    indices = torch.arange(MODULUS)
    diagonal_mask[indices, indices] = True

    diagonal_spectrum = torch.where(
        diagonal_mask.unsqueeze(-1),
        spectrum,
        torch.zeros_like(spectrum),
    )

    diagonal_hidden = torch.fft.ifft2(
        diagonal_spectrum,
        dim=(0, 1),
        norm="ortho",
    ).real.reshape(
        MODULUS * MODULUS,
        -1,
    )

    off_diagonal_hidden = (
        hidden - diagonal_hidden
    )

    power = (
        spectrum.abs()
        .square()
        .sum(dim=-1)
    )

    pair_power = []

    for frequency in range(
        1,
        (MODULUS + 1) // 2,
    ):
        conjugate = (-frequency) % MODULUS

        pair_power.append(
            power[frequency, frequency]
            + power[conjugate, conjugate]
        )

    pair_power_tensor = torch.stack(
        pair_power
    )

    non_dc_total = (
        power.sum() - power[0, 0]
    )

    if float(non_dc_total.item()) > 0:
        normalized_pair_power = (
            pair_power_tensor / non_dc_total
        )
    else:
        normalized_pair_power = (
            torch.zeros_like(pair_power_tensor)
        )

    return (
        diagonal_hidden,
        off_diagonal_hidden,
        normalized_pair_power,
    )


def decode(
    hidden: Tensor,
    unembedding: Tensor,
) -> Tensor:
    return hidden @ unembedding.T


def accuracy(
    logits: Tensor,
    targets: Tensor,
    indices: Tensor,
) -> float:
    selected_logits = logits.index_select(
        0,
        indices,
    )

    selected_targets = targets.index_select(
        0,
        indices,
    )

    return float(
        (
            selected_logits.argmax(dim=-1)
            == selected_targets
        )
        .float()
        .mean()
        .item()
    )


def mean_margin(
    logits: Tensor,
    targets: Tensor,
    indices: Tensor,
) -> float:
    selected_logits = logits.index_select(
        0,
        indices,
    )

    selected_targets = targets.index_select(
        0,
        indices,
    )

    correct_logits = selected_logits.gather(
        1,
        selected_targets.unsqueeze(1),
    ).squeeze(1)

    masked = selected_logits.clone()
    masked.scatter_(
        1,
        selected_targets.unsqueeze(1),
        float("-inf"),
    )

    largest_incorrect = masked.max(
        dim=1
    ).values

    return float(
        (
            correct_logits - largest_incorrect
        )
        .mean()
        .item()
    )


def sum_explained_variance(
    hidden: Tensor,
) -> float:
    hidden_grid = hidden.reshape(
        MODULUS,
        MODULUS,
        -1,
    )

    a = torch.arange(
        MODULUS
    ).view(MODULUS, 1)

    b = torch.arange(
        MODULUS
    ).view(1, MODULUS)

    sums = (
        a + b
    ).remainder(MODULUS)

    flat_sums = sums.reshape(-1)

    centroids = torch.zeros(
        MODULUS,
        hidden.shape[-1],
        dtype=hidden.dtype,
    )

    centroids.index_add_(
        0,
        flat_sums,
        hidden,
    )

    centroids = centroids / MODULUS

    reconstructed = centroids.index_select(
        0,
        flat_sums,
    )

    centered = hidden - hidden.mean(
        dim=0,
        keepdim=True,
    )

    total = centered.square().sum()
    residual = (
        hidden - reconstructed
    ).square().sum()

    if float(total.item()) <= 0:
        return 0.0

    return float(
        (
            1.0 - residual / total
        ).item()
    )


def diagonal_power_fraction(
    hidden: Tensor,
) -> float:
    hidden_grid = hidden.reshape(
        MODULUS,
        MODULUS,
        -1,
    )

    spectrum = torch.fft.fft2(
        hidden_grid,
        dim=(0, 1),
        norm="ortho",
    )

    power = (
        spectrum.abs()
        .square()
        .sum(dim=-1)
    )

    indices = torch.arange(MODULUS)
    diagonal = power[indices, indices]

    non_dc_total = (
        power.sum() - power[0, 0]
    )

    non_dc_diagonal = diagonal[1:].sum()

    if float(non_dc_total.item()) <= 0:
        return 0.0

    return float(
        (
            non_dc_diagonal
            / non_dc_total
        ).item()
    )


def top_frequencies(
    pair_power: Tensor,
    count: int,
) -> tuple[int, ...]:
    count = min(
        count,
        pair_power.numel(),
    )

    indices = torch.topk(
        pair_power,
        k=count,
    ).indices

    return tuple(
        int(index.item()) + 1
        for index in indices
    )


def cosine_similarity(
    first: Tensor,
    second: Tensor,
) -> float:
    first = first.reshape(-1).float()
    second = second.reshape(-1).float()

    first_norm = torch.linalg.vector_norm(
        first
    )

    second_norm = torch.linalg.vector_norm(
        second
    )

    denominator = first_norm * second_norm

    if float(denominator.item()) <= 0:
        return 0.0

    return float(
        (
            torch.dot(first, second)
            / denominator
        ).item()
    )


def relative_l2_drift(
    current: Tensor,
    reference: Tensor,
) -> float:
    denominator = torch.linalg.vector_norm(
        reference.float()
    )

    if float(denominator.item()) <= 0:
        return 0.0

    numerator = torch.linalg.vector_norm(
        current.float() - reference.float()
    )

    return float(
        (
            numerator / denominator
        ).item()
    )


def top_frequency_jaccard(
    current: tuple[int, ...],
    reference: tuple[int, ...],
) -> float:
    current_set = set(current)
    reference_set = set(reference)

    union = current_set | reference_set

    if not union:
        return 1.0

    return len(
        current_set & reference_set
    ) / len(union)


def interference_metrics(
    full_logits: Tensor,
    diagonal_logits: Tensor,
    off_diagonal_logits: Tensor,
    targets: Tensor,
    indices: Tensor,
) -> tuple[float, float]:
    full_logits = full_logits.index_select(
        0,
        indices,
    )

    diagonal_logits = diagonal_logits.index_select(
        0,
        indices,
    )

    off_diagonal_logits = (
        off_diagonal_logits.index_select(
            0,
            indices,
        )
    )

    targets = targets.index_select(
        0,
        indices,
    )

    full_predictions = full_logits.argmax(
        dim=-1
    )

    diagonal_predictions = (
        diagonal_logits.argmax(dim=-1)
    )

    rescued_mask = (
        (diagonal_predictions == targets)
        & (full_predictions != targets)
    )

    rescued_fraction = float(
        rescued_mask.float().mean().item()
    )

    if not bool(rescued_mask.any()):
        return rescued_fraction, 0.0

    selected_targets = targets[
        rescued_mask
    ]

    selected_competitors = full_predictions[
        rescued_mask
    ]

    diagonal_selected = diagonal_logits[
        rescued_mask
    ]

    off_selected = off_diagonal_logits[
        rescued_mask
    ]

    diagonal_true = diagonal_selected.gather(
        1,
        selected_targets.unsqueeze(1),
    ).squeeze(1)

    diagonal_competitor = (
        diagonal_selected.gather(
            1,
            selected_competitors.unsqueeze(1),
        ).squeeze(1)
    )

    off_true = off_selected.gather(
        1,
        selected_targets.unsqueeze(1),
    ).squeeze(1)

    off_competitor = off_selected.gather(
        1,
        selected_competitors.unsqueeze(1),
    ).squeeze(1)

    off_interference_advantage = (
        off_competitor
        - off_true
        - (
            diagonal_true
            - diagonal_competitor
        )
    )

    return (
        rescued_fraction,
        float(
            off_interference_advantage
            .mean()
            .item()
        ),
    )


def load_representation(
    checkpoint_path: Path,
    device: torch.device,
    inputs: Tensor,
    top_count: int,
) -> tuple[
    CheckpointRepresentation,
    Tensor,
]:
    model = build_model(
        checkpoint_path,
        device,
    )

    hidden, full_logits = (
        final_hidden_and_logits(
            model,
            inputs,
        )
    )

    (
        diagonal_hidden,
        off_diagonal_hidden,
        pair_power,
    ) = diagonal_projection(hidden)

    unembedding = (
        model.unembedding.weight
        .detach()
        .float()
        .cpu()
        .clone()
    )

    representation = CheckpointRepresentation(
        step=checkpoint_step(
            checkpoint_path
        ),
        hidden=hidden,
        diagonal_hidden=diagonal_hidden,
        off_diagonal_hidden=(
            off_diagonal_hidden
        ),
        unembedding=unembedding,
        diagonal_pair_power=pair_power,
        top_frequencies=top_frequencies(
            pair_power,
            top_count,
        ),
    )

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return representation, full_logits


def select_reference_step(
    paths: list[Path],
    test_accuracies: list[float],
    requested_step: int | None,
) -> tuple[int, int]:
    steps = [
        checkpoint_step(path)
        for path in paths
    ]

    if requested_step is not None:
        if requested_step not in steps:
            raise ValueError(
                "Requested reference step was not "
                "found among checkpoints."
            )

        reference_index = steps.index(
            requested_step
        )

        collapse_index = min(
            range(
                reference_index + 1,
                len(steps),
            ),
            key=lambda index: (
                test_accuracies[index]
            ),
            default=reference_index,
        )

        return (
            reference_index,
            collapse_index,
        )

    reached_high_accuracy = False

    for index, value in enumerate(
        test_accuracies
    ):
        if value >= 0.99:
            reached_high_accuracy = True

        if (
            reached_high_accuracy
            and value < 0.90
            and index > 0
        ):
            earlier_high = [
                earlier
                for earlier in range(index)
                if test_accuracies[earlier]
                >= 0.99
            ]

            if not earlier_high:
                continue

            return (
                earlier_high[-1],
                index,
            )

    high_indices = [
        index
        for index, value
        in enumerate(test_accuracies)
        if value >= 0.99
    ]

    if not high_indices:
        raise RuntimeError(
            "No checkpoint reached 99% test accuracy."
        )

    first_high = high_indices[0]

    later_indices = list(
        range(
            first_high + 1,
            len(paths),
        )
    )

    if not later_indices:
        raise RuntimeError(
            "No checkpoint exists after the first "
            "high-accuracy checkpoint."
        )

    collapse_index = min(
        later_indices,
        key=lambda index: test_accuracies[
            index
        ],
    )

    earlier_high = [
        index
        for index in range(
            first_high,
            collapse_index,
        )
        if test_accuracies[index] >= 0.99
    ]

    reference_index = (
        earlier_high[-1]
        if earlier_high
        else first_high
    )

    return reference_index, collapse_index


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_html(
    path: Path,
    rows: list[dict[str, object]],
    reference_step: int,
    collapse_step: int,
) -> None:
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError(
            "HTML output requires pandas and plotly."
        ) from error

    frame = pd.DataFrame(rows)

    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Full, diagonal-only, and off-diagonal-only test accuracy",
            "Diagonal structure",
            "Cross-decoding accuracies",
            "Reference-relative drift",
            "Spectral stability",
            "Off-diagonal interference",
        ),
        vertical_spacing=0.11,
        horizontal_spacing=0.11,
    )

    accuracy_fields = [
        (
            "test_accuracy",
            "Full model",
        ),
        (
            "diagonal_only_test_accuracy",
            "Diagonal only",
        ),
        (
            "off_diagonal_only_test_accuracy",
            "Off-diagonal only",
        ),
    ]

    for field, label in accuracy_fields:
        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame[field],
                mode="lines+markers",
                name=label,
            ),
            row=1,
            col=1,
        )

    structure_fields = [
        (
            "hidden_non_dc_diagonal_power_fraction",
            "Diagonal power",
        ),
        (
            "sum_explained_hidden_variance",
            "Sum-explained variance",
        ),
    ]

    for field, label in structure_fields:
        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame[field],
                mode="lines+markers",
                name=label,
            ),
            row=1,
            col=2,
        )

    cross_fields = [
        (
            "current_hidden_reference_readout_test_accuracy",
            "Current hidden + reference readout",
        ),
        (
            "reference_hidden_current_readout_test_accuracy",
            "Reference hidden + current readout",
        ),
        (
            "current_diagonal_reference_readout_test_accuracy",
            "Current diagonal + reference readout",
        ),
        (
            "reference_diagonal_current_readout_test_accuracy",
            "Reference diagonal + current readout",
        ),
    ]

    for field, label in cross_fields:
        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame[field],
                mode="lines+markers",
                name=label,
            ),
            row=2,
            col=1,
        )

    drift_fields = [
        (
            "relative_unembedding_drift",
            "Unembedding drift",
        ),
        (
            "relative_diagonal_hidden_drift",
            "Diagonal-hidden drift",
        ),
    ]

    for field, label in drift_fields:
        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame[field],
                mode="lines+markers",
                name=label,
            ),
            row=2,
            col=2,
        )

    spectral_fields = [
        (
            "diagonal_pair_power_cosine_similarity_to_reference",
            "Power-spectrum cosine",
        ),
        (
            "top_frequency_jaccard_to_reference",
            "Top-frequency Jaccard",
        ),
        (
            "diagonal_logit_cosine_similarity_to_reference",
            "Diagonal-logit cosine",
        ),
    ]

    for field, label in spectral_fields:
        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame[field],
                mode="lines+markers",
                name=label,
            ),
            row=3,
            col=1,
        )

    interference_fields = [
        (
            "diagonal_correct_full_wrong_fraction",
            "Diagonal correct, full wrong",
        ),
        (
            "off_diagonal_only_test_accuracy",
            "Off-diagonal accuracy",
        ),
    ]

    for field, label in interference_fields:
        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame[field],
                mode="lines+markers",
                name=label,
            ),
            row=3,
            col=2,
        )

    for row in range(1, 4):
        for column in range(1, 3):
            figure.add_vline(
                x=reference_step,
                line_dash="dash",
                annotation_text=(
                    "reference"
                    if row == 1
                    and column == 1
                    else None
                ),
                row=row,
                col=column,
            )

            figure.add_vline(
                x=collapse_step,
                line_dash="dot",
                annotation_text=(
                    "collapse"
                    if row == 1
                    and column == 1
                    else None
                ),
                row=row,
                col=column,
            )

    figure.update_layout(
        title=(
            "Spectral anatomy of the Muon collapse"
        ),
        height=1_200,
        hovermode="x unified",
    )

    figure.update_xaxes(
        title_text="Training step",
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(
        path,
        include_plotlyjs="cdn",
    )


def main() -> None:
    args = parse_arguments()

    if args.top_frequency_pairs < 1:
        raise ValueError(
            "--top-frequency-pairs must be at least 1."
        )

    paths = discover_checkpoints(
        args.checkpoint_directory
    )

    device = get_device(
        args.device
    )

    inputs, targets = ordered_full_grid(
        device
    )

    targets_cpu = targets.detach().cpu()

    train_indices, test_indices = (
        split_indices()
    )

    print(f"Device: {device}")
    print(
        f"Found {len(paths)} checkpoints."
    )

    preliminary_rows = []
    preliminary_accuracies = []

    for path in paths:
        representation, full_logits = (
            load_representation(
                path,
                device,
                inputs,
                args.top_frequency_pairs,
            )
        )

        test_accuracy = accuracy(
            full_logits,
            targets_cpu,
            test_indices,
        )

        diagonal_logits = decode(
            representation.diagonal_hidden,
            representation.unembedding,
        )

        off_diagonal_logits = decode(
            representation.off_diagonal_hidden,
            representation.unembedding,
        )

        preliminary_rows.append(
            {
                "path": path,
                "representation": representation,
                "full_logits": full_logits,
                "diagonal_logits": diagonal_logits,
                "off_diagonal_logits": (
                    off_diagonal_logits
                ),
            }
        )

        preliminary_accuracies.append(
            test_accuracy
        )

        print(
            f"step={representation.step:6d} | "
            f"test={test_accuracy:.4f} | "
            "diag="
            f"{accuracy(diagonal_logits, targets_cpu, test_indices):.4f}"
        )

    (
        reference_index,
        collapse_index,
    ) = select_reference_step(
        paths,
        preliminary_accuracies,
        args.reference_step,
    )

    reference = preliminary_rows[
        reference_index
    ]["representation"]

    reference_diagonal_logits = decode(
        reference.diagonal_hidden,
        reference.unembedding,
    )

    reference_full_logits = decode(
        reference.hidden,
        reference.unembedding,
    )

    reference_step = reference.step
    collapse_step = preliminary_rows[
        collapse_index
    ]["representation"].step

    print()
    print(
        f"Reference step: {reference_step}"
    )
    print(
        f"Collapse step: {collapse_step}"
    )

    rows: list[dict[str, object]] = []

    for item in preliminary_rows:
        representation = item[
            "representation"
        ]

        full_logits = item["full_logits"]
        diagonal_logits = item[
            "diagonal_logits"
        ]
        off_diagonal_logits = item[
            "off_diagonal_logits"
        ]

        current_hidden_reference_readout = (
            decode(
                representation.hidden,
                reference.unembedding,
            )
        )

        reference_hidden_current_readout = (
            decode(
                reference.hidden,
                representation.unembedding,
            )
        )

        current_diagonal_reference_readout = (
            decode(
                representation.diagonal_hidden,
                reference.unembedding,
            )
        )

        reference_diagonal_current_readout = (
            decode(
                reference.diagonal_hidden,
                representation.unembedding,
            )
        )

        (
            rescued_fraction,
            off_interference_advantage,
        ) = interference_metrics(
            full_logits,
            diagonal_logits,
            off_diagonal_logits,
            targets_cpu,
            test_indices,
        )

        row = {
            "step": representation.step,
            "is_reference": int(
                representation.step
                == reference_step
            ),
            "is_detected_collapse": int(
                representation.step
                == collapse_step
            ),
            "train_accuracy": accuracy(
                full_logits,
                targets_cpu,
                train_indices,
            ),
            "test_accuracy": accuracy(
                full_logits,
                targets_cpu,
                test_indices,
            ),
            "diagonal_only_train_accuracy": (
                accuracy(
                    diagonal_logits,
                    targets_cpu,
                    train_indices,
                )
            ),
            "diagonal_only_test_accuracy": (
                accuracy(
                    diagonal_logits,
                    targets_cpu,
                    test_indices,
                )
            ),
            "off_diagonal_only_train_accuracy": (
                accuracy(
                    off_diagonal_logits,
                    targets_cpu,
                    train_indices,
                )
            ),
            "off_diagonal_only_test_accuracy": (
                accuracy(
                    off_diagonal_logits,
                    targets_cpu,
                    test_indices,
                )
            ),
            "full_test_mean_margin": mean_margin(
                full_logits,
                targets_cpu,
                test_indices,
            ),
            "diagonal_test_mean_margin": (
                mean_margin(
                    diagonal_logits,
                    targets_cpu,
                    test_indices,
                )
            ),
            "off_diagonal_test_mean_margin": (
                mean_margin(
                    off_diagonal_logits,
                    targets_cpu,
                    test_indices,
                )
            ),
            "hidden_non_dc_diagonal_power_fraction": (
                diagonal_power_fraction(
                    representation.hidden
                )
            ),
            "sum_explained_hidden_variance": (
                sum_explained_variance(
                    representation.hidden
                )
            ),
            "top_frequency_pairs": "|".join(
                str(value)
                for value
                in representation.top_frequencies
            ),
            "current_hidden_reference_readout_test_accuracy": (
                accuracy(
                    current_hidden_reference_readout,
                    targets_cpu,
                    test_indices,
                )
            ),
            "reference_hidden_current_readout_test_accuracy": (
                accuracy(
                    reference_hidden_current_readout,
                    targets_cpu,
                    test_indices,
                )
            ),
            "current_diagonal_reference_readout_test_accuracy": (
                accuracy(
                    current_diagonal_reference_readout,
                    targets_cpu,
                    test_indices,
                )
            ),
            "reference_diagonal_current_readout_test_accuracy": (
                accuracy(
                    reference_diagonal_current_readout,
                    targets_cpu,
                    test_indices,
                )
            ),
            "relative_unembedding_drift": (
                relative_l2_drift(
                    representation.unembedding,
                    reference.unembedding,
                )
            ),
            "relative_diagonal_hidden_drift": (
                relative_l2_drift(
                    representation.diagonal_hidden,
                    reference.diagonal_hidden,
                )
            ),
            "diagonal_pair_power_cosine_similarity_to_reference": (
                cosine_similarity(
                    representation.diagonal_pair_power,
                    reference.diagonal_pair_power,
                )
            ),
            "top_frequency_jaccard_to_reference": (
                top_frequency_jaccard(
                    representation.top_frequencies,
                    reference.top_frequencies,
                )
            ),
            "diagonal_logit_cosine_similarity_to_reference": (
                cosine_similarity(
                    diagonal_logits,
                    reference_diagonal_logits,
                )
            ),
            "full_logit_cosine_similarity_to_reference": (
                cosine_similarity(
                    full_logits,
                    reference_full_logits,
                )
            ),
            "diagonal_correct_full_wrong_fraction": (
                rescued_fraction
            ),
            "mean_off_diagonal_interference_advantage_on_rescued_errors": (
                off_interference_advantage
            ),
        }

        rows.append(row)

    write_csv(
        args.summary_output,
        rows,
    )

    write_html(
        args.html_output,
        rows,
        reference_step,
        collapse_step,
    )

    collapse_row = rows[
        collapse_index
    ]

    print()
    print("Collapse summary")
    print(
        "Full test accuracy: "
        f"{collapse_row['test_accuracy']:.6f}"
    )
    print(
        "Diagonal-only test accuracy: "
        f"{collapse_row['diagonal_only_test_accuracy']:.6f}"
    )
    print(
        "Current hidden + reference readout: "
        f"{collapse_row['current_hidden_reference_readout_test_accuracy']:.6f}"
    )
    print(
        "Reference hidden + current readout: "
        f"{collapse_row['reference_hidden_current_readout_test_accuracy']:.6f}"
    )
    print(
        "Diagonal power fraction: "
        f"{collapse_row['hidden_non_dc_diagonal_power_fraction']:.6f}"
    )
    print(
        "Power-spectrum cosine to reference: "
        f"{collapse_row['diagonal_pair_power_cosine_similarity_to_reference']:.6f}"
    )
    print(
        "Diagonal-correct/full-wrong fraction: "
        f"{collapse_row['diagonal_correct_full_wrong_fraction']:.6f}"
    )
    print()
    print(
        f"Saved CSV: {args.summary_output}"
    )
    print(
        f"Saved HTML: {args.html_output}"
    )


if __name__ == "__main__":
    main()
