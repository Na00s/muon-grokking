from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from data import generate_modular_addition_data
from model import ModularAdditionTransformer


MODULUS = 113
TRAIN_FRACTION = 0.3
SEQUENCE_LENGTH = 3
D_MODEL = 128
NUMBER_OF_HEADS = 4
D_MLP = 512

CHECKPOINT_PATTERN = re.compile(r"step_(\d+)\.pt$")


@dataclass(frozen=True)
class RunSpecification:
    label: str
    checkpoint_directory: Path


@dataclass
class ForwardCache:
    logits: Tensor
    final_hidden: Tensor
    attention_output: Tensor
    mlp_activation: Tensor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track Fourier-circuit formation across AdamW, "
            "ordinary Muon, and stabilized Muon checkpoints."
        )
    )

    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=CHECKPOINT_DIRECTORY",
        help=(
            "Run label and checkpoint directory. Repeat once "
            "per run."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )

    parser.add_argument(
        "--checkpoint-stride",
        type=int,
        default=1_000,
        help=(
            "Analyze checkpoints whose step is divisible by "
            "this value."
        ),
    )

    parser.add_argument(
        "--include-step",
        type=int,
        action="append",
        default=[],
        help=(
            "Always analyze this exact checkpoint step, even "
            "when it is not divisible by the stride."
        ),
    )

    parser.add_argument(
        "--top-frequency-pairs",
        type=int,
        default=5,
        help=(
            "Number of nonzero conjugate Fourier-frequency "
            "pairs retained in the top-frequency sufficiency "
            "test."
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("runs/fourier_circuit_summary.csv"),
    )

    parser.add_argument(
        "--frequency-output",
        type=Path,
        default=Path("runs/fourier_mode_details.csv"),
    )

    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path("runs/fourier_circuit_analysis.html"),
    )

    return parser.parse_args()


def parse_run_specification(value: str) -> RunSpecification:
    if "=" not in value:
        raise ValueError(
            "--run must have the form "
            "LABEL=CHECKPOINT_DIRECTORY."
        )

    label, directory = value.split("=", 1)
    label = label.strip()
    directory = directory.strip()

    if not label:
        raise ValueError("Run label cannot be empty.")

    if not directory:
        raise ValueError(
            "Checkpoint directory cannot be empty."
        )

    path = Path(directory)

    if not path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {path}"
        )

    return RunSpecification(
        label=label,
        checkpoint_directory=path,
    )


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

    raise RuntimeError(
        "No CUDA or MPS device is available."
    )


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.search(path.name)

    if match is None:
        raise ValueError(
            f"Unrecognized checkpoint filename: {path.name}"
        )

    return int(match.group(1))


def discover_checkpoints(
    directory: Path,
    stride: int,
    included_steps: set[int],
) -> list[Path]:
    paths = []

    for path in directory.glob("step_*.pt"):
        try:
            step = checkpoint_step(path)
        except ValueError:
            continue

        if (
            step % stride == 0
            or step in included_steps
        ):
            paths.append(path)

    paths.sort(key=checkpoint_step)

    if not paths:
        raise FileNotFoundError(
            "No matching checkpoints found in "
            f"{directory}."
        )

    return paths


def unwrap_state_dict(loaded: object) -> dict[str, Tensor]:
    if not isinstance(loaded, dict):
        raise TypeError(
            "Checkpoint must contain a dictionary."
        )

    if "model_state_dict" in loaded:
        state = loaded["model_state_dict"]
    else:
        state = loaded

    if not isinstance(state, dict):
        raise TypeError(
            "model_state_dict must be a dictionary."
        )

    return state


def build_model(
    checkpoint_path: Path,
    device: torch.device,
) -> ModularAdditionTransformer:
    try:
        loaded = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        loaded = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    state = unwrap_state_dict(loaded)

    model = ModularAdditionTransformer(
        modulus=MODULUS,
        sequence_length=SEQUENCE_LENGTH,
        d_model=D_MODEL,
        num_heads=NUMBER_OF_HEADS,
        d_mlp=D_MLP,
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model


def ordered_full_grid(
    modulus: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    values = torch.arange(
        modulus,
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
        modulus,
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
    ).remainder(modulus).reshape(-1)

    return inputs, targets


def split_indices(
    modulus: int,
    train_fraction: float,
    seed: int,
    device: torch.device,
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
    )

    train_flat = (
        train_inputs[:, 0] * modulus
        + train_inputs[:, 1]
    ).to(device)

    test_flat = (
        test_inputs[:, 0] * modulus
        + test_inputs[:, 1]
    ).to(device)

    return train_flat, test_flat


@torch.no_grad()
def forward_with_cache(
    model: ModularAdditionTransformer,
    token_ids: Tensor,
) -> ForwardCache:
    _, sequence_length = token_ids.shape

    positions = torch.arange(
        sequence_length,
        device=token_ids.device,
    )

    token_embeddings = model.token_embedding(
        token_ids
    )

    position_embeddings = model.position_embedding(
        positions
    )

    residual_before_attention = (
        token_embeddings
        + position_embeddings
    )

    attention_output = (
        model
        .transformer_block
        .attention(
            residual_before_attention
        )
    )

    residual_after_attention = (
        residual_before_attention
        + attention_output
    )

    mlp_pre_activation = (
        model
        .transformer_block
        .mlp
        .input_projection(
            residual_after_attention
        )
    )

    mlp_activation = F.relu(
        mlp_pre_activation
    )

    mlp_output = (
        model
        .transformer_block
        .mlp
        .output_projection(
            mlp_activation
        )
    )

    final_residual = (
        residual_after_attention
        + mlp_output
    )

    final_hidden = final_residual[:, -1, :]
    logits = model.unembedding(final_hidden)

    return ForwardCache(
        logits=logits,
        final_hidden=final_hidden,
        attention_output=attention_output[:, -1, :],
        mlp_activation=mlp_activation[:, -1, :],
    )


def accuracy(
    logits: Tensor,
    targets: Tensor,
    indices: Tensor | None = None,
) -> float:
    if indices is not None:
        logits = logits.index_select(0, indices)
        targets = targets.index_select(0, indices)

    predictions = logits.argmax(dim=-1)

    return float(
        (
            predictions == targets
        )
        .float()
        .mean()
        .item()
    )


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    indices: Tensor | None = None,
) -> float:
    if indices is not None:
        logits = logits.index_select(0, indices)
        targets = targets.index_select(0, indices)

    return float(
        F.cross_entropy(
            logits,
            targets,
        ).item()
    )


def one_dimensional_power(
    values: Tensor,
) -> Tensor:
    # Prime-length FFTs are not consistently supported by
    # every MPS backend. Model inference stays on the requested
    # accelerator, while Fourier transforms are performed on
    # CPU for deterministic support of modulus 113.
    values = values.detach().float().cpu()

    spectrum = torch.fft.fft(
        values,
        dim=0,
        norm="ortho",
    )

    return (
        spectrum.abs()
        .square()
        .sum(dim=1)
    )


def two_dimensional_power(
    values: Tensor,
) -> tuple[Tensor, Tensor]:
    values = values.detach().float().cpu()

    spectrum = torch.fft.fft2(
        values,
        dim=(0, 1),
        norm="ortho",
    )

    power = (
        spectrum.abs()
        .square()
        .sum(dim=-1)
    )

    return spectrum, power


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


def diagonal_power_fraction(
    power: Tensor,
    include_dc: bool,
) -> float:
    modulus = power.shape[0]
    indices = torch.arange(
        modulus,
        device=power.device,
    )

    diagonal = power[indices, indices]

    if not include_dc:
        diagonal = diagonal[1:]

    total = power.sum()

    if not include_dc:
        total = total - power[0, 0]

    return safe_ratio(
        diagonal.sum(),
        total,
    )


def canonical_frequency_pairs(
    power: Tensor,
) -> list[tuple[int, float]]:
    modulus = power.shape[0]
    pairs: list[tuple[int, float]] = []

    for frequency in range(
        1,
        (modulus + 1) // 2,
    ):
        conjugate = (-frequency) % modulus

        pair_power = (
            power[frequency, frequency]
            + power[conjugate, conjugate]
        )

        pairs.append(
            (
                frequency,
                float(pair_power.item()),
            )
        )

    pairs.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return pairs


def spectral_mask(
    modulus: int,
    frequencies: Iterable[int],
    device: torch.device,
    include_dc: bool,
) -> Tensor:
    mask = torch.zeros(
        modulus,
        modulus,
        dtype=torch.bool,
        device=device,
    )

    if include_dc:
        mask[0, 0] = True

    for frequency in frequencies:
        frequency = int(frequency) % modulus
        conjugate = (-frequency) % modulus

        mask[frequency, frequency] = True
        mask[conjugate, conjugate] = True

    return mask


def all_diagonal_mask(
    modulus: int,
    device: torch.device,
) -> Tensor:
    mask = torch.zeros(
        modulus,
        modulus,
        dtype=torch.bool,
        device=device,
    )

    indices = torch.arange(
        modulus,
        device=device,
    )

    mask[indices, indices] = True

    return mask


def filter_hidden_spectrum(
    spectrum: Tensor,
    mask: Tensor,
    keep_masked: bool,
) -> Tensor:
    expanded_mask = mask.unsqueeze(-1)

    if keep_masked:
        filtered = torch.where(
            expanded_mask,
            spectrum,
            torch.zeros_like(spectrum),
        )
    else:
        filtered = torch.where(
            expanded_mask,
            torch.zeros_like(spectrum),
            spectrum,
        )

    reconstructed = torch.fft.ifft2(
        filtered,
        dim=(0, 1),
        norm="ortho",
    ).real

    return reconstructed


def logits_from_hidden(
    model: ModularAdditionTransformer,
    hidden_grid: Tensor,
) -> Tensor:
    model_device = (
        model.unembedding.weight.device
    )

    flat_hidden = hidden_grid.reshape(
        MODULUS * MODULUS,
        -1,
    ).to(model_device)

    return model.unembedding(flat_hidden)


def sum_centroid_hidden(
    hidden_grid: Tensor,
) -> Tensor:
    modulus = hidden_grid.shape[0]
    device = hidden_grid.device

    a = torch.arange(
        modulus,
        device=device,
    ).view(modulus, 1)

    b = torch.arange(
        modulus,
        device=device,
    ).view(1, modulus)

    sums = (a + b).remainder(modulus)

    flat_hidden = hidden_grid.reshape(
        modulus * modulus,
        -1,
    )
    flat_sums = sums.reshape(-1)

    centroids = torch.zeros(
        modulus,
        hidden_grid.shape[-1],
        dtype=hidden_grid.dtype,
        device=device,
    )

    centroids.index_add_(
        0,
        flat_sums,
        flat_hidden,
    )

    centroids = centroids / modulus

    reconstructed = centroids.index_select(
        0,
        flat_sums,
    )

    return reconstructed.reshape_as(
        hidden_grid
    )


def sum_explained_variance(
    hidden_grid: Tensor,
) -> float:
    global_mean = hidden_grid.mean(
        dim=(0, 1),
        keepdim=True,
    )

    total = (
        hidden_grid - global_mean
    ).square().sum()

    centroid_grid = sum_centroid_hidden(
        hidden_grid
    )

    residual = (
        hidden_grid - centroid_grid
    ).square().sum()

    if float(total.item()) <= 0.0:
        return 0.0

    return float(
        (
            1.0 - residual / total
        ).item()
    )


def aligned_logit_consistency(
    logits_grid: Tensor,
) -> float:
    modulus = logits_grid.shape[0]
    device = logits_grid.device

    a = torch.arange(
        modulus,
        device=device,
    ).view(modulus, 1, 1)

    b = torch.arange(
        modulus,
        device=device,
    ).view(1, modulus, 1)

    relative_classes = torch.arange(
        modulus,
        device=device,
    ).view(1, 1, modulus)

    absolute_classes = (
        relative_classes + a + b
    ).remainder(modulus)

    aligned = torch.gather(
        logits_grid,
        dim=2,
        index=absolute_classes.expand(
            modulus,
            modulus,
            modulus,
        ),
    )

    global_mean = aligned.mean()
    total = (
        aligned - global_mean
    ).square().sum()

    mean_profile = aligned.mean(
        dim=(0, 1),
        keepdim=True,
    )

    residual = (
        aligned - mean_profile
    ).square().sum()

    if float(total.item()) <= 0.0:
        return 0.0

    return float(
        (
            1.0 - residual / total
        ).item()
    )


def top_power_frequencies(
    power: Tensor,
    count: int,
) -> list[int]:
    pairs = canonical_frequency_pairs(
        power
    )

    return [
        frequency
        for frequency, _
        in pairs[:count]
    ]


def normalized_mode_power(
    power: Tensor,
    frequency: int,
) -> float:
    modulus = power.shape[0]
    conjugate = (-frequency) % modulus
    total = power.sum()

    pair_power = (
        power[frequency]
        + power[conjugate]
    )

    return safe_ratio(
        pair_power,
        total,
    )


def normalized_diagonal_pair_power(
    power: Tensor,
    frequency: int,
) -> float:
    modulus = power.shape[0]
    conjugate = (-frequency) % modulus
    total = power.sum()

    pair_power = (
        power[frequency, frequency]
        + power[conjugate, conjugate]
    )

    return safe_ratio(
        pair_power,
        total,
    )


def analyze_checkpoint(
    run_label: str,
    checkpoint_path: Path,
    device: torch.device,
    full_inputs: Tensor,
    full_targets: Tensor,
    train_indices: Tensor,
    test_indices: Tensor,
    top_pair_count: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    step = checkpoint_step(
        checkpoint_path
    )

    model = build_model(
        checkpoint_path,
        device,
    )

    cache = forward_with_cache(
        model,
        full_inputs,
    )

    hidden_grid = cache.final_hidden.reshape(
        MODULUS,
        MODULUS,
        -1,
    )

    attention_grid = (
        cache.attention_output.reshape(
            MODULUS,
            MODULUS,
            -1,
        )
    )

    mlp_grid = cache.mlp_activation.reshape(
        MODULUS,
        MODULUS,
        -1,
    )

    logits_grid = cache.logits.reshape(
        MODULUS,
        MODULUS,
        MODULUS,
    )

    hidden_spectrum, hidden_power = (
        two_dimensional_power(
            hidden_grid
        )
    )

    _, attention_power = (
        two_dimensional_power(
            attention_grid
        )
    )

    _, mlp_power = (
        two_dimensional_power(
            mlp_grid
        )
    )

    token_power = one_dimensional_power(
        model.token_embedding.weight[
            :MODULUS
        ]
    )

    unembedding_power = (
        one_dimensional_power(
            model.unembedding.weight
        )
    )

    top_frequencies = (
        top_power_frequencies(
            hidden_power,
            top_pair_count,
        )
    )

    spectral_device = (
        hidden_spectrum.device
    )

    top_mask = spectral_mask(
        modulus=MODULUS,
        frequencies=top_frequencies,
        device=spectral_device,
        include_dc=True,
    )

    diagonal_mask = all_diagonal_mask(
        modulus=MODULUS,
        device=spectral_device,
    )

    top_only_hidden = (
        filter_hidden_spectrum(
            hidden_spectrum,
            top_mask,
            keep_masked=True,
        )
    )

    top_ablated_hidden = (
        filter_hidden_spectrum(
            hidden_spectrum,
            top_mask,
            keep_masked=False,
        )
    )

    diagonal_only_hidden = (
        filter_hidden_spectrum(
            hidden_spectrum,
            diagonal_mask,
            keep_masked=True,
        )
    )

    diagonal_ablated_hidden = (
        filter_hidden_spectrum(
            hidden_spectrum,
            diagonal_mask,
            keep_masked=False,
        )
    )

    centroid_hidden = sum_centroid_hidden(
        hidden_grid
    )

    top_only_logits = logits_from_hidden(
        model,
        top_only_hidden,
    )

    top_ablated_logits = logits_from_hidden(
        model,
        top_ablated_hidden,
    )

    diagonal_only_logits = logits_from_hidden(
        model,
        diagonal_only_hidden,
    )

    diagonal_ablated_logits = logits_from_hidden(
        model,
        diagonal_ablated_hidden,
    )

    centroid_logits = logits_from_hidden(
        model,
        centroid_hidden,
    )

    retained_top_power = (
        hidden_power[top_mask].sum()
    )

    summary: dict[str, object] = {
        "run": run_label,
        "step": step,
        "checkpoint": str(
            checkpoint_path
        ),
        "train_loss": cross_entropy(
            cache.logits,
            full_targets,
            train_indices,
        ),
        "train_accuracy": accuracy(
            cache.logits,
            full_targets,
            train_indices,
        ),
        "test_loss": cross_entropy(
            cache.logits,
            full_targets,
            test_indices,
        ),
        "test_accuracy": accuracy(
            cache.logits,
            full_targets,
            test_indices,
        ),
        "full_accuracy": accuracy(
            cache.logits,
            full_targets,
        ),
        "hidden_diagonal_power_fraction": (
            diagonal_power_fraction(
                hidden_power,
                include_dc=True,
            )
        ),
        "hidden_non_dc_diagonal_power_fraction": (
            diagonal_power_fraction(
                hidden_power,
                include_dc=False,
            )
        ),
        "attention_diagonal_power_fraction": (
            diagonal_power_fraction(
                attention_power,
                include_dc=True,
            )
        ),
        "mlp_diagonal_power_fraction": (
            diagonal_power_fraction(
                mlp_power,
                include_dc=True,
            )
        ),
        "sum_explained_hidden_variance": (
            sum_explained_variance(
                hidden_grid
            )
        ),
        "aligned_logit_consistency": (
            aligned_logit_consistency(
                logits_grid
            )
        ),
        "top_frequency_pairs": "|".join(
            str(frequency)
            for frequency
            in top_frequencies
        ),
        "top_frequency_power_fraction": (
            safe_ratio(
                retained_top_power,
                hidden_power.sum(),
            )
        ),
        "top_frequency_sufficiency_train_accuracy": (
            accuracy(
                top_only_logits,
                full_targets,
                train_indices,
            )
        ),
        "top_frequency_sufficiency_test_accuracy": (
            accuracy(
                top_only_logits,
                full_targets,
                test_indices,
            )
        ),
        "top_frequency_ablation_train_accuracy": (
            accuracy(
                top_ablated_logits,
                full_targets,
                train_indices,
            )
        ),
        "top_frequency_ablation_test_accuracy": (
            accuracy(
                top_ablated_logits,
                full_targets,
                test_indices,
            )
        ),
        "diagonal_sufficiency_train_accuracy": (
            accuracy(
                diagonal_only_logits,
                full_targets,
                train_indices,
            )
        ),
        "diagonal_sufficiency_test_accuracy": (
            accuracy(
                diagonal_only_logits,
                full_targets,
                test_indices,
            )
        ),
        "diagonal_ablation_train_accuracy": (
            accuracy(
                diagonal_ablated_logits,
                full_targets,
                train_indices,
            )
        ),
        "diagonal_ablation_test_accuracy": (
            accuracy(
                diagonal_ablated_logits,
                full_targets,
                test_indices,
            )
        ),
        "sum_centroid_train_accuracy": (
            accuracy(
                centroid_logits,
                full_targets,
                train_indices,
            )
        ),
        "sum_centroid_test_accuracy": (
            accuracy(
                centroid_logits,
                full_targets,
                test_indices,
            )
        ),
    }

    frequency_rows: list[
        dict[str, object]
    ] = []

    ranked_pairs = canonical_frequency_pairs(
        hidden_power
    )

    rank_by_frequency = {
        frequency: rank
        for rank, (frequency, _)
        in enumerate(
            ranked_pairs,
            start=1,
        )
    }

    for frequency in range(
        1,
        (MODULUS + 1) // 2,
    ):
        frequency_rows.append(
            {
                "run": run_label,
                "step": step,
                "frequency": frequency,
                "hidden_rank": (
                    rank_by_frequency[
                        frequency
                    ]
                ),
                "hidden_diagonal_pair_power_fraction": (
                    normalized_diagonal_pair_power(
                        hidden_power,
                        frequency,
                    )
                ),
                "attention_diagonal_pair_power_fraction": (
                    normalized_diagonal_pair_power(
                        attention_power,
                        frequency,
                    )
                ),
                "mlp_diagonal_pair_power_fraction": (
                    normalized_diagonal_pair_power(
                        mlp_power,
                        frequency,
                    )
                ),
                "token_embedding_pair_power_fraction": (
                    normalized_mode_power(
                        token_power,
                        frequency,
                    )
                ),
                "unembedding_pair_power_fraction": (
                    normalized_mode_power(
                        unembedding_power,
                        frequency,
                    )
                ),
            }
        )

    del model
    del cache

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return summary, frequency_rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows available for {path}."
        )

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
    summaries: list[dict[str, object]],
) -> None:
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError(
            "The HTML report requires pandas "
            "and plotly."
        ) from error

    frame = pd.DataFrame(summaries)

    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Test accuracy",
            "Hidden diagonal Fourier power",
            "Sum-explained hidden variance",
            "Aligned-logit consistency",
            "Top-frequency sufficiency",
            "Top-frequency ablation",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    for run_name, run_frame in frame.groupby(
        "run",
        sort=False,
    ):
        run_frame = run_frame.sort_values(
            "step"
        )

        traces = [
            (
                1,
                1,
                "test_accuracy",
                run_name,
            ),
            (
                1,
                2,
                "hidden_non_dc_diagonal_power_fraction",
                run_name,
            ),
            (
                2,
                1,
                "sum_explained_hidden_variance",
                run_name,
            ),
            (
                2,
                2,
                "aligned_logit_consistency",
                run_name,
            ),
            (
                3,
                1,
                "top_frequency_sufficiency_test_accuracy",
                run_name,
            ),
            (
                3,
                2,
                "top_frequency_ablation_test_accuracy",
                run_name,
            ),
        ]

        for row, column, field, name in traces:
            figure.add_trace(
                go.Scatter(
                    x=run_frame["step"],
                    y=run_frame[field],
                    mode="lines+markers",
                    name=name,
                    legendgroup=name,
                    showlegend=(
                        row == 1
                        and column == 1
                    ),
                    hovertemplate=(
                        "run=%{fullData.name}<br>"
                        "step=%{x}<br>"
                        "value=%{y:.6f}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )

    figure.update_xaxes(
        title_text="Training step",
    )

    figure.update_yaxes(
        range=[-0.02, 1.02],
    )

    figure.update_layout(
        title=(
            "Fourier circuit formation and "
            "spectral causal tests"
        ),
        height=1_150,
        hovermode="x unified",
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

    if args.checkpoint_stride < 1:
        raise ValueError(
            "--checkpoint-stride must be at least 1."
        )

    if args.top_frequency_pairs < 1:
        raise ValueError(
            "--top-frequency-pairs must be at least 1."
        )

    run_specifications = [
        parse_run_specification(value)
        for value in args.run
    ]

    device = get_device(
        args.device
    )

    full_inputs, full_targets = (
        ordered_full_grid(
            modulus=MODULUS,
            device=device,
        )
    )

    train_indices, test_indices = (
        split_indices(
            modulus=MODULUS,
            train_fraction=TRAIN_FRACTION,
            seed=0,
            device=device,
        )
    )

    print(f"Device: {device}")
    print(
        f"Top frequency pairs: "
        f"{args.top_frequency_pairs}"
    )

    summaries: list[
        dict[str, object]
    ] = []

    frequency_rows: list[
        dict[str, object]
    ] = []

    included_steps = set(
        args.include_step
    )

    for specification in run_specifications:
        checkpoints = discover_checkpoints(
            directory=(
                specification
                .checkpoint_directory
            ),
            stride=args.checkpoint_stride,
            included_steps=included_steps,
        )

        print()
        print(
            f"{specification.label}: "
            f"{len(checkpoints)} checkpoints"
        )

        for checkpoint_path in checkpoints:
            step = checkpoint_step(
                checkpoint_path
            )

            print(
                f"Analyzing "
                f"{specification.label} "
                f"step {step}..."
            )

            summary, modes = analyze_checkpoint(
                run_label=(
                    specification.label
                ),
                checkpoint_path=(
                    checkpoint_path
                ),
                device=device,
                full_inputs=full_inputs,
                full_targets=full_targets,
                train_indices=train_indices,
                test_indices=test_indices,
                top_pair_count=(
                    args.top_frequency_pairs
                ),
            )

            summaries.append(summary)
            frequency_rows.extend(modes)

            print(
                "  "
                f"test={summary['test_accuracy']:.4f} | "
                "diag="
                f"{summary['hidden_non_dc_diagonal_power_fraction']:.4f} | "
                "top-k suff="
                f"{summary['top_frequency_sufficiency_test_accuracy']:.4f} | "
                "top-k ablate="
                f"{summary['top_frequency_ablation_test_accuracy']:.4f}"
            )

    write_csv(
        args.summary_output,
        summaries,
    )

    write_csv(
        args.frequency_output,
        frequency_rows,
    )

    write_html(
        args.html_output,
        summaries,
    )

    print()
    print(
        f"Saved summary to: "
        f"{args.summary_output}"
    )
    print(
        f"Saved frequency details to: "
        f"{args.frequency_output}"
    )
    print(
        f"Saved HTML report to: "
        f"{args.html_output}"
    )


if __name__ == "__main__":
    main()
