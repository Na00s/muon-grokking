from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data import generate_modular_addition_data
from model import ModularAdditionTransformer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize optimization, stability, and Fourier "
            "generality across the controlled variants."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "runs/generality_manifest.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "runs/generality_summary.csv"
        ),
    )

    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path(
            "runs/generality_analysis.html"
        ),
    )

    return parser.parse_args()


def first_sustained_step(
    frame: pd.DataFrame,
    field: str,
    threshold: float,
    consecutive: int = 5,
) -> int | None:
    values = frame[field].to_numpy()
    steps = frame["step"].to_numpy()

    for index in range(
        len(values)
        - consecutive
        + 1
    ):
        if (
            values[
                index:index + consecutive
            ]
            >= threshold
        ).all():
            return int(
                steps[index]
            )

    return None


def first_step(
    frame: pd.DataFrame,
    field: str,
    threshold: float,
) -> int | None:
    selected = frame[
        frame[field] >= threshold
    ]

    if selected.empty:
        return None

    return int(
        selected.iloc[0]["step"]
    )


def longest_consecutive_count(
    mask: list[bool],
) -> int:
    longest = 0
    current = 0

    for value in mask:
        if value:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return longest


def load_checkpoint(
    path: Path,
) -> dict[str, object]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            f"Invalid checkpoint: {path}"
        )

    return checkpoint


def build_model(
    checkpoint: dict[str, object],
) -> ModularAdditionTransformer:
    configuration = checkpoint.get(
        "model_config"
    )

    if not isinstance(
        configuration,
        dict,
    ):
        arguments = checkpoint[
            "arguments"
        ]

        configuration = {
            "modulus": int(
                arguments["modulus"]
            ),
            "sequence_length": 3,
            "d_model": int(
                arguments["d_model"]
            ),
            "num_heads": int(
                arguments["num_heads"]
            ),
            "d_mlp": int(
                arguments["d_mlp"]
            ),
        }

    model = ModularAdditionTransformer(
        **configuration
    )

    state = checkpoint[
        "model_state_dict"
    ]

    model.load_state_dict(state)
    model.eval()

    return model


@torch.no_grad()
def hidden_and_logits(
    model: ModularAdditionTransformer,
    inputs: Tensor,
) -> tuple[Tensor, Tensor]:
    positions = torch.arange(
        inputs.shape[1]
    )

    x = (
        model.token_embedding(inputs)
        + model.position_embedding(
            positions
        )
    )

    x = (
        x
        + model.transformer_block
        .attention(x)
    )

    x = (
        x
        + model.transformer_block
        .mlp(x)
    )

    hidden = x[:, -1, :]
    logits = model.unembedding(hidden)

    return hidden, logits


def ordered_full_grid(
    modulus: int,
) -> tuple[Tensor, Tensor]:
    values = torch.arange(
        modulus,
        dtype=torch.long,
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

    train_indices = (
        train_inputs[:, 0] * modulus
        + train_inputs[:, 1]
    )

    test_indices = (
        test_inputs[:, 0] * modulus
        + test_inputs[:, 1]
    )

    return (
        train_indices,
        test_indices,
    )


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


def spectral_metrics(
    model: ModularAdditionTransformer,
    modulus: int,
    train_fraction: float,
    seed: int,
) -> dict[str, float]:
    inputs, targets = ordered_full_grid(
        modulus
    )

    (
        train_indices,
        test_indices,
    ) = split_indices(
        modulus,
        train_fraction,
        seed,
    )

    hidden, logits = hidden_and_logits(
        model,
        inputs,
    )

    d_model = hidden.shape[-1]

    hidden_grid = hidden.reshape(
        modulus,
        modulus,
        d_model,
    )

    spectrum = torch.fft.fft2(
        hidden_grid.float(),
        dim=(0, 1),
        norm="ortho",
    )

    power = (
        spectrum.abs()
        .square()
        .sum(dim=-1)
    )

    diagonal_mask = torch.zeros(
        modulus,
        modulus,
        dtype=torch.bool,
    )

    diagonal_indices = torch.arange(
        modulus
    )

    diagonal_mask[
        diagonal_indices,
        diagonal_indices,
    ] = True

    diagonal_spectrum = torch.where(
        diagonal_mask.unsqueeze(-1),
        spectrum,
        torch.zeros_like(
            spectrum
        ),
    )

    off_diagonal_spectrum = torch.where(
        diagonal_mask.unsqueeze(-1),
        torch.zeros_like(
            spectrum
        ),
        spectrum,
    )

    diagonal_hidden = torch.fft.ifft2(
        diagonal_spectrum,
        dim=(0, 1),
        norm="ortho",
    ).real.reshape(
        modulus * modulus,
        d_model,
    )

    off_diagonal_hidden = torch.fft.ifft2(
        off_diagonal_spectrum,
        dim=(0, 1),
        norm="ortho",
    ).real.reshape(
        modulus * modulus,
        d_model,
    )

    diagonal_logits = (
        model.unembedding(
            diagonal_hidden
        )
    )

    off_diagonal_logits = (
        model.unembedding(
            off_diagonal_hidden
        )
    )

    non_dc_total = (
        power.sum()
        - power[0, 0]
    )

    diagonal_non_dc_power = (
        power[
            diagonal_indices[1:],
            diagonal_indices[1:],
        ].sum()
    )

    diagonal_power_fraction = (
        float(
            (
                diagonal_non_dc_power
                / non_dc_total
            ).item()
        )
        if float(
            non_dc_total.item()
        ) > 0
        else 0.0
    )

    pair_powers = []

    for frequency in range(
        1,
        (modulus + 1) // 2,
    ):
        conjugate = (
            -frequency
        ) % modulus

        pair_powers.append(
            power[
                frequency,
                frequency,
            ]
            + power[
                conjugate,
                conjugate,
            ]
        )

    pair_power = torch.stack(
        pair_powers
    )

    diagonal_pair_total = (
        pair_power.sum()
    )

    if float(
        diagonal_pair_total.item()
    ) > 0:
        normalized = (
            pair_power
            / diagonal_pair_total
        )
    else:
        normalized = torch.zeros_like(
            pair_power
        )

    participation_effective_count = (
        float(
            (
                1.0
                / normalized.square().sum()
            ).item()
        )
        if float(
            normalized.square().sum().item()
        ) > 0
        else 0.0
    )

    top_count = min(
        5,
        pair_power.numel(),
    )

    top_indices = torch.topk(
        pair_power,
        k=top_count,
    ).indices

    top_pair_power_fraction = float(
        normalized.index_select(
            0,
            top_indices,
        ).sum().item()
    )

    top_mask = torch.zeros(
        modulus,
        modulus,
        dtype=torch.bool,
    )
    top_mask[0, 0] = True

    for index in top_indices:
        frequency = int(
            index.item()
        ) + 1
        conjugate = (
            -frequency
        ) % modulus

        top_mask[
            frequency,
            frequency,
        ] = True
        top_mask[
            conjugate,
            conjugate,
        ] = True

    top_spectrum = torch.where(
        top_mask.unsqueeze(-1),
        spectrum,
        torch.zeros_like(
            spectrum
        ),
    )

    top_ablated_spectrum = (
        torch.where(
            top_mask.unsqueeze(-1),
            torch.zeros_like(
                spectrum
            ),
            spectrum,
        )
    )

    top_hidden = torch.fft.ifft2(
        top_spectrum,
        dim=(0, 1),
        norm="ortho",
    ).real.reshape(
        modulus * modulus,
        d_model,
    )

    top_ablated_hidden = torch.fft.ifft2(
        top_ablated_spectrum,
        dim=(0, 1),
        norm="ortho",
    ).real.reshape(
        modulus * modulus,
        d_model,
    )

    top_logits = model.unembedding(
        top_hidden
    )

    top_ablated_logits = (
        model.unembedding(
            top_ablated_hidden
        )
    )

    return {
        "final_full_test_accuracy": (
            accuracy(
                logits,
                targets,
                test_indices,
            )
        ),
        "diagonal_only_test_accuracy": (
            accuracy(
                diagonal_logits,
                targets,
                test_indices,
            )
        ),
        "diagonal_ablated_test_accuracy": (
            accuracy(
                off_diagonal_logits,
                targets,
                test_indices,
            )
        ),
        "hidden_non_dc_diagonal_power_fraction": (
            diagonal_power_fraction
        ),
        "effective_diagonal_frequency_pairs": (
            participation_effective_count
        ),
        "top5_diagonal_power_fraction": (
            top_pair_power_fraction
        ),
        "top5_only_test_accuracy": (
            accuracy(
                top_logits,
                targets,
                test_indices,
            )
        ),
        "top5_ablated_test_accuracy": (
            accuracy(
                top_ablated_logits,
                targets,
                test_indices,
            )
        ),
    }


def write_html(
    path: Path,
    summary: pd.DataFrame,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError(
            "HTML output requires plotly."
        ) from error

    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Sustained 95% test step",
            "Speedup versus AdamW",
            "Minimum post-grokking test accuracy",
            "Fourier sufficiency",
            "Fourier necessity",
            "Effective diagonal frequency pairs",
        ),
        vertical_spacing=0.11,
        horizontal_spacing=0.12,
    )

    variants = list(
        summary["variant"].drop_duplicates()
    )

    regimes = [
        "adamw",
        "muon",
        "stable_muon",
    ]

    labels = {
        "adamw": "AdamW",
        "muon": "Muon",
        "stable_muon": "Stable Muon",
    }

    fields = [
        (
            1,
            1,
            "sustained_95_test_step",
        ),
        (
            1,
            2,
            "speedup_vs_adamw",
        ),
        (
            2,
            1,
            "minimum_post_grokking_test_accuracy",
        ),
        (
            2,
            2,
            "diagonal_only_test_accuracy",
        ),
        (
            3,
            1,
            "diagonal_ablated_test_accuracy",
        ),
        (
            3,
            2,
            "effective_diagonal_frequency_pairs",
        ),
    ]

    for row, column, field in fields:
        for regime in regimes:
            selected = (
                summary[
                    summary["regime"]
                    == regime
                ]
                .set_index("variant")
                .reindex(variants)
            )

            figure.add_trace(
                go.Bar(
                    x=variants,
                    y=selected[field],
                    name=labels[regime],
                    legendgroup=regime,
                    showlegend=(
                        row == 1
                        and column == 1
                    ),
                ),
                row=row,
                col=column,
            )

    figure.update_layout(
        title=(
            "Controlled generality experiments"
        ),
        barmode="group",
        height=1_150,
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

    manifest = pd.read_csv(
        args.manifest
    )

    rows = []

    for record in manifest.to_dict(
        orient="records"
    ):
        frame = pd.read_csv(
            record["csv_path"]
        ).sort_values(
            "step"
        )

        sustained_train = (
            first_sustained_step(
                frame,
                "train_accuracy",
                0.999,
            )
        )

        sustained_test = (
            first_sustained_step(
                frame,
                "test_accuracy",
                0.95,
            )
        )

        if sustained_test is not None:
            post = frame[
                frame["step"]
                >= sustained_test
            ]
        else:
            post = frame.iloc[0:0]

        checkpoint = load_checkpoint(
            Path(
                record[
                    "final_checkpoint"
                ]
            )
        )

        model = build_model(
            checkpoint
        )

        spectral = spectral_metrics(
            model=model,
            modulus=int(
                record["modulus"]
            ),
            train_fraction=float(
                record[
                    "train_fraction"
                ]
            ),
            seed=int(
                record["seed"]
            ),
        )

        row = {
            **record,
            "sustained_99p9_train_step": (
                sustained_train
            ),
            "sustained_95_test_step": (
                sustained_test
            ),
            "memorization_plateau_steps": (
                None
                if (
                    sustained_train is None
                    or sustained_test
                    is None
                )
                else (
                    sustained_test
                    - sustained_train
                )
            ),
            "first_99_test_step": (
                first_step(
                    frame,
                    "test_accuracy",
                    0.99,
                )
            ),
            "first_100_test_step": (
                first_step(
                    frame,
                    "test_accuracy",
                    1.0,
                )
            ),
            "post_grokking_test_below_95_count": (
                0
                if post.empty
                else int(
                    (
                        post[
                            "test_accuracy"
                        ] < 0.95
                    ).sum()
                )
            ),
            "post_grokking_test_below_90_count": (
                0
                if post.empty
                else int(
                    (
                        post[
                            "test_accuracy"
                        ] < 0.90
                    ).sum()
                )
            ),
            "post_grokking_longest_below_95_run": (
                0
                if post.empty
                else longest_consecutive_count(
                    list(
                        post[
                            "test_accuracy"
                        ] < 0.95
                    )
                )
            ),
            "minimum_post_grokking_test_accuracy": (
                None
                if post.empty
                else float(
                    post[
                        "test_accuracy"
                    ].min()
                )
            ),
            "final_test_accuracy": float(
                frame.iloc[-1][
                    "test_accuracy"
                ]
            ),
            "actual_auxiliary_freeze_step": (
                None
            ),
            **spectral,
        }

        if (
            "auxiliary_frozen"
            in frame.columns
            and (
                frame[
                    "auxiliary_frozen"
                ] == 1
            ).any()
        ):
            row[
                "actual_auxiliary_freeze_step"
            ] = int(
                frame[
                    frame[
                        "auxiliary_frozen"
                    ] == 1
                ].iloc[0]["step"]
            )

        rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    adamw_steps = (
        summary[
            summary["regime"]
            == "adamw"
        ][
            [
                "variant",
                "sustained_95_test_step",
            ]
        ]
        .rename(
            columns={
                "sustained_95_test_step": (
                    "adamw_sustained_95_step"
                )
            }
        )
    )

    summary = summary.merge(
        adamw_steps,
        on="variant",
        how="left",
    )

    summary[
        "speedup_vs_adamw"
    ] = (
        summary[
            "adamw_sustained_95_step"
        ]
        / summary[
            "sustained_95_test_step"
        ]
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        args.summary_output,
        index=False,
    )

    write_html(
        args.html_output,
        summary,
    )

    display_columns = [
        "variant",
        "regime",
        "sustained_95_test_step",
        "speedup_vs_adamw",
        "minimum_post_grokking_test_accuracy",
        "final_test_accuracy",
        "diagonal_only_test_accuracy",
        "diagonal_ablated_test_accuracy",
        "effective_diagonal_frequency_pairs",
        "top5_only_test_accuracy",
        "top5_ablated_test_accuracy",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved summary: "
        f"{args.summary_output}"
    )
    print(
        f"Saved HTML: "
        f"{args.html_output}"
    )


if __name__ == "__main__":
    main()
