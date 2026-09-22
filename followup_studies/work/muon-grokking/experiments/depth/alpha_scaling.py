from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from depth_fourier_mode_identification import (
    accuracy,
    build_model,
    forward_with_layer_cache,
    get_device,
    load_checkpoint,
    mode_family_masks,
    ordered_full_grid,
    two_dimensional_spectrum,
)
from depth_fourier_hypothesis_tests import (
    dc_mask,
    readout_logits,
    reconstruct,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescale the task-aligned component of the final residual "
            "and measure full-model accuracy against the scale factor. "
            "Answers whether the family's share of representational "
            "power causes the masking failure or only accompanies it."
        )
    )
    parser.add_argument(
        "--paired-summary",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_mode_paired_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        default=300_000,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--family",
        default="addition",
        help=(
            "Mode family treated as task-aligned. The remainder is "
            "everything else, including the constant mode."
        ),
    )
    parser.add_argument(
        "--alpha-maximum",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--alpha-points",
        type=int,
        default=61,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--curve-output",
        type=Path,
        default=Path(
            "runs/alpha_scaling_curve_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--margin-output",
        type=Path,
        default=Path(
            "runs/alpha_scaling_margin_seed_0.csv"
        ),
    )
    return parser.parse_args()


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}.")

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
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def decompose(
    checkpoint_path: Path,
    depth: int,
    family: str,
    device: torch.device,
) -> dict[str, object]:
    """
    Split the final residual into the task-aligned component and the
    remainder, so that their sum is the original residual exactly.
    """
    checkpoint = load_checkpoint(checkpoint_path)
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
    )

    with torch.no_grad():
        logits, stages = forward_with_layer_cache(
            model,
            inputs,
        )

    final_stage = next(
        stage
        for stage in stages
        if stage.stage_name
        == f"layer_{depth}_post_block_residual"
    )

    hidden_grid = (
        final_stage.values
        .reshape(modulus, modulus, -1)
        .detach()
        .float()
        .cpu()
    )

    spectrum, power = two_dimensional_spectrum(
        hidden_grid
    )

    masks = mode_family_masks(modulus)
    family_mask = masks[family]

    if bool(family_mask[0, 0]):
        raise ValueError(
            "The task-aligned family unexpectedly contains the "
            "constant mode."
        )

    remainder_mask = ~family_mask

    family_component = reconstruct(
        spectrum,
        family_mask,
    )
    remainder_component = reconstruct(
        spectrum,
        remainder_mask,
    )

    reconstruction_error = float(
        (
            family_component
            + remainder_component
            - hidden_grid
        )
        .abs()
        .max()
        .item()
    )

    constant_mask = dc_mask(modulus)
    non_dc_mask = ~constant_mask

    family_power = float(
        power[family_mask].sum().item()
    )
    non_dc_power = float(
        power[non_dc_mask].sum().item()
    )
    other_non_dc_power = non_dc_power - family_power

    return {
        "model": model,
        "modulus": modulus,
        "targets": targets,
        "family_component": family_component,
        "remainder_component": remainder_component,
        "baseline_accuracy": accuracy(logits, targets),
        "reconstruction_error": reconstruction_error,
        "family_power": family_power,
        "other_non_dc_power": other_non_dc_power,
        "native_family_share": (
            family_power / non_dc_power
            if non_dc_power > 0.0
            else float("nan")
        ),
    }


def margin_rows(
    state: dict[str, object],
    branch: str,
    step: int,
) -> list[dict[str, object]]:
    """
    Decompose the correct-class margin through the readout.

    The unembedding is a bias-free linear map, so the logits of the
    two components sum exactly to the logits of the full residual and
    the margin decomposition is exact rather than approximate.
    """
    model = state["model"]
    targets = state["targets"].cpu()

    family_logits = readout_logits(
        state["family_component"],
        model,
    ).cpu()
    remainder_logits = readout_logits(
        state["remainder_component"],
        model,
    ).cpu()
    full_logits = family_logits + remainder_logits

    example_count = full_logits.shape[0]
    rows_index = torch.arange(example_count)

    masked = full_logits.clone()
    masked[rows_index, targets] = float("-inf")
    runner_up = masked.argmax(dim=-1)

    def margin_of(values: Tensor) -> Tensor:
        return (
            values[rows_index, targets]
            - values[rows_index, runner_up]
        )

    full_margin = margin_of(full_logits)
    family_margin = margin_of(family_logits)
    remainder_margin = margin_of(remainder_logits)

    correct = full_margin > 0

    rows = []
    for label, values in (
        ("full", full_margin),
        ("task_aligned", family_margin),
        ("remainder", remainder_margin),
    ):
        rows.append(
            {
                "branch": branch,
                "step": step,
                "component": label,
                "mean_margin": float(
                    values.mean().item()
                ),
                "mean_margin_correct": float(
                    values[correct].mean().item()
                )
                if bool(correct.any())
                else float("nan"),
                "mean_margin_incorrect": float(
                    values[~correct].mean().item()
                )
                if bool((~correct).any())
                else float("nan"),
                "fraction_positive": float(
                    (values > 0).float().mean().item()
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_arguments()

    if not args.paired_summary.is_file():
        raise FileNotFoundError(
            f"Missing paired summary: {args.paired_summary}"
        )

    paired = pd.read_csv(args.paired_summary)
    matching = paired[paired["step"] == args.step]

    if matching.empty:
        raise ValueError(
            f"Step {args.step} is not in "
            f"{args.paired_summary}."
        )

    row = matching.iloc[0]
    device = get_device(args.device)

    branches = {
        "control": Path(str(row["control_checkpoint"])),
        "freeze": Path(str(row["freeze_checkpoint"])),
    }

    alphas = torch.linspace(
        0.0,
        args.alpha_maximum,
        args.alpha_points,
    ).tolist()

    curve_rows: list[dict[str, object]] = []
    margin_output_rows: list[dict[str, object]] = []

    for branch, checkpoint_path in branches.items():
        print(f"Decomposing {branch}: {checkpoint_path}")

        state = decompose(
            checkpoint_path=checkpoint_path,
            depth=args.depth,
            family=args.family,
            device=device,
        )

        print(
            f"  baseline accuracy "
            f"{state['baseline_accuracy']:.4f}, "
            f"native family share "
            f"{state['native_family_share']:.4f}, "
            f"reconstruction error "
            f"{state['reconstruction_error']:.2e}"
        )

        margin_output_rows.extend(
            margin_rows(
                state,
                branch=branch,
                step=args.step,
            )
        )

        family_power = state["family_power"]
        other_power = state["other_non_dc_power"]

        for alpha in alphas:
            scaled = (
                alpha * state["family_component"]
                + state["remainder_component"]
            )

            logits = readout_logits(
                scaled,
                state["model"],
            )

            scaled_family_power = (
                alpha ** 2
            ) * family_power

            denominator = (
                scaled_family_power + other_power
            )

            curve_rows.append(
                {
                    "branch": branch,
                    "step": args.step,
                    "checkpoint": str(checkpoint_path),
                    "alpha": alpha,
                    "accuracy": accuracy(
                        logits,
                        state["targets"],
                    ),
                    "family_power_share_non_dc": (
                        scaled_family_power / denominator
                        if denominator > 0.0
                        else float("nan")
                    ),
                    "native_family_share": state[
                        "native_family_share"
                    ],
                    "baseline_accuracy": state[
                        "baseline_accuracy"
                    ],
                }
            )

        del state

    write_csv(args.curve_output, curve_rows)
    write_csv(args.margin_output, margin_output_rows)

    print()
    print(f"Saved alpha curve: {args.curve_output}")
    print(f"Saved margin decomposition: {args.margin_output}")


if __name__ == "__main__":
    main()
