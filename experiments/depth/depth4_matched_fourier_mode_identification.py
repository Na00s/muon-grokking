from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from depth_fourier_mode_identification import (
    MODE_FAMILIES,
    accuracy,
    build_model,
    conjugate_mode_pairs,
    family_statistics,
    forward_with_layer_cache,
    get_device,
    load_checkpoint,
    mode_family_masks,
    ordered_full_grid,
    release_accelerator_cache,
    sum_explained_variance,
    top_pair_statistics,
    two_dimensional_spectrum,
    write_csv,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify how the complete 2D Fourier mode structure "
            "diverges between the exact matched depth-four control "
            "and freeze branches. This phase performs no Fourier "
            "sufficiency, ablation, or cross-readout intervention."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--top-mode-pairs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_mode_paired_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--layer-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_mode_layerwise_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--mode-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_mode_inventory_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_fourier_mode_identification_seed_0.html"
        ),
    )
    return parser.parse_args()


def newest_metadata() -> Path:
    matches = sorted(
        Path("runs").glob(
            "depth4_matched_freeze_seed_*_from_*_metadata.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            "No depth-four matched-freeze metadata was found."
        )
    return matches[-1]


def checkpoint_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("step_"):
        raise ValueError(f"Unrecognized checkpoint: {path}")
    return int(stem.split("_", 1)[1])


def discover(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing branch checkpoint directory: {directory}"
        )
    return {
        checkpoint_step(path): path
        for path in directory.glob("step_*.pt")
    }


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
    if float(denominator.item()) <= 0.0:
        return 0.0
    return float(
        torch.dot(first, second).item()
        / denominator.item()
    )


def masked_power_cosine(
    first_power: Tensor,
    second_power: Tensor,
    mask: Tensor,
) -> float:
    return cosine_similarity(
        first_power[mask],
        second_power[mask],
    )


def top_mode_set(
    power: Tensor,
    count: int,
) -> set[tuple[int, int]]:
    pairs = [
        row
        for row in conjugate_mode_pairs(power)
        if row["family"] != "dc"
    ]
    return {
        (
            int(row["representative_k"]),
            int(row["representative_l"]),
        )
        for row in pairs[:count]
    }


def top_overlap(
    first_power: Tensor,
    second_power: Tensor,
    count: int,
) -> float:
    first = top_mode_set(first_power, count)
    second = top_mode_set(second_power, count)
    return len(first & second) / max(1, count)


def readout_weight_cosine(
    control_model,
    freeze_model,
) -> float:
    return cosine_similarity(
        control_model.unembedding.weight.detach().cpu(),
        freeze_model.unembedding.weight.detach().cpu(),
    )


def aligned_stage_map(stages) -> dict[str, object]:
    return {
        stage.stage_name: stage
        for stage in stages
    }


def prefixed_family_stats(
    prefix: str,
    statistics: dict[str, object],
) -> dict[str, object]:
    return {
        f"{prefix}_{key}": value
        for key, value in statistics.items()
    }


def main() -> None:
    args = parse_arguments()
    metadata_path = (
        args.metadata
        if args.metadata is not None
        else newest_metadata()
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing metadata: {metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    branch_step = int(metadata["branch_step"])
    shared_checkpoint = Path(metadata["shared_checkpoint"])

    prefix = metadata_path.name.removesuffix(
        "_metadata.json"
    )
    control_directory = (
        Path("checkpoints") / f"{prefix}_control"
    )
    freeze_directory = (
        Path("checkpoints") / f"{prefix}_freeze"
    )

    control_map = discover(control_directory)
    freeze_map = discover(freeze_directory)

    common_steps = sorted(
        set(control_map) & set(freeze_map)
    )
    if not common_steps:
        raise RuntimeError(
            "The control and freeze branches have no common "
            "checkpoint steps."
        )

    paired_paths = [
        (
            branch_step,
            shared_checkpoint,
            shared_checkpoint,
        )
    ]
    paired_paths.extend(
        (
            step,
            control_map[step],
            freeze_map[step],
        )
        for step in common_steps
        if step > branch_step
    )

    device = get_device(args.device)
    paired_rows = []
    layer_rows = []
    mode_rows = []

    print(f"Device: {device}")
    print(f"Matched branch step: {branch_step}")
    print(f"Paired checkpoints: {len(paired_paths)}")

    for step, control_path, freeze_path in paired_paths:
        print(f"Analyzing paired step {step}")

        control_checkpoint = load_checkpoint(control_path)
        freeze_checkpoint = load_checkpoint(freeze_path)

        control_model = build_model(
            control_checkpoint,
            expected_depth=4,
            device=device,
        )
        freeze_model = build_model(
            freeze_checkpoint,
            expected_depth=4,
            device=device,
        )

        modulus = int(
            control_checkpoint["model_config"]["modulus"]
        )
        masks = mode_family_masks(modulus)
        full_inputs, full_targets = ordered_full_grid(
            modulus,
            device,
        )

        with torch.no_grad():
            (
                control_logits,
                control_stages,
            ) = forward_with_layer_cache(
                control_model,
                full_inputs,
            )
            (
                freeze_logits,
                freeze_stages,
            ) = forward_with_layer_cache(
                freeze_model,
                full_inputs,
            )

        control_stage_map = aligned_stage_map(
            control_stages
        )
        freeze_stage_map = aligned_stage_map(
            freeze_stages
        )

        if set(control_stage_map) != set(freeze_stage_map):
            raise RuntimeError(
                "Matched branch stage names do not align."
            )

        final_stage_name = (
            "layer_4_post_block_residual"
        )

        paired_row: dict[str, object] = {
            "step": step,
            "branch_step": branch_step,
            "control_checkpoint": str(control_path),
            "freeze_checkpoint": str(freeze_path),
            "control_full_accuracy": accuracy(
                control_logits,
                full_targets,
            ),
            "freeze_full_accuracy": accuracy(
                freeze_logits,
                full_targets,
            ),
            "paired_accuracy_advantage_freeze": (
                accuracy(
                    freeze_logits,
                    full_targets,
                )
                - accuracy(
                    control_logits,
                    full_targets,
                )
            ),
            "readout_weight_cosine": (
                readout_weight_cosine(
                    control_model,
                    freeze_model,
                )
            ),
        }

        for stage_name in sorted(
            control_stage_map,
            key=lambda name: control_stage_map[name].order,
        ):
            control_stage = control_stage_map[stage_name]
            freeze_stage = freeze_stage_map[stage_name]

            control_grid = control_stage.values.reshape(
                modulus,
                modulus,
                -1,
            )
            freeze_grid = freeze_stage.values.reshape(
                modulus,
                modulus,
                -1,
            )

            _, control_power = two_dimensional_spectrum(
                control_grid
            )
            _, freeze_power = two_dimensional_spectrum(
                freeze_grid
            )

            control_family = family_statistics(
                control_power,
                masks,
            )
            freeze_family = family_statistics(
                freeze_power,
                masks,
            )

            control_pairs = conjugate_mode_pairs(
                control_power
            )
            freeze_pairs = conjugate_mode_pairs(
                freeze_power
            )

            control_top_summary, control_top_details = (
                top_pair_statistics(
                    control_pairs,
                    total_power=float(
                        control_family["total_spectral_power"]
                    ),
                    non_dc_power=float(
                        control_family["non_dc_spectral_power"]
                    ),
                    top_count=args.top_mode_pairs,
                )
            )
            freeze_top_summary, freeze_top_details = (
                top_pair_statistics(
                    freeze_pairs,
                    total_power=float(
                        freeze_family["total_spectral_power"]
                    ),
                    non_dc_power=float(
                        freeze_family["non_dc_spectral_power"]
                    ),
                    top_count=args.top_mode_pairs,
                )
            )

            row: dict[str, object] = {
                "step": step,
                "branch_step": branch_step,
                "stage_name": stage_name,
                "stage_kind": control_stage.stage_kind,
                "layer": control_stage.layer,
                "stage_order": control_stage.order,
                "control_sum_explained_variance": (
                    sum_explained_variance(control_grid)
                ),
                "freeze_sum_explained_variance": (
                    sum_explained_variance(freeze_grid)
                ),
                "full_power_spectral_cosine": (
                    cosine_similarity(
                        control_power,
                        freeze_power,
                    )
                ),
                "top_5_global_mode_overlap": (
                    top_overlap(
                        control_power,
                        freeze_power,
                        5,
                    )
                ),
                "top_20_global_mode_overlap": (
                    top_overlap(
                        control_power,
                        freeze_power,
                        20,
                    )
                ),
                **prefixed_family_stats(
                    "control",
                    control_family,
                ),
                **prefixed_family_stats(
                    "freeze",
                    freeze_family,
                ),
                **{
                    f"control_{key}": value
                    for key, value
                    in control_top_summary.items()
                },
                **{
                    f"freeze_{key}": value
                    for key, value
                    in freeze_top_summary.items()
                },
            }

            for family in MODE_FAMILIES:
                row[
                    f"{family}_power_spectral_cosine"
                ] = masked_power_cosine(
                    control_power,
                    freeze_power,
                    masks[family],
                )

            layer_rows.append(row)

            for branch_name, details in (
                ("control", control_top_details),
                ("freeze", freeze_top_details),
            ):
                for detail in details:
                    mode_rows.append(
                        {
                            "step": step,
                            "branch_step": branch_step,
                            "branch": branch_name,
                            "stage_name": stage_name,
                            "stage_kind": control_stage.stage_kind,
                            "layer": control_stage.layer,
                            "stage_order": control_stage.order,
                            **detail,
                        }
                    )

            if stage_name == final_stage_name:
                paired_row.update(
                    {
                        "final_full_power_spectral_cosine": (
                            cosine_similarity(
                                control_power,
                                freeze_power,
                            )
                        ),
                        "final_top_5_global_mode_overlap": (
                            top_overlap(
                                control_power,
                                freeze_power,
                                5,
                            )
                        ),
                        "final_top_20_global_mode_overlap": (
                            top_overlap(
                                control_power,
                                freeze_power,
                                20,
                            )
                        ),
                        **prefixed_family_stats(
                            "control_final",
                            control_family,
                        ),
                        **prefixed_family_stats(
                            "freeze_final",
                            freeze_family,
                        ),
                    }
                )
                for family in MODE_FAMILIES:
                    paired_row[
                        f"final_{family}_power_spectral_cosine"
                    ] = masked_power_cosine(
                        control_power,
                        freeze_power,
                        masks[family],
                    )

        paired_rows.append(paired_row)

        del control_model
        del freeze_model
        del control_stages
        del freeze_stages
        release_accelerator_cache(device)

    write_csv(args.paired_output, paired_rows)
    write_csv(args.layer_output, layer_rows)
    write_csv(args.mode_output, mode_rows)

    paired_frame = pd.DataFrame(paired_rows).sort_values(
        "step"
    )
    layer_frame = pd.DataFrame(layer_rows).sort_values(
        ["step", "stage_order"]
    )
    mode_frame = pd.DataFrame(mode_rows).sort_values(
        [
            "step",
            "stage_order",
            "branch",
            "rank",
        ]
    )

    try:
        import plotly.express as px
    except ImportError as error:
        raise RuntimeError(
            "Plotly is required for the matched Fourier HTML."
        ) from error

    accuracy_long = paired_frame.melt(
        id_vars=["step"],
        value_vars=[
            "control_full_accuracy",
            "freeze_full_accuracy",
        ],
        var_name="branch",
        value_name="full_accuracy",
    )

    family_long = paired_frame.melt(
        id_vars=["step"],
        value_vars=[
            (
                f"control_final_{family}_"
                "power_fraction_non_dc"
            )
            for family in MODE_FAMILIES
            if family != "dc"
        ]
        + [
            (
                f"freeze_final_{family}_"
                "power_fraction_non_dc"
            )
            for family in MODE_FAMILIES
            if family != "dc"
        ],
        var_name="branch_family",
        value_name="non_dc_power_fraction",
    )

    figures = [
        px.line(
            accuracy_long,
            x="step",
            y="full_accuracy",
            color="branch",
            markers=True,
            title="Matched depth-4 branch accuracy",
        ),
        px.line(
            paired_frame,
            x="step",
            y="final_full_power_spectral_cosine",
            markers=True,
            title=(
                "Final-hidden full-spectrum similarity "
                "between matched branches"
            ),
        ),
        px.line(
            paired_frame,
            x="step",
            y="final_top_20_global_mode_overlap",
            markers=True,
            title=(
                "Overlap of the 20 strongest final-hidden "
                "global Fourier-mode pairs"
            ),
        ),
        px.line(
            family_long,
            x="step",
            y="non_dc_power_fraction",
            color="branch_family",
            markers=True,
            title=(
                "Complete final-hidden mode-family decomposition "
                "after the matched intervention"
            ),
        ),
    ]

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Depth-4 matched Fourier mode identification</title>",
        "</head><body>",
        "<h1>Depth-4 matched Fourier mode identification</h1>",
        (
            "<p>This observational analysis compares complete 2D "
            "mode structure after the matched training intervention. "
            "It performs no Fourier sufficiency, ablation, or "
            "cross-readout test.</p>"
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
            "<h2>Paired summary</h2>",
            paired_frame.to_html(index=False),
            "<h2>Layerwise paired summary</h2>",
            layer_frame.to_html(index=False),
            "<h2>Top global conjugate mode pairs</h2>",
            mode_frame.to_html(index=False),
            "</body></html>",
        ]
    )

    args.html_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.html_output.write_text(
        "\n".join(html_parts),
        encoding="utf-8",
    )

    print()
    print(
        "Matched depth-four Fourier mode identification complete."
    )
    print(
        f"Saved paired summary: {args.paired_output}"
    )
    print(
        f"Saved layer summary: {args.layer_output}"
    )
    print(
        f"Saved mode inventory: {args.mode_output}"
    )
    print(f"Saved HTML report: {args.html_output}")


if __name__ == "__main__":
    main()
