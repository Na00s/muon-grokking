from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


RUNS = [
    {
        "label": "depth2_adamw",
        "depth": 2,
        "regime": "adamw",
        "csv_path": (
            "runs/depth_sweep_v2_depth_2_adamw_seed_0.csv"
        ),
        "checkpoint_directories": (
            "checkpoints/depth_sweep_v2_depth_2_adamw_seed_0"
        ),
    },
    {
        "label": "depth2_muon",
        "depth": 2,
        "regime": "muon",
        "csv_path": (
            "runs/depth_sweep_v2_depth_2_muon_seed_0.csv"
        ),
        "checkpoint_directories": (
            "checkpoints/depth_sweep_v2_depth_2_muon_seed_0"
        ),
    },
    {
        "label": "depth2_stable_muon",
        "depth": 2,
        "regime": "stable_muon",
        "csv_path": (
            "runs/depth_sweep_v2_depth_2_stable_muon_seed_0.csv"
        ),
        "checkpoint_directories": (
            "checkpoints/depth_sweep_v2_depth_2_stable_muon_seed_0"
        ),
    },
    {
        "label": "depth4_adamw",
        "depth": 4,
        "regime": "adamw",
        "csv_path": (
            "runs/depth_sweep_v3_depth_4_adamw_seed_0_to_300k.csv"
        ),
        "checkpoint_directories": (
            "checkpoints/depth_sweep_v2_depth_4_adamw_seed_0|"
            "checkpoints/depth_sweep_v3_depth_4_adamw_seed_0_to_300k"
        ),
    },
    {
        "label": "depth4_muon",
        "depth": 4,
        "regime": "muon",
        "csv_path": (
            "runs/depth_sweep_v3_depth_4_muon_seed_0_to_300k.csv"
        ),
        "checkpoint_directories": (
            "checkpoints/depth_sweep_v2_depth_4_muon_seed_0|"
            "checkpoints/depth_sweep_v3_depth_4_muon_seed_0_to_300k"
        ),
    },
    {
        "label": "depth4_stable_muon_observed",
        "depth": 4,
        "regime": "stable_muon",
        "csv_path": (
            "runs/depth_sweep_v3_depth_4_stable_muon_seed_0_to_300k.csv"
        ),
        "checkpoint_directories": (
            "checkpoints/depth_sweep_v2_depth_4_stable_muon_seed_0|"
            "checkpoints/depth_sweep_v3_depth_4_stable_muon_seed_0_to_300k"
        ),
    },
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run complete observational Fourier mode identification "
            "for depth 2 and depth 4, followed by the matched "
            "depth-four observational mode comparison."
        )
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
    )
    parser.add_argument(
        "--top-mode-pairs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--skip-matched",
        action="store_true",
    )
    parser.add_argument(
        "--include-all-checkpoints",
        action="store_true",
    )
    return parser.parse_args()


def require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required path does not exist: {path}"
        )


def main() -> None:
    args = parse_arguments()
    script_directory = Path(__file__).resolve().parent
    repository_root = script_directory.parents[1]

    for run in RUNS:
        require_path(repository_root / run["csv_path"])
        for part in run[
            "checkpoint_directories"
        ].split("|"):
            require_path(repository_root / part)

    manifest_path = (
        repository_root
        / "runs"
        / "depth_fourier_mode_manifest_seed_0.csv"
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "depth",
                "regime",
                "csv_path",
                "checkpoint_directories",
            ],
        )
        writer.writeheader()
        writer.writerows(RUNS)

    analysis_command = [
        sys.executable,
        str(
            script_directory
            / "depth_fourier_mode_identification.py"
        ),
        "--manifest",
        str(
            manifest_path.relative_to(
                repository_root
            )
        ),
        "--device",
        args.device,
        "--checkpoint-stride",
        str(args.checkpoint_stride),
        "--top-mode-pairs",
        str(args.top_mode_pairs),
    ]
    if args.include_all_checkpoints:
        analysis_command.append(
            "--include-all-checkpoints"
        )

    subprocess.run(
        analysis_command,
        cwd=repository_root,
        check=True,
    )

    if not args.skip_matched:
        matched_command = [
            sys.executable,
            str(
                script_directory
                / "depth4_matched_fourier_mode_identification.py"
            ),
            "--device",
            args.device,
            "--top-mode-pairs",
            str(args.top_mode_pairs),
        ]
        subprocess.run(
            matched_command,
            cwd=repository_root,
            check=True,
        )

    print()
    print(
        "Complete depth Fourier mode-identification study finished."
    )


if __name__ == "__main__":
    main()
