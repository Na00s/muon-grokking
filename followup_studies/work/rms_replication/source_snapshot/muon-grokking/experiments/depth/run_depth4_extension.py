from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


REGIMES = [
    "adamw",
    "muon",
    "stable_muon",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue all three depth-four trajectories "
            "from 100,000 to 300,000 total updates."
        )
    )
    parser.add_argument(
        "--source-step",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=300_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def completed_run(
    csv_path: Path,
    final_checkpoint: Path,
    total_steps: int,
) -> bool:
    if (
        not csv_path.is_file()
        or not final_checkpoint.is_file()
    ):
        return False

    try:
        frame = pd.read_csv(
            csv_path,
            usecols=["step"],
        )
    except Exception:
        return False

    if frame.empty:
        return False

    return int(frame["step"].max()) == total_steps


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )


def main() -> None:
    args = parse_arguments()

    if args.total_steps <= args.source_step:
        raise ValueError(
            "--total-steps must exceed "
            "--source-step."
        )

    script_directory = Path(
        __file__
    ).resolve().parent
    repository_root = (
        script_directory.parents[1]
    )
    resume_script = (
        script_directory
        / "resume_depth_variant.py"
    )
    require_file(resume_script)

    depth_four_paths: dict[
        str,
        str,
    ] = {}

    executed_count = 0
    skipped_count = 0

    for regime in REGIMES:
        source_run_name = (
            f"depth_sweep_v2_depth_4_"
            f"{regime}_seed_{args.seed}"
        )
        source_csv = (
            repository_root
            / "runs"
            / f"{source_run_name}.csv"
        )
        source_checkpoint = (
            repository_root
            / "checkpoints"
            / source_run_name
            / f"step_{args.source_step:06d}.pt"
        )

        require_file(source_csv)
        require_file(source_checkpoint)

        output_run_name = (
            f"depth_sweep_v3_depth_4_"
            f"{regime}_seed_{args.seed}_"
            f"to_{args.total_steps // 1000}k"
        )
        output_csv = (
            repository_root
            / "runs"
            / f"{output_run_name}.csv"
        )
        output_checkpoint = (
            repository_root
            / "checkpoints"
            / output_run_name
            / f"step_{args.total_steps:06d}.pt"
        )

        depth_four_paths[regime] = str(
            Path("runs")
            / f"{output_run_name}.csv"
        )

        if (
            completed_run(
                output_csv,
                output_checkpoint,
                args.total_steps,
            )
            and not args.overwrite
        ):
            print(
                "Skipping completed extension: "
                f"{output_run_name}"
            )
            skipped_count += 1
            continue

        if (
            not args.overwrite
            and (
                output_csv.exists()
                or output_checkpoint.parent.exists()
            )
        ):
            raise RuntimeError(
                "Incomplete continuation exists for "
                f"{output_run_name}. Delete its CSV "
                "and checkpoint directory, or rerun "
                "with --overwrite."
            )

        command = [
            sys.executable,
            str(resume_script),
            "--source-csv",
            str(source_csv),
            "--source-checkpoint",
            str(source_checkpoint),
            "--output-run-name",
            output_run_name,
            "--total-steps",
            str(args.total_steps),
            "--checkpoint-interval",
            str(args.checkpoint_interval),
            "--device",
            args.device,
        ]

        if args.overwrite:
            command.append("--overwrite")

        print()
        print("=" * 80)
        print(
            "Extending depth 4, "
            f"regime={regime}, "
            f"{args.source_step} -> "
            f"{args.total_steps}"
        )
        print("=" * 80)

        subprocess.run(
            command,
            check=True,
            cwd=repository_root,
        )
        executed_count += 1

    manifest_rows = []

    depth_one_paths = {
        "adamw": (
            "runs/"
            "adamw_sweep_lr_1em3_wd_3p0_seed_0.csv"
        ),
        "muon": (
            "runs/"
            "muon_sweep_lr_3em2_wd_0p1_seed_0.csv"
        ),
        "stable_muon": (
            "runs/"
            "stable_muon_tuned_lr_3em2_"
            "wd_0p1_seed_0.csv"
        ),
    }

    for regime in REGIMES:
        path = (
            repository_root
            / depth_one_paths[regime]
        )
        require_file(path)
        manifest_rows.append(
            {
                "depth": 1,
                "regime": regime,
                "seed": args.seed,
                "steps": 100_000,
                "csv_path": (
                    depth_one_paths[regime]
                ),
                "initial_state_path": (
                    "checkpoints/initial_states/"
                    "model_seed_0_"
                    "sweep_compatible.pt"
                ),
                "reused_depth_one": 1,
                "executed_now": 0,
                "continued_from_step": "",
            }
        )

    for regime in REGIMES:
        depth_two_path = (
            "runs/"
            f"depth_sweep_v2_depth_2_"
            f"{regime}_seed_{args.seed}.csv"
        )
        require_file(
            repository_root / depth_two_path
        )
        manifest_rows.append(
            {
                "depth": 2,
                "regime": regime,
                "seed": args.seed,
                "steps": 100_000,
                "csv_path": depth_two_path,
                "initial_state_path": (
                    "checkpoints/initial_states/"
                    f"depth_nested_v2_2_"
                    f"seed_{args.seed}.pt"
                ),
                "reused_depth_one": 0,
                "executed_now": 0,
                "continued_from_step": "",
            }
        )

    for regime in REGIMES:
        manifest_rows.append(
            {
                "depth": 4,
                "regime": regime,
                "seed": args.seed,
                "steps": args.total_steps,
                "csv_path": (
                    depth_four_paths[regime]
                ),
                "initial_state_path": (
                    "checkpoints/initial_states/"
                    f"depth_nested_v2_4_"
                    f"seed_{args.seed}.pt"
                ),
                "reused_depth_one": 0,
                "executed_now": 1,
                "continued_from_step": (
                    args.source_step
                ),
            }
        )

    manifest_path = (
        repository_root
        / "runs"
        / f"depth_sweep_v3_manifest_"
        f"seed_{args.seed}.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                manifest_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"Saved manifest: {manifest_path}")
    print(
        "Depth-four continuations executed: "
        f"{executed_count}"
    )
    print(
        "Completed continuations skipped: "
        f"{skipped_count}"
    )


if __name__ == "__main__":
    main()
