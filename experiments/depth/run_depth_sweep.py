from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEPTHS_TO_RUN = [2, 4]
REGIMES = ["adamw", "muon", "stable_muon"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled depth sweep using the hyperparameters "
            "selected at depth one."
        )
    )
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--depth-one-adamw-csv",
        type=Path,
        default=Path(
            "runs/adamw_sweep_lr_1em3_wd_3p0_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--depth-one-muon-csv",
        type=Path,
        default=Path(
            "runs/muon_sweep_lr_3em2_wd_0p1_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--depth-one-stable-muon-csv",
        type=Path,
        default=Path(
            "runs/stable_muon_tuned_lr_3em2_wd_0p1_seed_0.csv"
        ),
    )
    return parser.parse_args()


def completed_run(
    csv_path: Path,
    final_checkpoint: Path,
    expected_step: int,
) -> bool:
    if not csv_path.is_file() or not final_checkpoint.is_file():
        return False

    try:
        frame = pd.read_csv(csv_path, usecols=["step"])
    except Exception:
        return False

    if frame.empty:
        return False

    return int(frame["step"].max()) == expected_step


def add_depth_one_rows(
    args: argparse.Namespace,
    manifest_rows: list[dict[str, object]],
) -> None:
    existing = {
        "adamw": args.depth_one_adamw_csv,
        "muon": args.depth_one_muon_csv,
        "stable_muon": args.depth_one_stable_muon_csv,
    }

    missing = [
        str(path)
        for path in existing.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "The selected depth-one runs are required before the "
            "depth sweep. Missing: "
            + ", ".join(missing)
        )

    for regime, csv_path in existing.items():
        manifest_rows.append(
            {
                "depth": 1,
                "regime": regime,
                "seed": args.seed,
                "steps": args.steps,
                "csv_path": str(csv_path),
                "initial_state_path": (
                    "checkpoints/initial_states/"
                    "model_seed_0_sweep_compatible.pt"
                ),
                "reused_depth_one": 1,
                "executed_now": 0,
            }
        )


def main() -> None:
    args = parse_arguments()

    script_directory = Path(__file__).resolve().parent
    repository_root = script_directory.parents[1]
    trainer = script_directory / "train_depth_variant.py"

    if not trainer.is_file():
        raise FileNotFoundError(
            f"Missing depth trainer: {trainer}"
        )

    manifest_rows: list[dict[str, object]] = []
    add_depth_one_rows(args, manifest_rows)

    executed_count = 0
    skipped_completed_count = 0

    for depth in DEPTHS_TO_RUN:
        initial_state = (
            Path("checkpoints")
            / "initial_states"
            / f"depth_nested_v2_{depth}_seed_{args.seed}.pt"
        )

        for regime in REGIMES:
            run_name = (
                f"depth_sweep_v2_depth_{depth}_"
                f"{regime}_seed_{args.seed}"
            )
            csv_path = Path("runs") / f"{run_name}.csv"
            checkpoint_directory = (
                Path("checkpoints") / run_name
            )
            final_checkpoint = (
                checkpoint_directory
                / f"step_{args.steps:06d}.pt"
            )

            is_complete = completed_run(
                repository_root / csv_path,
                repository_root / final_checkpoint,
                args.steps,
            )

            if is_complete and not args.overwrite:
                print(f"Skipping completed run: {run_name}")
                skipped_completed_count += 1
                manifest_rows.append(
                    {
                        "depth": depth,
                        "regime": regime,
                        "seed": args.seed,
                        "steps": args.steps,
                        "csv_path": str(csv_path),
                        "initial_state_path": str(
                            initial_state
                        ),
                        "reused_depth_one": 0,
                        "executed_now": 0,
                    }
                )
                continue

            if (
                not args.overwrite
                and (
                    (repository_root / csv_path).exists()
                    or (
                        repository_root
                        / checkpoint_directory
                    ).exists()
                )
            ):
                raise RuntimeError(
                    "Incomplete existing depth run detected: "
                    f"{run_name}. Delete its CSV and checkpoint "
                    "directory, or rerun with --overwrite."
                )

            command = [
                sys.executable,
                str(trainer),
                "--regime",
                regime,
                "--num-layers",
                str(depth),
                "--modulus",
                "113",
                "--train-fraction",
                "0.3",
                "--d-model",
                "128",
                "--num-heads",
                "4",
                "--d-mlp",
                "512",
                "--seed",
                str(args.seed),
                "--steps",
                str(args.steps),
                "--device",
                args.device,
                "--run-name",
                run_name,
                "--initial-state-path",
                str(initial_state),
                "--depth-one-initial-state-path",
                (
                    "checkpoints/initial_states/"
                    "model_seed_0_sweep_compatible.pt"
                ),
                "--adamw-lr",
                "0.001",
                "--adamw-weight-decay",
                "3.0",
                "--muon-lr",
                "0.03",
                "--muon-weight-decay",
                "0.1",
                "--muon-momentum",
                "0.95",
                "--muon-ns-steps",
                "5",
                "--aux-lr",
                "0.001",
                "--aux-weight-decay",
                "1.0",
                "--unembedding-lr",
                "0.00025",
                "--unembedding-weight-decay",
                "1.0",
                "--freeze-trigger-test-accuracy",
                "0.95",
                "--freeze-trigger-consecutive",
                "5",
                "--freeze-delay",
                "2000",
            ]

            if args.overwrite:
                command.append("--overwrite")

            print()
            print("=" * 80)
            print(
                f"Running depth={depth}, regime={regime}"
            )
            print("=" * 80)

            subprocess.run(
                command,
                check=True,
                cwd=repository_root,
            )
            executed_count += 1

            manifest_rows.append(
                {
                    "depth": depth,
                    "regime": regime,
                    "seed": args.seed,
                    "steps": args.steps,
                    "csv_path": str(csv_path),
                    "initial_state_path": str(
                        initial_state
                    ),
                    "reused_depth_one": 0,
                    "executed_now": 1,
                }
            )

    manifest_path = (
        repository_root
        / "runs"
        / f"depth_sweep_v2_manifest_seed_{args.seed}.csv"
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
            fieldnames=list(manifest_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"Saved manifest: {manifest_path}")
    print("Depth-one runs reused: 3")
    print(
        "Previously completed depth runs skipped: "
        f"{skipped_completed_count}"
    )
    print(
        "New effective runs executed: "
        f"{executed_count}"
    )


if __name__ == "__main__":
    main()
