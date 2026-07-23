from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


VARIANTS = [
    {
        "variant": "mod97_tf0p3_w128",
        "modulus": 97,
        "train_fraction": 0.3,
        "d_model": 128,
        "num_heads": 4,
        "d_mlp": 512,
    },
    {
        "variant": "mod113_tf0p2_w128",
        "modulus": 113,
        "train_fraction": 0.2,
        "d_model": 128,
        "num_heads": 4,
        "d_mlp": 512,
    },
    {
        "variant": "mod113_tf0p3_w64",
        "modulus": 113,
        "train_fraction": 0.3,
        "d_model": 64,
        "num_heads": 4,
        "d_mlp": 256,
    },
]

REGIMES = [
    "adamw",
    "muon",
    "stable_muon",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete controlled generality suite."
        )
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "mps",
            "cpu",
        ],
        default="auto",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    repository_root = (
        Path(__file__).resolve().parent
    )

    trainer_path = (
        repository_root
        / "train_generality_variant.py"
    )

    if not trainer_path.is_file():
        raise FileNotFoundError(
            f"Trainer not found: {trainer_path}"
        )

    manifest_rows = []

    for variant in VARIANTS:
        initial_state_path = (
            Path("checkpoints")
            / "initial_states"
            / (
                "generality_"
                f"{variant['variant']}"
                f"_seed_{args.seed}.pt"
            )
        )

        for regime in REGIMES:
            run_name = (
                "generality_"
                f"{variant['variant']}_"
                f"{regime}_seed_{args.seed}"
            )

            command = [
                sys.executable,
                str(trainer_path),
                "--regime",
                regime,
                "--modulus",
                str(
                    variant["modulus"]
                ),
                "--train-fraction",
                str(
                    variant[
                        "train_fraction"
                    ]
                ),
                "--d-model",
                str(
                    variant["d_model"]
                ),
                "--num-heads",
                str(
                    variant["num_heads"]
                ),
                "--d-mlp",
                str(
                    variant["d_mlp"]
                ),
                "--seed",
                str(args.seed),
                "--steps",
                str(args.steps),
                "--device",
                args.device,
                "--run-name",
                run_name,
                "--initial-state-path",
                str(initial_state_path),
            ]

            if args.overwrite:
                command.append(
                    "--overwrite"
                )

            print()
            print("=" * 80)
            print(
                f"Running {run_name}"
            )
            print("=" * 80)

            subprocess.run(
                command,
                check=True,
                cwd=repository_root,
            )

            manifest_rows.append(
                {
                    "variant": (
                        variant["variant"]
                    ),
                    "regime": regime,
                    "run_name": run_name,
                    "modulus": (
                        variant["modulus"]
                    ),
                    "train_fraction": (
                        variant[
                            "train_fraction"
                        ]
                    ),
                    "d_model": (
                        variant["d_model"]
                    ),
                    "num_heads": (
                        variant["num_heads"]
                    ),
                    "d_mlp": (
                        variant["d_mlp"]
                    ),
                    "seed": args.seed,
                    "steps": args.steps,
                    "csv_path": (
                        f"runs/{run_name}.csv"
                    ),
                    "checkpoint_directory": (
                        "checkpoints/"
                        f"{run_name}"
                    ),
                    "final_checkpoint": (
                        "checkpoints/"
                        f"{run_name}/"
                        f"step_{args.steps:06d}.pt"
                    ),
                    "initial_state_path": (
                        str(
                            initial_state_path
                        )
                    ),
                }
            )

    manifest_path = (
        Path("runs")
        / "generality_manifest.csv"
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
        writer.writerows(
            manifest_rows
        )

    print()
    print(
        f"Saved manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
