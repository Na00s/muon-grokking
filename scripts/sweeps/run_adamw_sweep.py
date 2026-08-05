from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


LEARNING_RATES = [
    3e-4,
    1e-3,
    3e-3,
    1e-2,
]

WEIGHT_DECAYS = [
    0.1,
    1.0,
    3.0,
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a compact AdamW learning-rate and "
            "weight-decay sweep on the main modular-addition "
            "configuration."
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
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )

    parser.add_argument(
        "--initial-state-path",
        type=Path,
        default=Path(
            "checkpoints/initial_states/model_seed_0.pt"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def float_tag(value: float) -> str:
    text = f"{value:.0e}"
    coefficient, exponent = text.split("e")
    coefficient = coefficient.replace(".", "p")
    exponent_value = int(exponent)

    if exponent_value < 0:
        exponent_tag = f"m{abs(exponent_value)}"
    else:
        exponent_tag = f"p{exponent_value}"

    return f"{coefficient}e{exponent_tag}"


def decimal_tag(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    args = parse_arguments()

    repository_root = Path(__file__).resolve().parents[2]
    trainer_path = (
        repository_root
        / "scripts"
        / "training"
        / "train_generality_variant.py"
    )

    if not trainer_path.is_file():
        raise FileNotFoundError(
            "Place run_adamw_sweep.py beside "
            "train_generality_variant.py in scripts/training."
        )

    manifest_rows: list[dict[str, object]] = []

    for learning_rate in LEARNING_RATES:
        for weight_decay in WEIGHT_DECAYS:
            run_name = (
                "adamw_sweep_"
                f"lr_{float_tag(learning_rate)}_"
                f"wd_{decimal_tag(weight_decay)}_"
                f"seed_{args.seed}"
            )

            csv_path = Path("runs") / f"{run_name}.csv"
            checkpoint_directory = (
                Path("checkpoints") / run_name
            )

            command = [
                sys.executable,
                str(trainer_path),
                "--regime",
                "adamw",
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
                str(args.initial_state_path),
                "--adamw-lr",
                str(learning_rate),
                "--adamw-weight-decay",
                str(weight_decay),
            ]

            if args.overwrite:
                command.append("--overwrite")
            elif csv_path.exists():
                print(
                    f"Skipping completed run: {run_name}"
                )

                manifest_rows.append(
                    {
                        "run_name": run_name,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "seed": args.seed,
                        "steps": args.steps,
                        "csv_path": str(csv_path),
                        "checkpoint_directory": str(
                            checkpoint_directory
                        ),
                        "final_checkpoint": str(
                            checkpoint_directory
                            / f"step_{args.steps:06d}.pt"
                        ),
                    }
                )
                continue

            print()
            print("=" * 80)
            print(
                f"Running {run_name}: "
                f"lr={learning_rate}, "
                f"weight_decay={weight_decay}"
            )
            print("=" * 80)

            subprocess.run(
                command,
                check=True,
                cwd=repository_root,
            )

            manifest_rows.append(
                {
                    "run_name": run_name,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "seed": args.seed,
                    "steps": args.steps,
                    "csv_path": str(csv_path),
                    "checkpoint_directory": str(
                        checkpoint_directory
                    ),
                    "final_checkpoint": str(
                        checkpoint_directory
                        / f"step_{args.steps:06d}.pt"
                    ),
                }
            )

    manifest_path = (
        Path("runs")
        / f"adamw_sweep_manifest_seed_{args.seed}.csv"
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
            fieldnames=list(
                manifest_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
