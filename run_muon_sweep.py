from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


MUON_LEARNING_RATES = [
    3e-3,
    1e-2,
    3e-2,
]

MUON_WEIGHT_DECAYS = [
    3e-2,
    1e-1,
    3e-1,
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a compact ordinary-Muon learning-rate and weight-decay "
            "sweep on the main modular-addition condition."
        )
    )
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--initial-state-path",
        type=Path,
        default=Path(
            "checkpoints/initial_states/"
            "model_seed_0_sweep_compatible.pt"
        ),
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1_000,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scientific_tag(value: float) -> str:
    text = f"{value:.0e}"
    coefficient, exponent = text.split("e")
    coefficient = coefficient.replace(".", "p")
    exponent_value = int(exponent)
    sign = "m" if exponent_value < 0 else "p"
    return f"{coefficient}e{sign}{abs(exponent_value)}"


def decimal_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def csv_reached_final_step(
    csv_path: Path,
    expected_step: int,
) -> bool:
    if not csv_path.is_file():
        return False

    last_row: dict[str, str] | None = None
    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            last_row = row

    if last_row is None:
        return False

    try:
        return int(float(last_row["step"])) == expected_step
    except (KeyError, TypeError, ValueError):
        return False


def manifest_row(
    *,
    run_name: str,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    steps: int,
    csv_path: Path,
    checkpoint_directory: Path,
) -> dict[str, object]:
    return {
        "run_name": run_name,
        "regime": "muon",
        "muon_learning_rate": learning_rate,
        "muon_weight_decay": weight_decay,
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "aux_learning_rate": 1e-3,
        "aux_weight_decay": 1.0,
        "unembedding_learning_rate": 2.5e-4,
        "unembedding_weight_decay": 1.0,
        "seed": seed,
        "steps": steps,
        "csv_path": str(csv_path),
        "checkpoint_directory": str(checkpoint_directory),
        "final_checkpoint": str(
            checkpoint_directory / f"step_{steps:06d}.pt"
        ),
    }


def main() -> None:
    args = parse_arguments()

    repository_root = Path(__file__).resolve().parent
    trainer_path = (
        repository_root / "train_generality_variant.py"
    )

    if not trainer_path.is_file():
        raise FileNotFoundError(
            "Place run_muon_sweep.py beside "
            "train_generality_variant.py in the repository root."
        )

    manifest_rows: list[dict[str, object]] = []

    for learning_rate in MUON_LEARNING_RATES:
        for weight_decay in MUON_WEIGHT_DECAYS:
            run_name = (
                "muon_sweep_"
                f"lr_{scientific_tag(learning_rate)}_"
                f"wd_{decimal_tag(weight_decay)}_"
                f"seed_{args.seed}"
            )

            csv_path = Path("runs") / f"{run_name}.csv"
            checkpoint_directory = (
                Path("checkpoints") / run_name
            )

            if (
                not args.overwrite
                and csv_reached_final_step(
                    csv_path,
                    args.steps,
                )
            ):
                print(f"Skipping completed run: {run_name}")
                manifest_rows.append(
                    manifest_row(
                        run_name=run_name,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        seed=args.seed,
                        steps=args.steps,
                        csv_path=csv_path,
                        checkpoint_directory=checkpoint_directory,
                    )
                )
                continue

            if (
                not args.overwrite
                and (
                    csv_path.exists()
                    or checkpoint_directory.exists()
                )
            ):
                raise RuntimeError(
                    f"Incomplete existing run detected: {run_name}. "
                    "Delete its CSV and checkpoint directory, or rerun "
                    "the whole sweep with --overwrite."
                )

            command = [
                sys.executable,
                str(trainer_path),
                "--regime",
                "muon",
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
                "--evaluation-interval",
                str(args.evaluation_interval),
                "--checkpoint-interval",
                str(args.checkpoint_interval),
                "--device",
                args.device,
                "--run-name",
                run_name,
                "--initial-state-path",
                str(args.initial_state_path),
                "--muon-lr",
                str(learning_rate),
                "--muon-weight-decay",
                str(weight_decay),
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
            ]

            if args.overwrite:
                command.append("--overwrite")

            print()
            print("=" * 80)
            print(
                f"Running {run_name}: "
                f"Muon lr={learning_rate}, "
                f"Muon weight_decay={weight_decay}"
            )
            print("=" * 80)

            subprocess.run(
                command,
                check=True,
                cwd=repository_root,
            )

            if not csv_reached_final_step(
                csv_path,
                args.steps,
            ):
                raise RuntimeError(
                    f"Run did not reach step {args.steps}: {run_name}"
                )

            manifest_rows.append(
                manifest_row(
                    run_name=run_name,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    seed=args.seed,
                    steps=args.steps,
                    csv_path=csv_path,
                    checkpoint_directory=checkpoint_directory,
                )
            )

    manifest_path = (
        Path("runs")
        / f"muon_sweep_manifest_seed_{args.seed}.csv"
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


if __name__ == "__main__":
    main()
