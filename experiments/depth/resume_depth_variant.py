from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from data import generate_modular_addition_data
from optimizers.muon import Muon
from train_depth_variant import (
    DepthModularAdditionTransformer,
    split_parameter_groups,
    verify_architecture,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume one exact depth-sweep trajectory from a saved "
            "model and optimizer checkpoint."
        )
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-run-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=300_000,
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


def get_device(requested: str) -> torch.device:
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


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}"
        )

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

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must contain a dictionary."
        )

    required = {
        "step",
        "regime",
        "arguments",
        "model_config",
        "model_state_dict",
        "optimizer_state_dicts",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise KeyError(
            "Checkpoint is missing fields: "
            + ", ".join(sorted(missing))
        )

    return checkpoint


def read_records(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing source CSV: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {path}"
            )
        records = list(reader)
        fieldnames = list(reader.fieldnames)

    if not records:
        raise ValueError(
            f"CSV contains no records: {path}"
        )

    return fieldnames, records


def parameter_l2_norm(
    parameters: Iterable[Parameter],
) -> float:
    total = 0.0
    for parameter in parameters:
        value = torch.linalg.vector_norm(
            parameter.detach().float()
        ).item()
        total += value * value
    return math.sqrt(total)


def gradient_l2_norm(
    parameters: Iterable[Parameter],
) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(
                "A gradient contains NaN or infinity."
            )

        value = torch.linalg.vector_norm(
            gradient.float()
        ).item()
        total += value * value

    return math.sqrt(total)


@torch.no_grad()
def evaluate(
    model: DepthModularAdditionTransformer,
    inputs: Tensor,
    targets: Tensor,
) -> tuple[float, float]:
    model.eval()
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    accuracy = (
        logits.argmax(dim=-1) == targets
    ).float().mean()
    return float(loss.item()), float(accuracy.item())


def as_bool(value: object) -> bool:
    return str(value).strip() in {
        "1",
        "1.0",
        "True",
        "true",
    }


def optional_int(value: object) -> int | None:
    text = str(value).strip()
    if text in {"", "None", "nan", "NaN"}:
        return None
    return int(float(text))


def reconstruct_trigger_state(
    records: list[dict[str, str]],
    threshold: float,
) -> tuple[
    bool,
    bool,
    int,
    int | None,
    bool,
]:
    has_memorized = any(
        float(record["train_accuracy"]) >= 0.999
        for record in records
    )
    has_reached_95_test = any(
        float(record["test_accuracy"]) >= 0.95
        for record in records
    )

    last_record = records[-1]
    scheduled_freeze_step = optional_int(
        last_record.get(
            "scheduled_freeze_step",
            "",
        )
    )
    auxiliary_frozen = as_bool(
        last_record.get(
            "auxiliary_frozen",
            "0",
        )
    )

    consecutive_trigger_evaluations = 0
    if scheduled_freeze_step is None:
        for record in reversed(records):
            if (
                float(record["test_accuracy"])
                >= threshold
            ):
                consecutive_trigger_evaluations += 1
            else:
                break

    return (
        has_memorized,
        has_reached_95_test,
        consecutive_trigger_evaluations,
        scheduled_freeze_step,
        auxiliary_frozen,
    )


def instantiate_optimizers(
    regime: str,
    training_arguments: dict,
    model: DepthModularAdditionTransformer,
    hidden_parameters: list[Parameter],
    auxiliary_parameters: list[Parameter],
    unembedding_parameters: list[Parameter],
) -> dict[str, object]:
    beta1 = float(
        training_arguments.get(
            "adamw_beta1",
            0.9,
        )
    )
    beta2 = float(
        training_arguments.get(
            "adamw_beta2",
            0.999,
        )
    )

    if regime == "adamw":
        return {
            "adamw": torch.optim.AdamW(
                model.parameters(),
                lr=float(
                    training_arguments["adamw_lr"]
                ),
                weight_decay=float(
                    training_arguments[
                        "adamw_weight_decay"
                    ]
                ),
                betas=(beta1, beta2),
            )
        }

    return {
        "muon": Muon(
            hidden_parameters,
            learning_rate=float(
                training_arguments["muon_lr"]
            ),
            momentum=float(
                training_arguments["muon_momentum"]
            ),
            weight_decay=float(
                training_arguments[
                    "muon_weight_decay"
                ]
            ),
            newton_schulz_steps=int(
                training_arguments["muon_ns_steps"]
            ),
            nesterov=True,
        ),
        "auxiliary_adamw": torch.optim.AdamW(
            auxiliary_parameters,
            lr=float(
                training_arguments["aux_lr"]
            ),
            weight_decay=float(
                training_arguments[
                    "aux_weight_decay"
                ]
            ),
            betas=(beta1, beta2),
        ),
        "unembedding_adamw": torch.optim.AdamW(
            unembedding_parameters,
            lr=float(
                training_arguments[
                    "unembedding_lr"
                ]
            ),
            weight_decay=float(
                training_arguments[
                    "unembedding_weight_decay"
                ]
            ),
            betas=(beta1, beta2),
        ),
    }


def save_checkpoint(
    path: Path,
    step: int,
    output_run_name: str,
    regime: str,
    training_arguments: dict,
    model_configuration: dict,
    model: DepthModularAdditionTransformer,
    optimizers: dict[str, object],
    seed: int,
    actual_freeze_step: int | None,
    scheduled_freeze_step: int | None,
    auxiliary_frozen: bool,
    source_checkpoint: Path,
) -> None:
    torch.save(
        {
            "step": step,
            "run_name": output_run_name,
            "regime": regime,
            "arguments": training_arguments,
            "model_config": model_configuration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dicts": {
                name: optimizer.state_dict()
                for name, optimizer
                in optimizers.items()
            },
            "seed": seed,
            "actual_freeze_step": (
                actual_freeze_step
            ),
            "scheduled_freeze_step": (
                scheduled_freeze_step
            ),
            "auxiliary_frozen": (
                auxiliary_frozen
            ),
            "continued_from_checkpoint": str(
                source_checkpoint
            ),
        },
        path,
    )


def first_sustained_step(
    records: list[dict[str, object]],
    field: str,
    threshold: float,
    consecutive: int = 5,
) -> int | None:
    for index in range(
        len(records) - consecutive + 1
    ):
        window = records[
            index:index + consecutive
        ]
        if all(
            float(record[field]) >= threshold
            for record in window
        ):
            return int(
                float(window[0]["step"])
            )
    return None


def main() -> None:
    args = parse_arguments()

    if args.total_steps < 1:
        raise ValueError(
            "--total-steps must be positive."
        )
    if args.checkpoint_interval < 1:
        raise ValueError(
            "--checkpoint-interval must be positive."
        )

    checkpoint = load_checkpoint(
        args.source_checkpoint
    )
    fieldnames, source_records = read_records(
        args.source_csv
    )

    resume_step = int(checkpoint["step"])
    last_csv_step = int(
        float(source_records[-1]["step"])
    )

    if last_csv_step != resume_step:
        raise ValueError(
            "Source CSV and checkpoint disagree: "
            f"CSV ends at {last_csv_step}, "
            f"checkpoint is step {resume_step}."
        )
    if args.total_steps <= resume_step:
        raise ValueError(
            "--total-steps must exceed the source "
            f"checkpoint step {resume_step}."
        )

    training_arguments = dict(
        checkpoint["arguments"]
    )
    model_configuration = dict(
        checkpoint["model_config"]
    )
    regime = str(checkpoint["regime"])
    seed = int(checkpoint.get("seed", 0))

    if regime not in {
        "adamw",
        "muon",
        "stable_muon",
    }:
        raise ValueError(
            f"Unsupported regime: {regime}"
        )

    num_layers = int(
        model_configuration.get(
            "num_layers",
            training_arguments.get(
                "num_layers",
                -1,
            ),
        )
    )
    if num_layers != 4:
        raise ValueError(
            "This continuation is restricted to "
            f"depth 4, found depth {num_layers}."
        )

    evaluation_interval = int(
        training_arguments.get(
            "evaluation_interval",
            100,
        )
    )
    freeze_trigger_accuracy = float(
        training_arguments.get(
            "freeze_trigger_test_accuracy",
            0.95,
        )
    )
    freeze_trigger_consecutive = int(
        training_arguments.get(
            "freeze_trigger_consecutive",
            5,
        )
    )
    freeze_delay = int(
        training_arguments.get(
            "freeze_delay",
            2_000,
        )
    )

    output_csv = (
        Path("runs")
        / f"{args.output_run_name}.csv"
    )
    output_checkpoint_directory = (
        Path("checkpoints")
        / args.output_run_name
    )

    if args.overwrite:
        if output_csv.exists():
            output_csv.unlink()
        if output_checkpoint_directory.exists():
            shutil.rmtree(
                output_checkpoint_directory
            )

    if output_csv.exists():
        raise FileExistsError(
            f"Output CSV already exists: {output_csv}"
        )
    if output_checkpoint_directory.exists():
        raise FileExistsError(
            "Output checkpoint directory already "
            f"exists: {output_checkpoint_directory}"
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_checkpoint_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    device = get_device(args.device)

    (
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    ) = generate_modular_addition_data(
        modulus=int(
            training_arguments["modulus"]
        ),
        train_fraction=float(
            training_arguments["train_fraction"]
        ),
        seed=seed,
    )

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    model = DepthModularAdditionTransformer(
        **model_configuration
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(device)

    (
        hidden_parameters,
        auxiliary_parameters,
        unembedding_parameters,
    ) = split_parameter_groups(model)

    (
        parameter_count,
        hidden_matrix_count,
    ) = verify_architecture(
        model=model,
        hidden_parameters=hidden_parameters,
        expected_depth=4,
    )

    optimizers = instantiate_optimizers(
        regime=regime,
        training_arguments=training_arguments,
        model=model,
        hidden_parameters=hidden_parameters,
        auxiliary_parameters=(
            auxiliary_parameters
        ),
        unembedding_parameters=(
            unembedding_parameters
        ),
    )

    saved_optimizer_states = checkpoint[
        "optimizer_state_dicts"
    ]
    if set(optimizers) != set(
        saved_optimizer_states
    ):
        raise KeyError(
            "Optimizer groups in the checkpoint do not "
            "match the reconstructed optimizer groups."
        )

    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(
            saved_optimizer_states[name]
        )

    (
        has_memorized,
        has_reached_95_test,
        consecutive_trigger_evaluations,
        scheduled_freeze_step,
        auxiliary_frozen,
    ) = reconstruct_trigger_state(
        records=source_records,
        threshold=freeze_trigger_accuracy,
    )

    actual_freeze_step = checkpoint.get(
        "actual_freeze_step"
    )
    if actual_freeze_step is not None:
        actual_freeze_step = int(
            actual_freeze_step
        )

    if auxiliary_frozen:
        for parameter in (
            auxiliary_parameters
            + unembedding_parameters
        ):
            parameter.requires_grad_(False)
            parameter.grad = None

    if "elapsed_seconds" in fieldnames:
        elapsed_offset = float(
            source_records[-1][
                "elapsed_seconds"
            ]
        )
    else:
        elapsed_offset = 0.0

    evaluation_records: list[
        dict[str, object]
    ] = [
        dict(record)
        for record in source_records
    ]

    last_record = source_records[-1]
    last_hidden_gradient_norm = float(
        last_record.get(
            "hidden_gradient_norm",
            "nan",
        )
    )
    last_auxiliary_gradient_norm = float(
        last_record.get(
            "auxiliary_gradient_norm",
            "nan",
        )
    )
    last_unembedding_gradient_norm = float(
        last_record.get(
            "unembedding_gradient_norm",
            "nan",
        )
    )
    last_muon_applied_update_norm = float(
        last_record.get(
            "muon_applied_update_norm",
            "nan",
        )
    )
    last_muon_max_abs_applied_update = float(
        last_record.get(
            "muon_max_abs_applied_update",
            "nan",
        )
    )

    print(f"Device: {device}")
    print(f"Continuation run: {args.output_run_name}")
    print(f"Regime: {regime}")
    print(
        f"Resuming exact trajectory at step "
        f"{resume_step}"
    )
    print(
        "Verified architecture: "
        f"blocks=4, "
        f"hidden_matrices={hidden_matrix_count}, "
        f"parameters={parameter_count}"
    )
    print(
        "Reconstructed stable-Muon state: "
        f"scheduled_freeze_step="
        f"{scheduled_freeze_step}, "
        f"actual_freeze_step="
        f"{actual_freeze_step}, "
        f"auxiliary_frozen="
        f"{int(auxiliary_frozen)}, "
        f"trailing_trigger_evaluations="
        f"{consecutive_trigger_evaluations}"
    )

    started_at = time.perf_counter()

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(source_records)
        handle.flush()

        initial_output_checkpoint = (
            output_checkpoint_directory
            / f"step_{resume_step:06d}.pt"
        )
        save_checkpoint(
            path=initial_output_checkpoint,
            step=resume_step,
            output_run_name=(
                args.output_run_name
            ),
            regime=regime,
            training_arguments=(
                training_arguments
            ),
            model_configuration=(
                model_configuration
            ),
            model=model,
            optimizers=optimizers,
            seed=seed,
            actual_freeze_step=(
                actual_freeze_step
            ),
            scheduled_freeze_step=(
                scheduled_freeze_step
            ),
            auxiliary_frozen=(
                auxiliary_frozen
            ),
            source_checkpoint=(
                args.source_checkpoint
            ),
        )

        for completed_step in range(
            resume_step,
            args.total_steps,
        ):
            if (
                regime == "stable_muon"
                and not auxiliary_frozen
                and scheduled_freeze_step
                is not None
                and completed_step
                >= scheduled_freeze_step
            ):
                for parameter in (
                    auxiliary_parameters
                    + unembedding_parameters
                ):
                    parameter.requires_grad_(
                        False
                    )
                    parameter.grad = None

                auxiliary_frozen = True
                actual_freeze_step = (
                    completed_step
                )
                print(
                    "Froze all AdamW-managed "
                    f"parameters at step "
                    f"{completed_step}."
                )

            model.train()
            for optimizer in optimizers.values():
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(train_inputs)
            loss = F.cross_entropy(
                logits,
                train_targets,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss at update "
                    f"{completed_step}."
                )

            loss.backward()

            last_hidden_gradient_norm = (
                gradient_l2_norm(
                    hidden_parameters
                )
            )
            last_auxiliary_gradient_norm = (
                gradient_l2_norm(
                    auxiliary_parameters
                )
            )
            last_unembedding_gradient_norm = (
                gradient_l2_norm(
                    unembedding_parameters
                )
            )

            if regime == "adamw":
                optimizers["adamw"].step()
            else:
                muon_optimizer = optimizers[
                    "muon"
                ]
                muon_optimizer.step()

                if not auxiliary_frozen:
                    optimizers[
                        "auxiliary_adamw"
                    ].step()
                    optimizers[
                        "unembedding_adamw"
                    ].step()

                statistics = (
                    muon_optimizer
                    .last_step_stats
                )
                last_muon_applied_update_norm = (
                    statistics[
                        "applied_update_norm"
                    ]
                )
                last_muon_max_abs_applied_update = (
                    statistics[
                        "max_abs_applied_update"
                    ]
                )

            step = completed_step + 1

            if (
                step % evaluation_interval
                == 0
            ):
                (
                    train_loss,
                    train_accuracy,
                ) = evaluate(
                    model,
                    train_inputs,
                    train_targets,
                )
                (
                    test_loss,
                    test_accuracy,
                ) = evaluate(
                    model,
                    test_inputs,
                    test_targets,
                )

                if train_accuracy >= 0.999:
                    has_memorized = True
                if test_accuracy >= 0.95:
                    has_reached_95_test = True

                if (
                    regime == "stable_muon"
                    and scheduled_freeze_step
                    is None
                ):
                    if (
                        test_accuracy
                        >= freeze_trigger_accuracy
                    ):
                        consecutive_trigger_evaluations += 1
                    else:
                        consecutive_trigger_evaluations = 0

                    if (
                        consecutive_trigger_evaluations
                        >= freeze_trigger_consecutive
                    ):
                        first_trigger_step = (
                            step
                            - (
                                freeze_trigger_consecutive
                                - 1
                            )
                            * evaluation_interval
                        )
                        scheduled_freeze_step = (
                            first_trigger_step
                            + freeze_delay
                        )
                        print(
                            "Scheduled auxiliary "
                            f"freeze for step "
                            f"{scheduled_freeze_step}."
                        )

                train_collapse = (
                    has_memorized
                    and train_accuracy < 0.90
                )
                generalization_collapse = (
                    has_reached_95_test
                    and test_accuracy < 0.90
                )

                elapsed_seconds = (
                    elapsed_offset
                    + (
                        time.perf_counter()
                        - started_at
                    )
                )

                row_values = {
                    "step": step,
                    "regime": regime,
                    "num_layers": 4,
                    "modulus": int(
                        training_arguments[
                            "modulus"
                        ]
                    ),
                    "train_fraction": float(
                        training_arguments[
                            "train_fraction"
                        ]
                    ),
                    "d_model": int(
                        training_arguments[
                            "d_model"
                        ]
                    ),
                    "num_heads": int(
                        training_arguments[
                            "num_heads"
                        ]
                    ),
                    "d_mlp": int(
                        training_arguments[
                            "d_mlp"
                        ]
                    ),
                    "parameter_count": (
                        parameter_count
                    ),
                    "hidden_matrix_count": (
                        hidden_matrix_count
                    ),
                    "initialization_scheme": (
                        "nested_depth_one_shared_v2"
                    ),
                    "train_loss": train_loss,
                    "train_accuracy": (
                        train_accuracy
                    ),
                    "test_loss": test_loss,
                    "test_accuracy": (
                        test_accuracy
                    ),
                    "elapsed_seconds": (
                        elapsed_seconds
                    ),
                    "hidden_parameter_norm": (
                        parameter_l2_norm(
                            hidden_parameters
                        )
                    ),
                    "auxiliary_parameter_norm": (
                        parameter_l2_norm(
                            auxiliary_parameters
                        )
                    ),
                    "unembedding_parameter_norm": (
                        parameter_l2_norm(
                            unembedding_parameters
                        )
                    ),
                    "hidden_gradient_norm": (
                        last_hidden_gradient_norm
                    ),
                    "auxiliary_gradient_norm": (
                        last_auxiliary_gradient_norm
                    ),
                    "unembedding_gradient_norm": (
                        last_unembedding_gradient_norm
                    ),
                    "muon_applied_update_norm": (
                        last_muon_applied_update_norm
                    ),
                    "muon_max_abs_applied_update": (
                        last_muon_max_abs_applied_update
                    ),
                    "train_collapse_detected": int(
                        train_collapse
                    ),
                    "generalization_collapse_detected": int(
                        generalization_collapse
                    ),
                    "auxiliary_frozen": int(
                        auxiliary_frozen
                    ),
                    "scheduled_freeze_step": (
                        ""
                        if scheduled_freeze_step
                        is None
                        else scheduled_freeze_step
                    ),
                }

                row = {
                    field: row_values.get(
                        field,
                        "",
                    )
                    for field in fieldnames
                }

                writer.writerow(row)
                handle.flush()
                evaluation_records.append(row)

                print(
                    f"step={step:6d} | "
                    f"train={train_accuracy:.4f} | "
                    f"test={test_accuracy:.4f} | "
                    f"aux_frozen="
                    f"{int(auxiliary_frozen)}"
                )

            if (
                step % args.checkpoint_interval
                == 0
                or step == args.total_steps
            ):
                checkpoint_path = (
                    output_checkpoint_directory
                    / f"step_{step:06d}.pt"
                )
                save_checkpoint(
                    path=checkpoint_path,
                    step=step,
                    output_run_name=(
                        args.output_run_name
                    ),
                    regime=regime,
                    training_arguments=(
                        training_arguments
                    ),
                    model_configuration=(
                        model_configuration
                    ),
                    model=model,
                    optimizers=optimizers,
                    seed=seed,
                    actual_freeze_step=(
                        actual_freeze_step
                    ),
                    scheduled_freeze_step=(
                        scheduled_freeze_step
                    ),
                    auxiliary_frozen=(
                        auxiliary_frozen
                    ),
                    source_checkpoint=(
                        args.source_checkpoint
                    ),
                )

    memorization_step = first_sustained_step(
        evaluation_records,
        "train_accuracy",
        0.999,
    )
    grokking_step = first_sustained_step(
        evaluation_records,
        "test_accuracy",
        0.95,
    )

    print()
    print(
        "Sustained 99.9% train step: "
        f"{memorization_step}"
    )
    print(
        "Sustained 95% test step: "
        f"{grokking_step}"
    )
    print(
        "Actual auxiliary freeze step: "
        f"{actual_freeze_step}"
    )
    print(
        "Final test accuracy: "
        f"{float(evaluation_records[-1]['test_accuracy']):.6f}"
    )
    print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
    main()
