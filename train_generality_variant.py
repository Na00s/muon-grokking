from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


REPOSITORY_ROOT = Path(__file__).resolve().parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data import generate_modular_addition_data
from model import ModularAdditionTransformer
from optimizers.muon import Muon


SEQUENCE_LENGTH = 3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one controlled modular-addition generality "
            "condition with AdamW, Muon, or stabilized Muon."
        )
    )

    parser.add_argument(
        "--regime",
        choices=[
            "adamw",
            "muon",
            "stable_muon",
        ],
        required=True,
    )

    parser.add_argument(
        "--modulus",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--d-mlp",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
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
        "--run-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--initial-state-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--adamw-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--adamw-weight-decay",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--adamw-beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--adamw-beta2",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--muon-lr",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--muon-weight-decay",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--muon-ns-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--aux-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--aux-weight-decay",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--unembedding-lr",
        type=float,
        default=2.5e-4,
    )

    parser.add_argument(
        "--unembedding-weight-decay",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--freeze-trigger-test-accuracy",
        type=float,
        default=0.95,
        help=(
            "For stable_muon only: schedule the auxiliary "
            "freeze after this test threshold is sustained."
        ),
    )

    parser.add_argument(
        "--freeze-trigger-consecutive",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--freeze-delay",
        type=int,
        default=2_000,
        help=(
            "For stable_muon only: additional updates after "
            "the sustained test threshold before freezing all "
            "AdamW-managed parameters."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    if args.modulus < 3:
        raise ValueError(
            "--modulus must be at least 3."
        )

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError(
            "--train-fraction must lie in (0, 1)."
        )

    if args.d_model < 1:
        raise ValueError(
            "--d-model must be positive."
        )

    if args.num_heads < 1:
        raise ValueError(
            "--num-heads must be positive."
        )

    if args.d_model % args.num_heads != 0:
        raise ValueError(
            "--d-model must be divisible by "
            "--num-heads."
        )

    if args.d_mlp < 1:
        raise ValueError(
            "--d-mlp must be positive."
        )

    positive_integers = {
        "--steps": args.steps,
        "--evaluation-interval": (
            args.evaluation_interval
        ),
        "--checkpoint-interval": (
            args.checkpoint_interval
        ),
        "--muon-ns-steps": (
            args.muon_ns_steps
        ),
        "--freeze-trigger-consecutive": (
            args.freeze_trigger_consecutive
        ),
    }

    for name, value in positive_integers.items():
        if value < 1:
            raise ValueError(
                f"{name} must be at least 1."
            )

    if args.freeze_delay < 0:
        raise ValueError(
            "--freeze-delay cannot be negative."
        )

    if not (
        0.0
        <= args.freeze_trigger_test_accuracy
        <= 1.0
    ):
        raise ValueError(
            "--freeze-trigger-test-accuracy must "
            "lie in [0, 1]."
        )

    nonnegative_values = {
        "--adamw-lr": args.adamw_lr,
        "--adamw-weight-decay": (
            args.adamw_weight_decay
        ),
        "--muon-lr": args.muon_lr,
        "--muon-weight-decay": (
            args.muon_weight_decay
        ),
        "--aux-lr": args.aux_lr,
        "--aux-weight-decay": (
            args.aux_weight_decay
        ),
        "--unembedding-lr": (
            args.unembedding_lr
        ),
        "--unembedding-weight-decay": (
            args.unembedding_weight_decay
        ),
    }

    for name, value in nonnegative_values.items():
        if value < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )


def get_device(
    requested: str,
) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

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

    return torch.device("cpu")


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: ModularAdditionTransformer,
    inputs: Tensor,
    targets: Tensor,
) -> tuple[float, float]:
    model.eval()

    logits = model(inputs)
    loss = F.cross_entropy(
        logits,
        targets,
    )

    accuracy = (
        logits.argmax(dim=-1)
        == targets
    ).float().mean()

    return (
        float(loss.item()),
        float(accuracy.item()),
    )


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

        if not torch.isfinite(
            gradient
        ).all():
            raise FloatingPointError(
                "A gradient contains NaN or infinity."
            )

        value = torch.linalg.vector_norm(
            gradient.float()
        ).item()

        total += value * value

    return math.sqrt(total)


def split_parameter_groups(
    model: ModularAdditionTransformer,
) -> tuple[
    list[Parameter],
    list[Parameter],
    list[Parameter],
]:
    hidden_parameters = [
        model
        .transformer_block
        .attention
        .qkv_projection
        .weight,
        model
        .transformer_block
        .attention
        .output_projection
        .weight,
        model
        .transformer_block
        .mlp
        .input_projection
        .weight,
        model
        .transformer_block
        .mlp
        .output_projection
        .weight,
    ]

    unembedding_parameters = [
        model.unembedding.weight
    ]

    hidden_ids = {
        id(parameter)
        for parameter in hidden_parameters
    }

    unembedding_ids = {
        id(parameter)
        for parameter
        in unembedding_parameters
    }

    auxiliary_parameters = [
        parameter
        for parameter in model.parameters()
        if (
            id(parameter) not in hidden_ids
            and id(parameter)
            not in unembedding_ids
        )
    ]

    covered_ids = {
        id(parameter)
        for parameter in (
            hidden_parameters
            + auxiliary_parameters
            + unembedding_parameters
        )
    }

    model_ids = {
        id(parameter)
        for parameter in model.parameters()
    }

    if covered_ids != model_ids:
        raise RuntimeError(
            "Parameter groups do not exactly cover "
            "the model."
        )

    return (
        hidden_parameters,
        auxiliary_parameters,
        unembedding_parameters,
    )


def model_configuration(
    args: argparse.Namespace,
) -> dict[str, int]:
    return {
        "modulus": args.modulus,
        "sequence_length": (
            SEQUENCE_LENGTH
        ),
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_mlp": args.d_mlp,
    }


def load_or_create_initial_state(
    model: ModularAdditionTransformer,
    path: Path,
    configuration: dict[str, int],
    seed: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        try:
            loaded = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            loaded = torch.load(
                path,
                map_location="cpu",
            )

        if not isinstance(
            loaded,
            dict,
        ):
            raise TypeError(
                "Initial-state checkpoint must "
                "contain a dictionary."
            )

        saved_configuration = loaded.get(
            "model_config"
        )

        if (
            saved_configuration is not None
            and saved_configuration
            != configuration
        ):
            raise ValueError(
                "Initial-state model configuration "
                "does not match this run."
            )

        state = loaded.get(
            "model_state_dict",
            loaded,
        )

        model.load_state_dict(state)

        print(
            f"Loaded shared initial state: {path}"
        )
        return

    torch.save(
        {
            "seed": seed,
            "model_config": configuration,
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value
                in model.state_dict().items()
            },
        },
        path,
    )

    print(
        f"Created shared initial state: {path}"
    )


def prepare_outputs(
    run_name: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    runs_directory = Path("runs")
    csv_path = (
        runs_directory
        / f"{run_name}.csv"
    )
    checkpoint_directory = (
        Path("checkpoints")
        / run_name
    )

    if overwrite:
        if csv_path.exists():
            csv_path.unlink()

        if checkpoint_directory.exists():
            shutil.rmtree(
                checkpoint_directory
            )

    if csv_path.exists():
        raise FileExistsError(
            f"CSV already exists: {csv_path}"
        )

    if checkpoint_directory.exists():
        raise FileExistsError(
            "Checkpoint directory already "
            f"exists: {checkpoint_directory}"
        )

    runs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return (
        csv_path,
        checkpoint_directory,
    )


def first_sustained_step(
    records: list[dict[str, object]],
    field: str,
    threshold: float,
    consecutive: int = 5,
) -> int | None:
    for index in range(
        len(records)
        - consecutive
        + 1
    ):
        window = records[
            index:index + consecutive
        ]

        if all(
            float(record[field])
            >= threshold
            for record in window
        ):
            return int(
                window[0]["step"]
            )

    return None


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    set_seed(args.seed)

    device = get_device(
        args.device
    )

    (
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    ) = generate_modular_addition_data(
        modulus=args.modulus,
        train_fraction=(
            args.train_fraction
        ),
        seed=args.seed,
    )

    train_inputs = train_inputs.to(
        device
    )
    train_targets = train_targets.to(
        device
    )
    test_inputs = test_inputs.to(
        device
    )
    test_targets = test_targets.to(
        device
    )

    configuration = model_configuration(
        args
    )

    model = ModularAdditionTransformer(
        **configuration
    )

    load_or_create_initial_state(
        model=model,
        path=args.initial_state_path,
        configuration=configuration,
        seed=args.seed,
    )

    model = model.to(device)

    (
        hidden_parameters,
        auxiliary_parameters,
        unembedding_parameters,
    ) = split_parameter_groups(
        model
    )

    (
        csv_path,
        checkpoint_directory,
    ) = prepare_outputs(
        run_name=args.run_name,
        overwrite=args.overwrite,
    )

    optimizers: dict[str, object] = {}

    if args.regime == "adamw":
        optimizers["adamw"] = (
            torch.optim.AdamW(
                model.parameters(),
                lr=args.adamw_lr,
                weight_decay=(
                    args.adamw_weight_decay
                ),
                betas=(
                    args.adamw_beta1,
                    args.adamw_beta2,
                ),
            )
        )
    else:
        optimizers["muon"] = Muon(
            hidden_parameters,
            learning_rate=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=(
                args.muon_weight_decay
            ),
            newton_schulz_steps=(
                args.muon_ns_steps
            ),
            nesterov=True,
        )

        optimizers[
            "auxiliary_adamw"
        ] = torch.optim.AdamW(
            auxiliary_parameters,
            lr=args.aux_lr,
            weight_decay=(
                args.aux_weight_decay
            ),
            betas=(
                args.adamw_beta1,
                args.adamw_beta2,
            ),
        )

        optimizers[
            "unembedding_adamw"
        ] = torch.optim.AdamW(
            unembedding_parameters,
            lr=args.unembedding_lr,
            weight_decay=(
                args
                .unembedding_weight_decay
            ),
            betas=(
                args.adamw_beta1,
                args.adamw_beta2,
            ),
        )

    fieldnames = [
        "step",
        "regime",
        "modulus",
        "train_fraction",
        "d_model",
        "num_heads",
        "d_mlp",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "hidden_parameter_norm",
        "auxiliary_parameter_norm",
        "unembedding_parameter_norm",
        "hidden_gradient_norm",
        "auxiliary_gradient_norm",
        "unembedding_gradient_norm",
        "muon_applied_update_norm",
        "muon_max_abs_applied_update",
        "train_collapse_detected",
        "generalization_collapse_detected",
        "auxiliary_frozen",
        "scheduled_freeze_step",
    ]

    evaluation_records: list[
        dict[str, object]
    ] = []

    last_hidden_gradient_norm = (
        float("nan")
    )
    last_auxiliary_gradient_norm = (
        float("nan")
    )
    last_unembedding_gradient_norm = (
        float("nan")
    )
    last_muon_applied_update_norm = (
        float("nan")
    )
    last_muon_max_abs_applied_update = (
        float("nan")
    )

    has_memorized = False
    has_reached_95_test = False
    consecutive_trigger_evaluations = 0

    auxiliary_frozen = False
    scheduled_freeze_step: int | None = (
        None
    )
    actual_freeze_step: int | None = None

    print(f"Device: {device}")
    print(f"Run: {args.run_name}")
    print(f"Regime: {args.regime}")
    print(
        "Configuration: "
        f"modulus={args.modulus}, "
        "train_fraction="
        f"{args.train_fraction}, "
        f"d_model={args.d_model}, "
        f"d_mlp={args.d_mlp}"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for step in range(
            args.steps + 1
        ):
            if (
                step
                % args.evaluation_interval
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
                    args.regime
                    == "stable_muon"
                    and scheduled_freeze_step
                    is None
                ):
                    if (
                        test_accuracy
                        >= args
                        .freeze_trigger_test_accuracy
                    ):
                        consecutive_trigger_evaluations += 1
                    else:
                        consecutive_trigger_evaluations = 0

                    if (
                        consecutive_trigger_evaluations
                        >= args
                        .freeze_trigger_consecutive
                    ):
                        first_trigger_step = (
                            step
                            - (
                                args
                                .freeze_trigger_consecutive
                                - 1
                            )
                            * args
                            .evaluation_interval
                        )

                        scheduled_freeze_step = (
                            first_trigger_step
                            + args.freeze_delay
                        )

                        print(
                            "Scheduled auxiliary freeze "
                            f"for step "
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

                row = {
                    "step": step,
                    "regime": args.regime,
                    "modulus": args.modulus,
                    "train_fraction": (
                        args.train_fraction
                    ),
                    "d_model": args.d_model,
                    "num_heads": (
                        args.num_heads
                    ),
                    "d_mlp": args.d_mlp,
                    "train_loss": train_loss,
                    "train_accuracy": (
                        train_accuracy
                    ),
                    "test_loss": test_loss,
                    "test_accuracy": (
                        test_accuracy
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

                writer.writerow(row)
                handle.flush()
                evaluation_records.append(
                    row
                )

                print(
                    f"step={step:6d} | "
                    f"train={train_accuracy:.4f} | "
                    f"test={test_accuracy:.4f} | "
                    f"aux_frozen="
                    f"{int(auxiliary_frozen)}"
                )

            if (
                step
                % args.checkpoint_interval
                == 0
            ):
                optimizer_states = {
                    name: optimizer.state_dict()
                    for name, optimizer
                    in optimizers.items()
                }

                checkpoint_path = (
                    checkpoint_directory
                    / f"step_{step:06d}.pt"
                )

                torch.save(
                    {
                        "step": step,
                        "run_name": (
                            args.run_name
                        ),
                        "regime": args.regime,
                        "arguments": vars(args),
                        "model_config": (
                            configuration
                        ),
                        "model_state_dict": (
                            model.state_dict()
                        ),
                        "optimizer_state_dicts": (
                            optimizer_states
                        ),
                        "seed": args.seed,
                        "actual_freeze_step": (
                            actual_freeze_step
                        ),
                    },
                    checkpoint_path,
                )

            if step == args.steps:
                break

            if (
                args.regime
                == "stable_muon"
                and not auxiliary_frozen
                and scheduled_freeze_step
                is not None
                and step
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
                actual_freeze_step = step

                print(
                    "Froze all AdamW-managed "
                    f"parameters at step {step}."
                )

            model.train()

            for optimizer in optimizers.values():
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(
                train_inputs
            )

            loss = F.cross_entropy(
                logits,
                train_targets,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at step "
                    f"{step}."
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

            if args.regime == "adamw":
                optimizers[
                    "adamw"
                ].step()
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

                muon_statistics = (
                    muon_optimizer
                    .last_step_stats
                )

                last_muon_applied_update_norm = (
                    muon_statistics[
                        "applied_update_norm"
                    ]
                )

                last_muon_max_abs_applied_update = (
                    muon_statistics[
                        "max_abs_applied_update"
                    ]
                )

    memorization_step = (
        first_sustained_step(
            evaluation_records,
            "train_accuracy",
            0.999,
        )
    )

    grokking_step = (
        first_sustained_step(
            evaluation_records,
            "test_accuracy",
            0.95,
        )
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
        f"{evaluation_records[-1]['test_accuracy']:.6f}"
    )
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
