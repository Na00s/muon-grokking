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
from torch.optim import Optimizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data import generate_modular_addition_data
from model import ModularAdditionTransformer
from optimizers.muon import Muon


MODULUS = 113
TRAIN_FRACTION = 0.3
SEQUENCE_LENGTH = 3

D_MODEL = 128
NUMBER_OF_HEADS = 4
D_MLP = 512

DEFAULT_NUMBER_OF_STEPS = 100_000
DEFAULT_EVALUATION_INTERVAL = 100
DEFAULT_CHECKPOINT_INTERVAL = 1_000

# These match the completed original AdamW run.
DEFAULT_ADAMW_LEARNING_RATE = 1e-3
DEFAULT_ADAMW_WEIGHT_DECAY = 1.0
DEFAULT_ADAMW_BETAS = (0.9, 0.999)

DEFAULT_MUON_LEARNING_RATE = 0.02
DEFAULT_MUON_MOMENTUM = 0.95
DEFAULT_MUON_NEWTON_SCHULZ_STEPS = 5

DEFAULT_AUXILIARY_LEARNING_RATE = 1e-3
DEFAULT_AUXILIARY_WEIGHT_DECAY = 1.0
DEFAULT_AUXILIARY_BETAS = (0.9, 0.999)

# The original AdamW run used:
#
# learning rate * weight decay
# = 0.001 * 1.0
# = 0.001 per step
#
# When Muon weight decay is not given explicitly, we preserve
# this same per-step decay multiplier.
REFERENCE_DECAY_PER_STEP = (
    DEFAULT_ADAMW_LEARNING_RATE
    * DEFAULT_ADAMW_WEIGHT_DECAY
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the modular-addition Transformer "
            "using AdamW or Muon."
        )
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "muon"],
        required=True,
        help="Optimizer regime to use.",
    )

    parser.add_argument(
        "--operation",
        type=str,
        choices=["addition", "subtraction"],
        default="addition",
        help="Modular operation the targets are drawn from.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_NUMBER_OF_STEPS,
    )

    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=DEFAULT_EVALUATION_INTERVAL,
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps"],
        default="auto",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete an existing run with the same name "
            "before starting."
        ),
    )

    parser.add_argument(
        "--initial-state-path",
        type=str,
        default=None,
        help=(
            "Optional path for the shared initial model state."
        ),
    )

    # AdamW-only regime.
    parser.add_argument(
        "--adamw-lr",
        type=float,
        default=DEFAULT_ADAMW_LEARNING_RATE,
    )

    parser.add_argument(
        "--adamw-weight-decay",
        type=float,
        default=DEFAULT_ADAMW_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--adamw-beta1",
        type=float,
        default=DEFAULT_ADAMW_BETAS[0],
    )

    parser.add_argument(
        "--adamw-beta2",
        type=float,
        default=DEFAULT_ADAMW_BETAS[1],
    )

    # Muon hidden-matrix optimizer.
    parser.add_argument(
        "--muon-lr",
        type=float,
        default=DEFAULT_MUON_LEARNING_RATE,
    )

    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=DEFAULT_MUON_MOMENTUM,
    )

    parser.add_argument(
        "--muon-weight-decay",
        type=float,
        default=None,
        help=(
            "Muon weight decay. When omitted, it is chosen "
            "to preserve the original AdamW per-step decay."
        ),
    )

    parser.add_argument(
        "--muon-ns-steps",
        type=int,
        default=DEFAULT_MUON_NEWTON_SCHULZ_STEPS,
    )

    parser.add_argument(
        "--disable-muon-nesterov",
        action="store_true",
    )

    # AdamW used for embeddings and the unembedding under Muon.
    parser.add_argument(
        "--aux-lr",
        type=float,
        default=DEFAULT_AUXILIARY_LEARNING_RATE,
    )

    parser.add_argument(
        "--aux-weight-decay",
        type=float,
        default=DEFAULT_AUXILIARY_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--aux-beta1",
        type=float,
        default=DEFAULT_AUXILIARY_BETAS[0],
    )

    parser.add_argument(
        "--aux-beta2",
        type=float,
        default=DEFAULT_AUXILIARY_BETAS[1],
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    if args.steps < 1:
        raise ValueError(
            "--steps must be at least 1."
        )

    if args.evaluation_interval < 1:
        raise ValueError(
            "--evaluation-interval must be at least 1."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            "--checkpoint-interval must be at least 1."
        )

    positive_values = {
        "--adamw-lr": args.adamw_lr,
        "--muon-lr": args.muon_lr,
        "--aux-lr": args.aux_lr,
    }

    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(
                f"{name} must be positive."
            )

    nonnegative_values = {
        "--adamw-weight-decay": (
            args.adamw_weight_decay
        ),
        "--aux-weight-decay": (
            args.aux_weight_decay
        ),
    }

    if args.muon_weight_decay is not None:
        nonnegative_values[
            "--muon-weight-decay"
        ] = args.muon_weight_decay

    for name, value in nonnegative_values.items():
        if value < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )

    beta_values = {
        "--adamw-beta1": args.adamw_beta1,
        "--adamw-beta2": args.adamw_beta2,
        "--aux-beta1": args.aux_beta1,
        "--aux-beta2": args.aux_beta2,
        "--muon-momentum": args.muon_momentum,
    }

    for name, value in beta_values.items():
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"{name} must be in [0, 1)."
            )

    if args.muon_ns_steps < 1:
        raise ValueError(
            "--muon-ns-steps must be at least 1."
        )


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(
    requested_device: str,
) -> torch.device:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return torch.device("cuda")

    if requested_device == "mps":
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
        "No CUDA or MPS GPU was found."
    )


@torch.no_grad()
def evaluate(
    model: ModularAdditionTransformer,
    inputs: Tensor,
    targets: Tensor,
) -> tuple[float, float]:
    model.eval()

    logits = model(inputs)  # (B, P)

    loss = F.cross_entropy(
        logits,
        targets,
    )

    predictions = logits.argmax(
        dim=-1
    )  # (B,)

    accuracy = (
        predictions == targets
    ).float().mean()

    return (
        loss.item(),
        accuracy.item(),
    )


def split_hidden_and_auxiliary_parameters(
    model: ModularAdditionTransformer,
) -> tuple[list[Parameter], list[Parameter]]:
    """
    Muon receives the hidden Transformer matrices.

    AdamW receives:
        token embeddings
        position embeddings
        unembedding
    """
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

    hidden_parameter_ids = {
        id(parameter)
        for parameter in hidden_parameters
    }

    auxiliary_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in hidden_parameter_ids
    ]

    all_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
    }

    auxiliary_parameter_ids = {
        id(parameter)
        for parameter in auxiliary_parameters
    }

    if (
        hidden_parameter_ids
        & auxiliary_parameter_ids
    ):
        raise RuntimeError(
            "Hidden and auxiliary parameter groups overlap."
        )

    covered_parameter_ids = (
        hidden_parameter_ids
        | auxiliary_parameter_ids
    )

    if covered_parameter_ids != all_parameter_ids:
        raise RuntimeError(
            "Optimizer parameter groups do not cover "
            "all model parameters."
        )

    return (
        hidden_parameters,
        auxiliary_parameters,
    )


def resolve_muon_weight_decay(
    args: argparse.Namespace,
) -> float:
    if args.muon_weight_decay is not None:
        return args.muon_weight_decay

    return (
        REFERENCE_DECAY_PER_STEP
        / args.muon_lr
    )


def build_optimizers(
    model: ModularAdditionTransformer,
    args: argparse.Namespace,
    muon_weight_decay: float,
) -> tuple[
    dict[str, Optimizer],
    list[Parameter],
    list[Parameter],
]:
    (
        hidden_parameters,
        auxiliary_parameters,
    ) = split_hidden_and_auxiliary_parameters(
        model
    )

    if args.optimizer == "adamw":
        optimizers: dict[str, Optimizer] = {
            "adamw": torch.optim.AdamW(
                model.parameters(),
                lr=args.adamw_lr,
                weight_decay=args.adamw_weight_decay,
                betas=(
                    args.adamw_beta1,
                    args.adamw_beta2,
                ),
            )
        }

        return (
            optimizers,
            hidden_parameters,
            auxiliary_parameters,
        )

    muon_optimizer = Muon(
        hidden_parameters,
        learning_rate=args.muon_lr,
        momentum=args.muon_momentum,
        weight_decay=muon_weight_decay,
        newton_schulz_steps=args.muon_ns_steps,
        nesterov=not args.disable_muon_nesterov,
    )

    auxiliary_optimizer = torch.optim.AdamW(
        auxiliary_parameters,
        lr=args.aux_lr,
        weight_decay=args.aux_weight_decay,
        betas=(
            args.aux_beta1,
            args.aux_beta2,
        ),
    )

    return (
        {
            "muon": muon_optimizer,
            "auxiliary_adamw": auxiliary_optimizer,
        },
        hidden_parameters,
        auxiliary_parameters,
    )


def validate_state_dict_shapes(
    model: ModularAdditionTransformer,
    state_dict: dict[str, Tensor],
) -> None:
    current_state_dict = model.state_dict()

    if set(state_dict) != set(current_state_dict):
        raise RuntimeError(
            "The saved initial state does not have the "
            "same parameter names as the current model. "
            "Delete the initial-state file and rerun."
        )

    for name, current_tensor in current_state_dict.items():
        saved_tensor = state_dict[name]

        if saved_tensor.shape != current_tensor.shape:
            raise RuntimeError(
                f"Initial-state shape mismatch for {name}: "
                f"saved={tuple(saved_tensor.shape)}, "
                f"current={tuple(current_tensor.shape)}. "
                "Delete the initial-state file and rerun."
            )


def load_or_create_initial_state(
    model: ModularAdditionTransformer,
    path: Path,
    seed: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        payload = torch.load(
            path,
            map_location="cpu",
        )

        if "model_state_dict" in payload:
            state_dict = payload[
                "model_state_dict"
            ]
        else:
            state_dict = payload

        validate_state_dict_shapes(
            model,
            state_dict,
        )

        model.load_state_dict(
            state_dict
        )

        print(
            f"Loaded shared initial state: {path}"
        )

        return

    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    torch.save(
        {
            "seed": seed,
            "model_state_dict": state_dict,
            "model_config": {
                "modulus": MODULUS,
                "sequence_length": SEQUENCE_LENGTH,
                "d_model": D_MODEL,
                "number_of_heads": NUMBER_OF_HEADS,
                "d_mlp": D_MLP,
            },
        },
        path,
    )

    print(
        f"Saved shared initial state: {path}"
    )


def parameter_l2_norm(
    parameters: Iterable[Parameter],
) -> float:
    total_squared_norm = 0.0

    for parameter in parameters:
        norm = torch.linalg.vector_norm(
            parameter.detach().float()
        ).item()

        total_squared_norm += norm ** 2

    return math.sqrt(
        total_squared_norm
    )


def gradient_l2_norm(
    parameters: Iterable[Parameter],
) -> float:
    total_squared_norm = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        if not torch.isfinite(gradient).all():
            raise FloatingPointError(
                "A gradient contains NaN or infinity."
            )

        norm = torch.linalg.vector_norm(
            gradient.float()
        ).item()

        total_squared_norm += norm ** 2

    return math.sqrt(
        total_squared_norm
    )


def float_to_name(
    value: float,
) -> str:
    return (
        f"{value:g}"
        .replace("-", "m")
        .replace(".", "p")
    )


def operation_suffix(
    operation: str,
) -> str:
    """
    Addition keeps the original naming so existing runs are
    unchanged. Every other operation gets a suffix, which
    separates both the run log and the checkpoint directory.
    """
    if operation == "addition":
        return ""

    return f"_{operation}"


def resolve_run_name(
    args: argparse.Namespace,
    muon_weight_decay: float,
) -> str:
    suffix = operation_suffix(args.operation)

    if args.run_name is not None:
        if suffix and not args.run_name.endswith(suffix):
            return f"{args.run_name}{suffix}"

        return args.run_name

    if args.optimizer == "adamw":
        return (
            f"adamw"
            f"_lr_{float_to_name(args.adamw_lr)}"
            f"_wd_{float_to_name(args.adamw_weight_decay)}"
            f"_seed_{args.seed}"
            f"{suffix}"
        )

    return (
        f"muon"
        f"_lr_{float_to_name(args.muon_lr)}"
        f"_wd_{float_to_name(muon_weight_decay)}"
        f"_seed_{args.seed}"
        f"{suffix}"
    )


def prepare_run_directories(
    run_name: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    checkpoint_directory = (
        Path("checkpoints")
        / run_name
    )

    runs_directory = Path("runs")

    log_path = (
        runs_directory
        / f"{run_name}.csv"
    )

    if overwrite:
        if checkpoint_directory.exists():
            shutil.rmtree(
                checkpoint_directory
            )

        if log_path.exists():
            log_path.unlink()

    if checkpoint_directory.exists():
        raise FileExistsError(
            f"Checkpoint directory already exists: "
            f"{checkpoint_directory}. "
            "Use --overwrite to replace it."
        )

    if log_path.exists():
        raise FileExistsError(
            f"Run log already exists: {log_path}. "
            "Use --overwrite to replace it."
        )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    runs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        checkpoint_directory,
        log_path,
    )


def first_sustained_step(
    history: list[dict[str, float]],
    metric_name: str,
    threshold: float,
    consecutive_evaluations: int = 5,
) -> int | None:
    for index in range(
        len(history)
        - consecutive_evaluations
        + 1
    ):
        window = history[
            index:
            index + consecutive_evaluations
        ]

        if all(
            row[metric_name] >= threshold
            for row in window
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

    muon_weight_decay = (
        resolve_muon_weight_decay(args)
    )

    if (
        args.muon_lr
        * muon_weight_decay
        >= 1.0
    ):
        raise ValueError(
            "Muon learning rate multiplied by Muon "
            "weight decay must be below 1."
        )

    run_name = resolve_run_name(
        args,
        muon_weight_decay,
    )

    (
        checkpoint_directory,
        log_path,
    ) = prepare_run_directories(
        run_name=run_name,
        overwrite=args.overwrite,
    )

    print(f"Device: {device}")
    print(f"Optimizer regime: {args.optimizer}")
    print(f"Operation: {args.operation}")
    print(f"Run name: {run_name}")
    print(f"Number of steps: {args.steps}")
    print(
        f"Evaluation interval: "
        f"{args.evaluation_interval}"
    )

    if args.optimizer == "muon":
        print(
            f"Muon learning rate: "
            f"{args.muon_lr}"
        )

        print(
            f"Muon momentum: "
            f"{args.muon_momentum}"
        )

        print(
            f"Muon weight decay: "
            f"{muon_weight_decay}"
        )

        print(
            "Muon per-step decay: "
            f"{args.muon_lr * muon_weight_decay}"
        )

        print(
            f"Auxiliary AdamW learning rate: "
            f"{args.aux_lr}"
        )

        print(
            f"Auxiliary AdamW weight decay: "
            f"{args.aux_weight_decay}"
        )

    (
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    ) = generate_modular_addition_data(
        modulus=MODULUS,
        train_fraction=TRAIN_FRACTION,
        seed=args.seed,
        operation=args.operation,
    )

    train_inputs = train_inputs.to(
        device
    )  # (B_train, T)

    train_targets = train_targets.to(
        device
    )  # (B_train,)

    test_inputs = test_inputs.to(
        device
    )  # (B_test, T)

    test_targets = test_targets.to(
        device
    )  # (B_test,)

    model = ModularAdditionTransformer(
        modulus=MODULUS,
        sequence_length=SEQUENCE_LENGTH,
        d_model=D_MODEL,
        num_heads=NUMBER_OF_HEADS,
        d_mlp=D_MLP,
    )

    if args.initial_state_path is None:
        initial_state_path = (
            Path("checkpoints")
            / "initial_states"
            / f"model_seed_{args.seed}.pt"
        )
    else:
        initial_state_path = Path(
            args.initial_state_path
        )

    load_or_create_initial_state(
        model=model,
        path=initial_state_path,
        seed=args.seed,
    )

    model = model.to(
        device
    )

    (
        optimizers,
        hidden_parameters,
        auxiliary_parameters,
    ) = build_optimizers(
        model=model,
        args=args,
        muon_weight_decay=muon_weight_decay,
    )

    fieldnames = [
        "step",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "total_parameter_norm",
        "hidden_parameter_norm",
        "auxiliary_parameter_norm",
        "hidden_gradient_norm",
        "auxiliary_gradient_norm",
        "muon_pre_ns_update_norm",
        "muon_post_ns_update_norm",
        "muon_applied_update_norm",
        "muon_max_abs_applied_update",
        "collapse_detected",
    ]

    history: list[dict[str, float]] = []

    has_memorized = False

    last_diagnostics = {
        "hidden_gradient_norm": float("nan"),
        "auxiliary_gradient_norm": float("nan"),
        "muon_pre_ns_update_norm": float("nan"),
        "muon_post_ns_update_norm": float("nan"),
        "muon_applied_update_norm": float("nan"),
        "muon_max_abs_applied_update": float("nan"),
    }

    with log_path.open(
        "w",
        newline="",
    ) as log_file:
        writer = csv.DictWriter(
            log_file,
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

                collapse_detected = (
                    has_memorized
                    and train_accuracy < 0.90
                )

                total_parameter_norm = (
                    parameter_l2_norm(
                        model.parameters()
                    )
                )

                hidden_parameter_norm = (
                    parameter_l2_norm(
                        hidden_parameters
                    )
                )

                auxiliary_parameter_norm = (
                    parameter_l2_norm(
                        auxiliary_parameters
                    )
                )

                row = {
                    "step": step,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "total_parameter_norm": (
                        total_parameter_norm
                    ),
                    "hidden_parameter_norm": (
                        hidden_parameter_norm
                    ),
                    "auxiliary_parameter_norm": (
                        auxiliary_parameter_norm
                    ),
                    **last_diagnostics,
                    "collapse_detected": int(
                        collapse_detected
                    ),
                }

                writer.writerow(row)
                log_file.flush()

                history.append(
                    {
                        "step": float(step),
                        "train_accuracy": (
                            train_accuracy
                        ),
                        "test_accuracy": (
                            test_accuracy
                        ),
                    }
                )

                print(
                    f"step={step:6d} | "
                    f"train_loss={train_loss:.6f} | "
                    f"train_acc={train_accuracy:.4f} | "
                    f"test_loss={test_loss:.6f} | "
                    f"test_acc={test_accuracy:.4f} | "
                    f"hidden_norm="
                    f"{hidden_parameter_norm:.4f} | "
                    f"aux_norm="
                    f"{auxiliary_parameter_norm:.4f}"
                )

                if collapse_detected:
                    print(
                        "COLLAPSE DETECTED: "
                        "training accuracy fell below 90% "
                        "after memorization."
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
                        "run_name": run_name,
                        "optimizer_name": (
                            args.optimizer
                        ),
                        "arguments": vars(args),
                        "resolved_muon_weight_decay": (
                            muon_weight_decay
                        ),
                        "model_state_dict": (
                            model.state_dict()
                        ),
                        "optimizer_state_dicts": (
                            optimizer_states
                        ),
                        "seed": args.seed,
                        "modulus": MODULUS,
                        "operation": args.operation,
                        "train_fraction": (
                            TRAIN_FRACTION
                        ),
                    },
                    checkpoint_path,
                )

            if step == args.steps:
                break

            model.train()

            for optimizer in optimizers.values():
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(
                train_inputs
            )  # (B_train, P)

            loss = F.cross_entropy(
                logits,
                train_targets,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss "
                    f"at step {step}."
                )

            loss.backward()

            capture_diagnostics = (
                (step + 1)
                % args.evaluation_interval
                == 0
            )

            if capture_diagnostics:
                hidden_gradient_norm = (
                    gradient_l2_norm(
                        hidden_parameters
                    )
                )

                auxiliary_gradient_norm = (
                    gradient_l2_norm(
                        auxiliary_parameters
                    )
                )

            for optimizer in optimizers.values():
                optimizer.step()

            if capture_diagnostics:
                last_diagnostics[
                    "hidden_gradient_norm"
                ] = hidden_gradient_norm

                last_diagnostics[
                    "auxiliary_gradient_norm"
                ] = auxiliary_gradient_norm

                if "muon" in optimizers:
                    muon_optimizer = optimizers[
                        "muon"
                    ]

                    if not isinstance(
                        muon_optimizer,
                        Muon,
                    ):
                        raise TypeError(
                            "The Muon optimizer has an "
                            "unexpected type."
                        )

                    muon_stats = (
                        muon_optimizer
                        .last_step_stats
                    )

                    last_diagnostics[
                        "muon_pre_ns_update_norm"
                    ] = muon_stats[
                        "pre_ns_update_norm"
                    ]

                    last_diagnostics[
                        "muon_post_ns_update_norm"
                    ] = muon_stats[
                        "post_ns_update_norm"
                    ]

                    last_diagnostics[
                        "muon_applied_update_norm"
                    ] = muon_stats[
                        "applied_update_norm"
                    ]

                    last_diagnostics[
                        "muon_max_abs_applied_update"
                    ] = muon_stats[
                        "max_abs_applied_update"
                    ]

    memorization_step = first_sustained_step(
        history=history,
        metric_name="train_accuracy",
        threshold=0.999,
        consecutive_evaluations=5,
    )

    grokking_step = first_sustained_step(
        history=history,
        metric_name="test_accuracy",
        threshold=0.95,
        consecutive_evaluations=5,
    )

    print()
    print(f"Saved metrics to: {log_path}")
    print(
        f"Saved checkpoints to: "
        f"{checkpoint_directory}"
    )

    print(
        f"Sustained memorization step: "
        f"{memorization_step}"
    )

    print(
        f"Sustained 95% test step: "
        f"{grokking_step}"
    )

    if (
        memorization_step is not None
        and grokking_step is not None
    ):
        print(
            "Memorization plateau: "
            f"{grokking_step - memorization_step} "
            "steps"
        )


if __name__ == "__main__":
    main()