from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from data import generate_modular_addition_data
from model import ModularAdditionTransformer
from optimizers.muon import Muon


MODULUS = 113
TRAIN_FRACTION = 0.3
SEQUENCE_LENGTH = 3

D_MODEL = 128
NUMBER_OF_HEADS = 4
D_MLP = 512


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Muon regime from scratch with a separate "
            "learning rate for the unembedding."
        )
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
        choices=["auto", "cuda", "mps"],
        default="auto",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--initial-state-path",
        type=Path,
        default=None,
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
        help=(
            "AdamW learning rate for token and position embeddings."
        ),
    )

    parser.add_argument(
        "--aux-weight-decay",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--aux-beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--aux-beta2",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--unembedding-lr",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--unembedding-weight-decay",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--freeze-unembedding-step",
        type=int,
        default=8_000,
        help=(
            "Freeze the unembedding after this many completed "
            "training updates. At step 8000, the evaluation uses "
            "the readout produced by updates 0 through 7999, and "
            "all subsequent training updates leave it unchanged."
        ),
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    positive_values = {
        "--steps": args.steps,
        "--evaluation-interval": args.evaluation_interval,
        "--checkpoint-interval": args.checkpoint_interval,
        "--muon-ns-steps": args.muon_ns_steps,
    }

    positive_values[
        "--freeze-unembedding-step"
    ] = args.freeze_unembedding_step

    for name, value in positive_values.items():
        if value < 1:
            raise ValueError(
                f"{name} must be at least 1."
            )

    if args.freeze_unembedding_step > args.steps:
        raise ValueError(
            "--freeze-unembedding-step cannot exceed --steps."
        )

    nonnegative_values = {
        "--muon-lr": args.muon_lr,
        "--muon-weight-decay": args.muon_weight_decay,
        "--aux-lr": args.aux_lr,
        "--aux-weight-decay": args.aux_weight_decay,
        "--unembedding-lr": args.unembedding_lr,
        "--unembedding-weight-decay": (
            args.unembedding_weight_decay
        ),
    }

    for name, value in nonnegative_values.items():
        if value < 0:
            raise ValueError(
                f"{name} must be nonnegative."
            )


def get_device(requested_device: str) -> torch.device:
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
        "No CUDA or MPS device is available."
    )


def set_seed(seed: int) -> None:
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
    loss = F.cross_entropy(logits, targets)

    accuracy = (
        logits.argmax(dim=-1) == targets
    ).float().mean()

    return loss.item(), accuracy.item()


def get_component_parameters(
    model: nn.Module,
    candidates: list[str],
    component_name: str,
) -> tuple[str, list[Parameter]]:
    for attribute_name in candidates:
        if not hasattr(model, attribute_name):
            continue

        value = getattr(model, attribute_name)

        if isinstance(value, Parameter):
            return attribute_name, [value]

        if isinstance(value, nn.Module):
            parameters = list(value.parameters())

            if parameters:
                return attribute_name, parameters

    available_names = [
        name
        for name, _ in model.named_parameters()
    ]

    raise AttributeError(
        f"Could not find {component_name}. "
        f"Tried {candidates}. "
        f"Available parameter names: {available_names}"
    )


def split_parameter_groups(
    model: ModularAdditionTransformer,
) -> tuple[
    list[Parameter],
    list[Parameter],
    list[Parameter],
]:
    hidden_parameters = [
        model.transformer_block.attention.qkv_projection.weight,
        model.transformer_block.attention.output_projection.weight,
        model.transformer_block.mlp.input_projection.weight,
        model.transformer_block.mlp.output_projection.weight,
    ]

    (
        token_attribute,
        token_parameters,
    ) = get_component_parameters(
        model,
        [
            "token_embedding",
            "token_embeddings",
        ],
        "token embedding",
    )

    (
        position_attribute,
        position_parameters,
    ) = get_component_parameters(
        model,
        [
            "position_embedding",
            "position_embeddings",
            "positional_embedding",
            "positional_embeddings",
            "pos_embedding",
        ],
        "position embedding",
    )

    (
        unembedding_attribute,
        unembedding_parameters,
    ) = get_component_parameters(
        model,
        [
            "unembedding",
            "unembed",
        ],
        "unembedding",
    )

    hidden_ids = {
        id(parameter)
        for parameter in hidden_parameters
    }

    unembedding_ids = {
        id(parameter)
        for parameter in unembedding_parameters
    }

    auxiliary_parameters = [
        parameter
        for parameter in model.parameters()
        if (
            id(parameter) not in hidden_ids
            and id(parameter) not in unembedding_ids
        )
    ]

    expected_auxiliary_ids = {
        id(parameter)
        for parameter in (
            token_parameters
            + position_parameters
        )
    }

    actual_auxiliary_ids = {
        id(parameter)
        for parameter in auxiliary_parameters
    }

    if expected_auxiliary_ids != actual_auxiliary_ids:
        names = {
            id(parameter): name
            for name, parameter in model.named_parameters()
        }

        unexpected = [
            names[id(parameter)]
            for parameter in auxiliary_parameters
            if id(parameter) not in expected_auxiliary_ids
        ]

        raise RuntimeError(
            "Unexpected auxiliary parameters found: "
            f"{unexpected}"
        )

    print(f"Token embedding attribute: {token_attribute}")
    print(f"Position embedding attribute: {position_attribute}")
    print(f"Unembedding attribute: {unembedding_attribute}")

    return (
        hidden_parameters,
        auxiliary_parameters,
        unembedding_parameters,
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

    return math.sqrt(total_squared_norm)


def parameter_l2_norm(
    parameters: Iterable[Parameter],
) -> float:
    total_squared_norm = 0.0

    for parameter in parameters:
        norm = torch.linalg.vector_norm(
            parameter.detach().float()
        ).item()

        total_squared_norm += norm ** 2

    return math.sqrt(total_squared_norm)


def first_sustained_step(
    records: list[dict],
    key: str,
    threshold: float,
    consecutive: int = 5,
) -> int | None:
    for index in range(
        len(records) - consecutive + 1
    ):
        values = [
            records[position][key]
            for position in range(
                index,
                index + consecutive,
            )
        ]

        if all(
            value >= threshold
            for value in values
        ):
            return int(
                records[index]["step"]
            )

    return None


def prepare_outputs(
    run_name: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    runs_directory = Path("runs")
    checkpoint_directory = (
        Path("checkpoints") / run_name
    )
    csv_path = (
        runs_directory / f"{run_name}.csv"
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
            f"CSV already exists: {csv_path}. "
            "Use --overwrite to replace it."
        )

    if checkpoint_directory.exists():
        raise FileExistsError(
            f"Checkpoint directory already exists: "
            f"{checkpoint_directory}. "
            "Use --overwrite to replace it."
        )

    runs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return csv_path, checkpoint_directory


def get_initial_state_path(
    args: argparse.Namespace,
) -> Path:
    if args.initial_state_path is not None:
        return args.initial_state_path

    return (
        Path("checkpoints")
        / "initial_states"
        / f"model_seed_{args.seed}.pt"
    )


def load_or_create_initial_state(
    model: ModularAdditionTransformer,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        try:
            state = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            state = torch.load(
                path,
                map_location="cpu",
            )

        if (
            isinstance(state, dict)
            and "model_state_dict" in state
        ):
            state = state["model_state_dict"]

        model.load_state_dict(state)
        print(f"Loaded initial state: {path}")
        return

    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    torch.save(
        state,
        path,
    )

    print(f"Created initial state: {path}")


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
        modulus=MODULUS,
        train_fraction=TRAIN_FRACTION,
        seed=args.seed,
    )

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    model = ModularAdditionTransformer(
        modulus=MODULUS,
        sequence_length=SEQUENCE_LENGTH,
        d_model=D_MODEL,
        num_heads=NUMBER_OF_HEADS,
        d_mlp=D_MLP,
    )

    initial_state_path = get_initial_state_path(
        args
    )

    load_or_create_initial_state(
        model,
        initial_state_path,
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

    muon_optimizer = Muon(
        hidden_parameters,
        learning_rate=args.muon_lr,
        momentum=args.muon_momentum,
        weight_decay=args.muon_weight_decay,
        newton_schulz_steps=args.muon_ns_steps,
        nesterov=True,
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

    unembedding_optimizer = torch.optim.AdamW(
        unembedding_parameters,
        lr=args.unembedding_lr,
        weight_decay=args.unembedding_weight_decay,
        betas=(
            args.aux_beta1,
            args.aux_beta2,
        ),
    )

    print(f"Device: {device}")
    print(f"Run name: {args.run_name}")
    print(f"Muon LR: {args.muon_lr}")
    print(f"Auxiliary LR: {args.aux_lr}")
    print(
        f"Unembedding LR before freeze: "
        f"{args.unembedding_lr}"
    )
    print(
        f"Freeze unembedding step: "
        f"{args.freeze_unembedding_step}"
    )
    print(f"Steps: {args.steps}")
    print(f"Output CSV: {csv_path}")

    fieldnames = [
        "step",
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
        "collapse_detected",
        "generalization_collapse_detected",
        "unembedding_learning_rate",
        "unembedding_frozen",
    ]

    evaluation_records: list[dict] = []

    last_hidden_gradient_norm = float("nan")
    last_auxiliary_gradient_norm = float("nan")
    last_unembedding_gradient_norm = float("nan")
    last_muon_applied_update_norm = float("nan")
    last_muon_max_abs_applied_update = float("nan")

    has_memorized = False
    has_reached_95_test = False
    unembedding_is_frozen = False

    with csv_path.open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
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

                collapse_detected = (
                    has_memorized
                    and train_accuracy < 0.90
                )

                generalization_collapse_detected = (
                    has_reached_95_test
                    and test_accuracy < 0.90
                )

                row = {
                    "step": step,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
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
                    "collapse_detected": int(
                        collapse_detected
                    ),
                    "generalization_collapse_detected": int(
                        generalization_collapse_detected
                    ),
                    "unembedding_learning_rate": (
                        0.0
                        if unembedding_is_frozen
                        else args.unembedding_lr
                    ),
                    "unembedding_frozen": int(
                        unembedding_is_frozen
                    ),
                }

                writer.writerow(row)
                csv_file.flush()
                evaluation_records.append(row)

                print(
                    f"step={step:6d} | "
                    f"train_acc={train_accuracy:.4f} | "
                    f"test_acc={test_accuracy:.4f} | "
                    f"collapse={int(collapse_detected)} | "
                    f"gen_collapse="
                    f"{int(generalization_collapse_detected)} | "
                    f"unembed_frozen="
                    f"{int(unembedding_is_frozen)}"
                )

            if (
                step
                % args.checkpoint_interval
                == 0
            ):
                checkpoint_path = (
                    checkpoint_directory
                    / f"step_{step:06d}.pt"
                )

                torch.save(
                    {
                        "step": step,
                        "optimizer_name": "muon",
                        "model_state_dict": (
                            model.state_dict()
                        ),
                        "optimizer_state_dicts": {
                            "muon": (
                                muon_optimizer.state_dict()
                            ),
                            "auxiliary_adamw": (
                                auxiliary_optimizer.state_dict()
                            ),
                            "unembedding_adamw": (
                                unembedding_optimizer.state_dict()
                            ),
                        },
                        "seed": args.seed,
                        "arguments": vars(args),
                    },
                    checkpoint_path,
                )

            if step == args.steps:
                break

            if (
                not unembedding_is_frozen
                and step
                >= args.freeze_unembedding_step
            ):
                for parameter in unembedding_parameters:
                    parameter.requires_grad_(False)
                    parameter.grad = None

                unembedding_is_frozen = True

                print(
                    f"Froze unembedding at step {step}."
                )

            model.train()

            muon_optimizer.zero_grad(
                set_to_none=True
            )

            auxiliary_optimizer.zero_grad(
                set_to_none=True
            )

            unembedding_optimizer.zero_grad(
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
                    f"Non-finite loss at step {step}."
                )

            loss.backward()

            last_hidden_gradient_norm = gradient_l2_norm(
                hidden_parameters
            )

            last_auxiliary_gradient_norm = gradient_l2_norm(
                auxiliary_parameters
            )

            last_unembedding_gradient_norm = gradient_l2_norm(
                unembedding_parameters
            )

            muon_optimizer.step()
            auxiliary_optimizer.step()

            if not unembedding_is_frozen:
                unembedding_optimizer.step()

            muon_stats = (
                muon_optimizer.last_step_stats
            )

            last_muon_applied_update_norm = (
                muon_stats["applied_update_norm"]
            )

            last_muon_max_abs_applied_update = (
                muon_stats[
                    "max_abs_applied_update"
                ]
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

    collapse_count = sum(
        int(record["collapse_detected"])
        for record in evaluation_records
    )

    generalization_collapse_count = sum(
        int(
            record[
                "generalization_collapse_detected"
            ]
        )
        for record in evaluation_records
    )

    post_grokking_records = (
        [
            record
            for record in evaluation_records
            if (
                grokking_step is not None
                and record["step"] >= grokking_step
            )
        ]
    )

    minimum_post_grokking_test = (
        min(
            record["test_accuracy"]
            for record in post_grokking_records
        )
        if post_grokking_records
        else None
    )

    print()
    print(
        f"Sustained memorization step: "
        f"{memorization_step}"
    )
    print(
        f"Sustained 95% test step: "
        f"{grokking_step}"
    )
    print(
        f"Train-collapse evaluations: "
        f"{collapse_count}"
    )
    print(
        f"Generalization-collapse evaluations: "
        f"{generalization_collapse_count}"
    )
    print(
        f"Minimum post-grokking test accuracy: "
        f"{minimum_post_grokking_test}"
    )
    print(
        f"Final test accuracy: "
        f"{evaluation_records[-1]['test_accuracy']:.6f}"
    )
    print(f"Saved metrics to: {csv_path}")


if __name__ == "__main__":
    main()
