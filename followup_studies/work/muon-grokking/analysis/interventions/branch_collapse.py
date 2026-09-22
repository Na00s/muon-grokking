from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path
from typing import Iterable

import torch
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
            "Branch from a Muon checkpoint and freeze either the hidden "
            "Muon group or the auxiliary AdamW group."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint from the original Muon run.",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "control",
            "freeze_hidden",
            "freeze_auxiliary",
        ],
        required=True,
        help=(
            "control: update both groups; "
            "freeze_hidden: update auxiliary AdamW only; "
            "freeze_auxiliary: update hidden Muon matrices only."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2_000,
        help="Number of additional optimization steps.",
    )

    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--device",
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
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {args.checkpoint}"
        )

    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")

    if args.evaluation_interval < 1:
        raise ValueError(
            "--evaluation-interval must be at least 1."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            "--checkpoint-interval must be at least 1."
        )


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
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


def split_hidden_and_auxiliary_parameters(
    model: ModularAdditionTransformer,
) -> tuple[list[Parameter], list[Parameter]]:
    hidden_parameters = [
        model.transformer_block.attention.qkv_projection.weight,
        model.transformer_block.attention.output_projection.weight,
        model.transformer_block.mlp.input_projection.weight,
        model.transformer_block.mlp.output_projection.weight,
    ]

    hidden_ids = {
        id(parameter)
        for parameter in hidden_parameters
    }

    auxiliary_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in hidden_ids
    ]

    auxiliary_ids = {
        id(parameter)
        for parameter in auxiliary_parameters
    }

    all_ids = {
        id(parameter)
        for parameter in model.parameters()
    }

    if hidden_ids & auxiliary_ids:
        raise RuntimeError(
            "Hidden and auxiliary parameter groups overlap."
        )

    if hidden_ids | auxiliary_ids != all_ids:
        raise RuntimeError(
            "Optimizer groups do not cover all parameters."
        )

    return hidden_parameters, auxiliary_parameters


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


def parameter_delta_l2_norm(
    parameters: list[Parameter],
    starting_values: list[Tensor],
) -> float:
    total_squared_norm = 0.0

    for parameter, starting_value in zip(
        parameters,
        starting_values,
        strict=True,
    ):
        difference = (
            parameter.detach().float()
            - starting_value
        )

        norm = torch.linalg.vector_norm(
            difference
        ).item()

        total_squared_norm += norm ** 2

    return math.sqrt(total_squared_norm)


def clone_parameter_values(
    parameters: list[Parameter],
) -> list[Tensor]:
    return [
        parameter.detach().float().clone()
        for parameter in parameters
    ]


@torch.no_grad()
def applied_update_statistics(
    parameters: list[Parameter],
    values_before_step: list[Tensor],
    learning_rate: float,
    weight_decay: float,
) -> dict[str, float]:
    """
    Decompose the tensor the optimizer actually subtracted.

    Muon and decoupled AdamW both apply

        parameter <- parameter * (1 - lr * wd) - lr * update

    so the decay contribution is exactly lr * wd * parameter_before
    and the gradient-driven contribution is the remainder. The decay
    term is not a response to the gradient, so it is reported
    separately rather than counted as one.
    """
    total_squared = 0.0
    gradient_squared = 0.0
    decay_squared = 0.0
    parameter_squared = 0.0
    element_count = 0

    for parameter, before in zip(
        parameters,
        values_before_step,
        strict=True,
    ):
        after = parameter.detach().float()

        total = before - after
        decay = before * (learning_rate * weight_decay)
        gradient = total - decay

        total_squared += float(
            total.pow(2).sum().item()
        )
        gradient_squared += float(
            gradient.pow(2).sum().item()
        )
        decay_squared += float(
            decay.pow(2).sum().item()
        )
        parameter_squared += float(
            after.pow(2).sum().item()
        )
        element_count += parameter.numel()

    applied_norm = math.sqrt(total_squared)

    return {
        "applied_update_norm": applied_norm,
        "gradient_component_norm": math.sqrt(
            gradient_squared
        ),
        "decay_component_norm": math.sqrt(
            decay_squared
        ),
        "applied_update_rms": (
            applied_norm / math.sqrt(element_count)
            if element_count
            else float("nan")
        ),
        "parameter_norm": math.sqrt(
            parameter_squared
        ),
    }


ZERO_UPDATE_STATISTICS = {
    "applied_update_norm": 0.0,
    "gradient_component_norm": 0.0,
    "decay_component_norm": 0.0,
    "applied_update_rms": 0.0,
    "parameter_norm": float("nan"),
}


def build_optimizers(
    hidden_parameters: list[Parameter],
    auxiliary_parameters: list[Parameter],
    checkpoint: dict,
) -> tuple[Muon, torch.optim.AdamW]:
    saved_arguments = checkpoint.get(
        "arguments",
        {},
    )

    resolved_muon_weight_decay = checkpoint.get(
        "resolved_muon_weight_decay",
        saved_arguments.get(
            "muon_weight_decay",
            0.10,
        ),
    )

    if resolved_muon_weight_decay is None:
        muon_learning_rate = saved_arguments.get(
            "muon_lr",
            0.01,
        )

        resolved_muon_weight_decay = (
            0.001 / muon_learning_rate
        )

    muon_optimizer = Muon(
        hidden_parameters,
        learning_rate=saved_arguments.get(
            "muon_lr",
            0.01,
        ),
        momentum=saved_arguments.get(
            "muon_momentum",
            0.95,
        ),
        weight_decay=resolved_muon_weight_decay,
        newton_schulz_steps=saved_arguments.get(
            "muon_ns_steps",
            5,
        ),
        nesterov=not saved_arguments.get(
            "disable_muon_nesterov",
            False,
        ),
    )

    auxiliary_optimizer = torch.optim.AdamW(
        auxiliary_parameters,
        lr=saved_arguments.get(
            "aux_lr",
            1e-3,
        ),
        weight_decay=saved_arguments.get(
            "aux_weight_decay",
            1.0,
        ),
        betas=(
            saved_arguments.get(
                "aux_beta1",
                0.9,
            ),
            saved_arguments.get(
                "aux_beta2",
                0.999,
            ),
        ),
    )

    optimizer_states = checkpoint.get(
        "optimizer_state_dicts"
    )

    if optimizer_states is None:
        raise KeyError(
            "Checkpoint is missing optimizer_state_dicts."
        )

    if "muon" not in optimizer_states:
        raise KeyError(
            "Checkpoint is missing the Muon optimizer state."
        )

    if "auxiliary_adamw" not in optimizer_states:
        raise KeyError(
            "Checkpoint is missing the auxiliary AdamW state."
        )

    muon_optimizer.load_state_dict(
        optimizer_states["muon"]
    )

    auxiliary_optimizer.load_state_dict(
        optimizer_states["auxiliary_adamw"]
    )

    return muon_optimizer, auxiliary_optimizer


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


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    checkpoint = load_checkpoint(
        args.checkpoint
    )

    if checkpoint.get("optimizer_name") != "muon":
        raise ValueError(
            "This experiment requires a checkpoint from "
            "the Muon regime."
        )

    start_step = int(
        checkpoint["step"]
    )

    seed = int(
        checkpoint.get("seed", 0)
    )

    set_seed(seed)

    device = get_device(
        args.device
    )

    run_name = (
        args.run_name
        or (
            f"branch_{args.mode}"
            f"_from_{start_step}"
        )
    )

    csv_path, checkpoint_directory = (
        prepare_outputs(
            run_name=run_name,
            overwrite=args.overwrite,
        )
    )

    (
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    ) = generate_modular_addition_data(
        modulus=MODULUS,
        train_fraction=TRAIN_FRACTION,
        seed=seed,
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
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        hidden_parameters,
        auxiliary_parameters,
    ) = split_hidden_and_auxiliary_parameters(
        model
    )

    (
        muon_optimizer,
        auxiliary_optimizer,
    ) = build_optimizers(
        hidden_parameters=hidden_parameters,
        auxiliary_parameters=auxiliary_parameters,
        checkpoint=checkpoint,
    )

    hidden_start = clone_parameter_values(
        hidden_parameters
    )

    auxiliary_start = clone_parameter_values(
        auxiliary_parameters
    )

    # The readout shares the auxiliary AdamW group, so it is not a
    # separate optimizer group here. It is reported separately anyway,
    # because the question is whether the per-parameter separation
    # tracks the update rule or the learning rate.
    readout_parameters = [
        model.unembedding.weight
    ]

    readout_ids = {
        id(parameter)
        for parameter in readout_parameters
    }

    embedding_parameters = [
        parameter
        for parameter in auxiliary_parameters
        if id(parameter) not in readout_ids
    ]

    muon_learning_rate = muon_optimizer.param_groups[0][
        "learning_rate"
    ]
    muon_weight_decay = muon_optimizer.param_groups[0][
        "weight_decay"
    ]
    auxiliary_learning_rate = (
        auxiliary_optimizer.param_groups[0]["lr"]
    )
    auxiliary_weight_decay = (
        auxiliary_optimizer.param_groups[0]["weight_decay"]
    )

    print(
        "Instrumented groups: "
        f"hidden (Muon, lr={muon_learning_rate}, "
        f"wd={muon_weight_decay}), "
        f"embeddings and readout "
        f"(AdamW, lr={auxiliary_learning_rate}, "
        f"wd={auxiliary_weight_decay})"
    )
    print(
        "Learning-rate ratio hidden:auxiliary = "
        f"{muon_learning_rate / auxiliary_learning_rate:.1f}"
    )

    update_hidden = (
        args.mode != "freeze_hidden"
    )

    update_auxiliary = (
        args.mode != "freeze_auxiliary"
    )

    for parameter in hidden_parameters:
        parameter.requires_grad_(
            update_hidden
        )

    for parameter in auxiliary_parameters:
        parameter.requires_grad_(
            update_auxiliary
        )

    print(f"Device: {device}")
    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Starting step: {start_step}")
    print(f"Branch mode: {args.mode}")
    print(f"Update hidden Muon group: {update_hidden}")
    print(
        "Update auxiliary AdamW group: "
        f"{update_auxiliary}"
    )
    print(f"Additional steps: {args.steps}")
    print(
        f"Evaluation interval: "
        f"{args.evaluation_interval}"
    )
    print(f"Output CSV: {csv_path}")

    fieldnames = [
        "step",
        "local_step",
        "branch",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "hidden_gradient_norm",
        "auxiliary_gradient_norm",
        "hidden_delta_norm",
        "auxiliary_delta_norm",
        "muon_applied_update_norm",
        "muon_max_abs_applied_update",
        "collapse_detected",
    ]

    instrumented_groups = (
        "hidden",
        "embeddings",
        "readout",
    )

    instrumented_fields = (
        "applied_update_norm",
        "gradient_component_norm",
        "decay_component_norm",
        "applied_update_rms",
        "parameter_norm",
    )

    fieldnames.extend(
        f"{group}_{field}"
        for group in instrumented_groups
        for field in instrumented_fields
    )

    last_hidden_gradient_norm = float("nan")
    last_auxiliary_gradient_norm = float("nan")
    last_muon_applied_update_norm = float("nan")
    last_muon_max_abs_applied_update = float("nan")

    last_update_statistics = {
        group: dict(ZERO_UPDATE_STATISTICS)
        for group in instrumented_groups
    }

    has_memorized = True

    with csv_path.open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for local_step in range(
            args.steps + 1
        ):
            global_step = (
                start_step + local_step
            )

            if (
                local_step
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

                hidden_delta_norm = (
                    parameter_delta_l2_norm(
                        hidden_parameters,
                        hidden_start,
                    )
                )

                auxiliary_delta_norm = (
                    parameter_delta_l2_norm(
                        auxiliary_parameters,
                        auxiliary_start,
                    )
                )

                row = {
                    "step": global_step,
                    "local_step": local_step,
                    "branch": args.mode,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "hidden_gradient_norm": (
                        last_hidden_gradient_norm
                    ),
                    "auxiliary_gradient_norm": (
                        last_auxiliary_gradient_norm
                    ),
                    "hidden_delta_norm": (
                        hidden_delta_norm
                    ),
                    "auxiliary_delta_norm": (
                        auxiliary_delta_norm
                    ),
                    "muon_applied_update_norm": (
                        last_muon_applied_update_norm
                    ),
                    **{
                        f"{group}_{field}": (
                            last_update_statistics
                            [group][field]
                        )
                        for group in instrumented_groups
                        for field in instrumented_fields
                    },
                    "muon_max_abs_applied_update": (
                        last_muon_max_abs_applied_update
                    ),
                    "collapse_detected": int(
                        collapse_detected
                    ),
                }

                writer.writerow(row)
                csv_file.flush()

                print(
                    f"step={global_step:6d} | "
                    f"train_acc={train_accuracy:.4f} | "
                    f"test_acc={test_accuracy:.4f} | "
                    f"hidden_delta={hidden_delta_norm:.4f} | "
                    f"aux_delta={auxiliary_delta_norm:.4f}"
                )

                if collapse_detected:
                    print(
                        "COLLAPSE DETECTED"
                    )

            if (
                local_step
                % args.checkpoint_interval
                == 0
            ):
                branch_checkpoint_path = (
                    checkpoint_directory
                    / f"step_{global_step:06d}.pt"
                )

                torch.save(
                    {
                        "step": global_step,
                        "local_step": local_step,
                        "branch_mode": args.mode,
                        "source_checkpoint": str(
                            args.checkpoint
                        ),
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
                        },
                        "seed": seed,
                    },
                    branch_checkpoint_path,
                )

            if local_step == args.steps:
                break

            model.train()

            muon_optimizer.zero_grad(
                set_to_none=True
            )

            auxiliary_optimizer.zero_grad(
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
                    f"{global_step}."
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

            hidden_before_step = clone_parameter_values(
                hidden_parameters
            )
            embeddings_before_step = (
                clone_parameter_values(
                    embedding_parameters
                )
            )
            readout_before_step = clone_parameter_values(
                readout_parameters
            )

            if update_hidden:
                muon_optimizer.step()

                last_muon_applied_update_norm = (
                    muon_optimizer
                    .last_step_stats[
                        "applied_update_norm"
                    ]
                )

                last_muon_max_abs_applied_update = (
                    muon_optimizer
                    .last_step_stats[
                        "max_abs_applied_update"
                    ]
                )
            else:
                last_muon_applied_update_norm = 0.0
                last_muon_max_abs_applied_update = 0.0

            if update_auxiliary:
                auxiliary_optimizer.step()

            last_update_statistics["hidden"] = (
                applied_update_statistics(
                    hidden_parameters,
                    hidden_before_step,
                    learning_rate=muon_learning_rate,
                    weight_decay=muon_weight_decay,
                )
                if update_hidden
                else dict(ZERO_UPDATE_STATISTICS)
            )

            last_update_statistics["embeddings"] = (
                applied_update_statistics(
                    embedding_parameters,
                    embeddings_before_step,
                    learning_rate=(
                        auxiliary_learning_rate
                    ),
                    weight_decay=(
                        auxiliary_weight_decay
                    ),
                )
                if update_auxiliary
                else dict(ZERO_UPDATE_STATISTICS)
            )

            last_update_statistics["readout"] = (
                applied_update_statistics(
                    readout_parameters,
                    readout_before_step,
                    learning_rate=(
                        auxiliary_learning_rate
                    ),
                    weight_decay=(
                        auxiliary_weight_decay
                    ),
                )
                if update_auxiliary
                else dict(ZERO_UPDATE_STATISTICS)
            )

    print()
    print(f"Saved branch metrics to: {csv_path}")
    print(
        "Saved branch checkpoints to: "
        f"{checkpoint_directory}"
    )


if __name__ == "__main__":
    main()