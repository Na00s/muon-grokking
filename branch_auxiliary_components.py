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

BRANCH_MODES = [
    "control",
    "freeze_token_embedding",
    "freeze_position_embedding",
    "freeze_unembedding",
    "freeze_other_auxiliary",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Branch from a pre-collapse Muon checkpoint and freeze one "
            "auxiliary component at a time."
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
        choices=BRANCH_MODES,
        required=True,
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


def get_attribute_parameters(
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
        f"Could not find the {component_name} component. "
        f"Tried attributes {candidates}. "
        f"Available parameter names: {available_names}"
    )


def split_parameter_groups(
    model: ModularAdditionTransformer,
) -> tuple[
    list[Parameter],
    list[Parameter],
    dict[str, list[Parameter]],
    dict[int, str],
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
    ) = get_attribute_parameters(
        model=model,
        candidates=[
            "token_embedding",
            "token_embeddings",
        ],
        component_name="token embedding",
    )

    (
        position_attribute,
        position_parameters,
    ) = get_attribute_parameters(
        model=model,
        candidates=[
            "position_embedding",
            "position_embeddings",
            "positional_embedding",
            "positional_embeddings",
            "pos_embedding",
        ],
        component_name="position embedding",
    )

    (
        unembedding_attribute,
        unembedding_parameters,
    ) = get_attribute_parameters(
        model=model,
        candidates=[
            "unembedding",
            "unembed",
        ],
        component_name="unembedding",
    )

    parameter_names = {
        id(parameter): name
        for name, parameter in model.named_parameters()
    }

    hidden_ids = {
        id(parameter)
        for parameter in hidden_parameters
    }

    auxiliary_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in hidden_ids
    ]

    token_ids = {
        id(parameter)
        for parameter in token_parameters
    }

    position_ids = {
        id(parameter)
        for parameter in position_parameters
    }

    unembedding_ids = {
        id(parameter)
        for parameter in unembedding_parameters
    }

    named_component_ids = (
        token_ids
        | position_ids
        | unembedding_ids
    )

    auxiliary_ids = {
        id(parameter)
        for parameter in auxiliary_parameters
    }

    if not named_component_ids <= auxiliary_ids:
        raise RuntimeError(
            "A named auxiliary component overlaps the hidden Muon group."
        )

    if (
        token_ids & position_ids
        or token_ids & unembedding_ids
        or position_ids & unembedding_ids
    ):
        raise RuntimeError(
            "Auxiliary component groups overlap."
        )

    other_auxiliary_parameters = [
        parameter
        for parameter in auxiliary_parameters
        if id(parameter) not in named_component_ids
    ]

    component_groups = {
        "token_embedding": token_parameters,
        "position_embedding": position_parameters,
        "unembedding": unembedding_parameters,
        "other_auxiliary": other_auxiliary_parameters,
    }

    all_ids = {
        id(parameter)
        for parameter in model.parameters()
    }

    covered_ids = (
        hidden_ids
        | auxiliary_ids
    )

    if covered_ids != all_ids:
        raise RuntimeError(
            "The hidden and auxiliary groups do not cover all parameters."
        )

    print(
        f"Token embedding attribute: {token_attribute}"
    )

    print(
        f"Position embedding attribute: {position_attribute}"
    )

    print(
        f"Unembedding attribute: {unembedding_attribute}"
    )

    for group_name, parameters in component_groups.items():
        names = [
            parameter_names[id(parameter)]
            for parameter in parameters
        ]

        print(
            f"{group_name} parameters: {names}"
        )

    return (
        hidden_parameters,
        auxiliary_parameters,
        component_groups,
        parameter_names,
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


def clone_parameter_values(
    parameters: list[Parameter],
) -> list[Tensor]:
    return [
        parameter.detach().float().clone()
        for parameter in parameters
    ]


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


def build_optimizers(
    hidden_parameters: list[Parameter],
    auxiliary_parameters: list[Parameter],
    checkpoint: dict,
) -> tuple[Muon, torch.optim.AdamW]:
    saved_arguments = checkpoint.get(
        "arguments",
        {},
    )

    muon_learning_rate = saved_arguments.get(
        "muon_lr",
        0.01,
    )

    resolved_muon_weight_decay = checkpoint.get(
        "resolved_muon_weight_decay",
        saved_arguments.get(
            "muon_weight_decay",
            None,
        ),
    )

    if resolved_muon_weight_decay is None:
        resolved_muon_weight_decay = (
            0.001 / muon_learning_rate
        )

    muon_optimizer = Muon(
        hidden_parameters,
        learning_rate=muon_learning_rate,
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


def mode_to_frozen_group(
    mode: str,
) -> str | None:
    mapping = {
        "control": None,
        "freeze_token_embedding": "token_embedding",
        "freeze_position_embedding": "position_embedding",
        "freeze_unembedding": "unembedding",
        "freeze_other_auxiliary": "other_auxiliary",
    }

    return mapping[mode]


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
            f"aux_component_{args.mode}"
            f"_from_{start_step}"
        )
    )

    (
        csv_path,
        checkpoint_directory,
    ) = prepare_outputs(
        run_name=run_name,
        overwrite=args.overwrite,
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
        component_groups,
        _,
    ) = split_parameter_groups(
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

    frozen_group_name = mode_to_frozen_group(
        args.mode
    )

    if (
        frozen_group_name == "other_auxiliary"
        and not component_groups["other_auxiliary"]
    ):
        raise ValueError(
            "There are no other auxiliary parameters to freeze. "
            "You can skip the freeze_other_auxiliary branch."
        )

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    if frozen_group_name is not None:
        for parameter in component_groups[
            frozen_group_name
        ]:
            parameter.requires_grad_(False)

    hidden_start = clone_parameter_values(
        hidden_parameters
    )

    group_starts = {
        name: clone_parameter_values(parameters)
        for name, parameters in component_groups.items()
    }

    print(f"Device: {device}")
    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Starting step: {start_step}")
    print(f"Branch mode: {args.mode}")
    print(f"Frozen group: {frozen_group_name}")
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
        "frozen_group",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "hidden_gradient_norm",
        "auxiliary_gradient_norm",
        "token_embedding_gradient_norm",
        "position_embedding_gradient_norm",
        "unembedding_gradient_norm",
        "other_auxiliary_gradient_norm",
        "hidden_delta_norm",
        "token_embedding_delta_norm",
        "position_embedding_delta_norm",
        "unembedding_delta_norm",
        "other_auxiliary_delta_norm",
        "muon_applied_update_norm",
        "muon_max_abs_applied_update",
        "collapse_detected",
    ]

    last_diagnostics = {
        "hidden_gradient_norm": float("nan"),
        "auxiliary_gradient_norm": float("nan"),
        "token_embedding_gradient_norm": float("nan"),
        "position_embedding_gradient_norm": float("nan"),
        "unembedding_gradient_norm": float("nan"),
        "other_auxiliary_gradient_norm": float("nan"),
        "muon_applied_update_norm": float("nan"),
        "muon_max_abs_applied_update": float("nan"),
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

                component_deltas = {
                    f"{name}_delta_norm": (
                        parameter_delta_l2_norm(
                            component_groups[name],
                            group_starts[name],
                        )
                    )
                    for name in component_groups
                }

                row = {
                    "step": global_step,
                    "local_step": local_step,
                    "branch": args.mode,
                    "frozen_group": (
                        frozen_group_name
                        if frozen_group_name is not None
                        else "none"
                    ),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    **last_diagnostics,
                    "hidden_delta_norm": (
                        parameter_delta_l2_norm(
                            hidden_parameters,
                            hidden_start,
                        )
                    ),
                    **component_deltas,
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
                    f"token_delta="
                    f"{component_deltas['token_embedding_delta_norm']:.4f} | "
                    f"position_delta="
                    f"{component_deltas['position_embedding_delta_norm']:.4f} | "
                    f"unembedding_delta="
                    f"{component_deltas['unembedding_delta_norm']:.4f}"
                )

                if collapse_detected:
                    print("COLLAPSE DETECTED")

            if (
                local_step
                % args.checkpoint_interval
                == 0
            ):
                output_checkpoint_path = (
                    checkpoint_directory
                    / f"step_{global_step:06d}.pt"
                )

                torch.save(
                    {
                        "step": global_step,
                        "local_step": local_step,
                        "branch_mode": args.mode,
                        "frozen_group": frozen_group_name,
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
                    output_checkpoint_path,
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

            last_diagnostics[
                "hidden_gradient_norm"
            ] = gradient_l2_norm(
                hidden_parameters
            )

            last_diagnostics[
                "auxiliary_gradient_norm"
            ] = gradient_l2_norm(
                auxiliary_parameters
            )

            for group_name, parameters in component_groups.items():
                last_diagnostics[
                    f"{group_name}_gradient_norm"
                ] = gradient_l2_norm(
                    parameters
                )

            muon_optimizer.step()
            auxiliary_optimizer.step()

            muon_stats = (
                muon_optimizer.last_step_stats
            )

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

    print()
    print(f"Saved branch metrics to: {csv_path}")
    print(
        "Saved branch checkpoints to: "
        f"{checkpoint_directory}"
    )


if __name__ == "__main__":
    main()
