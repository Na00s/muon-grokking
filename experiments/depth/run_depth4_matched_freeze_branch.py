from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
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


BRANCH_NAMES = ("control", "freeze")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one exact pre-freeze depth-four checkpoint and "
            "fork it into matched control and auxiliary-freeze branches."
        )
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path(
            "runs/depth_sweep_v3_depth_4_"
            "stable_muon_seed_0_to_300k.csv"
        ),
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/depth_sweep_v3_depth_4_"
            "stable_muon_seed_0_to_300k/"
            "step_100000.pt"
        ),
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=300_000,
        help=(
            "Final global training step for both matched branches."
        ),
    )
    parser.add_argument(
        "--maximum-branch-search-step",
        type=int,
        default=150_000,
        help=(
            "Abort if the replay has not scheduled its freeze by "
            "this global step."
        ),
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=None,
        help=(
            "Defaults to the interval stored in the source checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        "No CUDA or MPS device is available. "
        "This experiment intentionally has no CPU fallback."
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing source checkpoint: {path}"
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
            "Source checkpoint must contain a dictionary."
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
            "Source checkpoint is missing fields: "
            + ", ".join(sorted(missing))
        )

    return checkpoint


def read_source_records(
    path: Path,
    source_step: int,
) -> list[dict[str, str]]:
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
        records = [
            row
            for row in reader
            if int(float(row["step"])) <= source_step
        ]

    if not records:
        raise ValueError(
            "The source CSV contains no records through "
            f"step {source_step}."
        )

    final_csv_step = int(
        float(records[-1]["step"])
    )
    if final_csv_step != source_step:
        raise ValueError(
            "The source CSV does not contain an evaluation at "
            f"the checkpoint step {source_step}."
        )

    return records


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


def optimizer_arguments(
    checkpoint: dict,
) -> dict:
    arguments = checkpoint["arguments"]
    if not isinstance(arguments, dict):
        raise TypeError(
            "Checkpoint arguments must be a dictionary."
        )
    return dict(arguments)


def instantiate_optimizers(
    arguments: dict,
    model: DepthModularAdditionTransformer,
    hidden_parameters: list[Parameter],
    auxiliary_parameters: list[Parameter],
    unembedding_parameters: list[Parameter],
) -> dict[str, object]:
    beta1 = float(
        arguments.get("adamw_beta1", 0.9)
    )
    beta2 = float(
        arguments.get("adamw_beta2", 0.999)
    )

    return {
        "muon": Muon(
            hidden_parameters,
            learning_rate=float(
                arguments["muon_lr"]
            ),
            momentum=float(
                arguments["muon_momentum"]
            ),
            weight_decay=float(
                arguments["muon_weight_decay"]
            ),
            newton_schulz_steps=int(
                arguments["muon_ns_steps"]
            ),
            nesterov=True,
        ),
        "auxiliary_adamw": torch.optim.AdamW(
            auxiliary_parameters,
            lr=float(arguments["aux_lr"]),
            weight_decay=float(
                arguments["aux_weight_decay"]
            ),
            betas=(beta1, beta2),
        ),
        "unembedding_adamw": torch.optim.AdamW(
            unembedding_parameters,
            lr=float(
                arguments["unembedding_lr"]
            ),
            weight_decay=float(
                arguments[
                    "unembedding_weight_decay"
                ]
            ),
            betas=(beta1, beta2),
        ),
    }


def load_optimizer_states(
    optimizers: dict[str, object],
    checkpoint: dict,
) -> None:
    saved = checkpoint["optimizer_state_dicts"]

    if set(optimizers) != set(saved):
        raise KeyError(
            "The source checkpoint optimizer groups do not "
            "match the reconstructed Muon optimizer groups."
        )

    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(saved[name])


def trailing_threshold_count(
    records: list[dict[str, str]],
    threshold: float,
) -> int:
    count = 0
    for record in reversed(records):
        if float(record["test_accuracy"]) >= threshold:
            count += 1
        else:
            break
    return count


def train_one_update(
    model: DepthModularAdditionTransformer,
    optimizers: dict[str, object],
    hidden_parameters: list[Parameter],
    auxiliary_parameters: list[Parameter],
    unembedding_parameters: list[Parameter],
    train_inputs: Tensor,
    train_targets: Tensor,
    freeze_auxiliary: bool,
) -> dict[str, float]:
    model.train()

    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)

    logits = model(train_inputs)
    loss = F.cross_entropy(logits, train_targets)

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Encountered a non-finite training loss."
        )

    loss.backward()

    hidden_gradient_norm = gradient_l2_norm(
        hidden_parameters
    )
    auxiliary_gradient_norm = gradient_l2_norm(
        auxiliary_parameters
    )
    unembedding_gradient_norm = gradient_l2_norm(
        unembedding_parameters
    )

    muon_optimizer = optimizers["muon"]
    muon_optimizer.step()

    if not freeze_auxiliary:
        optimizers["auxiliary_adamw"].step()
        optimizers["unembedding_adamw"].step()

    statistics = muon_optimizer.last_step_stats

    return {
        "training_loss": float(loss.item()),
        "hidden_gradient_norm": (
            hidden_gradient_norm
        ),
        "auxiliary_gradient_norm": (
            auxiliary_gradient_norm
        ),
        "unembedding_gradient_norm": (
            unembedding_gradient_norm
        ),
        "muon_applied_update_norm": float(
            statistics["applied_update_norm"]
        ),
        "muon_max_abs_applied_update": float(
            statistics[
                "max_abs_applied_update"
            ]
        ),
    }


def save_training_state(
    path: Path,
    step: int,
    label: str,
    model: DepthModularAdditionTransformer,
    optimizers: dict[str, object],
    checkpoint: dict,
    branch_step: int,
    freeze_auxiliary: bool,
) -> None:
    torch.save(
        {
            "step": step,
            "label": label,
            "regime": "muon",
            "branch_step": branch_step,
            "freeze_auxiliary": freeze_auxiliary,
            "arguments": checkpoint["arguments"],
            "model_config": checkpoint["model_config"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dicts": {
                name: optimizer.state_dict()
                for name, optimizer
                in optimizers.items()
            },
            "seed": checkpoint.get("seed", 0),
            "source_checkpoint_step": int(
                checkpoint["step"]
            ),
            "matched_checkpoint_intervention": True,
        },
        path,
    )


def max_parameter_difference(
    first: DepthModularAdditionTransformer,
    second: DepthModularAdditionTransformer,
) -> float:
    maximum = 0.0

    for (
        first_name,
        first_parameter,
    ), (
        second_name,
        second_parameter,
    ) in zip(
        first.named_parameters(),
        second.named_parameters(),
        strict=True,
    ):
        if first_name != second_name:
            raise RuntimeError(
                "Branch parameter names do not align."
            )

        difference = (
            first_parameter.detach()
            - second_parameter.detach()
        ).abs().max().item()
        maximum = max(maximum, float(difference))

    return maximum


def independent_parameter_storage(
    first: DepthModularAdditionTransformer,
    second: DepthModularAdditionTransformer,
) -> bool:
    for first_parameter, second_parameter in zip(
        first.parameters(),
        second.parameters(),
        strict=True,
    ):
        if (
            first_parameter.untyped_storage().data_ptr()
            == second_parameter.untyped_storage().data_ptr()
        ):
            return False
    return True


def prepare_output_path(
    path: Path,
    overwrite: bool,
) -> None:
    if overwrite and path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}"
        )


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    device = get_device(args.device)
    source_checkpoint = load_checkpoint(
        args.source_checkpoint
    )

    if source_checkpoint["regime"] != "stable_muon":
        raise ValueError(
            "The source checkpoint must come from the "
            "depth-four stable_muon trajectory."
        )

    source_step = int(source_checkpoint["step"])
    if source_step != 100_000:
        raise ValueError(
            "This intervention expects the exact step-100000 "
            f"checkpoint, found step {source_step}."
        )

    configuration = dict(
        source_checkpoint["model_config"]
    )
    if int(configuration.get("num_layers", -1)) != 4:
        raise ValueError(
            "The source model is not depth 4."
        )

    arguments = optimizer_arguments(
        source_checkpoint
    )
    evaluation_interval = (
        int(args.evaluation_interval)
        if args.evaluation_interval is not None
        else int(
            arguments.get(
                "evaluation_interval",
                100,
            )
        )
    )
    trigger_accuracy = float(
        arguments.get(
            "freeze_trigger_test_accuracy",
            0.95,
        )
    )
    trigger_consecutive = int(
        arguments.get(
            "freeze_trigger_consecutive",
            5,
        )
    )
    freeze_delay = int(
        arguments.get(
            "freeze_delay",
            2_000,
        )
    )

    if evaluation_interval < 1:
        raise ValueError(
            "Evaluation interval must be positive."
        )
    if args.total_steps <= source_step:
        raise ValueError(
            "--total-steps must exceed 100000."
        )
    if (
        args.maximum_branch_search_step
        >= args.total_steps
    ):
        raise ValueError(
            "--maximum-branch-search-step must be "
            "smaller than --total-steps."
        )

    source_records = read_source_records(
        args.source_csv,
        source_step,
    )

    (
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    ) = generate_modular_addition_data(
        modulus=int(arguments["modulus"]),
        train_fraction=float(
            arguments["train_fraction"]
        ),
        seed=int(
            source_checkpoint.get("seed", 0)
        ),
    )

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    replay_model = (
        DepthModularAdditionTransformer(
            **configuration
        )
    )
    replay_model.load_state_dict(
        source_checkpoint["model_state_dict"]
    )
    replay_model = replay_model.to(device)

    (
        replay_hidden,
        replay_auxiliary,
        replay_unembedding,
    ) = split_parameter_groups(replay_model)

    (
        parameter_count,
        hidden_matrix_count,
    ) = verify_architecture(
        model=replay_model,
        hidden_parameters=replay_hidden,
        expected_depth=4,
    )

    replay_optimizers = instantiate_optimizers(
        arguments=arguments,
        model=replay_model,
        hidden_parameters=replay_hidden,
        auxiliary_parameters=replay_auxiliary,
        unembedding_parameters=(
            replay_unembedding
        ),
    )
    load_optimizer_states(
        replay_optimizers,
        source_checkpoint,
    )

    consecutive = trailing_threshold_count(
        source_records,
        trigger_accuracy,
    )
    scheduled_branch_step: int | None = None
    current_step = source_step

    print(f"Device: {device}")
    print(
        "Replaying one shared trajectory from "
        f"step {source_step}."
    )
    print(
        "Verified architecture: "
        f"blocks=4, "
        f"hidden_matrices={hidden_matrix_count}, "
        f"parameters={parameter_count}"
    )
    print(
        "Freeze rule: "
        f"{trigger_consecutive} consecutive "
        f"evaluations >= {trigger_accuracy:.2f}, "
        f"then {freeze_delay} additional updates."
    )

    replay_started_at = time.perf_counter()

    while (
        scheduled_branch_step is None
        or current_step < scheduled_branch_step
    ):
        if (
            current_step
            >= args.maximum_branch_search_step
        ):
            raise RuntimeError(
                "The replay did not schedule a freeze by "
                f"step {args.maximum_branch_search_step}."
            )

        train_one_update(
            model=replay_model,
            optimizers=replay_optimizers,
            hidden_parameters=replay_hidden,
            auxiliary_parameters=(
                replay_auxiliary
            ),
            unembedding_parameters=(
                replay_unembedding
            ),
            train_inputs=train_inputs,
            train_targets=train_targets,
            freeze_auxiliary=False,
        )
        current_step += 1

        if (
            current_step % evaluation_interval
            == 0
        ):
            _, test_accuracy = evaluate(
                replay_model,
                test_inputs,
                test_targets,
            )

            if test_accuracy >= trigger_accuracy:
                consecutive += 1
            else:
                consecutive = 0

            print(
                f"replay step={current_step:6d} | "
                f"test={test_accuracy:.4f} | "
                f"trigger_count={consecutive}"
            )

            if (
                scheduled_branch_step is None
                and consecutive
                >= trigger_consecutive
            ):
                first_trigger_step = (
                    current_step
                    - (
                        trigger_consecutive - 1
                    )
                    * evaluation_interval
                )
                scheduled_branch_step = (
                    first_trigger_step
                    + freeze_delay
                )
                print(
                    "Matched branch scheduled for "
                    f"step {scheduled_branch_step}."
                )

    branch_step = current_step
    if branch_step != scheduled_branch_step:
        raise RuntimeError(
            "Replay stopped at the wrong branch step."
        )

    synchronize(device)
    replay_seconds = (
        time.perf_counter()
        - replay_started_at
    )

    branch_train_loss, branch_train_accuracy = (
        evaluate(
            replay_model,
            train_inputs,
            train_targets,
        )
    )
    branch_test_loss, branch_test_accuracy = (
        evaluate(
            replay_model,
            test_inputs,
            test_targets,
        )
    )

    output_prefix = (
        f"depth4_matched_freeze_seed_"
        f"{args.seed}_from_{branch_step}"
    )
    metadata_path = (
        Path("runs")
        / f"{output_prefix}_metadata.json"
    )
    control_csv = (
        Path("runs")
        / f"{output_prefix}_control.csv"
    )
    freeze_csv = (
        Path("runs")
        / f"{output_prefix}_freeze.csv"
    )
    branch_checkpoint_directory = (
        Path("checkpoints")
        / f"{output_prefix}_shared"
    )
    control_checkpoint_directory = (
        Path("checkpoints")
        / f"{output_prefix}_control"
    )
    freeze_checkpoint_directory = (
        Path("checkpoints")
        / f"{output_prefix}_freeze"
    )

    for path in (
        metadata_path,
        control_csv,
        freeze_csv,
        branch_checkpoint_directory,
        control_checkpoint_directory,
        freeze_checkpoint_directory,
    ):
        prepare_output_path(
            path,
            args.overwrite,
        )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    branch_checkpoint_directory.mkdir(
        parents=True,
        exist_ok=False,
    )
    control_checkpoint_directory.mkdir(
        parents=True,
        exist_ok=False,
    )
    freeze_checkpoint_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    shared_checkpoint_path = (
        branch_checkpoint_directory
        / f"step_{branch_step:06d}.pt"
    )
    save_training_state(
        path=shared_checkpoint_path,
        step=branch_step,
        label="shared_pre_intervention",
        model=replay_model,
        optimizers=replay_optimizers,
        checkpoint=source_checkpoint,
        branch_step=branch_step,
        freeze_auxiliary=False,
    )

    shared_model_state = copy.deepcopy(
        replay_model.state_dict()
    )
    shared_optimizer_states = {
        name: copy.deepcopy(
            optimizer.state_dict()
        )
        for name, optimizer
        in replay_optimizers.items()
    }

    branches: dict[str, dict[str, object]] = {}

    for branch_name in BRANCH_NAMES:
        model = DepthModularAdditionTransformer(
            **configuration
        )
        model.load_state_dict(
            shared_model_state
        )
        model = model.to(device)

        (
            hidden_parameters,
            auxiliary_parameters,
            unembedding_parameters,
        ) = split_parameter_groups(model)

        optimizers = instantiate_optimizers(
            arguments=arguments,
            model=model,
            hidden_parameters=(
                hidden_parameters
            ),
            auxiliary_parameters=(
                auxiliary_parameters
            ),
            unembedding_parameters=(
                unembedding_parameters
            ),
        )

        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(
                copy.deepcopy(
                    shared_optimizer_states[name]
                )
            )

        freeze_auxiliary = (
            branch_name == "freeze"
        )
        if freeze_auxiliary:
            for parameter in (
                auxiliary_parameters
                + unembedding_parameters
            ):
                parameter.requires_grad_(False)
                parameter.grad = None

        branches[branch_name] = {
            "model": model,
            "hidden_parameters": (
                hidden_parameters
            ),
            "auxiliary_parameters": (
                auxiliary_parameters
            ),
            "unembedding_parameters": (
                unembedding_parameters
            ),
            "optimizers": optimizers,
            "freeze_auxiliary": (
                freeze_auxiliary
            ),
            "last_statistics": {
                "hidden_gradient_norm": (
                    float("nan")
                ),
                "auxiliary_gradient_norm": (
                    float("nan")
                ),
                "unembedding_gradient_norm": (
                    float("nan")
                ),
                "muon_applied_update_norm": (
                    float("nan")
                ),
                "muon_max_abs_applied_update": (
                    float("nan")
                ),
            },
        }

    branch_parameter_difference = (
        max_parameter_difference(
            branches["control"]["model"],
            branches["freeze"]["model"],
        )
    )
    if branch_parameter_difference != 0.0:
        raise RuntimeError(
            "Matched branches are not parameter-identical "
            "at the branch point."
        )
    if not independent_parameter_storage(
        branches["control"]["model"],
        branches["freeze"]["model"],
    ):
        raise RuntimeError(
            "Matched branches unexpectedly share "
            "parameter storage."
        )

    fieldnames = [
        "step",
        "branch",
        "branch_step",
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
        "auxiliary_frozen",
        "generalization_collapse_detected",
        "elapsed_post_branch_seconds",
    ]

    csv_paths = {
        "control": control_csv,
        "freeze": freeze_csv,
    }
    handles = {}
    writers = {}

    try:
        for branch_name in BRANCH_NAMES:
            handle = csv_paths[
                branch_name
            ].open(
                "w",
                newline="",
                encoding="utf-8",
            )
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            handles[branch_name] = handle
            writers[branch_name] = writer

            writer.writerow(
                {
                    "step": branch_step,
                    "branch": branch_name,
                    "branch_step": branch_step,
                    "train_loss": (
                        branch_train_loss
                    ),
                    "train_accuracy": (
                        branch_train_accuracy
                    ),
                    "test_loss": (
                        branch_test_loss
                    ),
                    "test_accuracy": (
                        branch_test_accuracy
                    ),
                    "hidden_parameter_norm": (
                        parameter_l2_norm(
                            branches[
                                branch_name
                            ][
                                "hidden_parameters"
                            ]
                        )
                    ),
                    "auxiliary_parameter_norm": (
                        parameter_l2_norm(
                            branches[
                                branch_name
                            ][
                                "auxiliary_parameters"
                            ]
                        )
                    ),
                    "unembedding_parameter_norm": (
                        parameter_l2_norm(
                            branches[
                                branch_name
                            ][
                                "unembedding_parameters"
                            ]
                        )
                    ),
                    "hidden_gradient_norm": (
                        float("nan")
                    ),
                    "auxiliary_gradient_norm": (
                        float("nan")
                    ),
                    "unembedding_gradient_norm": (
                        float("nan")
                    ),
                    "muon_applied_update_norm": (
                        float("nan")
                    ),
                    "muon_max_abs_applied_update": (
                        float("nan")
                    ),
                    "auxiliary_frozen": int(
                        branches[
                            branch_name
                        ][
                            "freeze_auxiliary"
                        ]
                    ),
                    "generalization_collapse_detected": (
                        int(
                            branch_test_accuracy
                            < 0.90
                        )
                    ),
                    "elapsed_post_branch_seconds": (
                        0.0
                    ),
                }
            )
            handle.flush()

        print()
        print(
            "Created exact matched checkpoint:"
        )
        print(
            f"  branch_step={branch_step}"
        )
        print(
            "  maximum parameter difference "
            "between branches="
            f"{branch_parameter_difference:.1f}"
        )
        print(
            "  control: continue Muon plus both "
            "AdamW-managed groups"
        )
        print(
            "  freeze: continue Muon, freeze "
            "auxiliary plus unembedding"
        )
        print(
            f"Running both branches through "
            f"step {args.total_steps}."
        )

        branch_started_at = time.perf_counter()

        for completed_step in range(
            branch_step,
            args.total_steps,
        ):
            for branch_name in BRANCH_NAMES:
                branch = branches[
                    branch_name
                ]
                statistics = train_one_update(
                    model=branch["model"],
                    optimizers=(
                        branch["optimizers"]
                    ),
                    hidden_parameters=(
                        branch[
                            "hidden_parameters"
                        ]
                    ),
                    auxiliary_parameters=(
                        branch[
                            "auxiliary_parameters"
                        ]
                    ),
                    unembedding_parameters=(
                        branch[
                            "unembedding_parameters"
                        ]
                    ),
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    freeze_auxiliary=bool(
                        branch[
                            "freeze_auxiliary"
                        ]
                    ),
                )
                branch[
                    "last_statistics"
                ] = statistics
                synchronize(device)

            step = completed_step + 1

            if (
                step % evaluation_interval
                == 0
            ):
                elapsed = (
                    time.perf_counter()
                    - branch_started_at
                )

                printed = []
                for branch_name in BRANCH_NAMES:
                    branch = branches[
                        branch_name
                    ]
                    (
                        train_loss,
                        train_accuracy,
                    ) = evaluate(
                        branch["model"],
                        train_inputs,
                        train_targets,
                    )
                    (
                        test_loss,
                        test_accuracy,
                    ) = evaluate(
                        branch["model"],
                        test_inputs,
                        test_targets,
                    )

                    statistics = branch[
                        "last_statistics"
                    ]

                    writers[
                        branch_name
                    ].writerow(
                        {
                            "step": step,
                            "branch": (
                                branch_name
                            ),
                            "branch_step": (
                                branch_step
                            ),
                            "train_loss": (
                                train_loss
                            ),
                            "train_accuracy": (
                                train_accuracy
                            ),
                            "test_loss": (
                                test_loss
                            ),
                            "test_accuracy": (
                                test_accuracy
                            ),
                            "hidden_parameter_norm": (
                                parameter_l2_norm(
                                    branch[
                                        "hidden_parameters"
                                    ]
                                )
                            ),
                            "auxiliary_parameter_norm": (
                                parameter_l2_norm(
                                    branch[
                                        "auxiliary_parameters"
                                    ]
                                )
                            ),
                            "unembedding_parameter_norm": (
                                parameter_l2_norm(
                                    branch[
                                        "unembedding_parameters"
                                    ]
                                )
                            ),
                            "hidden_gradient_norm": (
                                statistics[
                                    "hidden_gradient_norm"
                                ]
                            ),
                            "auxiliary_gradient_norm": (
                                statistics[
                                    "auxiliary_gradient_norm"
                                ]
                            ),
                            "unembedding_gradient_norm": (
                                statistics[
                                    "unembedding_gradient_norm"
                                ]
                            ),
                            "muon_applied_update_norm": (
                                statistics[
                                    "muon_applied_update_norm"
                                ]
                            ),
                            "muon_max_abs_applied_update": (
                                statistics[
                                    "muon_max_abs_applied_update"
                                ]
                            ),
                            "auxiliary_frozen": int(
                                branch[
                                    "freeze_auxiliary"
                                ]
                            ),
                            "generalization_collapse_detected": int(
                                test_accuracy < 0.90
                            ),
                            "elapsed_post_branch_seconds": (
                                elapsed
                            ),
                        }
                    )
                    handles[
                        branch_name
                    ].flush()
                    printed.append(
                        f"{branch_name}="
                        f"{test_accuracy:.4f}"
                    )

                print(
                    f"step={step:6d} | "
                    + " | ".join(printed)
                )

            if (
                step % args.checkpoint_interval
                == 0
                or step == args.total_steps
            ):
                for branch_name in BRANCH_NAMES:
                    branch = branches[
                        branch_name
                    ]
                    directory = (
                        control_checkpoint_directory
                        if branch_name == "control"
                        else freeze_checkpoint_directory
                    )
                    save_training_state(
                        path=(
                            directory
                            / f"step_{step:06d}.pt"
                        ),
                        step=step,
                        label=branch_name,
                        model=branch["model"],
                        optimizers=(
                            branch["optimizers"]
                        ),
                        checkpoint=(
                            source_checkpoint
                        ),
                        branch_step=(
                            branch_step
                        ),
                        freeze_auxiliary=bool(
                            branch[
                                "freeze_auxiliary"
                            ]
                        ),
                    )
    finally:
        for handle in handles.values():
            handle.close()

    metadata = {
        "source_csv": str(args.source_csv),
        "source_checkpoint": str(
            args.source_checkpoint
        ),
        "source_step": source_step,
        "branch_step": branch_step,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "device": str(device),
        "trigger_accuracy": trigger_accuracy,
        "trigger_consecutive": (
            trigger_consecutive
        ),
        "freeze_delay": freeze_delay,
        "branch_train_accuracy": (
            branch_train_accuracy
        ),
        "branch_test_accuracy": (
            branch_test_accuracy
        ),
        "branch_parameter_max_abs_difference": (
            branch_parameter_difference
        ),
        "independent_parameter_storage": True,
        "replay_seconds": replay_seconds,
        "control_csv": str(control_csv),
        "freeze_csv": str(freeze_csv),
        "shared_checkpoint": str(
            shared_checkpoint_path
        ),
        "intervention": {
            "control": (
                "Continue Muon hidden updates and both "
                "AdamW-managed optimizer groups."
            ),
            "freeze": (
                "Continue Muon hidden updates while "
                "freezing auxiliary parameters and "
                "the unembedding."
            ),
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved control CSV: {control_csv}")
    print(f"Saved freeze CSV: {freeze_csv}")


if __name__ == "__main__":
    main()
