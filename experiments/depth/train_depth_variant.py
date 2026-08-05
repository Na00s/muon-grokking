from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data import generate_modular_addition_data
from model import TransformerBlock
from optimizers.muon import Muon


SEQUENCE_LENGTH = 3


class DepthModularAdditionTransformer(nn.Module):
    """The existing architecture generalized to an arbitrary number of blocks."""

    def __init__(
        self,
        modulus: int,
        sequence_length: int,
        d_model: int,
        num_heads: int,
        d_mlp: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.modulus = modulus
        self.sequence_length = sequence_length
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(
            modulus + 1,
            d_model,
        )
        self.position_embedding = nn.Embedding(
            sequence_length,
            d_model,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_mlp=d_mlp,
                    sequence_length=sequence_length,
                )
                for _ in range(num_layers)
            ]
        )
        self.unembedding = nn.Linear(
            d_model,
            modulus,
            bias=False,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        _, sequence_length = token_ids.shape
        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        hidden = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        for block in self.transformer_blocks:
            hidden = block(hidden)

        final_token = hidden[:, -1, :]
        return self.unembedding(final_token)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one controlled depth condition with tuned AdamW, "
            "ordinary Muon, or stabilized Muon."
        )
    )
    parser.add_argument(
        "--regime",
        choices=["adamw", "muon", "stable_muon"],
        required=True,
    )
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument(
        "--operation",
        choices=["addition", "subtraction"],
        default="addition",
    )
    parser.add_argument("--modulus", type=int, default=113)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--initial-state-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--depth-one-initial-state-path",
        type=Path,
        required=True,
        help=(
            "Selected depth-one initial state. Its embeddings, first "
            "block, and unembedding are copied into every depth so "
            "shared parameters start identically."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    # Locked depth-one selections.
    parser.add_argument("--adamw-lr", type=float, default=1e-3)
    parser.add_argument(
        "--adamw-weight-decay",
        type=float,
        default=3.0,
    )
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.999)

    parser.add_argument("--muon-lr", type=float, default=0.03)
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
    parser.add_argument("--muon-ns-steps", type=int, default=5)

    parser.add_argument("--aux-lr", type=float, default=1e-3)
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
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.num_layers < 1:
        raise ValueError("--num-layers must be at least 1.")
    if args.modulus < 3:
        raise ValueError("--modulus must be at least 3.")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must lie in (0, 1).")
    if args.d_model < 1 or args.d_mlp < 1:
        raise ValueError("Model dimensions must be positive.")
    if args.num_heads < 1:
        raise ValueError("--num-heads must be positive.")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads.")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.evaluation_interval < 1:
        raise ValueError("--evaluation-interval must be at least 1.")
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be at least 1.")
    if args.freeze_trigger_consecutive < 1:
        raise ValueError("--freeze-trigger-consecutive must be at least 1.")
    if args.freeze_delay < 0:
        raise ValueError("--freeze-delay cannot be negative.")


def get_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    raise RuntimeError("No CUDA or MPS device is available.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
) -> tuple[float, float]:
    model.eval()
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return float(loss.item()), float(accuracy.item())


def parameter_l2_norm(parameters: Iterable[Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        value = torch.linalg.vector_norm(
            parameter.detach().float()
        ).item()
        total += value * value
    return math.sqrt(total)


def gradient_l2_norm(parameters: Iterable[Parameter]) -> float:
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


def split_parameter_groups(
    model: DepthModularAdditionTransformer,
) -> tuple[list[Parameter], list[Parameter], list[Parameter]]:
    hidden_parameters: list[Parameter] = []

    for block in model.transformer_blocks:
        hidden_parameters.extend(
            [
                block.attention.qkv_projection.weight,
                block.attention.output_projection.weight,
                block.mlp.input_projection.weight,
                block.mlp.output_projection.weight,
            ]
        )

    unembedding_parameters = [model.unembedding.weight]

    hidden_ids = {id(parameter) for parameter in hidden_parameters}
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
            "Parameter groups do not exactly cover the model."
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
        "sequence_length": SEQUENCE_LENGTH,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_mlp": args.d_mlp,
        "num_layers": args.num_layers,
    }


def load_checkpoint_dictionary(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Initial-state checkpoint does not exist: {path}"
        )

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

    if not isinstance(loaded, dict):
        raise TypeError(
            "Initial-state checkpoint must contain a dictionary."
        )

    return loaded


@torch.no_grad()
def apply_nested_depth_one_initialization(
    model: DepthModularAdditionTransformer,
    depth_one_path: Path,
    configuration: dict[str, int],
) -> None:
    loaded = load_checkpoint_dictionary(depth_one_path)
    depth_one_state = loaded.get(
        "model_state_dict",
        loaded,
    )
    if not isinstance(depth_one_state, dict):
        raise TypeError(
            "Depth-one checkpoint is missing a model state dictionary."
        )

    expected_depth_one_configuration = {
        "modulus": configuration["modulus"],
        "sequence_length": configuration["sequence_length"],
        "d_model": configuration["d_model"],
        "num_heads": configuration["num_heads"],
        "d_mlp": configuration["d_mlp"],
    }
    saved_configuration = loaded.get("model_config")
    if (
        saved_configuration is not None
        and saved_configuration
        != expected_depth_one_configuration
    ):
        raise ValueError(
            "Depth-one initial-state configuration does not match "
            "the depth sweep."
        )

    target_state = model.state_dict()
    mapped_state: dict[str, Tensor] = {}

    for source_name, source_value in depth_one_state.items():
        if source_name.startswith("transformer_block."):
            target_name = source_name.replace(
                "transformer_block.",
                "transformer_blocks.0.",
                1,
            )
        else:
            target_name = source_name

        if target_name not in target_state:
            raise KeyError(
                "Cannot map depth-one parameter "
                f"{source_name} to the depth model."
            )
        if target_state[target_name].shape != source_value.shape:
            raise ValueError(
                f"Shape mismatch for shared parameter {target_name}: "
                f"{tuple(source_value.shape)} versus "
                f"{tuple(target_state[target_name].shape)}."
            )

        mapped_state[target_name] = source_value

    required_shared_names = {
        "token_embedding.weight",
        "position_embedding.weight",
        "unembedding.weight",
        "transformer_blocks.0.attention.qkv_projection.weight",
        "transformer_blocks.0.attention.output_projection.weight",
        "transformer_blocks.0.mlp.input_projection.weight",
        "transformer_blocks.0.mlp.output_projection.weight",
    }
    missing = required_shared_names - mapped_state.keys()
    if missing:
        raise KeyError(
            "Depth-one checkpoint is missing shared parameters: "
            + ", ".join(sorted(missing))
        )

    for target_name, source_value in mapped_state.items():
        target_state[target_name].copy_(source_value)

    model.load_state_dict(target_state)


def load_or_create_initial_state(
    model: DepthModularAdditionTransformer,
    path: Path,
    configuration: dict[str, int],
    seed: int,
    depth_one_initial_state_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        loaded = load_checkpoint_dictionary(path)

        saved_configuration = loaded.get("model_config")
        if saved_configuration != configuration:
            raise ValueError(
                "Initial-state model configuration does not match "
                "this depth condition."
            )

        if (
            loaded.get("initialization_scheme")
            != "nested_depth_one_shared_v2"
        ):
            raise RuntimeError(
                "The existing depth initial state was created by the "
                "older, non-nested initialization. Delete it or use "
                "the v2 initial-state path."
            )

        state = loaded.get("model_state_dict")
        if not isinstance(state, dict):
            raise KeyError(
                "Depth initial state is missing model_state_dict."
            )

        model.load_state_dict(state)
        print(f"Loaded nested shared initial state: {path}")
        return

    apply_nested_depth_one_initialization(
        model=model,
        depth_one_path=depth_one_initial_state_path,
        configuration=configuration,
    )

    torch.save(
        {
            "seed": seed,
            "model_config": configuration,
            "initialization_scheme": (
                "nested_depth_one_shared_v2"
            ),
            "depth_one_initial_state_path": str(
                depth_one_initial_state_path
            ),
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
        },
        path,
    )
    print(f"Created nested shared initial state: {path}")


def verify_architecture(
    model: DepthModularAdditionTransformer,
    hidden_parameters: list[Parameter],
    expected_depth: int,
) -> tuple[int, int]:
    block_count = len(model.transformer_blocks)
    if block_count != expected_depth:
        raise RuntimeError(
            f"Expected {expected_depth} blocks, found {block_count}."
        )

    if len({id(block) for block in model.transformer_blocks}) != block_count:
        raise RuntimeError("Transformer blocks are unexpectedly shared.")

    expected_hidden_matrix_count = 4 * expected_depth
    hidden_matrix_count = len(hidden_parameters)
    if hidden_matrix_count != expected_hidden_matrix_count:
        raise RuntimeError(
            "Incorrect Muon hidden-matrix count: "
            f"expected {expected_hidden_matrix_count}, "
            f"found {hidden_matrix_count}."
        )

    if (
        len({id(parameter) for parameter in hidden_parameters})
        != hidden_matrix_count
    ):
        raise RuntimeError(
            "Hidden matrices are unexpectedly shared between layers."
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    return parameter_count, hidden_matrix_count


def prepare_outputs(
    run_name: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    csv_path = Path("runs") / f"{run_name}.csv"
    checkpoint_directory = Path("checkpoints") / run_name

    if overwrite:
        if csv_path.exists():
            csv_path.unlink()
        if checkpoint_directory.exists():
            shutil.rmtree(checkpoint_directory)

    if csv_path.exists():
        raise FileExistsError(f"CSV already exists: {csv_path}")
    if checkpoint_directory.exists():
        raise FileExistsError(
            f"Checkpoint directory already exists: "
            f"{checkpoint_directory}"
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=False)
    return csv_path, checkpoint_directory


def first_sustained_step(
    records: list[dict[str, object]],
    field: str,
    threshold: float,
    consecutive: int = 5,
) -> int | None:
    for index in range(len(records) - consecutive + 1):
        window = records[index:index + consecutive]
        if all(
            float(record[field]) >= threshold
            for record in window
        ):
            return int(window[0]["step"])
    return None


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    set_seed(args.seed)

    device = get_device(args.device)

    (
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    ) = generate_modular_addition_data(
        modulus=args.modulus,
        train_fraction=args.train_fraction,
        seed=args.seed,
        operation=args.operation,
    )

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    configuration = model_configuration(args)
    model = DepthModularAdditionTransformer(**configuration)

    load_or_create_initial_state(
        model=model,
        path=args.initial_state_path,
        configuration=configuration,
        seed=args.seed,
        depth_one_initial_state_path=(
            args.depth_one_initial_state_path
        ),
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
        expected_depth=args.num_layers,
    )

    csv_path, checkpoint_directory = prepare_outputs(
        args.run_name,
        args.overwrite,
    )

    optimizers: dict[str, object] = {}

    if args.regime == "adamw":
        optimizers["adamw"] = torch.optim.AdamW(
            model.parameters(),
            lr=args.adamw_lr,
            weight_decay=args.adamw_weight_decay,
            betas=(args.adamw_beta1, args.adamw_beta2),
        )
    else:
        optimizers["muon"] = Muon(
            hidden_parameters,
            learning_rate=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=args.muon_weight_decay,
            newton_schulz_steps=args.muon_ns_steps,
            nesterov=True,
        )
        optimizers["auxiliary_adamw"] = torch.optim.AdamW(
            auxiliary_parameters,
            lr=args.aux_lr,
            weight_decay=args.aux_weight_decay,
            betas=(args.adamw_beta1, args.adamw_beta2),
        )
        optimizers["unembedding_adamw"] = torch.optim.AdamW(
            unembedding_parameters,
            lr=args.unembedding_lr,
            weight_decay=args.unembedding_weight_decay,
            betas=(args.adamw_beta1, args.adamw_beta2),
        )

    fieldnames = [
        "step",
        "regime",
        "num_layers",
        "modulus",
        "train_fraction",
        "d_model",
        "num_heads",
        "d_mlp",
        "parameter_count",
        "hidden_matrix_count",
        "initialization_scheme",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "elapsed_seconds",
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

    evaluation_records: list[dict[str, object]] = []

    last_hidden_gradient_norm = float("nan")
    last_auxiliary_gradient_norm = float("nan")
    last_unembedding_gradient_norm = float("nan")
    last_muon_applied_update_norm = float("nan")
    last_muon_max_abs_applied_update = float("nan")

    has_memorized = False
    has_reached_95_test = False
    consecutive_trigger_evaluations = 0

    auxiliary_frozen = False
    scheduled_freeze_step: int | None = None
    actual_freeze_step: int | None = None

    print(f"Device: {device}")
    print(f"Run: {args.run_name}")
    print(f"Regime: {args.regime}")
    print(f"Operation: {args.operation}")
    print(
        "Configuration: "
        f"depth={args.num_layers}, "
        f"modulus={args.modulus}, "
        f"train_fraction={args.train_fraction}, "
        f"d_model={args.d_model}, "
        f"d_mlp={args.d_mlp}"
    )
    print(
        "Verified architecture: "
        f"blocks={len(model.transformer_blocks)}, "
        f"hidden_matrices={hidden_matrix_count}, "
        f"parameters={parameter_count}"
    )

    started_at = time.perf_counter()

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

        for step in range(args.steps + 1):
            if step % args.evaluation_interval == 0:
                train_loss, train_accuracy = evaluate(
                    model,
                    train_inputs,
                    train_targets,
                )
                test_loss, test_accuracy = evaluate(
                    model,
                    test_inputs,
                    test_targets,
                )

                if train_accuracy >= 0.999:
                    has_memorized = True
                if test_accuracy >= 0.95:
                    has_reached_95_test = True

                if (
                    args.regime == "stable_muon"
                    and scheduled_freeze_step is None
                ):
                    if (
                        test_accuracy
                        >= args.freeze_trigger_test_accuracy
                    ):
                        consecutive_trigger_evaluations += 1
                    else:
                        consecutive_trigger_evaluations = 0

                    if (
                        consecutive_trigger_evaluations
                        >= args.freeze_trigger_consecutive
                    ):
                        first_trigger_step = (
                            step
                            - (
                                args.freeze_trigger_consecutive
                                - 1
                            )
                            * args.evaluation_interval
                        )
                        scheduled_freeze_step = (
                            first_trigger_step
                            + args.freeze_delay
                        )
                        print(
                            "Scheduled auxiliary freeze for step "
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
                    "num_layers": args.num_layers,
                    "modulus": args.modulus,
                    "train_fraction": args.train_fraction,
                    "d_model": args.d_model,
                    "num_heads": args.num_heads,
                    "d_mlp": args.d_mlp,
                    "parameter_count": parameter_count,
                    "hidden_matrix_count": hidden_matrix_count,
                    "initialization_scheme": (
                        "nested_depth_one_shared_v2"
                    ),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "elapsed_seconds": (
                        time.perf_counter() - started_at
                    ),
                    "hidden_parameter_norm": parameter_l2_norm(
                        hidden_parameters
                    ),
                    "auxiliary_parameter_norm": parameter_l2_norm(
                        auxiliary_parameters
                    ),
                    "unembedding_parameter_norm": parameter_l2_norm(
                        unembedding_parameters
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
                        if scheduled_freeze_step is None
                        else scheduled_freeze_step
                    ),
                }

                writer.writerow(row)
                handle.flush()
                evaluation_records.append(row)

                print(
                    f"step={step:6d} | "
                    f"train={train_accuracy:.4f} | "
                    f"test={test_accuracy:.4f} | "
                    f"aux_frozen={int(auxiliary_frozen)}"
                )

            if step % args.checkpoint_interval == 0:
                optimizer_states = {
                    name: optimizer.state_dict()
                    for name, optimizer in optimizers.items()
                }
                checkpoint_path = (
                    checkpoint_directory
                    / f"step_{step:06d}.pt"
                )
                torch.save(
                    {
                        "step": step,
                        "run_name": args.run_name,
                        "regime": args.regime,
                        "operation": args.operation,
                        "arguments": vars(args),
                        "model_config": configuration,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dicts": optimizer_states,
                        "seed": args.seed,
                        "parameter_count": parameter_count,
                        "hidden_matrix_count": (
                            hidden_matrix_count
                        ),
                        "initialization_scheme": (
                            "nested_depth_one_shared_v2"
                        ),
                        "actual_freeze_step": (
                            actual_freeze_step
                        ),
                    },
                    checkpoint_path,
                )

            if step == args.steps:
                break

            if (
                args.regime == "stable_muon"
                and not auxiliary_frozen
                and scheduled_freeze_step is not None
                and step >= scheduled_freeze_step
            ):
                for parameter in (
                    auxiliary_parameters
                    + unembedding_parameters
                ):
                    parameter.requires_grad_(False)
                    parameter.grad = None

                auxiliary_frozen = True
                actual_freeze_step = step
                print(
                    "Froze all AdamW-managed parameters at step "
                    f"{step}."
                )

            model.train()
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)

            logits = model(train_inputs)
            loss = F.cross_entropy(logits, train_targets)

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

            if args.regime == "adamw":
                optimizers["adamw"].step()
            else:
                muon_optimizer = optimizers["muon"]
                muon_optimizer.step()

                if not auxiliary_frozen:
                    optimizers["auxiliary_adamw"].step()
                    optimizers["unembedding_adamw"].step()

                statistics = muon_optimizer.last_step_stats
                last_muon_applied_update_norm = statistics[
                    "applied_update_norm"
                ]
                last_muon_max_abs_applied_update = statistics[
                    "max_abs_applied_update"
                ]

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
        f"{evaluation_records[-1]['test_accuracy']:.6f}"
    )
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
