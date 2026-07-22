import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from data import generate_modular_addition_data
from model import ModularAdditionTransformer
from optimizers.muon import Muon


MODULUS = 113
TRAIN_FRACTION = 0.3

NUMBER_OF_STEPS = 100_000
EVALUATION_INTERVAL = 100
CHECKPOINT_INTERVAL = 1_000

ADAMW_LEARNING_RATE = 1e-3
ADAMW_WEIGHT_DECAY = 1.0
ADAMW_BETAS = (0.9, 0.999)

MUON_LEARNING_RATE = 0.02
MUON_MOMENTUM = 0.95
MUON_WEIGHT_DECAY = 0.05
MUON_NEWTON_SCHULZ_STEPS = 5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "muon"],
        required=True,
        help="Optimizer to use: adamw or muon",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=NUMBER_OF_STEPS,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    raise RuntimeError("No CUDA or MPS GPU was found.")


@torch.no_grad()
def evaluate(
    model: ModularAdditionTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    model.eval()

    logits = model(inputs)  # (B, P), B = examples, P = 113 classes

    loss = F.cross_entropy(
        logits,
        targets,
    )

    predictions = logits.argmax(dim=-1)  # (B,)

    accuracy = (
        predictions == targets
    ).float().mean()

    return loss.item(), accuracy.item()


def split_muon_parameters(
    model: ModularAdditionTransformer,
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
]:
    # Hidden 2D matrices optimized using Muon.
    muon_parameters = [
        model.transformer_block.attention.qkv_projection.weight,
        model.transformer_block.attention.output_projection.weight,
        model.transformer_block.mlp.input_projection.weight,
        model.transformer_block.mlp.output_projection.weight,
    ]

    muon_parameter_ids = {
        id(parameter)
        for parameter in muon_parameters
    }

    # Embeddings and unembedding remain under AdamW.
    auxiliary_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in muon_parameter_ids
    ]

    return muon_parameters, auxiliary_parameters


def build_optimizers(
    model: ModularAdditionTransformer,
    optimizer_name: str,
) -> dict[str, torch.optim.Optimizer]:
    if optimizer_name == "adamw":
        return {
            "adamw": torch.optim.AdamW(
                model.parameters(),
                lr=ADAMW_LEARNING_RATE,
                weight_decay=ADAMW_WEIGHT_DECAY,
                betas=ADAMW_BETAS,
            )
        }

    if optimizer_name == "muon":
        muon_parameters, auxiliary_parameters = (
            split_muon_parameters(model)
        )

        return {
            "muon": Muon(
                muon_parameters,
                learning_rate=MUON_LEARNING_RATE,
                momentum=MUON_MOMENTUM,
                weight_decay=MUON_WEIGHT_DECAY,
                newton_schulz_steps=MUON_NEWTON_SCHULZ_STEPS,
                nesterov=True,
            ),
            "auxiliary_adamw": torch.optim.AdamW(
                auxiliary_parameters,
                lr=ADAMW_LEARNING_RATE,
                weight_decay=ADAMW_WEIGHT_DECAY,
                betas=ADAMW_BETAS,
            ),
        }

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )


def main() -> None:
    args = parse_arguments()

    set_seed(args.seed)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Using optimizer: {args.optimizer}")
    print(f"Number of steps: {args.steps}")

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

    train_inputs = train_inputs.to(device)  # (B_train, T)
    train_targets = train_targets.to(device)  # (B_train,)

    test_inputs = test_inputs.to(device)  # (B_test, T)
    test_targets = test_targets.to(device)  # (B_test,)

    model = ModularAdditionTransformer(
        modulus=MODULUS,
        sequence_length=3,
        d_model=128,
        num_heads=4,
        d_mlp=512,
    ).to(device)

    optimizers = build_optimizers(
        model=model,
        optimizer_name=args.optimizer,
    )

    run_name = (
        f"{args.optimizer}_seed_{args.seed}"
    )

    checkpoint_directory = (
        Path("checkpoints") / run_name
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs_directory = Path("runs")

    runs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        runs_directory / f"{run_name}.csv"
    )

    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)

        writer.writerow(
            [
                "step",
                "train_loss",
                "train_accuracy",
                "test_loss",
                "test_accuracy",
            ]
        )

        for step in range(args.steps + 1):
            if step % EVALUATION_INTERVAL == 0:
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

                print(
                    f"step={step:6d} | "
                    f"train_loss={train_loss:.6f} | "
                    f"train_acc={train_accuracy:.4f} | "
                    f"test_loss={test_loss:.6f} | "
                    f"test_acc={test_accuracy:.4f}"
                )

                writer.writerow(
                    [
                        step,
                        train_loss,
                        train_accuracy,
                        test_loss,
                        test_accuracy,
                    ]
                )

                log_file.flush()

            if step % CHECKPOINT_INTERVAL == 0:
                checkpoint_path = (
                    checkpoint_directory
                    / f"step_{step:06d}.pt"
                )

                optimizer_states = {
                    name: optimizer.state_dict()
                    for name, optimizer in optimizers.items()
                }

                torch.save(
                    {
                        "step": step,
                        "optimizer_name": args.optimizer,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dicts": optimizer_states,
                        "seed": args.seed,
                        "modulus": MODULUS,
                        "train_fraction": TRAIN_FRACTION,
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

            loss.backward()

            for optimizer in optimizers.values():
                optimizer.step()


if __name__ == "__main__":
    main()