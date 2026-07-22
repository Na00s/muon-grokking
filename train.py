from pathlib import Path
import random

import torch
import torch.nn.functional as F

from data import generate_modular_addition_data
from model import ModularAdditionTransformer


MODULUS = 113
TRAIN_FRACTION = 0.3
SEED = 0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.0
NUMBER_OF_STEPS = 100_000
EVALUATION_INTERVAL = 100
CHECKPOINT_INTERVAL = 1_000


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

    logits = model(inputs)  # (B, P), B = examples, P = 113 possible answers

    loss = F.cross_entropy(logits, targets)

    predictions = logits.argmax(dim=-1)  # (B,), one predicted answer per example

    accuracy = (predictions == targets).float().mean()

    return loss.item(), accuracy.item()


def main() -> None:
    set_seed(SEED)

    device = get_device()
    print(f"Using device: {device}")

    (
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    ) = generate_modular_addition_data(
        modulus=MODULUS,
        train_fraction=TRAIN_FRACTION,
        seed=SEED,
    )

    train_inputs = train_inputs.to(device)  # (B_train, T), B_train = 3,830, T = 3
    train_targets = train_targets.to(device)  # (B_train,), one target per example

    test_inputs = test_inputs.to(device)  # (B_test, T), B_test = 8,939, T = 3
    test_targets = test_targets.to(device)  # (B_test,), one target per example

    model = ModularAdditionTransformer(
        modulus=MODULUS,
        sequence_length=3,
        d_model=128,
        num_heads=4,
        d_mlp=512,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    checkpoint_directory = Path("checkpoints/adamw_seed_0")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    for step in range(NUMBER_OF_STEPS + 1):
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

        if step % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = checkpoint_directory / f"step_{step:06d}.pt"

            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "seed": SEED,
                    "modulus": MODULUS,
                    "train_fraction": TRAIN_FRACTION,
                },
                checkpoint_path,
            )

        if step == NUMBER_OF_STEPS:
            break

        model.train()

        optimizer.zero_grad()

        logits = model(train_inputs)  # (B_train, P), P = 113 answer classes

        loss = F.cross_entropy(
            logits,
            train_targets,
        )

        loss.backward()

        optimizer.step()


if __name__ == "__main__":
    main()