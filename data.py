import torch


def generate_modular_addition_data(
    modulus: int = 113,
    train_fraction: float = 0.3,
    seed: int = 0,
):
    values = torch.arange(modulus, dtype=torch.long)  # (P,), P = number of values modulo 113

    a, b = torch.meshgrid(values, values, indexing="ij")  # each: (P, P), all ordered pairs

    a = a.flatten()  # (B,), B = P² = 12,769 total examples
    b = b.flatten()  # (B,), B = P² = 12,769 total examples

    equals_token = torch.full_like(a, modulus)  # (B,), one equals token per example

    inputs = torch.stack([a, b, equals_token], dim=1)  # (B, T), B = examples, T = 3 input tokens

    targets = (a + b) % modulus  # (B,), one target class per example

    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(len(inputs), generator=generator)  # (B,), shuffled example indices

    inputs = inputs[permutation]  # (B, T), shuffled inputs
    targets = targets[permutation]  # (B,), shuffled targets

    number_of_training_examples = int(train_fraction * len(inputs))

    train_inputs = inputs[:number_of_training_examples]  # (B_train, T), B_train = 3,830, T = 3
    train_targets = targets[:number_of_training_examples]  # (B_train,), B_train = 3,830

    test_inputs = inputs[number_of_training_examples:]  # (B_test, T), B_test = 8,939, T = 3
    test_targets = targets[number_of_training_examples:]  # (B_test,), B_test = 8,939

    return train_inputs, train_targets, test_inputs, test_targets