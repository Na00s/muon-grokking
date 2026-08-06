from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Parameter


UPDATE_STATISTIC_FIELDS = (
    "applied_update_norm",
    "gradient_component_norm",
    "decay_component_norm",
    "applied_update_rms",
    "parameter_norm",
)

ZERO_UPDATE_STATISTICS = {
    "applied_update_norm": 0.0,
    "gradient_component_norm": 0.0,
    "decay_component_norm": 0.0,
    "applied_update_rms": 0.0,
    "parameter_norm": float("nan"),
}


def clone_parameter_values(
    parameters: list[Parameter],
) -> list[Tensor]:
    """
    Snapshot in double precision.

    The applied update is a difference of two nearly equal numbers, so
    the snapshot must not be taken at lower precision than the
    parameters themselves.
    """
    return [
        parameter.detach().double().clone()
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
    Decompose the tensor an optimizer actually subtracted this step.

    Muon, the Newton-Schulz ablation, and decoupled AdamW all apply

        parameter <- parameter * (1 - lr * wd) - lr * update

    so the decay contribution is exactly lr * wd * parameter_before and
    the gradient-driven contribution is the remainder. The decay term
    is not a response to the gradient, so it is reported separately
    rather than counted as one.

    Measuring the difference across the step, rather than trusting a
    quantity the optimizer reports internally, keeps this comparable
    across optimizers whose internal bookkeeping differs.
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
        after = parameter.detach().double()

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
