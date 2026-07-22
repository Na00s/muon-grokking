from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


@torch.no_grad()
def zeroth_power_newton_schulz(
    matrix: Tensor,
    steps: int = 5,
    epsilon: float = 1e-7,
) -> Tensor:
    """
    Approximate the zeroth power of a 2D matrix.

    For a singular value decomposition:

        matrix = U @ S @ V.T

    this approximately returns:

        U @ V.T

    The Newton-Schulz iteration avoids computing a full SVD.
    """
    if matrix.ndim != 2:
        raise ValueError(
            "Newton-Schulz orthogonalization requires a 2D matrix."
        )

    if steps < 1:
        raise ValueError(
            "steps must be at least 1."
        )

    original_dtype = matrix.dtype

    # Float32 is used intentionally because the experiment runs on Apple MPS.
    x = matrix.float()

    if not torch.isfinite(x).all():
        raise FloatingPointError(
            "Muon received a gradient containing NaN or infinity."
        )

    was_transposed = False

    # The iteration is applied with rows <= columns.
    if x.shape[0] > x.shape[1]:
        x = x.T
        was_transposed = True

    matrix_norm = torch.linalg.vector_norm(x)

    if matrix_norm.item() == 0.0:
        return torch.zeros_like(matrix)

    x = x / (matrix_norm + epsilon)

    # Quintic Newton-Schulz coefficients used by Muon.
    coefficient_a = 3.4445
    coefficient_b = -4.7750
    coefficient_c = 2.0315

    for _ in range(steps):
        gram = x @ x.T  # (rows, rows)

        polynomial = (
            coefficient_b * gram
            + coefficient_c * (gram @ gram)
        )  # (rows, rows)

        x = (
            coefficient_a * x
            + polynomial @ x
        )  # (rows, columns)

    if was_transposed:
        x = x.T

    if not torch.isfinite(x).all():
        raise FloatingPointError(
            "Newton-Schulz produced NaN or infinity."
        )

    return x.to(dtype=original_dtype)


class Muon(Optimizer):
    """
    Single-device Muon optimizer for hidden 2D weight matrices.

    Embeddings, output layers, biases, and other non-hidden parameters
    should be handled by a separate optimizer such as AdamW.
    """

    def __init__(
        self,
        parameters,
        learning_rate: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.05,
        newton_schulz_steps: int = 5,
        nesterov: bool = True,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "momentum must be between 0 and 1."
            )

        if weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if newton_schulz_steps < 1:
            raise ValueError(
                "newton_schulz_steps must be at least 1."
            )

        if learning_rate * weight_decay >= 1.0:
            raise ValueError(
                "learning_rate * weight_decay must be below 1."
            )

        defaults = {
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "newton_schulz_steps": newton_schulz_steps,
            "nesterov": nesterov,
        }

        super().__init__(
            parameters,
            defaults,
        )

        # Aggregated statistics from the most recent optimizer step.
        # These are logged by train.py to diagnose instability.
        self.last_step_stats: dict[str, float] = {
            "pre_ns_update_norm": float("nan"),
            "post_ns_update_norm": float("nan"),
            "applied_update_norm": float("nan"),
            "max_abs_applied_update": float("nan"),
        }

    @torch.no_grad()
    def step(
        self,
        closure=None,
    ) -> Any:
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_pre_ns_norm_squared = 0.0
        total_post_ns_norm_squared = 0.0
        total_applied_update_norm_squared = 0.0
        maximum_absolute_applied_update = 0.0

        for group in self.param_groups:
            learning_rate = group["learning_rate"]
            momentum_coefficient = group["momentum"]
            weight_decay = group["weight_decay"]
            newton_schulz_steps = group[
                "newton_schulz_steps"
            ]
            use_nesterov = group["nesterov"]

            decay_multiplier = (
                1.0
                - learning_rate * weight_decay
            )

            if decay_multiplier < 0.0:
                raise ValueError(
                    "Muon weight decay produced a negative "
                    "parameter multiplier."
                )

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                if parameter.ndim != 2:
                    raise ValueError(
                        "Muon should only receive 2D hidden "
                        "weight matrices."
                    )

                gradient = parameter.grad.detach()

                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(
                        "Muon received a non-finite gradient."
                    )

                state = self.state[parameter]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = (
                        torch.zeros_like(parameter)
                    )

                momentum_buffer = state[
                    "momentum_buffer"
                ]

                # Exponential moving average:
                #
                # m_t = beta * m_(t-1) + (1 - beta) * g_t
                momentum_buffer.mul_(
                    momentum_coefficient
                )

                momentum_buffer.add_(
                    gradient,
                    alpha=1.0 - momentum_coefficient,
                )

                if use_nesterov:
                    # Equivalent to:
                    #
                    # update =
                    #     (1 - beta) * gradient
                    #     + beta * momentum_buffer
                    update = gradient.lerp(
                        momentum_buffer,
                        momentum_coefficient,
                    )
                else:
                    update = momentum_buffer.clone()

                pre_ns_norm = torch.linalg.vector_norm(
                    update.float()
                ).item()

                orthogonalized_update = (
                    zeroth_power_newton_schulz(
                        update,
                        steps=newton_schulz_steps,
                    )
                )

                post_ns_norm = torch.linalg.vector_norm(
                    orthogonalized_update.float()
                ).item()

                rows, columns = parameter.shape

                # Shape-dependent Muon scaling.
                shape_scale = math.sqrt(
                    max(
                        1.0,
                        rows / columns,
                    )
                )

                orthogonalized_update.mul_(
                    shape_scale
                )

                applied_update = (
                    learning_rate
                    * orthogonalized_update
                )

                applied_update_norm = (
                    torch.linalg.vector_norm(
                        applied_update.float()
                    ).item()
                )

                maximum_absolute_update_for_parameter = (
                    applied_update
                    .float()
                    .abs()
                    .max()
                    .item()
                )

                total_pre_ns_norm_squared += (
                    pre_ns_norm ** 2
                )

                total_post_ns_norm_squared += (
                    post_ns_norm ** 2
                )

                total_applied_update_norm_squared += (
                    applied_update_norm ** 2
                )

                maximum_absolute_applied_update = max(
                    maximum_absolute_applied_update,
                    maximum_absolute_update_for_parameter,
                )

                # Decoupled weight decay.
                parameter.mul_(
                    decay_multiplier
                )

                parameter.add_(
                    orthogonalized_update,
                    alpha=-learning_rate,
                )

        self.last_step_stats = {
            "pre_ns_update_norm": math.sqrt(
                total_pre_ns_norm_squared
            ),
            "post_ns_update_norm": math.sqrt(
                total_post_ns_norm_squared
            ),
            "applied_update_norm": math.sqrt(
                total_applied_update_norm_squared
            ),
            "max_abs_applied_update": (
                maximum_absolute_applied_update
            ),
        }

        return loss


if __name__ == "__main__":
    torch.manual_seed(0)

    parameter = torch.nn.Parameter(
        torch.randn(128, 128)
    )

    optimizer = Muon(
        [parameter],
        learning_rate=0.02,
        momentum=0.95,
        weight_decay=0.05,
        newton_schulz_steps=5,
        nesterov=True,
    )

    loss = parameter.square().mean()

    loss.backward()
    optimizer.step()

    print("Muon optimizer test passed.")
    print(optimizer.last_step_stats)