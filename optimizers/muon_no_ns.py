from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer


class MuonWithoutNewtonSchulz(Optimizer):
    """
    Muon with the Newton-Schulz orthogonalization removed.

    Every other element is identical to optimizers/muon.py: the same
    momentum buffer, the same Nesterov option, the same shape-dependent
    scaling, the same decoupled weight decay, and the same order of
    operations on the parameter.

    The single difference is that the update is the momentum buffer
    itself rather than its approximate zeroth power. Muon normalizes
    its input before orthogonalizing, so the orthogonalized update has
    a magnitude set by the matrix shape rather than by the gradient.
    Here the update keeps the magnitude of the momentum buffer, so the
    step scales with the gradient.

    That difference makes the learning rate non-transferable: the same
    learning rate produces a completely different effective step, so a
    run of this optimizer is only interpretable against a learning-rate
    sweep or an explicit calibration.

    last_step_stats carries the same keys as Muon so existing logging
    paths work unchanged. pre_ns_update_norm and post_ns_update_norm
    are equal here by construction, and that equality is the signature
    of the ablation.
    """

    def __init__(
        self,
        parameters,
        learning_rate: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.05,
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

        if learning_rate * weight_decay >= 1.0:
            raise ValueError(
                "learning_rate * weight_decay must be below 1."
            )

        defaults = {
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }

        super().__init__(
            parameters,
            defaults,
        )

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
        total_applied_update_norm_squared = 0.0
        maximum_absolute_applied_update = 0.0

        for group in self.param_groups:
            learning_rate = group["learning_rate"]
            momentum_coefficient = group["momentum"]
            weight_decay = group["weight_decay"]
            use_nesterov = group["nesterov"]

            decay_multiplier = (
                1.0
                - learning_rate * weight_decay
            )

            if decay_multiplier < 0.0:
                raise ValueError(
                    "Weight decay produced a negative "
                    "parameter multiplier."
                )

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                if parameter.ndim != 2:
                    raise ValueError(
                        "This optimizer should only receive 2D "
                        "hidden weight matrices."
                    )

                gradient = parameter.grad.detach()

                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(
                        "Received a non-finite gradient."
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

                # Newton-Schulz is skipped here. This is the entire
                # ablation: the update is never renormalized, so it
                # carries the magnitude of the momentum buffer.
                unnormalized_update = update.clone()

                rows, columns = parameter.shape

                # Shape-dependent scaling, retained so that the
                # orthogonalization is the only difference from Muon.
                shape_scale = math.sqrt(
                    max(
                        1.0,
                        rows / columns,
                    )
                )

                unnormalized_update.mul_(
                    shape_scale
                )

                applied_update = (
                    learning_rate
                    * unnormalized_update
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
                    unnormalized_update,
                    alpha=-learning_rate,
                )

        pre_ns_norm_total = math.sqrt(
            total_pre_ns_norm_squared
        )

        self.last_step_stats = {
            "pre_ns_update_norm": pre_ns_norm_total,
            # Equal to the pre-iteration norm by construction,
            # since no orthogonalization is applied.
            "post_ns_update_norm": pre_ns_norm_total,
            "applied_update_norm": math.sqrt(
                total_applied_update_norm_squared
            ),
            "max_abs_applied_update": (
                maximum_absolute_applied_update
            ),
        }

        return loss
