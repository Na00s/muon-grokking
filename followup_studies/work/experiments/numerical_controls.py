"""Numerical controls for the Muon grokking checkpoint experiments.

Original source: work/muon-grokking at commit
6d64a981af75f1300d9060109e81552d48a81360. No repository files are changed.

``PrecisionMuon`` inherits the original constructor and checkpoint schema.
Its step is copied verbatim from the source except for the NS function name.
Thus the only update-rule difference is NS compute precision. For float32
parameters it retains the original arithmetic, including the 1e-7 epsilon.
The original float32 casts in logging statistics are deliberately retained;
they do not feed into parameter or momentum updates.

``accurate_cross_entropy`` supports the exact task interface: finite floating
logits [batch, classes], integer targets [batch], no weights or smoothing.
It preserves the logits dtype unless compute_dtype is supplied explicitly.
Promoting only the loss leaves model backward and parameter storage precision
unchanged. A full float64 control must also convert the model before restoring
optimizer state and must use PrecisionMuon.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1] / "muon-grokking"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from optimizers.muon import Muon


@torch.no_grad()
def zeroth_power_newton_schulz_precision(
    matrix: Tensor,
    steps: int = 5,
    epsilon: float = 1e-7,
) -> Tensor:
    """Original quintic NS with float64 computation for float64 input.

For float16/bfloat16/float32 input computation uses float32, as in source.
This intentionally keeps the source epsilon and finite iteration count.
It is a precision control, not a more accurate polar decomposition method.
"""
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization requires a 2D matrix.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if not matrix.is_floating_point():
        raise TypeError("The NS input must be floating point.")
    original_dtype = matrix.dtype
    compute_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    x = matrix.to(dtype=compute_dtype)
    if not torch.isfinite(x).all():
        raise FloatingPointError("Muon received a gradient containing NaN or infinity.")
    was_transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        was_transposed = True
    matrix_norm = torch.linalg.vector_norm(x)
    if matrix_norm.item() == 0.0:
        return torch.zeros_like(matrix)
    x = x / (matrix_norm + epsilon)
    coefficient_a = 3.4445
    coefficient_b = -4.7750
    coefficient_c = 2.0315
    for _ in range(steps):
        gram = x @ x.T
        polynomial = coefficient_b * gram + coefficient_c * (gram @ gram)
        x = coefficient_a * x + polynomial @ x
    if was_transposed:
        x = x.T
    if not torch.isfinite(x).all():
        raise FloatingPointError("Newton-Schulz produced NaN or infinity.")
    return x.to(dtype=original_dtype)


def accurate_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    reduction: str = "mean",
    *,
    compute_dtype: torch.dtype | None = None,
) -> Tensor:
    """Cross-entropy with accurate small losses and correct-class gradients.

For rows where the correct class has a maximal logit, compute
log1p(sum_{j != y} exp(logit_j - logit_y)). This avoids subtraction of
near-equal values in both loss and the correct-class derivative. Every
exponent in this branch is nonpositive. Other rows use PyTorch CE.

The stable expression is protected with clamp(max=0) before it is evaluated
for mixed batches, so the unused branch cannot overflow on incorrect rows.
All operations remain differentiable; no probability or loss is detached.
Input logits should be finite. NaNs propagate rather than being repaired.
"""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Expected logits [batch, classes] and targets [batch].")
    if logits.shape[1] < 1:
        raise ValueError("At least one class is required.")
    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long.")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point.")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
    values = logits if compute_dtype is None else logits.to(dtype=compute_dtype)
    selected = values.gather(1, targets[:, None])
    correct_is_maximum = selected[:, 0] >= values.max(dim=1).values
    differences = (values - selected).scatter(1, targets[:, None], float("-inf"))
    protected_differences = differences.clamp(max=0)
    stable_losses = torch.log1p(torch.exp(protected_differences).sum(dim=1))
    if bool(correct_is_maximum.all()):
        losses = stable_losses
    else:
        ordinary_losses = F.cross_entropy(values, targets, reduction="none")
        losses = torch.where(correct_is_maximum, stable_losses, ordinary_losses)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


class PrecisionMuon(Muon):
    """Original Muon update with NS dtype following parameter dtype.

    Instantiate with the same keywords as Muon. load_state_dict accepts
    the source Muon state directly. Convert the model first, then create
    optimizers and load their states; PyTorch maps floating optimizer
    buffers onto the parameter dtype during load_state_dict.
    """

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
                    zeroth_power_newton_schulz_precision(
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

