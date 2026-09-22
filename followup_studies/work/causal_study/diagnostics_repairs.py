"""Targeted backward-only controls for the stock cross-entropy objective.

Both functions return the original PyTorch forward loss. Their custom backward
operates at the logits and preserves the incoming reduction and example weights.
Use these functions as the sole loss consumer of their wrapped logit tensor.
No model, parameter, optimizer-state, or Newton-Schulz arithmetic is changed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def target_repair_gradient(gradient: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Keep stock wrong-class entries; reconstruct each correct-class entry.

    Sum wrong-class entries in float64, then cast back to the incoming dtype.
    The correction restores the exact CE derivative identity to rounding error.
    """
    wrong = gradient.scatter(1, targets[:, None], 0.0)
    repaired_target = -wrong.double().sum(dim=-1, keepdim=True).to(gradient.dtype)
    return gradient.scatter(1, targets[:, None], repaired_target)


def row_project_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Orthogonally remove the class-common component, with float64 reduction."""
    values = gradient.double()
    return (values-values.mean(dim=-1, keepdim=True)).to(gradient.dtype)


class _TargetRepair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        ctx.save_for_backward(targets)
        return logits.view_as(logits)

    @staticmethod
    def backward(ctx, gradient):
        targets, = ctx.saved_tensors
        return target_repair_gradient(gradient, targets), None


class _RowProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        return logits.view_as(logits)

    @staticmethod
    def backward(ctx, gradient):
        return row_project_gradient(gradient)


def ce_target_repair(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Stock CE forward with corrected target-logit derivative only."""
    return F.cross_entropy(_TargetRepair.apply(logits, targets), targets, reduction=reduction)


def ce_row_projection(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Stock CE forward with the class-common gradient component removed."""
    return F.cross_entropy(_RowProjection.apply(logits), targets, reduction=reduction)
