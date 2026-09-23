"""Train-only linear alignment and readout transport for checkpoint comparisons.

Rows index examples: H has shape (N, d), W has shape (d, classes).
If H1 = H0 @ A, its compatible healthy readout is solve(A, W0).
The backward fit H1 @ B ~= H0 supplies transport B @ W0 without inversion.
All fitting is float64. The returned report contains only JSON-compatible values.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


DEFAULT_RIDGE_GRID = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)


@dataclass
class AlignmentResult:
    report: dict
    transforms: Dict[str, np.ndarray]
    transported_readouts: Dict[str, np.ndarray]


def _matrix(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _mask(value, length, name):
    result = np.asarray(value)
    if result.dtype != np.bool_ or result.shape != (length,):
        raise ValueError(f"{name} must be an explicit boolean mask of shape ({length},)")
    if not result.any():
        raise ValueError(f"{name} must select at least one example")
    return result


def _relative(numerator, denominator):
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    return float(numerator / denominator)


def _cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return None
    return float(np.clip(np.sum(left * right) / denominator, -1.0, 1.0))


def classification_metrics(logits, labels):
    """Accuracy, stable cross-entropy, and margin against the strongest wrong class."""
    logits = _matrix(logits, "logits")
    labels = np.asarray(labels)
    if labels.shape != (len(logits),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be an integer vector aligned with logits")
    if logits.shape[1] < 2 or np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("labels must index at least two logit classes")
    row = np.arange(len(logits))
    competitors = logits.copy()
    competitors[row, labels] = -np.inf
    # Preserve tiny positive losses once the correct class dominates.
    wrong_delta = competitors - logits[row, labels, None]
    loss = np.logaddexp(0.0, np.logaddexp.reduce(wrong_delta, axis=1))
    margin = logits[row, labels] - competitors.max(axis=1)
    correct = logits.argmax(axis=1) == labels
    return {
        "n": int(len(logits)),
        "accuracy": float(correct.mean()),
        "correct_count": int(correct.sum()),
        "cross_entropy": float(loss.mean()),
        "margin_mean": float(margin.mean()),
        "margin_median": float(np.median(margin)),
        "margin_min": float(margin.min()),
        "margin_p05": float(np.quantile(margin, 0.05)),
        "margin_p95": float(np.quantile(margin, 0.95)),
        "positive_margin_fraction": float((margin > 0).mean()),
    }


def reconstruction_metrics(prediction, target, *, prediction_train_mean, target_train_mean):
    """Report absolute and relative errors, including training-mean centering.

    ``centered_relative_error`` removes each matrix's own training mean.
    ``error_over_centered_target_norm`` keeps the original error numerator.
    These expose constant components that can dominate uncentered residual norms.
    """
    prediction = _matrix(prediction, "prediction")
    target = _matrix(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    prediction_mean = np.asarray(prediction_train_mean, dtype=np.float64)
    target_mean = np.asarray(target_train_mean, dtype=np.float64)
    if prediction_mean.shape != (target.shape[1],) or target_mean.shape != prediction_mean.shape:
        raise ValueError("training means must match the feature dimension")
    error = np.linalg.norm(prediction - target)
    target_norm = np.linalg.norm(target)
    centered_prediction = prediction - prediction_mean
    centered_target = target - target_mean
    centered_norm = np.linalg.norm(centered_target)
    return {
        "frobenius_error": float(error),
        "rmse": float(error / np.sqrt(target.size)),
        "relative_error": _relative(error, target_norm),
        "centered_relative_error": _relative(
            np.linalg.norm(centered_prediction - centered_target), centered_norm
        ),
        "error_over_centered_target_norm": _relative(error, centered_norm),
        "target_frobenius_norm": float(target_norm),
        "prediction_frobenius_norm": float(np.linalg.norm(prediction)),
        "target_centered_frobenius_norm": float(centered_norm),
        "training_mean_error_norm": float(np.linalg.norm(prediction_mean - target_mean)),
        "frobenius_cosine": _cosine(prediction, target),
        "centered_frobenius_cosine": _cosine(centered_prediction, centered_target),
    }


def logit_comparison_metrics(prediction, target):
    """Compare functions after removing per-example common class offsets."""
    prediction = _matrix(prediction, "prediction")
    target = _matrix(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("logit arrays must have identical shapes")
    centered_prediction = prediction - prediction.mean(axis=1, keepdims=True)
    centered_target = target - target.mean(axis=1, keepdims=True)
    return {
        "relative_error": _relative(np.linalg.norm(prediction - target), np.linalg.norm(target)),
        "class_centered_relative_error": _relative(
            np.linalg.norm(centered_prediction - centered_target),
            np.linalg.norm(centered_target),
        ),
        "class_centered_cosine": _cosine(centered_prediction, centered_target),
        "prediction_agreement": float((prediction.argmax(axis=1) == target.argmax(axis=1)).mean()),
    }


def spectrum_metrics(matrix, *, rtol=None):
    """Numerical rank, conditioning and energy-based ranks of a matrix."""
    matrix = _matrix(matrix, "matrix")
    values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = (max(matrix.shape) * np.finfo(np.float64).eps if rtol is None else rtol)
    if tolerance < 0:
        raise ValueError("rtol must be nonnegative")
    threshold = float(tolerance * values[0])
    positive = values[values > threshold]
    energy = values ** 2
    total = energy.sum()
    full_rank = len(positive) == min(matrix.shape)
    return {
        "shape": list(matrix.shape),
        "singular_values": values.tolist(),
        "rank_threshold": threshold,
        "numerical_rank": int(len(positive)),
        "full_rank": bool(full_rank),
        "condition_number": float(values[0] / values[-1]) if full_rank else None,
        "nonzero_condition_number": float(positive[0] / positive[-1]) if len(positive) else None,
        "stable_rank": float(total / energy[0]) if total else 0.0,
        "energy_participation_rank": float(total ** 2 / np.sum(energy ** 2)) if total else 0.0,
    }


def fit_orthogonal_forward(healthy, collapsed):
    """Fit R minimizing ||healthy @ R - collapsed||, with R.T @ R = I."""
    healthy = _matrix(healthy, "healthy")
    collapsed = _matrix(collapsed, "collapsed")
    if healthy.shape != collapsed.shape:
        raise ValueError("orthogonal alignment requires equal shapes")
    left, _, right_t = np.linalg.svd(healthy.T @ collapsed, full_matrices=False)
    return left @ right_t


def _ridge_from_svd(left, singular, right_t, target, relative_ridge, source_shape):
    if not np.isfinite(relative_ridge) or relative_ridge < 0:
        raise ValueError("relative_ridge must be finite and nonnegative")
    penalty = float(relative_ridge * np.sum(singular ** 2) / source_shape[1])
    if penalty > 0:
        gain = singular / (singular ** 2 + penalty)
    else:
        cutoff = max(source_shape) * np.finfo(np.float64).eps * singular[0]
        gain = np.zeros_like(singular)
        np.divide(1.0, singular, out=gain, where=singular > cutoff)
    return (right_t.T * gain) @ (left.T @ target)


def fit_ridge_map(source, target, relative_ridge=0.0):
    """Fit source @ map ~= target with a Frobenius-scaled ridge penalty.

    The absolute penalty is relative_ridge * ||source||_F**2 / d.
    A zero penalty uses a rank-aware SVD least-squares solution.
    """
    source = _matrix(source, "source")
    target = _matrix(target, "target")
    if len(source) != len(target):
        raise ValueError("source and target must have matching example counts")
    left, singular, right_t = np.linalg.svd(source, full_matrices=False)
    return _ridge_from_svd(left, singular, right_t, target, relative_ridge, source.shape)


def _select_ridge(source, target, fit_mask, validation_mask, ridge_grid):
    train_source = source[fit_mask]
    train_target = target[fit_mask]
    left, singular, right_t = np.linalg.svd(train_source, full_matrices=False)
    candidates = []
    for penalty in ridge_grid:
        mapping = _ridge_from_svd(left, singular, right_t, train_target, penalty, train_source.shape)
        error = np.linalg.norm(source[validation_mask] @ mapping - target[validation_mask])
        candidates.append({
            "relative_ridge": float(penalty),
            "validation_frobenius_error": float(error),
            "validation_relative_error": _relative(error, np.linalg.norm(target[validation_mask])),
        })
    selected = min(candidates, key=lambda candidate: (candidate["validation_frobenius_error"], candidate["relative_ridge"]))
    return selected["relative_ridge"], candidates


def analyze_alignment(
    H0, H1, W0, W1, y, train_mask, heldout_mask, *,
    inner_validation_mask: Optional[np.ndarray] = None,
    validation_fraction: float = 0.2,
    validation_seed: int = 0,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    inverse_condition_limit: float = 1e8,
):
    """Fit and evaluate checkpoint alignment without consulting held-out examples.

    If supplied, inner_validation_mask must be a nonempty strict subset of the
    original training mask. Otherwise a deterministic random 20% subset is used.
    Ridge selection uses representation reconstruction alone. Selected maps are
    refit on all original training examples. All result accuracies are fractions.

    Returns AlignmentResult(report, transforms, transported_readouts). The report
    is JSON-compatible, including None for undefined ratios or singular condition
    numbers. ``transforms`` contains orthogonal_forward, linear_forward, and
    linear_backward. No decoder is trained by this function.
    """
    H0, H1 = _matrix(H0, "H0"), _matrix(H1, "H1")
    W0, W1 = _matrix(W0, "W0"), _matrix(W1, "W1")
    if H0.shape != H1.shape or W0.shape != W1.shape or H0.shape[1] != W0.shape[0]:
        raise ValueError("representations and readouts have incompatible shapes")
    n, d = H0.shape
    train_mask = _mask(train_mask, n, "train_mask")
    heldout_mask = _mask(heldout_mask, n, "heldout_mask")
    if np.any(train_mask & heldout_mask):
        raise ValueError("train_mask and heldout_mask must be disjoint")
    labels = np.asarray(y)
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y must be an integer vector of length N")
    if W0.shape[1] < 2 or np.any(labels < 0) or np.any(labels >= W0.shape[1]):
        raise ValueError("y contains an invalid class index")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if not np.isfinite(inverse_condition_limit) or inverse_condition_limit < 1:
        raise ValueError("inverse_condition_limit must be finite and at least one")
    ridge_grid = tuple(float(value) for value in ridge_grid)
    if not ridge_grid or any(not np.isfinite(value) or value < 0 for value in ridge_grid):
        raise ValueError("ridge_grid must contain finite nonnegative penalties")
    if inner_validation_mask is None:
        indices = np.flatnonzero(train_mask)
        if len(indices) < 2:
            raise ValueError("at least two training examples are required")
        indices = np.random.default_rng(validation_seed).permutation(indices)
        count = min(len(indices) - 1, max(1, int(np.ceil(validation_fraction * len(indices)))))
        validation_mask = np.zeros(n, dtype=bool)
        validation_mask[indices[:count]] = True
    else:
        validation_mask = _mask(inner_validation_mask, n, "inner_validation_mask")
        if np.any(validation_mask & ~train_mask):
            raise ValueError("inner_validation_mask must be a subset of train_mask")
    fitting_mask = train_mask & ~validation_mask
    if not fitting_mask.any():
        raise ValueError("inner validation must leave training examples for map fitting")

    forward_penalty, forward_candidates = _select_ridge(H0, H1, fitting_mask, validation_mask, ridge_grid)
    backward_penalty, backward_candidates = _select_ridge(H1, H0, fitting_mask, validation_mask, ridge_grid)
    R = fit_orthogonal_forward(H0[train_mask], H1[train_mask])
    A = fit_ridge_map(H0[train_mask], H1[train_mask], forward_penalty)
    B = fit_ridge_map(H1[train_mask], H0[train_mask], backward_penalty)
    transforms = {"orthogonal_forward": R, "linear_forward": A, "linear_backward": B}
    readouts = {"orthogonal_transport": R.T @ W0, "linear_transport": B @ W0}
    train_spectra = {"healthy": spectrum_metrics(H0[train_mask]), "collapsed": spectrum_metrics(H1[train_mask])}
    map_spectra = {name: spectrum_metrics(mapping) for name, mapping in transforms.items()}
    condition = map_spectra["linear_forward"]["condition_number"]
    identifiable = train_spectra["healthy"]["numerical_rank"] == d and train_spectra["collapsed"]["numerical_rank"] == d
    invertible = identifiable and condition is not None and condition <= inverse_condition_limit
    if invertible:
        readouts["inverse_forward_transport"] = np.linalg.solve(A, W0)

    logits = {
        "healthy_original": H0 @ W0,
        "collapsed_original": H1 @ W1,
        "healthy_readout_on_collapsed": H1 @ W0,
        "collapsed_readout_on_healthy": H0 @ W1,
    }
    logits.update({name: H1 @ readout for name, readout in readouts.items()})
    reconstructions = {
        "orthogonal_forward": (H0 @ R, H1),
        "orthogonal_backward": (H1 @ R.T, H0),
        "linear_forward": (H0 @ A, H1),
        "linear_backward": (H1 @ B, H0),
        "healthy_roundtrip": (H0 @ A @ B, H0),
        "collapsed_roundtrip": (H1 @ B @ A, H1),
    }
    report = {
        "schema_version": 1,
        "shape": {"examples": n, "features": d, "classes": int(W0.shape[1])},
        "split": {
            "original_train_count": int(train_mask.sum()),
            "heldout_count": int(heldout_mask.sum()),
            "inner_fit_count": int(fitting_mask.sum()),
            "inner_validation_count": int(validation_mask.sum()),
            "inner_validation_indices": np.flatnonzero(validation_mask).tolist(),
            "validation_seed": int(validation_seed),
        },
        "selection": {
            "criterion": "inner-validation residual Frobenius error, without labels",
            "forward_relative_ridge": forward_penalty,
            "backward_relative_ridge": backward_penalty,
            "forward_candidates": forward_candidates,
            "backward_candidates": backward_candidates,
        },
        "training_representation_spectra": train_spectra,
        "map_spectra": map_spectra,
        "inverse_transport": {
            "performed": bool(invertible),
            "training_features_full_column_rank": bool(identifiable),
            "condition_limit": float(inverse_condition_limit),
            "interpretation": "invertible fitted map" if invertible else "linear reconstruction; invertible transport is unestablished",
        },
        "classification": {},
        "reconstruction": {},
        "healthy_function_reconstruction": {},
        "current_readout_discrepancy": {},
    }
    for split_name, mask in (("train", train_mask), ("heldout", heldout_mask)):
        report["classification"][split_name] = {name: classification_metrics(value[mask], labels[mask]) for name, value in logits.items()}
        report["reconstruction"][split_name] = {
            name: reconstruction_metrics(
                prediction[mask], target[mask],
                prediction_train_mean=prediction[train_mask].mean(axis=0),
                target_train_mean=target[train_mask].mean(axis=0),
            ) for name, (prediction, target) in reconstructions.items()
        }
        report["healthy_function_reconstruction"][split_name] = {
            name: logit_comparison_metrics(logits[name][mask], logits["healthy_original"][mask])
            for name in readouts
        }
        report["current_readout_discrepancy"][split_name] = {
            name: logit_comparison_metrics(logits["collapsed_original"][mask], logits[name][mask])
            for name in readouts
        }
    return AlignmentResult(report=report, transforms=transforms, transported_readouts=readouts)
