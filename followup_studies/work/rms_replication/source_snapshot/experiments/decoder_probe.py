"""Bounded, bias-free multiclass logistic probes of frozen representations.

Requires NumPy and SciPy. Feature scaling and regularization selection use only
original training examples. Reported convergence is the optimizer's actual status.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from alignment_core import classification_metrics


DEFAULT_REGULARIZATION_GRID = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)


@dataclass
class ProbeResult:
    readout: np.ndarray
    scaled_readout: np.ndarray
    report: dict
    feature_matrix: Optional[np.ndarray] = None


def _loss_and_gradient(flat_weights, features, labels, n_classes, regularization):
    """Mean cross-entropy plus 0.5 * lambda * ||W||_F^2, with exact gradient."""
    weights = flat_weights.reshape(features.shape[1], n_classes)
    logits = features @ weights
    rows = np.arange(len(features))
    wrong_delta = logits - logits[rows, labels, None]
    wrong_delta[rows, labels] = -np.inf
    per_example_loss = np.logaddexp(0.0, logsumexp(wrong_delta, axis=1))
    loss = np.mean(per_example_loss)
    residual = np.exp(wrong_delta - per_example_loss[:, None])
    # Compute the target gradient from the wrong-class mass. Subtracting one
    # from a near-unit target probability loses small but valid gradients.
    residual[rows, labels] = -residual.sum(axis=1)
    gradient = features.T @ residual / len(features)
    if regularization:
        loss += 0.5 * regularization * np.sum(weights ** 2)
        gradient += regularization * weights
    return float(loss), gradient.ravel()


def _training_scale(features):
    scale = float(np.sqrt(np.mean(features ** 2)))
    if not np.isfinite(scale):
        raise ValueError("training feature RMS is non-finite")
    return scale if scale > 0 else 1.0


def _feature_preconditioner(features, mode, relative_floor, native_dtype):
    """Return an invertible feature transform, inverse, and training-only audit."""
    scale = _training_scale(features)
    d = features.shape[1]
    if mode == "rms":
        return np.eye(d) / scale, np.eye(d) * scale, {
            "mode": "rms", "global_rms": scale, "centered": False,
            "definition": "H_transformed = H / training_feature_global_rms",
            "dimension": d, "capacity_preserved": True,
        }
    _, singular, Vt = np.linalg.svd(features, full_matrices=len(features) < d)
    if len(singular) < d:
        singular = np.pad(singular, (0, d - len(singular)))
    largest = float(singular[0])
    # A zero matrix has no identifiable representation directions, but the
    # invertible identity transform still preserves its original linear capacity.
    floor = max(largest * relative_floor, np.finfo(np.float64).tiny)
    regularized_singular = np.maximum(singular, floor) if largest else np.ones(d) * np.sqrt(len(features))
    factors = np.sqrt(len(features)) / regularized_singular
    matrix = (Vt.T * factors) @ Vt
    inverse = (Vt.T / factors) @ Vt
    eps_native = np.finfo(np.dtype(native_dtype)).eps
    cutoffs = sorted(set([float(eps_native), float(d * eps_native), float(max(features.shape) * eps_native), 1e-6, 1e-5, 1e-4]))
    transformed = features @ matrix
    covariance_error = np.linalg.norm(transformed.T @ transformed / len(features) - np.eye(d)) / np.sqrt(d)
    report = {
        "mode": "whiten", "global_rms": scale, "centered": False,
        "definition": "P = V diag(sqrt(n_train) / max(singular_value, floor)) V.T; H_transformed = H @ P",
        "readout_conversion": "W_original = P @ W_transformed",
        "dimension": d, "capacity_preserved": True,
        "relative_singular_floor": relative_floor,
        "absolute_singular_floor": float(floor),
        "floored_singular_values": int(np.sum(singular < floor)),
        "singular_values": singular.tolist(),
        "numerical_rank_float64": int(np.sum(singular > largest * max(features.shape) * np.finfo(np.float64).eps)),
        "native_feature_dtype_declaration": str(np.dtype(native_dtype)),
        "rank_sensitivity": [{"relative_cutoff": cutoff, "rank": int(np.sum(singular > largest * cutoff))} for cutoff in cutoffs],
        "preconditioner_condition_number": float(factors.max() / factors.min()),
        "transformed_covariance_relative_identity_error": float(covariance_error),
        "regularization_prior": "L2 on transformed coefficients equals ||P^-1 W_original||_F^2; with no flooring this is covariance-weighted L2 in original coordinates.",
    }
    return matrix, inverse, report


def _fit(features, labels, n_classes, regularization, max_iter, initial_readout,
         feature_transform="rms", whitening_relative_floor=1e-6, native_feature_dtype="float64"):
    matrix, inverse, transform_report = _feature_preconditioner(features, feature_transform, whitening_relative_floor, native_feature_dtype)
    scale = transform_report["global_rms"]
    scaled = np.ascontiguousarray(features / scale if feature_transform == "rms" else features @ matrix)
    shape = (features.shape[1], n_classes)
    initial = np.zeros(shape, dtype=np.float64) if initial_readout is None else (
        initial_readout * scale if feature_transform == "rms" else inverse @ initial_readout
    )
    initial_objective, _ = _loss_and_gradient(initial.ravel(), scaled, labels, n_classes, regularization)
    optimized = minimize(
        _loss_and_gradient,
        initial.ravel(),
        args=(scaled, labels, n_classes, regularization),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iter),
            "maxfun": int(3 * max_iter),
            "maxls": 20,
            "maxcor": 10,
            "ftol": 1e-10,
            "gtol": 1e-7,
        },
    )
    scaled_readout = optimized.x.reshape(shape)
    readout = scaled_readout / scale if feature_transform == "rms" else matrix @ scaled_readout
    if not np.isfinite(readout).all() or not np.isfinite(optimized.fun):
        raise FloatingPointError("logistic probe optimization produced non-finite values")
    report = {
        "converged": bool(optimized.success),
        "status": int(optimized.status),
        "message": str(optimized.message),
        "iterations": int(optimized.nit),
        "function_evaluations": int(optimized.nfev),
        "iteration_limit": int(max_iter),
        "function_evaluation_limit": int(3 * max_iter),
        "initial_objective": initial_objective,
        "final_objective": float(optimized.fun),
        "gradient_infinity_norm": float(np.linalg.norm(optimized.jac, ord=np.inf)),
        "gradient_l2_norm": float(np.linalg.norm(optimized.jac)),
        "training_feature_global_rms": scale,
        "feature_transform": transform_report,
        "scaled_readout_frobenius_norm": float(np.linalg.norm(scaled_readout)),
        "original_coordinate_readout_frobenius_norm": float(np.linalg.norm(readout)),
    }
    return readout, scaled_readout, report, matrix


def fit_logistic_probe(
    H, y, train_mask, heldout_mask, *,
    n_classes: Optional[int] = None,
    inner_validation_mask: Optional[np.ndarray] = None,
    validation_fraction: float = 0.2,
    validation_seed: int = 0,
    regularization_grid: Sequence[float] = DEFAULT_REGULARIZATION_GRID,
    max_iter: int = 100,
    refit_max_iter: int = 200,
    initial_readout: Optional[np.ndarray] = None,
    feature_transform: str = "rms",
    whitening_relative_floor: float = 1e-6,
    native_feature_dtype: Optional[str] = None,
):
    """Fit a fresh bias-free decoder with train-only validation and bounded effort.

    ``readout`` maps the original unscaled H directly to logits. Each candidate
    starts at zero unless initial_readout is supplied explicitly. Regularization
    is selected by the lowest inner-validation cross-entropy, with smaller lambda
    breaking exact ties. The chosen configuration is independently refit on all
    original training rows. The held-out data never affect a fitted parameter.

    Pass the known class count when some task classes may be absent from training.
    The default uses one global training RMS scalar. Optional uncentered SVD
    whitening preserves full linear capacity and applies a positive singular-value
    floor without dropping dimensions. Each inner fit learns its own transform.
    With nonzero regularization, whitening changes the coefficient prior; an
    unregularized fit isolates preconditioning of the original CE objective.
    Unregularized separable problems may exhaust their budget without convergence;
    report['refit']['converged'] and the candidate statuses retain that outcome.
    """
    original_dtype = np.asarray(H).dtype
    H = np.asarray(H, dtype=np.float64)
    labels = np.asarray(y)
    if H.ndim != 2 or min(H.shape) == 0 or not np.isfinite(H).all():
        raise ValueError("H must be a finite nonempty N x d matrix")
    if labels.shape != (len(H),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y must be an integer vector aligned with H")
    masks = {}
    for name, value in (("train", train_mask), ("heldout", heldout_mask)):
        mask = np.asarray(value)
        if mask.dtype != np.bool_ or mask.shape != (len(H),) or not mask.any():
            raise ValueError(f"{name}_mask must be a nonempty explicit boolean mask")
        masks[name] = mask
    train, heldout = masks["train"], masks["heldout"]
    if np.any(train & heldout):
        raise ValueError("train and heldout masks must be disjoint")
    if n_classes is None:
        n_classes = int(labels[train].max()) + 1
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 2:
        raise ValueError("n_classes must be an integer of at least two")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels exceed the specified class range; pass the known n_classes")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if not isinstance(max_iter, (int, np.integer)) or not isinstance(refit_max_iter, (int, np.integer)) or min(max_iter, refit_max_iter) < 1:
        raise ValueError("iteration limits must be positive integers")
    if feature_transform not in ("rms", "whiten"):
        raise ValueError("feature_transform must be 'rms' or 'whiten'")
    if not np.isfinite(whitening_relative_floor) or not 0 < whitening_relative_floor <= 1:
        raise ValueError("whitening_relative_floor must lie in (0, 1]")
    native_feature_dtype = str(original_dtype) if native_feature_dtype is None else str(native_feature_dtype)
    if np.dtype(native_feature_dtype).kind != "f":
        raise ValueError("native_feature_dtype must identify a floating-point dtype")
    grid = tuple(float(value) for value in regularization_grid)
    if not grid or any(not np.isfinite(value) or value < 0 for value in grid):
        raise ValueError("regularization_grid must contain finite nonnegative values")
    if initial_readout is not None:
        initial_readout = np.asarray(initial_readout, dtype=np.float64)
        if initial_readout.shape != (H.shape[1], n_classes) or not np.isfinite(initial_readout).all():
            raise ValueError("initial_readout must be a finite d x n_classes matrix")
    if inner_validation_mask is None:
        indices = np.flatnonzero(train)
        if len(indices) < 2:
            raise ValueError("at least two original training examples are required")
        indices = np.random.default_rng(validation_seed).permutation(indices)
        count = min(len(indices) - 1, max(1, int(np.ceil(len(indices) * validation_fraction))))
        validation = np.zeros(len(H), dtype=bool)
        validation[indices[:count]] = True
    else:
        validation = np.asarray(inner_validation_mask)
        if validation.dtype != np.bool_ or validation.shape != (len(H),) or not validation.any() or np.any(validation & ~train):
            raise ValueError("inner_validation_mask must be a nonempty subset of train_mask")
    inner_fit = train & ~validation
    if not inner_fit.any():
        raise ValueError("inner validation must leave rows for candidate training")

    candidates = []
    for penalty in grid:
        readout, _, optimization_report, _ = _fit(
            H[inner_fit], labels[inner_fit], n_classes, penalty, max_iter, initial_readout,
            feature_transform, whitening_relative_floor, native_feature_dtype,
        )
        candidates.append({
            "regularization": penalty,
            "validation": classification_metrics(H[validation] @ readout, labels[validation]),
            "optimization": optimization_report,
        })
    selected = min(candidates, key=lambda candidate: (candidate["validation"]["cross_entropy"], candidate["regularization"]))
    penalty = selected["regularization"]
    readout, scaled_readout, refit_report, matrix = _fit(
        H[train], labels[train], n_classes, penalty, refit_max_iter, initial_readout,
        feature_transform, whitening_relative_floor, native_feature_dtype,
    )
    report = {
        "schema_version": 1,
        "model": "bias-free multinomial logistic regression",
        "initialization": "zero" if initial_readout is None else "supplied readout",
        "selection_criterion": "lowest inner-validation cross-entropy; exact ties use smaller regularization",
        "selected_regularization": penalty,
        "regularization_definition": "mean CE + 0.5 * lambda * squared Frobenius norm of transformed-coordinate readout",
        "feature_transform": {
            **refit_report["feature_transform"],
            "original_training_feature_global_rms": refit_report["training_feature_global_rms"],
        },
        "split": {
            "original_train_count": int(train.sum()),
            "heldout_count": int(heldout.sum()),
            "inner_fit_count": int(inner_fit.sum()),
            "inner_validation_count": int(validation.sum()),
            "inner_validation_indices": np.flatnonzero(validation).tolist(),
            "validation_seed": int(validation_seed),
        },
        "candidates": candidates,
        "refit": refit_report,
        "classification": {
            "train": classification_metrics(H[train] @ readout, labels[train]),
            "heldout": classification_metrics(H[heldout] @ readout, labels[heldout]),
        },
        "limitation": "A failed bounded fit does not establish that task information is absent; inspect convergence and the healthy comparator.",
    }
    return ProbeResult(readout=readout, scaled_readout=scaled_readout, report=report, feature_matrix=matrix)


# Integration-friendly name used by the checkpoint analysis pipeline.
fit_decoder = fit_logistic_probe
