"""Geometric tests of a shared linear change of basis between checkpoints.

Rows are examples. The exact hypothesis H1 = H0 @ A for invertible A implies
equal column spaces. Oracle residuals deliberately use every example and are
descriptive lower bounds, never held-out prediction results. All predictive
maps fit the original training mask. Rank sensitivity is reported because the
activations were originally computed in float32, then saved as float64.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from alignment_core import classification_metrics, reconstruction_metrics, logit_comparison_metrics


def factor(x):
    return svd(np.asarray(x, dtype=np.float64), full_matrices=False,
               check_finite=True, lapack_driver='gesdd')


def thresholds(shape):
    return {
        'float64_solver': max(shape) * np.finfo(np.float64).eps,
        'eps32': float(np.finfo(np.float32).eps),
        'relative_1e-6': 1e-6,
        'relative_1e-5': 1e-5,
        'feature_count_times_eps32': shape[1] * float(np.finfo(np.float32).eps),
        'relative_1e-4': 1e-4,
        'max_shape_times_eps32': max(shape) * float(np.finfo(np.float32).eps),
    }


def rank_at(s, tolerance):
    return int(np.count_nonzero(s > s[0] * tolerance)) if s[0] > 0 else 0


def solve_factor(decomp, y, tolerance):
    u, s, vt = decomp
    rank = rank_at(s, tolerance)
    return (vt[:rank].T / s[:rank]) @ (u[:, :rank].T @ y)


def spectrum(x, decomp=None):
    _, s, _ = factor(x) if decomp is None else decomp
    energy = s * s
    return {
        'shape': list(x.shape), 'singular_values': s.tolist(),
        'condition_number': float(s[0] / s[-1]) if s[-1] > 0 else None,
        'stable_rank': float(energy.sum() / energy[0]) if energy[0] else 0.0,
        'ranks': {key: rank_at(s, value) for key, value in thresholds(x.shape).items()},
        'thresholds': thresholds(x.shape),
    }


def errors(prediction, target, pred_train_mean, target_train_mean):
    return reconstruction_metrics(prediction, target,
        prediction_train_mean=pred_train_mean, target_train_mean=target_train_mean)


def column_space_diagnostic(x, y):
    """Oracle subspace geometry on the supplied rows, no predictive claims.

    Principal-angle cosines equal canonical correlations after whitening at
    each declared numerical rank. If ranks differ, additional dimensions have
    no counterpart; report rank difference and symmetric projection distances.
    """
    dx, dy = factor(x), factor(y)
    ux, sx, _ = dx
    uy, sy, _ = dy
    result = {'x_spectrum': spectrum(x, dx), 'y_spectrum': spectrum(y, dy), 'by_rank_tolerance': {}}
    # These exact projection residuals avoid error amplification in an explicit A.
    for key, rtol in thresholds(x.shape).items():
        rx, ry = rank_at(sx, rtol), rank_at(sy, rtol)
        qx, qy = ux[:, :rx], uy[:, :ry]
        cosines = np.clip(svd(qx.T @ qy, compute_uv=False), 0.0, 1.0)
        angles = np.degrees(np.arccos(cosines))
        residual_y = y - qx @ (qx.T @ y)
        residual_x = x - qy @ (qy.T @ x)
        # Both matrices are represented in a common example-index space.
        sum_cos2 = float(np.sum(cosines ** 2))
        result['by_rank_tolerance'][key] = {
            'relative_tolerance': float(rtol), 'rank_x': rx, 'rank_y': ry,
            'rank_difference': abs(rx - ry),
            'canonical_correlations': cosines.tolist(),
            'principal_angles_degrees': angles.tolist(),
            'angle_min_degrees': float(angles.min()) if len(angles) else None,
            'angle_median_degrees': float(np.median(angles)) if len(angles) else None,
            'angle_max_degrees': float(angles.max()) if len(angles) else None,
            'angles_above_1_degree': int(np.count_nonzero(angles > 1)),
            'angles_above_10_degrees': int(np.count_nonzero(angles > 10)),
            'projector_frobenius_distance': float(np.sqrt(max(0.0, rx + ry - 2 * sum_cos2))),
            'y_outside_x_column_space_relative': float(np.linalg.norm(residual_y) / np.linalg.norm(y)),
            'x_outside_y_column_space_relative': float(np.linalg.norm(residual_x) / np.linalg.norm(x)),
        }
    return result


def functional_decomposition(H0, H1, W0, W1, mapping, y, train, heldout):
    """Exact additive logit identity with a train-only forward representation map.

    D = H1 W1 - H0 W0 = H0(A W1-W0) + (H1-H0 A)W1.
    Component norms are not additive variance shares because their cross term
    can be large and negative. Classification counterfactuals are descriptive.
    """
    z0, z1 = H0 @ W0, H1 @ W1
    basis_z = (H0 @ mapping) @ W1
    residual_z = (H1 - H0 @ mapping) @ W1
    delta = z1 - z0
    basis_delta = basis_z - z0
    identity_error = delta - basis_delta - residual_z
    prediction_logits = {
        'healthy': z0,
        'actual_later': z1,
        'basis_predicted': basis_z,
        'residual_only': z0 + residual_z,
        'readout_change_only': H0 @ W1,
    }
    report = {
        'identity': 'H1 W1 - H0 W0 = H0(A W1-W0) + (H1-H0 A) W1',
        'map_fit': 'Forward, uncentered, unregularized SVD OLS, original training rows only.',
        'interpretation': 'Norm ratios are nonadditive. Squared ratios plus normalized cross term sum to one. Counterfactual logits establish functional sufficiency at these endpoints, not an optimizer-level causal history.',
        'splits': {},
    }
    for split, mask in [('train', train), ('heldout', heldout)]:
        def centered(a):
            a = a[mask]
            return a - a.mean(axis=1, keepdims=True)
        dc, bc, rc = [centered(a) for a in (delta, basis_delta, residual_z)]
        nd, nb, nr = [float(np.linalg.norm(a)) for a in (dc, bc, rc)]
        denominator = nd ** 2
        healthy_predictions = z0[mask].argmax(axis=1)
        actual_predictions = z1[mask].argmax(axis=1)
        actual_flip = actual_predictions != healthy_predictions
        models = {}
        for label, logits in prediction_logits.items():
            pred = logits[mask].argmax(axis=1)
            flips = pred != healthy_predictions
            tp = int(np.count_nonzero(flips & actual_flip))
            fp = int(np.count_nonzero(flips & ~actual_flip))
            fn = int(np.count_nonzero(~flips & actual_flip))
            models[label] = {
                'classification': classification_metrics(logits[mask], y[mask]),
                'argmax_agreement_with_actual': float(np.mean(pred == actual_predictions)),
                'flips_from_healthy': int(flips.sum()),
                'actual_flip_overlap': tp,
                'extra_flips': fp,
                'missed_actual_flips': fn,
                'actual_flip_precision': tp / (tp + fp) if tp + fp else None,
                'actual_flip_recall': tp / (tp + fn) if tp + fn else None,
                'same_destination_on_actual_flips': float(np.mean(pred[actual_flip] == actual_predictions[actual_flip])) if actual_flip.any() else None,
            }
        report['splits'][split] = {
            'actual_flip_count': int(actual_flip.sum()),
            'delta_class_centered_norm': nd,
            'basis_component_class_centered_norm': nb,
            'residual_component_class_centered_norm': nr,
            'basis_norm_over_delta': nb / nd if nd else None,
            'residual_norm_over_delta': nr / nd if nd else None,
            'basis_squared_norm_over_delta_squared': nb * nb / denominator if denominator else None,
            'residual_squared_norm_over_delta_squared': nr * nr / denominator if denominator else None,
            'normalized_cross_term': float(2 * np.sum(bc * rc) / denominator) if denominator else None,
            'component_cosine': float(np.sum(bc * rc) / (nb * nr)) if nb and nr else None,
            'identity_error_max_abs': float(np.max(np.abs(identity_error[mask]))),
            'identity_error_relative_to_delta': float(np.linalg.norm(identity_error[mask]) / np.linalg.norm(delta[mask])) if np.linalg.norm(delta[mask]) else None,
            'counterfactuals': models,
        }
    return report


def analyze_pair(H0, H1, W0, W1, y, train, heldout):
    H0, H1, W0, W1 = [np.asarray(v, dtype=np.float64) for v in (H0, H1, W0, W1)]
    y, train, heldout = np.asarray(y), np.asarray(train), np.asarray(heldout)
    if H0.shape != H1.shape or train.dtype != bool or heldout.dtype != bool:
        raise ValueError('Equal-shaped representations and explicit boolean masks required')
    if np.any(train & heldout) or not train.any() or not heldout.any():
        raise ValueError('Training and heldout masks must be nonempty and disjoint')
    masks = {'train': train, 'heldout': heldout}
    means = [h[train].mean(axis=0) for h in (H0, H1)]
    report = {
        'protocol': {
            'hypothesis': 'H1 = H0 @ A, with a single invertible d-by-d A shared by all examples.',
            'fit_rows': 'Original training mask only. No fitting or rank threshold selection uses heldout rows.',
            'oracle_rows': 'All examples, explicitly transductive. Descriptive minimum-residual diagnostics only.',
            'fitting_dtype': 'float64', 'activation_origin_dtype': 'float32',
            'affine_note': 'The affine model adds a learned intercept and is a distinct, weaker hypothesis.',
            'cca_note': 'Canonical correlations describe column-space geometry and depend on truncation rank.',
        },
        'dimensions': {'examples': len(H0), 'features': H0.shape[1],
                       'train': int(train.sum()), 'heldout': int(heldout.sum())},
        'classification': {split: {
            'native_healthy': classification_metrics((H0 @ W0)[mask], y[mask]),
            'native_later': classification_metrics((H1 @ W1)[mask], y[mask]),
            'old_readout_on_later': classification_metrics((H1 @ W0)[mask], y[mask]),
        } for split, mask in masks.items()},
        'train_only_maps': {},
        'oracle_geometry': {},
    }
    arrays = {}
    for direction, x, target, xm, tm, readout in (
        ('forward', H0, H1, means[0], means[1], W1),
        ('backward', H1, H0, means[1], means[0], W0),
    ):
        direction_report = {}
        for model in ('linear', 'affine'):
            centered = model == 'affine'
            xs = x - xm if centered else x
            ts = target - tm if centered else target
            decomp = factor(xs[train])
            model_report = {'source_training_spectrum': spectrum(xs[train], decomp), 'rank_sensitivity': {}}
            for key, rtol in thresholds(xs[train].shape).items():
                mapping = solve_factor(decomp, ts[train], rtol)
                intercept = tm - xm @ mapping if centered else np.zeros(target.shape[1])
                prediction = x @ mapping + intercept
                pmean = prediction[train].mean(axis=0)
                native_function = target @ readout
                transformed_function = prediction @ readout
                item = {
                    'relative_tolerance': rtol,
                    'source_fit_rank': rank_at(decomp[1], rtol),
                    'mapping_spectrum': spectrum(mapping),
                    'intercept_norm': float(np.linalg.norm(intercept)),
                    'reconstruction': {split: errors(prediction[mask], target[mask], pmean, tm)
                                       for split, mask in masks.items()},
                    'function_reconstruction': {split: logit_comparison_metrics(transformed_function[mask], native_function[mask])
                                                for split, mask in masks.items()},
                    'transported_classification': {split: classification_metrics(transformed_function[mask], y[mask])
                                                  for split, mask in masks.items()},
                }
                model_report['rank_sensitivity'][key] = item
                if key == 'float64_solver':
                    arrays[f'{direction}_{model}_mapping'] = mapping
                    arrays[f'{direction}_{model}_intercept'] = intercept
            direction_report[model] = model_report
        report['train_only_maps'][direction] = direction_report
    # Compute both uncentered and mean-centered column spaces. Centering uses
    # original training means. An affine map could explain the centered case.
    for name, x, target in (
        ('uncentered', H0, H1),
        ('training_mean_centered', H0 - means[0], H1 - means[1]),
    ):
        oracle = column_space_diagnostic(x, target)
        oracle['label'] = 'Transductive all-example geometric diagnostic; not held-out prediction evidence.'
        report['oracle_geometry'][name] = oracle
    report['functional_decomposition'] = functional_decomposition(
        H0, H1, W0, W1, arrays['forward_linear_mapping'], y, train, heldout)
    return report, arrays


def simple_fit_report(x, target, train, heldout):
    decomp = factor(x[train])
    rtol = thresholds(x[train].shape)['float64_solver']
    a = solve_factor(decomp, target[train], rtol)
    pred = x @ a
    pm, tm = pred[train].mean(0), target[train].mean(0)
    return {
        'map_condition_number': spectrum(a)['condition_number'],
        'train': errors(pred[train], target[train], pm, tm),
        'heldout': errors(pred[heldout], target[heldout], pm, tm),
    }


def controls(H0, train, heldout, *, seed=0):
    """Representational recovery and native rounding floors on observed data."""
    rng = np.random.default_rng(seed)
    H0 = np.asarray(H0, dtype=np.float64)
    q, _ = np.linalg.qr(rng.normal(size=(H0.shape[1], H0.shape[1])))
    p, _ = np.linalg.qr(rng.normal(size=(H0.shape[1], H0.shape[1])))
    result = {'random_seed': seed, 'identity': simple_fit_report(H0, H0, train, heldout), 'planted': {}}
    for condition in (1, 10, 100):
        a = (q * np.geomspace(1.0, float(condition), H0.shape[1])) @ p.T
        target64 = H0 @ a
        target32 = (H0.astype(np.float32) @ a.astype(np.float32)).astype(np.float64)
        record = {'target_condition_number': condition, 'double_precision': {}, 'float32_matmul': {}}
        for label, target in [('double_precision', target64), ('float32_matmul', target32)]:
            record[label] = {
                'forward': simple_fit_report(H0, target, train, heldout),
                'backward': simple_fit_report(target, H0, train, heldout),
                'all_example_forward_oracle_relative_error': float(
                    np.linalg.norm(target - (lambda u: u @ (u.T @ target))(factor(H0)[0])) / np.linalg.norm(target)),
            }
        result['planted'][str(condition)] = record
    # Only training correspondences are shuffled. Held-out rows remain untouched.
    corrupted_targets = H0.copy()
    corrupted_targets[train] = H0[train][rng.permutation(train.sum())]
    d = factor(H0[train])
    a = solve_factor(d, corrupted_targets[train], thresholds(H0[train].shape)['float64_solver'])
    prediction = H0 @ a
    result['permuted_training_correspondence'] = {
        split: errors(prediction[mask], H0[mask], prediction[train].mean(0), H0[train].mean(0))
        for split, mask in [('train', train), ('heldout', heldout)]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    data = dict(np.load(args.features))
    report, arrays = analyze_pair(**data)
    report['source'] = str(args.features.resolve())
    report['source_sha256'] = hashlib.sha256(args.features.read_bytes()).hexdigest()
    if args.controls:
        report['controls'] = controls(data['H0'], data['train'], data['heldout'])
    report['elapsed_seconds'] = time.perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'geometry.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(args.out / 'geometry_maps.npz', **arrays)
    output = {'source': str(args.features), 'elapsed_seconds': report['elapsed_seconds'],
        'backward_linear_heldout': report['train_only_maps']['backward']['linear']['rank_sensitivity']['float64_solver']['reconstruction']['heldout'],
        'oracle_fullrank': report['oracle_geometry']['uncentered']['by_rank_tolerance']['float64_solver']}
    output['oracle_fullrank'] = {k: v for k, v in output['oracle_fullrank'].items()
                                if k not in ('canonical_correlations', 'principal_angles_degrees')}
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
