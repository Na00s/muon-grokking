"""Exact feature/readout decomposition and identity-versus-linear predictions.

All model counterfactuals use observed feature and readout matrices. Only the
linear-map predictor involves fitting, on the original training examples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

STUDY = Path(__file__).resolve().parent
EXPERIMENTS = STUDY.parent / 'experiments'
sys.path.insert(0, str(EXPERIMENTS))
from alignment_core import classification_metrics, logit_comparison_metrics
from geometry import factor, solve_factor, spectrum, thresholds


def centered(logits):
    return logits - logits.mean(axis=1, keepdims=True)


def analyze_bilinear(H0, H1, W0, W1, y, train, heldout):
    H0, H1, W0, W1 = [np.asarray(a, dtype=np.float64) for a in (H0, H1, W0, W1)]
    y, train, heldout = np.asarray(y), np.asarray(train), np.asarray(heldout)
    if train.dtype != bool or heldout.dtype != bool or np.any(train & heldout):
        raise ValueError('Explicit nonoverlapping boolean training and heldout masks required')
    if not train.any() or not heldout.any() or H0.shape != H1.shape or W0.shape != W1.shape:
        raise ValueError('Nonempty splits and shape-compatible checkpoints required')
    dh, dw = H1 - H0, W1 - W0
    z0, z1 = H0 @ W0, H1 @ W1
    delta = z1 - z0
    terms = {
        'readout_change': H0 @ dw,
        'feature_change': dh @ W0,
        'interaction': dh @ dw,
    }
    identity_error = delta - sum(terms.values())
    models = {
        'preceding': z0,
        'feature_change_only': H1 @ W0,
        'readout_change_only': H0 @ W1,
        'actual_later': z1,
    }
    decomp = factor(H0[train])
    tolerance = thresholds(H0[train].shape)['float64_solver']
    a = solve_factor(decomp, H1[train], tolerance)
    predictors = {
        'identity_features_with_later_readout': H0 @ W1,
        'fitted_linear_features_with_later_readout': (H0 @ a) @ W1,
    }
    report = {
        'identity': 'H1 W1 - H0 W0 = H0 deltaW + deltaH W0 + deltaH deltaW',
        'definitions': {'deltaH': 'H1-H0', 'deltaW': 'W1-W0'},
        'interpretation': 'Components describe endpoint changes. Squared norms require pairwise cross terms. Their ratios and accuracy differences are not additive causal attribution shares.',
        'class_centering': 'Subtract each example\'s mean across classes, preserving all predictions and margins.',
        'linear_predictor_fit': {
            'rows': 'Original training examples only; untouched test rows and no labels used for fitting.',
            'type': 'Uncentered, unregularized, rank-aware float64 SVD least squares',
            'source_spectrum': spectrum(H0[train], decomp),
            'mapping_spectrum': spectrum(a),
        },
        'splits': {},
    }
    for split, mask in [('train', train), ('heldout', heldout)]:
        dc = centered(delta[mask])
        term_values = {name: centered(value[mask]) for name, value in terms.items()}
        norms = {name: float(np.linalg.norm(value)) for name, value in term_values.items()}
        nd = float(np.linalg.norm(dc))
        scale = max(float(np.linalg.norm(centered(z0[mask]))),
                    float(np.linalg.norm(centered(z1[mask]))), sum(norms.values()), 1.0)
        near_zero_threshold = 64 * np.finfo(np.float64).eps * scale
        near_zero_delta = bool(nd <= near_zero_threshold)
        ratios = {name: {
            'class_centered_norm': norm,
            'norm_over_total_change': None if near_zero_delta else norm / nd,
            'squared_norm_over_total_change_squared': None if near_zero_delta else (norm / nd) ** 2,
        } for name, norm in norms.items()}
        cross_terms = {}
        names = list(terms)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                dot = float(np.sum(term_values[left] * term_values[right]))
                cross_terms[f'{left}__{right}'] = {
                    'twice_inner_product': 2 * dot,
                    'normalized_by_total_change_squared': None if near_zero_delta else 2 * dot / (nd * nd),
                    'cosine': float(np.clip(dot / (norms[left] * norms[right]), -1, 1)) if norms[left] and norms[right] else None,
                }
        actual_predictions = z1[mask].argmax(1)
        counterfactuals = {
            name: {
                'classification': classification_metrics(value[mask], y[mask]),
                'comparison_to_actual': logit_comparison_metrics(value[mask], z1[mask]),
                'argmax_agreement_with_actual': float(np.mean(value[mask].argmax(1) == actual_predictions)),
            } for name, value in models.items()
        }
        predictor_metrics = {
            name: {
                'classification': classification_metrics(value[mask], y[mask]),
                'prediction_error_to_actual': logit_comparison_metrics(value[mask], z1[mask]),
                'argmax_agreement_with_actual': float(np.mean(value[mask].argmax(1) == actual_predictions)),
            } for name, value in predictors.items()
        }
        norm_check = sum(norm ** 2 for norm in norms.values()) + sum(
            cross['twice_inner_product'] for cross in cross_terms.values())
        report['splits'][split] = {
            'n': int(mask.sum()), 'total_change_class_centered_norm': nd,
            'near_zero_total_change': near_zero_delta,
            'near_zero_threshold': float(near_zero_threshold),
            'components': ratios, 'pairwise_cross_terms': cross_terms,
            'squared_norm_identity_error_absolute': float(abs(nd ** 2 - norm_check)),
            'squared_norm_identity_relative_error': None if near_zero_delta else float(abs(nd ** 2 - norm_check) / nd ** 2),
            'logit_identity_error_max_abs': float(np.max(np.abs(identity_error[mask]))),
            'logit_identity_error_relative_to_total_change': None if near_zero_delta else float(np.linalg.norm(centered(identity_error[mask])) / nd),
            'counterfactuals_2x2': counterfactuals,
            'identity_vs_linear_prediction': predictor_metrics,
        }
    return report, a


def compact_row(name, report):
    d = report['splits']['heldout']
    models = d['counterfactuals_2x2']
    predictors = d['identity_vs_linear_prediction']
    row = {'pair': name, 'heldout_examples': d['n'],
           'delta_class_centered_norm': d['total_change_class_centered_norm']}
    for key, value in models.items():
        row[key + '_accuracy'] = value['classification']['accuracy']
    for key, value in d['components'].items():
        row[key + '_norm_over_delta'] = value['norm_over_total_change']
        row[key + '_squared_norm_over_delta_squared'] = value['squared_norm_over_total_change_squared']
    for key, value in d['pairwise_cross_terms'].items():
        row[key + '_normalized_cross_term'] = value['normalized_by_total_change_squared']
    for key, label in [('identity_features_with_later_readout', 'identity'),
                       ('fitted_linear_features_with_later_readout', 'linear')]:
        p = predictors[key]
        row[label + '_predicted_accuracy'] = p['classification']['accuracy']
        row[label + '_actual_class_agreement'] = p['argmax_agreement_with_actual']
        row[label + '_class_centered_relative_error_to_actual'] = p['prediction_error_to_actual']['class_centered_relative_error']
        row[label + '_cross_entropy'] = p['classification']['cross_entropy']
    row['squared_norm_identity_relative_error'] = d['squared_norm_identity_relative_error']
    row['logit_identity_error_max_abs'] = d['logit_identity_error_max_abs']
    return row


def run_one(features, out):
    data = dict(np.load(features))
    report, mapping = analyze_bilinear(**data)
    report['source_features'] = str(features.resolve())
    report['source_sha256'] = hashlib.sha256(features.read_bytes()).hexdigest()
    if (features.parent / 'metadata.json').exists():
        report['source_metadata'] = json.loads((features.parent / 'metadata.json').read_text())
    destination = out / features.parent.name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'bilinear.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.save(destination / 'forward_map.npy', mapping)
    print(json.dumps(compact_row(features.parent.name, report)), flush=True)


def summarize(out):
    rows = [compact_row(p.parent.name, json.loads(p.read_text()))
            for p in sorted(out.glob('*/bilinear.json'))]
    if not rows:
        return
    with (out / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(rows, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', nargs='+', type=Path)
    parser.add_argument('--discover-existing', action='store_true')
    parser.add_argument('--out', type=Path, default=STUDY / 'bilinear')
    args = parser.parse_args()
    paths = args.features or []
    if args.discover_existing:
        paths += list(EXPERIMENTS.glob('*/features.npz')) + list(STUDY.rglob('features.npz'))
    paths = sorted(set(p.resolve() for p in paths))
    names = [p.parent.name for p in paths]
    if len(names) != len(set(names)):
        raise ValueError('Source parent directory names must be unique within one output directory')
    for path in paths:
        run_one(path, args.out)
    summarize(args.out)


if __name__ == '__main__':
    main()
