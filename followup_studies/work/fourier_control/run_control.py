"""Task-structured Fourier averaging controls.

The synthetic suite needs only the public repository and NumPy/Torch.
Optional checkpoint diagnostics reuse archived feature pairs without training.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def labels_grid(p, operation):
    a, b = np.indices((p, p))
    return (a + b if operation == 'addition' else a - b) % p


def fft_project(h, operation, include_dc=True):
    p = h.shape[0]
    transformed = np.fft.fft2(h, axes=(0, 1))
    k, ell = np.indices((p, p))
    mask = (k == ell) if operation == 'addition' else ((k + ell) % p == 0)
    if not include_dc:
        mask[0, 0] = False
    projected = np.fft.ifft2(transformed * mask[..., None], axes=(0, 1))
    assert np.max(np.abs(projected.imag)) < 1e-10 * max(1.0, np.max(np.abs(h)))
    return projected.real


def orbit_average(h, operation, donor_mask=None):
    p = h.shape[0]
    labels = labels_grid(p, operation).reshape(-1)
    values = h.reshape(p * p, -1)
    selected = np.ones(p * p, dtype=bool) if donor_mask is None else donor_mask.reshape(-1)
    sums = np.zeros((p, values.shape[1]), dtype=np.float64)
    np.add.at(sums, labels[selected], values[selected])
    counts = np.bincount(labels[selected], minlength=p)
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] != 0)
    return means[labels].reshape(h.shape), sums, counts


def translated_average(h, operation):
    p = h.shape[0]
    result = np.zeros_like(h, dtype=np.float64)
    # roll(..., -t, axis=0)[a,b] = h[a+t,b].
    for t in range(p):
        result += np.roll(np.roll(h, -t, axis=0), t if operation == 'addition' else -t, axis=1)
    return result / p


def metrics(logits, labels, mask):
    z = logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)]
    y = labels.reshape(-1)[mask.reshape(-1)]
    prediction = z.argmax(axis=1)
    maxima = z.max(axis=1)
    tie_count = (z == maxima[:, None]).sum(axis=1)
    correct_is_max = z[np.arange(len(z)), y] == maxima
    other = z.copy()
    other[np.arange(len(z)), y] = -np.inf
    margin = z[np.arange(len(z)), y] - other.max(axis=1)
    return dict(n=int(len(y)), correct=int((prediction == y).sum()),
                accuracy=float((prediction == y).mean()),
                uniform_tie_expected_accuracy=float(np.mean(correct_is_max / tie_count)),
                tied_maximum_rows=int((tie_count > 1).sum()),
                positive_margin_rows=int((margin > 0).sum()),
                minimum_margin=float(margin.min()), mean_margin=float(margin.mean()))


def all_metrics(logits, labels, train):
    return {name: metrics(logits, labels, mask) for name, mask in (
        ('training', train), ('heldout', ~train), ('full_grid', np.ones_like(train)))}


def generate_split(generate, p, fraction, seed, operation):
    tx, ty, vx, vy = generate(modulus=p, train_fraction=fraction, seed=seed, operation=operation)
    inputs = np.concatenate([tx.numpy(), vx.numpy()])
    labels = np.concatenate([ty.numpy(), vy.numpy()])
    indices = inputs[:, 0] * p + inputs[:, 1]
    train = np.zeros(p * p, dtype=bool)
    train[indices[:len(tx)]] = True
    assert np.array_equal(labels_grid(p, operation).reshape(-1)[indices], labels)
    return train.reshape(p, p), inputs, labels, indices


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(args):
    sys.path.insert(0, str(args.repo))
    from data import generate_modular_addition_data
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checks = []

    def compare(name, actual, expected, rtol=2e-12, atol=2e-12):
        passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
        checks.append(dict(name=name, passed=passed, max_absolute_error=float(np.max(np.abs(actual - expected))), rtol=rtol, atol=atol))
        if not passed:
            raise AssertionError(name)

    def condition(name, value):
        checks.append(dict(name=name, passed=bool(value)))
        if not value:
            raise AssertionError(name)

    p, fraction = 113, .30
    rng = np.random.default_rng(9222026)
    identity_results = []
    for modulus in (7, 113):
        h = rng.normal(size=(modulus, modulus, 5))
        w = rng.normal(size=(5, 9))
        bias = rng.normal(size=9)
        for operation in ('addition', 'subtraction'):
            average, _, _ = orbit_average(h, operation)
            fourier = fft_project(h, operation)
            compare(f'identity_p{modulus}_{operation}_fft_orbits', fourier, average)
            compare(f'identity_p{modulus}_{operation}_translations', translated_average(h, operation), average)
            compare(f'identity_p{modulus}_{operation}_idempotence', fft_project(fourier, operation), fourier)
            compare(f'identity_p{modulus}_{operation}_affine_readout', fourier @ w + bias, fft_project(h @ w + bias, operation))
            compare(f'identity_p{modulus}_{operation}_without_dc', fft_project(h, operation, False), average - h.mean(axis=(0, 1)))
            identity_results.append(dict(modulus=modulus, operation=operation, max_fft_orbit_error=float(np.max(np.abs(fourier - average)))))

    memorizer_results = []
    memorizer_rows = []
    for operation in ('addition', 'subtraction'):
        y = labels_grid(p, operation)
        for seed in range(5):
            train, inputs, ordered_y, _ = generate_split(generate_modular_addition_data, p, fraction, seed, operation)
            z = np.zeros((p, p, p), dtype=np.float64)
            a, b = np.nonzero(train)
            z[a, b, y[a, b]] = 1.0
            projected, sums, counts = orbit_average(z, operation)
            training_donors, _, train_counts = orbit_average(z, operation, train)
            heldout_donors, _, test_counts = orbit_average(z, operation, ~train)
            fft = fft_project(z, operation)
            compare(f'memorizer_{operation}_seed{seed}_fft_orbits', fft, projected)
            compare(f'memorizer_{operation}_seed{seed}_projected_counts', projected[a, b, y[a, b]], train_counts[y[a, b]] / p)
            condition(f'memorizer_{operation}_seed{seed}_heldout_donors_zero', np.count_nonzero(heldout_donors) == 0)
            covered = train_counts > 0
            condition(f'memorizer_{operation}_seed{seed}_all_orbits_covered', np.all(covered))
            condition(f'memorizer_{operation}_seed{seed}_full_projection_correct', np.all(projected.argmax(axis=-1) == y))
            permutation = np.random.default_rng(6000 + seed).permutation(p)
            permuted_z = np.zeros_like(z)
            permuted_z[a, b, permutation[y[a, b]]] = 1.0
            permuted_projected, _, _ = orbit_average(permuted_z, operation)
            conditions = {
                'raw_lookup': (z, y),
                'complete_family_with_dc': (projected, y),
                'complete_family_without_dc': (projected - z.mean(axis=(0, 1)), y),
                'training_donors_only': (training_donors, y),
                'heldout_donors_only': (heldout_donors, y),
                'opposite_family_with_dc': (orbit_average(z, 'subtraction' if operation == 'addition' else 'addition')[0], y),
                'permuted_orbit_labels_raw': (permuted_z, permutation[y]),
                'permuted_orbit_labels_projected': (permuted_projected, permutation[y]),
            }
            m = int(train.sum())
            probability_uncovered = float(np.prod([(p * p - m - i) / (p * p - i) for i in range(p)]))
            result = dict(seed=seed, operation=operation, p=p, nominal_training_fraction=fraction,
                          training_examples=m, heldout_examples=int((~train).sum()),
                          dataset_sha256=hashlib.sha256(inputs.tobytes() + ordered_y.tobytes()).hexdigest(),
                          training_orbit_counts=train_counts.tolist(), heldout_orbit_counts=test_counts.tolist(),
                          covered_orbits=int(covered.sum()), total_orbits=p,
                          minimum_training_donors=int(train_counts.min()), maximum_training_donors=int(train_counts.max()),
                          probability_one_orbit_uncovered_fixed_size=probability_uncovered,
                          union_bound_probability_any_orbit_uncovered=p * probability_uncovered,
                          conditions={label: all_metrics(values, labels, train) for label, (values, labels) in conditions.items()})
            memorizer_results.append(result)
            for label, split_results in result['conditions'].items():
                for split, metric in split_results.items():
                    memorizer_rows.append(dict(seed=seed, operation=operation, condition=label, split=split, **metric))
            print(f'memorizer {operation} seed={seed} raw heldout={result["conditions"]["raw_lookup"]["heldout"]["correct"]}/{int((~train).sum())}, projected=100%, donor_min={train_counts.min()}', flush=True)

    checkpoint_results, inputs_manifest = [], []
    if args.work_root:
        pairs = ['adjacent_seed0', 'seed1_adjacent_event', 'seed2_adjacent_event', 'seed3_adjacent_event', 'adjacent_seed4']
        for seed, pair in enumerate(pairs):
            directory = args.work_root / 'basis_study' / pair
            path = directory / 'features.npz'
            metadata_path = directory / 'metadata.json'
            metadata = json.loads(metadata_path.read_text())
            inputs_manifest.append(dict(pair=pair, path=str(path), sha256=sha256(path), metadata_sha256=sha256(metadata_path)))
            train, inputs, ordered_y, indices = generate_split(generate_modular_addition_data, p, fraction, seed, 'addition')
            y = labels_grid(p, 'addition')
            data = np.load(path)
            condition(f'{pair}_ordered_labels_match', np.array_equal(data['y'], ordered_y))
            condition(f'{pair}_training_mask_matches', np.array_equal(data['train'], np.arange(p * p) < int(train.sum())))
            for endpoint in (0, 1):
                h = np.zeros((p * p, data[f'H{endpoint}'].shape[1]), dtype=np.float64)
                h[indices] = data[f'H{endpoint}']
                h = h.reshape(p, p, -1)
                w = data[f'W{endpoint}'].astype(np.float64)
                z = h @ w
                projected, _, _ = orbit_average(z, 'addition')
                h_projected = fft_project(h, 'addition')
                compare(f'{pair}_endpoint{endpoint}_feature_logit_projection', h_projected @ w, projected, rtol=2e-10, atol=2e-10)
                training_donors, _, _ = orbit_average(z, 'addition', train)
                heldout_donors, heldout_sums, heldout_counts = orbit_average(z, 'addition', ~train)
                condition(f'{pair}_heldout_donors_allow_leave_one_out', heldout_counts.min() > 1)
                # Every row's LOO value removes that row only if it is a held-out donor.
                loo = (heldout_sums[y] - z * (~train)[..., None]) / (heldout_counts[y] - (~train))[..., None]
                conditions = dict(raw=z, complete_family_with_dc=projected,
                                  training_donors_only=training_donors,
                                  heldout_donors_only=heldout_donors,
                                  heldout_donors_leave_one_out=loo)
                step = metadata.get('first_step' if endpoint == 0 else 'second_step', metadata.get('healthy_step' if endpoint == 0 else 'collapsed_step'))
                result = dict(seed=seed, pair=pair, endpoint=endpoint, step=step,
                              inference='Float64 matrix product of archived float32-origin residuals and head.',
                              conditions={label: all_metrics(values, y, train) for label, values in conditions.items()})
                checkpoint_results.append(result)
                print(f'checkpoint {pair} endpoint={endpoint} step={step}: ' + ', '.join(f'{label}={100*values["heldout"]["accuracy"]:.4f}%' for label, values in result['conditions'].items()), flush=True)

    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                    python=sys.version, numpy=np.__version__, torch=torch.__version__, platform=platform.platform(),
                    script_sha256=sha256(__file__), data_py_sha256=sha256(args.repo / 'data.py'),
                    protocol_sha256=sha256(Path(__file__).with_name('protocol.md')),
                    tie_convention='Smallest class index, matching NumPy/PyTorch argmax; exact ties additionally evaluated under uniform random tie breaking.',
                    raw_memorizer_note='Raw held-out logits are identically zero; deterministic accuracy is the held-out frequency of class zero.',
                    checkpoint_inputs=inputs_manifest)
    write_json(args.out / 'manifest.json', manifest)
    write_json(args.out / 'identity_checks.json', identity_results)
    write_json(args.out / 'memorizer_results.json', memorizer_results)
    write_json(args.out / 'checkpoint_donor_results.json', checkpoint_results)
    write_json(args.out / 'verification.json', dict(check_count=len(checks), all_passed=all(c['passed'] for c in checks), checks=checks))
    with (args.out / 'memorizer_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(memorizer_rows[0]))
        writer.writeheader()
        writer.writerows(memorizer_rows)
    if checkpoint_results:
        rows = [dict(seed=r['seed'], pair=r['pair'], endpoint=r['endpoint'], step=r['step'], condition=condition, split=split, **metric)
                for r in checkpoint_results for condition, splits in r['conditions'].items() for split, metric in splits.items()]
        with (args.out / 'checkpoint_donor_metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'Completed {len(checks)} checks, all passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, help='Root containing basis_study; omit for synthetic controls only.')
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
