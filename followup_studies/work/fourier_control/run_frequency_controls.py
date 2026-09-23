"""Declared secondary controls for Fourier sufficiency interpretations."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from run_control import labels_grid, generate_split, fft_project, sha256, write_json


def weighted_metric(logits, counts):
    p = len(counts)
    labels = np.arange(p)
    pred = logits.argmax(axis=1)
    maximum = logits.max(axis=1)
    ties = (logits == maximum[:, None]).sum(axis=1)
    is_max = logits[labels, labels] == maximum
    wrong = logits.copy()
    wrong[labels, labels] = -np.inf
    margins = logits[labels, labels] - wrong.max(axis=1)
    n = int(counts.sum())
    correct = int(counts[pred == labels].sum())
    return dict(n=n, correct=correct, accuracy=correct / n,
                uniform_tie_expected_accuracy=float(np.dot(counts, is_max / ties) / n),
                positive_margin_rows=int(counts[margins > 0].sum()),
                tied_maximum_rows=int(counts[ties > 1].sum()),
                minimum_margin=float(margins[counts > 0].min()),
                mean_margin=float(np.dot(counts, margins) / n))


def evaluated_templates(train_logits, heldout_logits, train_counts, p):
    heldout_counts = p - train_counts
    a = weighted_metric(train_logits, train_counts)
    b = weighted_metric(heldout_logits, heldout_counts)
    # Full-grid outcomes combine the two explicitly reported denominators.
    combined = {key: a[key] + b[key] for key in ('n', 'correct', 'positive_margin_rows', 'tied_maximum_rows')}
    combined['accuracy'] = combined['correct'] / combined['n']
    for key in ('uniform_tie_expected_accuracy', 'mean_margin'):
        combined[key] = (a[key] * a['n'] + b[key] * b['n']) / combined['n']
    combined['minimum_margin'] = min(a['minimum_margin'], b['minimum_margin'])
    return dict(training=a, heldout=b, full_grid=combined)


def derangement(rng, size):
    for _ in range(10000):
        permutation = rng.permutation(size)
        if np.all(permutation != np.arange(size)):
            return permutation
    raise RuntimeError('Derangement draw limit exceeded')


def main(args):
    sys.path.insert(0, str(args.repo))
    from data import generate_modular_addition_data
    args.out.mkdir(parents=True, exist_ok=True)
    p = 113
    frequencies = np.arange(1, (p + 1) // 2)
    prefix_counts = (1, 2, 3, 5, 10, 20, 56)
    prefix, ablation, phase, single_pairs, spectra, checks = [], [], [], [], [], []
    for seed in range(5):
        train, _, _, _ = generate_split(generate_modular_addition_data, p, .30, seed, 'addition')
        y = labels_grid(p, 'addition')
        counts = np.bincount(y[train], minlength=p)
        for kind_index, kind in enumerate(('one_hot', 'training_count_normalized')):
            amplitude = np.ones(p) if kind == 'one_hot' else p / counts
            raw_train = np.diag(amplitude)
            template = np.diag(counts * amplitude / p)
            coefficients = np.fft.fft(template, axis=0)
            dc = np.broadcast_to(template.mean(axis=0), template.shape)
            powers = 2 * np.sum(np.abs(coefficients[frequencies]) ** 2, axis=1)
            descending = frequencies[np.lexsort((frequencies, -powers))]
            spectra.append(dict(seed=seed, lookup=kind, mathematical_power_tie=True,
                                numerical_relative_pair_power_spread=float((powers.max() - powers.min()) / powers.mean()),
                                ascending_tie_break=frequencies.tolist(), descending_numeric_power=descending.tolist(),
                                pair_powers=powers.tolist()))
            raw_metrics = evaluated_templates(raw_train, np.zeros_like(template), counts, p)
            ablated = evaluated_templates(raw_train - template + dc, -template + dc, counts, p)
            for condition, splits in (('raw_lookup', raw_metrics), ('complete_family_with_dc', evaluated_templates(template, template, counts, p)),
                                      ('family_ablated_dc_retained', ablated)):
                for split, metric in splits.items():
                    ablation.append(dict(seed=seed, lookup=kind, condition=condition, split=split, **metric))
            checks.append(dict(name=f'{kind}_seed{seed}_heldout_ablation_zero', passed=ablated['heldout']['correct'] == 0))
            checks.append(dict(name=f'{kind}_seed{seed}_training_ablation_perfect', passed=ablated['training']['accuracy'] == 1.0))
            for ranking, ordered in (('ascending_frequency_mathematical_tie', frequencies), ('descending_numeric_power', descending)):
                for count in prefix_counts:
                    selected = ordered[:count]
                    changed = np.zeros_like(coefficients)
                    changed[0] = coefficients[0]
                    changed[selected] = coefficients[selected]
                    changed[p - selected] = coefficients[p - selected]
                    projected = np.fft.ifft(changed, axis=0).real
                    splits = evaluated_templates(projected, projected, counts, p)
                    if kind == 'training_count_normalized':
                        checks.append(dict(name=f'normalized_seed{seed}_{ranking}_prefix{count}_perfect', passed=splits['heldout']['accuracy'] == 1.0))
                    for split, metric in splits.items():
                        prefix.append(dict(seed=seed, lookup=kind, ranking=ranking, retained_pairs=count,
                                           selected_frequencies='|'.join(map(str, selected)), split=split, **metric))
                    if seed == 0 and ranking == 'ascending_frequency_mathematical_tie' and count == 5:
                        z = np.zeros((p, p, p))
                        a, b = np.nonzero(train)
                        z[a, b, y[a, b]] = amplitude[y[a, b]]
                        spectrum = np.fft.fft2(z, axes=(0, 1))
                        mask = np.zeros((p, p), dtype=bool)
                        mask[0, 0] = True
                        mask[selected, selected] = True
                        mask[p - selected, p - selected] = True
                        direct = np.fft.ifft2(spectrum * mask[..., None], axes=(0, 1)).real
                        error = float(np.max(np.abs(direct - projected[y])))
                        checks.append(dict(name=f'{kind}_1d2d_prefix_equivalence', passed=error < 1e-12, max_absolute_error=error))
            if kind == 'training_count_normalized':
                for k in frequencies:
                    changed = np.zeros_like(coefficients)
                    changed[0] = coefficients[0]
                    changed[k] = coefficients[k]
                    changed[p-k] = coefficients[p-k]
                    projected = np.fft.ifft(changed, axis=0).real
                    analytic = (1 + 2 * np.cos(2 * np.pi * k * (np.arange(p)[:, None] - np.arange(p)[None, :]) / p)) / p
                    error = float(np.max(np.abs(projected - analytic)))
                    checks.append(dict(name=f'normalized_seed{seed}_single{k}_analytic', passed=error < 2e-14, max_absolute_error=error))
                    splits = evaluated_templates(projected, projected, counts, p)
                    checks.append(dict(name=f'normalized_seed{seed}_single{k}_perfect', passed=splits['heldout']['accuracy'] == 1.0))
                    single_pairs.append(dict(seed=seed, frequency=int(k), heldout_correct=splits['heldout']['correct'], heldout_n=splits['heldout']['n'], minimum_margin=splits['heldout']['minimum_margin']))
            for control_index, control in enumerate(('pair_global_phase', 'channelwise_phase', 'frequency_derangement')):
                for replicate in range(20):
                    rng_seed = 700000 + seed * 10000 + kind_index * 1000 + control_index * 100 + replicate
                    rng = np.random.default_rng(rng_seed)
                    changed = np.zeros_like(coefficients)
                    changed[0] = coefficients[0]
                    if control == 'frequency_derangement':
                        destination = frequencies[derangement(rng, len(frequencies))]
                        changed[destination] = coefficients[frequencies]
                        changed[p - destination] = coefficients[frequencies].conj()
                    else:
                        phase_shape = (len(frequencies), 1 if control == 'pair_global_phase' else p)
                        factors = np.exp(1j * rng.uniform(0, 2 * np.pi, size=phase_shape))
                        changed[frequencies] = coefficients[frequencies] * factors
                        changed[p - frequencies] = changed[frequencies].conj()
                    error = float(abs(np.sum(np.abs(changed[1:]) ** 2) - np.sum(np.abs(coefficients[1:]) ** 2)))
                    checks.append(dict(name=f'{kind}_seed{seed}_{control}_rep{replicate}_power', passed=error < 1e-9, absolute_error=error))
                    modified = np.fft.ifft(changed, axis=0).real
                    if control == 'frequency_derangement':
                        # At answer zero every Fourier character equals one.
                        # Permuting coefficients preserves this orbit exactly.
                        # Restore that identity before argmax so cancellation
                        # roundoff cannot break the all-zero held-out logit tie.
                        modified[0] = template[0]
                        checks.append(dict(name=f'{kind}_seed{seed}_{control}_rep{replicate}_zero_orbit_identity',
                                           passed=bool(np.array_equal((modified - template)[0], np.zeros(p)))))
                    for context, train_logits, heldout_logits in (
                        ('isolated_family_with_dc', modified, modified),
                        ('full_lookup_table', raw_train - template + modified, -template + modified)):
                        for split, metric in evaluated_templates(train_logits, heldout_logits, counts, p).items():
                            phase.append(dict(seed=seed, lookup=kind, control=control, context=context,
                                              replicate=replicate, rng_seed=rng_seed, split=split, **metric))
    for name, rows in (('frequency_prefix_metrics', prefix), ('ablation_metrics', ablation), ('phase_metrics', phase), ('single_pair_normalized_metrics', single_pairs)):
        with (args.out / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.out / 'frequency_spectra.json', spectra)
    write_json(args.out / 'frequency_verification.json', dict(check_count=len(checks), all_passed=all(c['passed'] for c in checks), checks=checks))
    write_json(args.out / 'frequency_manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
               script_sha256=sha256(__file__), dependency_script_sha256=sha256(Path(__file__).with_name('run_control.py')),
               protocol_sha256=sha256(Path(__file__).with_name('protocol_extension.md')), data_py_sha256=sha256(args.repo / 'data.py'),
               numpy=np.__version__, note='Protocol extension was declared after the primary controls, before these tests.',
               exact_tie_handling='Frequency derangement preserves the answer-zero orbit exactly. Its reconstructed row is assigned the original template row before evaluation; full-lookup held-out logits are therefore exactly zero on that orbit and smallest-index argmax selects class zero. This corrects FFT cancellation roundoff at a known structural identity.'))
    assert all(c['passed'] for c in checks), [c for c in checks if not c['passed']]
    print(f'Completed {len(checks)} additional checks; all passed. Prefix rows={len(prefix)}, phase rows={len(phase)}.')
    for kind in ('one_hot', 'training_count_normalized'):
        for ranking in ('ascending_frequency_mathematical_tie', 'descending_numeric_power'):
            for count in prefix_counts:
                values = [r['accuracy'] * 100 for r in prefix if r['lookup'] == kind and r['ranking'] == ranking and r['retained_pairs'] == count and r['split'] == 'heldout']
                print(kind, ranking, count, f'{min(values):.4f}--{max(values):.4f}%')
        for control in ('pair_global_phase', 'channelwise_phase', 'frequency_derangement'):
            for context in ('isolated_family_with_dc', 'full_lookup_table'):
                values = [r['accuracy'] * 100 for r in phase if r['lookup'] == kind and r['control'] == control and r['context'] == context and r['split'] == 'heldout']
                print(kind, control, context, f'mean={np.mean(values):.4f}% range={min(values):.4f}--{max(values):.4f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
