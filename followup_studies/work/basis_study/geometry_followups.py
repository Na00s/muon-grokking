"""Adjacent-step pairs, matched healthy control, and fixed-reference timecourses."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

STUDY = Path(__file__).resolve().parent
EXP = STUDY.parent / 'experiments'
sys.path.insert(0, str(EXP))
from run_collapse import generate_modular_addition_data, make_model_optimizers, residuals, evaluate
from run_precision_branches import make_arm, run_branch
from alignment_core import classification_metrics, fit_orthogonal_forward
from geometry import analyze_pair, controls, factor, functional_decomposition, solve_factor, spectrum, thresholds


def dump_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False))


def read_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def extract(checkpoint_path, inputs, split_data):
    state = read_checkpoint(checkpoint_path)
    model, _ = make_model_optimizers(int(state['seed']), checkpoint=state)
    h = residuals(model, inputs).double().numpy()
    w = model.unembedding.weight.detach().T.double().numpy()
    train_x, train_y, test_x, test_y = split_data
    metrics = {'train': evaluate(model, train_x, train_y),
               'heldout': evaluate(model, test_x, test_y)}
    return state, h, w, metrics


def dataset(seed):
    data = generate_modular_addition_data(seed=seed)
    train_x, train_y, test_x, test_y = data
    inputs = torch.cat([train_x, test_x])
    labels = torch.cat([train_y, test_y]).numpy()
    train = np.arange(len(inputs)) < len(train_x)
    return data, inputs, labels, train, ~train


def prepare_pair(name, first, second, reference_role, require_healthy=False):
    out = STUDY / name
    out.mkdir(exist_ok=True)
    seed = int(read_checkpoint(first)['seed'])
    split_data, inputs, y, train, heldout = dataset(seed)
    s0, h0, w0, m0 = extract(first, inputs, split_data)
    s1, h1, w1, m1 = extract(second, inputs, split_data)
    if s1['seed'] != seed:
        raise ValueError('Pair seeds differ')
    if require_healthy and min(m0['heldout']['accuracy'], m1['heldout']['accuracy']) < .99:
        raise ValueError(f'Matched control fails preregistered >=99% endpoint criterion: {m0}, {m1}')
    metadata = {
        'seed': seed, 'first_checkpoint': str(Path(first).resolve()),
        'second_checkpoint': str(Path(second).resolve()),
        'first_step': s0['step'], 'second_step': s1['step'],
        'step_interval': s1['step'] - s0['step'], 'reference_role': reference_role,
        'first_metrics': m0, 'second_metrics': m1,
        'healthy_endpoint_criterion': 'Both test accuracies >=99%' if require_healthy else None,
        'dataset_sha256': hashlib.sha256(inputs.numpy().tobytes() + y.tobytes()).hexdigest(),
        'note': 'H0 denotes the first checkpoint. A preceding-step checkpoint may already have poor generalization. Geometry field names containing healthy retain the mathematical reference-role convention only.',
    }
    dump_json(out / 'metadata.json', metadata)
    np.savez_compressed(out / 'features.npz', H0=h0, H1=h1, W0=w0, W1=w1, y=y,
                        train=train, heldout=heldout)
    report, maps = analyze_pair(h0, h1, w0, w1, y, train, heldout)
    report['source'] = str((out / 'features.npz').resolve())
    report['pair_metadata'] = metadata
    report['controls'] = controls(h0, train, heldout)
    dump_json(out / 'geometry.json', report)
    np.savez_compressed(out / 'geometry_maps.npz', **maps)
    print(json.dumps({'pair': name, 'metadata': metadata}), flush=True)
    return out


def prepare_pairs():
    replay_pre = STUDY / 'seed0_preceding_step_replay'
    if not (replay_pre / 'final.pt').exists():
        run_branch(EXP / 'seed0_dense_capture' / 'pre_event_latest.pt', replay_pre,
                   arm='original32', steps=3, threads=1, eval_every=1, save_every=1)
    if read_checkpoint(replay_pre / 'final.pt')['step'] != 17493:
        raise ValueError('Unexpected seed0 preceding checkpoint step')
    # A fourth replayed update must reproduce the stored collapse state exactly.
    s = read_checkpoint(replay_pre / 'final.pt')
    model, opts, loss = make_arm(s, 'original32')
    from run_collapse import train_step
    train_x, train_y, _, _ = generate_modular_addition_data(seed=0)
    train_step(model, opts, train_x, train_y, loss_fn=loss)
    expected = read_checkpoint(EXP / 'seed0_dense_capture' / 'collapse.pt')
    matches = {k: torch.equal(v, expected['model_state_dict'][k])
               for k, v in model.state_dict().items()}
    dump_json(replay_pre / 'next_step_verification.json', {
        'expected_step': expected['step'], 'all_model_tensors_bitwise_equal': all(matches.values()),
        'tensor_matches': matches})
    if not all(matches.values()):
        raise RuntimeError('Seed0 adjacent replay does not match stored collapse exactly')
    prepare_pair('adjacent_seed4', EXP / 'seed4_one_step_attribution' / 'pre_step.pt',
                 EXP / 'seed4_dense_capture' / 'collapse.pt', 'preceding_step')
    prepare_pair('adjacent_seed0', replay_pre / 'final.pt',
                 EXP / 'seed0_dense_capture' / 'collapse.pt', 'preceding_step')
    replay_healthy = STUDY / 'seed0_matched_healthy_start_replay'
    if not (replay_healthy / 'final.pt').exists():
        run_branch(EXP / 'seed0_original' / 'event_window_016900.pt', replay_healthy,
                   arm='original32', steps=6, threads=1, eval_every=1, save_every=6)
    if read_checkpoint(replay_healthy / 'final.pt')['step'] != 16906:
        raise ValueError('Unexpected matched healthy starting step')
    prepare_pair('seed0_matched_healthy_16906_17200', replay_healthy / 'final.pt',
                 EXP / 'seed0_dense_capture' / 'pre_event_joint_healthy.pt',
                 'matched_294_update_healthy_endpoints', require_healthy=True)


def checkpoint_paths(seed):
    # Dense snapshots take precedence at duplicated steps; they are exact
    # continuations of the corresponding original training run.
    paths = list(sorted((EXP / f'seed{seed}_original').glob('event_window_*.pt')))
    paths += list(sorted((EXP / f'seed{seed}_dense_capture' / 'event_window').glob('step_*.pt')))
    # Add the already-selected adjacent-step reference and peak endpoint.
    if seed == 4:
        paths += [EXP / 'seed4_one_step_attribution' / 'pre_step.pt',
                  EXP / 'seed4_peak_replay' / 'final.pt']
    elif (STUDY / 'seed0_preceding_step_replay' / 'final.pt').exists():
        paths += [STUDY / 'seed0_preceding_step_replay' / 'final.pt']
    by_step = {}
    for p in paths:
        s = read_checkpoint(p)
        if s['seed'] != seed:
            raise ValueError('Mismatched seed in event-window source')
        by_step[int(s['step'])] = p
    return by_step


def timecourse(seed):
    out = STUDY / f'timecourse_seed{seed}'
    out.mkdir(exist_ok=True)
    pair = 'seed4_dense_event' if seed == 4 else 'seed0_dense_event'
    source = dict(np.load(EXP / pair / 'features.npz'))
    source_meta = json.loads((EXP / pair / 'metadata.json').read_text())
    h0, w0, y, train, heldout = [source[k] for k in ('H0', 'W0', 'y', 'train', 'heldout')]
    ref_step = source_meta['healthy_step']
    split_data, inputs, check_y, check_train, check_heldout = dataset(seed)
    np.testing.assert_array_equal(y, check_y)
    np.testing.assert_array_equal(train, check_train)
    forward_decomp = factor(h0[train])
    forward_tolerance = thresholds(h0[train].shape)['float64_solver']
    h0_mean = h0[train].mean(0)
    paths = checkpoint_paths(seed)
    reference_path = Path(source_meta['healthy_checkpoint'])
    paths.setdefault(ref_step, reference_path)
    metadata = {
        'seed': seed, 'fixed_reference_step': ref_step,
        'fixed_reference_checkpoint': str(reference_path),
        'selection': 'All saved dense and coarse event-window steps, the fixed healthy reference, the already-selected adjacent preceding step, and the existing seed4 peak endpoint. No test-selected hyperparameters.',
        'fits': 'Original training examples only; unregularized SVD OLS forward and backward, orthogonal Procrustes.',
        'time_offsets': 'Negative offsets precede the fixed reference.',
        'native_metrics': 'Original float32 model evaluation; mapped classifications use float64 products of saved float32 activations and weights.',
        'source_checkpoints': {str(k): str(v.resolve()) for k, v in sorted(paths.items())},
    }
    dump_json(out / 'metadata.json', metadata)
    details, rows = {}, []
    start = time.perf_counter()
    for step, path in sorted(paths.items()):
        _, h1, w1, native = extract(path, inputs, split_data)
        forward = solve_factor(forward_decomp, h1[train], forward_tolerance)
        backward = solve_factor(factor(h1[train]), h0[train], thresholds(h1[train].shape)['float64_solver'])
        ortho = fit_orthogonal_forward(h0[train], h1[train])
        prediction = h1 @ backward
        centered_error = np.linalg.norm((prediction[heldout] - prediction[train].mean(0)) - (h0[heldout] - h0_mean)) / np.linalg.norm(h0[heldout] - h0_mean)
        functional = functional_decomposition(h0, h1, w0, w1, forward, y, train, heldout)
        details[str(step)] = functional
        f = functional['splits']['heldout']
        c = f['counterfactuals']
        row = {
            'seed': seed, 'step': step, 'reference_step': ref_step, 'offset': step - ref_step,
            'native_train_accuracy': native['train']['accuracy'],
            'native_heldout_accuracy': native['heldout']['accuracy'],
            'native_train_loss64': native['train']['loss64'],
            'native_heldout_loss64': native['heldout']['loss64'],
            'backward_centered_heldout_error': float(centered_error),
            'forward_map_condition': spectrum(forward)['condition_number'],
            'backward_map_condition': spectrum(backward)['condition_number'],
            'orthogonal_repair_accuracy': classification_metrics((h1 @ ortho.T @ w0)[heldout], y[heldout])['accuracy'],
            'backward_repair_accuracy': classification_metrics((prediction @ w0)[heldout], y[heldout])['accuracy'],
            'old_head_accuracy': classification_metrics((h1 @ w0)[heldout], y[heldout])['accuracy'],
            'basis_predicted_accuracy': c['basis_predicted']['classification']['accuracy'],
            'residual_only_accuracy': c['residual_only']['classification']['accuracy'],
            'readout_change_only_accuracy': c['readout_change_only']['classification']['accuracy'],
            'basis_argmax_agreement': c['basis_predicted']['argmax_agreement_with_actual'],
            'basis_actual_flip_recall': c['basis_predicted']['actual_flip_recall'],
            'basis_norm_over_delta': f['basis_norm_over_delta'],
            'residual_norm_over_delta': f['residual_norm_over_delta'],
            'normalized_cross_term': f['normalized_cross_term'],
            'decomposition_identity_error_max_abs': f['identity_error_max_abs'],
        }
        rows.append(row)
        print(json.dumps({'timecourse_seed': seed, 'step': step, 'test_accuracy': native['heldout']['accuracy'],
                          'backward_centered_error': centered_error}), flush=True)
    with (out / 'trajectory.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump_json(out / 'functional_decompositions.json', details)
    dump_json(out / 'completion.json', {'steps': len(rows), 'elapsed_seconds': time.perf_counter() - start})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-pairs', action='store_true')
    parser.add_argument('--timecourse', type=int, choices=[0, 4])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.prepare_pairs:
        prepare_pairs()
    if args.timecourse is not None:
        timecourse(args.timecourse)


if __name__ == '__main__':
    main()
