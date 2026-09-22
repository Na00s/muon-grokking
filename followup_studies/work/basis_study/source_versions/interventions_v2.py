"""Train-only basis compensation and checkpoint continuation interventions.

Rows are examples: H1 ~= H0 A, so the corresponding readout is A^-1 W0.
Every continuation restores all optimizer buffers and changes only the head.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / 'experiments'
sys.path.insert(0, str(EXPERIMENTS))
import run_collapse as original
from run_precision_branches import configure_runtime, make_arm, file_sha256

EVENTS = {
    'seed4_first': EXPERIMENTS / 'seed4_dense_event',
    'seed0_first': EXPERIMENTS / 'seed0_dense_event',
    'seed4_peak': EXPERIMENTS / 'seed4_peak_event',
}
CONTINUATION_ARMS = ['original', 'old_head', 'orthogonal', 'inverse_gl', 'random_correction_0']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def metrics(logits, labels, healthy_logits):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    ref = np.asarray(healthy_logits, dtype=np.float64)
    correct = z[np.arange(len(y)), y]
    wrong = z.copy()
    wrong[np.arange(len(y)), y] = -np.inf
    margins = correct - wrong.max(axis=1)
    # Stable CE, including very small positive losses for correctly classified rows.
    d = z - correct[:, None]
    correct_maximum = np.max(d, axis=1) <= 0
    d[np.arange(len(y)), y] = -np.inf
    ce = np.empty(len(y))
    ce[correct_maximum] = np.log1p(np.exp(d[correct_maximum]).sum(axis=1))
    other = ~correct_maximum
    if np.any(other):
        maximum = np.maximum(0, np.max(d[other], axis=1))
        ce[other] = maximum + np.log(np.exp(-maximum) + np.exp(d[other] - maximum[:, None]).sum(axis=1))
    centered = z - z.mean(axis=1, keepdims=True)
    centered_ref = ref - ref.mean(axis=1, keepdims=True)
    error = np.linalg.norm(centered - centered_ref)
    return dict(accuracy=float(np.mean(np.argmax(z, axis=1) == y)), cross_entropy=float(ce.mean()),
                minimum_margin=float(margins.min()), mean_margin=float(margins.mean()),
                margin_quantiles={str(q): float(np.quantile(margins, q)) for q in [0, .01, .05, .25, .5, .75, .95, 1]},
                centered_healthy_logit_relative_error=float(error / np.linalg.norm(centered_ref)),
                centered_healthy_logit_rmse=float(np.sqrt(np.mean((centered - centered_ref) ** 2))))


def fit_heads(H0_train, H1_train, W0, W1):
    """Inputs contain original training rows only; no labels are used."""
    A, _, rank0, s0 = np.linalg.lstsq(H0_train, H1_train, rcond=None)
    B, _, rank1, s1 = np.linalg.lstsq(H1_train, H0_train, rcond=None)
    U, _, Vt = np.linalg.svd(H0_train.T @ H1_train, full_matrices=False)
    R = U @ Vt
    sa = np.linalg.svd(A, compute_uv=False)
    rank_A = int(np.linalg.matrix_rank(A))
    if rank_A != A.shape[0]:
        raise ValueError('Full-rank forward map is required for inverse_gl')
    heads = {'original': W1.copy(), 'old_head': W0.copy(),
             'orthogonal': R.T @ W0, 'inverse_gl': np.linalg.solve(A, W0),
             'backward_ols': B @ W0,
             'old_head_norm_matched': W0 * (np.linalg.norm(W1) / np.linalg.norm(W0))}
    distill, _, _, _ = np.linalg.lstsq(H1_train, H0_train @ W0, rcond=None)
    heads['healthy_logit_distillation'] = distill
    correction_norm = np.linalg.norm(heads['inverse_gl'] - W1)
    for seed in range(3):
        noise = np.random.default_rng(seed).normal(size=W1.shape)
        heads[f'random_correction_{seed}'] = W1 + noise * correction_norm / np.linalg.norm(noise)
    report = dict(training_rows=len(H0_train), hidden_dimension=H0_train.shape[1],
                  healthy_rank=int(rank0), current_rank=int(rank1), forward_rank=rank_A,
                  healthy_condition=float(s0[0] / s0[-1]), current_condition=float(s1[0] / s1[-1]),
                  forward_condition=float(sa[0] / sa[-1]), forward_singular_values=sa.tolist(),
                  healthy_singular_values=s0.tolist(), current_singular_values=s1.tolist(),
                  backward_distillation_relative_difference=float(np.linalg.norm(B @ W0 - distill) / np.linalg.norm(distill)),
                  inverse_backward_relative_difference=float(np.linalg.norm(heads['inverse_gl'] - heads['backward_ols']) / np.linalg.norm(heads['backward_ols'])),
                  inverse_head_correction_norm=float(correction_norm),
                  fit_dtype='float64 from recorded float32 residuals',
                  regularization='None; numpy.linalg.lstsq rcond=None',
                  condition_warning='Numerical full rank and invertibility on this dataset do not establish an exact global change of basis.')
    return heads, {'forward_A': A, 'backward_B': B, 'orthogonal_R': R}, report


def synthetic_checks():
    rng = np.random.default_rng(937)
    H0 = rng.normal(size=(600, 12))
    W0 = rng.normal(size=(12, 7))
    Q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    V, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    report = {}
    for name, A in [('orthogonal', Q), ('invertible_cond10', Q @ np.diag(np.geomspace(1, 10, 12)) @ V.T)]:
        H1 = H0 @ A
        W1 = np.linalg.solve(A, W0)
        heads, maps, info = fit_heads(H0[:400], H1[:400], W0, W1)
        error = np.linalg.norm(H1[400:] @ heads['inverse_gl'] - H0[400:] @ W0) / np.linalg.norm(H0[400:] @ W0)
        assert error < 1e-11
        assert np.linalg.norm(maps['forward_A'] - A) / np.linalg.norm(A) < 1e-11
        if name == 'orthogonal':
            orth_error = np.linalg.norm(H1[400:] @ heads['orthogonal'] - H0[400:] @ W0) / np.linalg.norm(H0[400:] @ W0)
            assert orth_error < 1e-11
        else:
            orth_error = None
        report[name] = dict(inverse_heldout_logit_relative_error=float(error), orthogonal_heldout_logit_relative_error=orth_error)
    example = np.array([[30., 0.], [0., 30.], [3., 2.]])
    labels = np.array([0, 1, 1])
    measured = metrics(example, labels, example)
    expected_ce = np.logaddexp(0, np.array([-30., -30., 1.])).mean()
    assert abs(measured['cross_entropy'] - expected_ce) < 1e-14
    report['stable_metric_cross_entropy_error'] = float(abs(measured['cross_entropy'] - expected_ce))
    report['passed'] = True
    return report


def prepare_event(event, destination):
    source_dir = EVENTS[event]
    metadata = json.loads((source_dir / 'metadata.json').read_text())
    arrays = np.load(source_dir / 'features.npz')
    H0, H1, W0, W1, labels, train, heldout = [arrays[k] for k in ['H0', 'H1', 'W0', 'W1', 'y', 'train', 'heldout']]
    heads, maps, fit_report = fit_heads(H0[train], H1[train], W0, W1)
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / 'heads.npz', **heads)
    np.savez_compressed(destination / 'maps.npz', **maps)
    source = torch.load(metadata['collapsed_checkpoint'], map_location='cpu', weights_only=False)
    model, optimizers, _ = make_arm(source, 'original32')
    train_x, train_y, test_x, test_y = original.generate_modular_addition_data(seed=metadata['seed'])
    assert np.array_equal(np.concatenate([train_y.numpy(), test_y.numpy()]), labels)
    assert np.array_equal(train, np.arange(len(labels)) < len(train_y))
    input_hash = hashlib.sha256(torch.cat([train_x, test_x]).numpy().tobytes() + labels.tobytes()).hexdigest()
    assert input_hash == metadata['dataset_sha256']
    healthy_logits = H0 @ W0
    report = {}
    for arm, W in heads.items():
        double_logits = H1 @ W
        with torch.no_grad():
            model.unembedding.weight.copy_(torch.from_numpy(W.T).to(dtype=torch.float32))
            native_logits = torch.cat([model(torch.cat([train_x, test_x])[i:i + 1024]) for i in range(0, len(labels), 1024)]).double().numpy()
        actual_W = model.unembedding.weight.detach().double().numpy().T
        report[arm] = dict(
            train=metrics(native_logits[train], labels[train], healthy_logits[train]),
            heldout=metrics(native_logits[heldout], labels[heldout], healthy_logits[heldout]),
            fitted_float64_heldout=metrics(double_logits[heldout], labels[heldout], healthy_logits[heldout]),
            head_norm=float(np.linalg.norm(actual_W)),
            head_change_norm=float(np.linalg.norm(actual_W - W1)),
            relative_head_change=float(np.linalg.norm(actual_W - W1) / np.linalg.norm(W1)),
            head_cast_relative_error=float(np.linalg.norm(actual_W - W) / np.linalg.norm(W)),
            full_logits_vs_feature_product_relative_error=float(np.linalg.norm(native_logits - H1 @ actual_W) / np.linalg.norm(native_logits)))
    metadata.update(event=event, source_feature_path=str(source_dir / 'features.npz'),
                    source_feature_sha256=file_sha256(source_dir / 'features.npz'),
                    collapsed_checkpoint_sha256=file_sha256(metadata['collapsed_checkpoint']),
                    healthy_checkpoint_sha256=file_sha256(metadata['healthy_checkpoint']),
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    head_fit_protocol='Fit maps on original training rows only, without labels. Readout interventions cast to float32; primary metrics use actual model forward pass.',
                    random_control='Three fixed Gaussian directions around collapsed W1, each matched to Frobenius norm of inverse_gl minus W1. Continuation uses seed 0 by advance specification.',
                    forward_map=fit_report)
    write_json(destination / 'metadata.json', metadata)
    write_json(destination / 'instant.json', report)
    return metadata


def monitored_step(model, optimizers, x, y, monitor=None):
    """Return pre-update train metrics while retaining the original update order."""
    model.train()
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    with torch.no_grad():
        value = dict(train_accuracy=float((logits.argmax(-1) == y).double().mean()), train_loss=float(loss.detach()))
    if monitor is not None:
        monitor(value)
    loss.backward()
    for opt in optimizers.values():
        opt.step()
    return value


def replay_checks(destination):
    meta = json.loads((EVENTS['seed4_first'] / 'metadata.json').read_text())
    healthy = torch.load(meta['healthy_checkpoint'], map_location='cpu', weights_only=False)
    expected = torch.load(meta['collapsed_checkpoint'], map_location='cpu', weights_only=False)
    model, optimizers, _ = make_arm(healthy, 'original32')
    tr_x, tr_y, te_x, te_y = original.generate_modular_addition_data(seed=healthy['seed'])
    for _ in range(int(expected['step']) - int(healthy['step'])):
        monitored_step(model, optimizers, tr_x, tr_y,
                       monitor=lambda value: original.evaluate(model, te_x, te_y))
    replay_state = original.snapshot(model, optimizers, expected['step'], expected['seed'])
    matches = {key: state_equal(replay_state[key], expected[key]) for key in ['model_state_dict', 'optimizer_state_dicts', 'torch_rng_state']}
    assert all(matches.values()), matches
    a, opts_a, _ = make_arm(expected, 'original32')
    b, opts_b, _ = make_arm(expected, 'original32')
    original.train_step(a, opts_a, tr_x, tr_y)
    monitored_step(b, opts_b, tr_x, tr_y, monitor=lambda value: original.evaluate(b, te_x, te_y))
    one_step = state_equal(a.state_dict(), b.state_dict()) and state_equal({k: v.state_dict() for k, v in opts_a.items()}, {k: v.state_dict() for k, v in opts_b.items()})
    assert one_step
    result = dict(synthetic=synthetic_checks(), source_replay_steps=int(expected['step']) - int(healthy['step']),
                  replay_exact_matches=matches, monitored_step_matches_source_update=one_step,
                  original_monitoring_preserves_update=True)
    write_json(destination, result)
    return result


def continue_arm(event_directory, arm, steps, optimizer_control='preserve'):
    configure_runtime(1)
    event_directory = Path(event_directory)
    metadata = json.loads((event_directory / 'metadata.json').read_text())
    source = torch.load(metadata['collapsed_checkpoint'], map_location='cpu', weights_only=False)
    heads = np.load(event_directory / 'heads.npz')
    model, optimizers, _ = make_arm(source, 'original32')
    W = heads[arm]
    with torch.no_grad():
        model.unembedding.weight.copy_(torch.from_numpy(W.T).to(dtype=torch.float32))
    head_optimizer = optimizers['unembedding_adamw']
    if optimizer_control == 'clear_head_state':
        head_optimizer.state.clear()
    elif optimizer_control == 'zero_head_first_moment':
        for state in head_optimizer.state.values():
            state['exp_avg'].zero_()
    elif optimizer_control != 'preserve':
        raise ValueError(f'Unknown optimizer control: {optimizer_control}')
    head_key = 'unembedding.weight'
    untouched_other_parameters = all(torch.equal(value, source['model_state_dict'][key]) for key, value in model.state_dict().items() if key != head_key)
    untouched_optimizers = state_equal({k: v.state_dict() for k, v in optimizers.items()}, source['optimizer_state_dicts'])
    other_optimizer_states_untouched = all(state_equal(value.state_dict(), source['optimizer_state_dicts'][key]) for key, value in optimizers.items() if key != 'unembedding_adamw')
    head_state_matches_specification = False
    actual_head_state = head_optimizer.state_dict()
    expected_head_state = copy.deepcopy(source['optimizer_state_dicts']['unembedding_adamw'])
    if optimizer_control == 'clear_head_state':
        expected_head_state['state'] = {}
    elif optimizer_control == 'zero_head_first_moment':
        for state in expected_head_state['state'].values():
            state['exp_avg'].zero_()
    head_state_matches_specification = state_equal(actual_head_state, expected_head_state)
    assert untouched_other_parameters and other_optimizer_states_untouched and head_state_matches_specification
    if optimizer_control == 'preserve':
        assert untouched_optimizers
    rng_unchanged = torch.equal(torch.get_rng_state(), source['torch_rng_state'])
    assert rng_unchanged
    out = event_directory / 'continuation' / arm
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(event=metadata['event'], arm=arm, seed=source['seed'], start_step=source['step'], steps=steps,
                source_checkpoint=metadata['collapsed_checkpoint'], source_checkpoint_sha256=metadata['collapsed_checkpoint_sha256'],
                only_head_parameters_changed=untouched_other_parameters, every_optimizer_state_preserved=untouched_optimizers,
                hidden_and_embedding_optimizer_states_preserved=other_optimizer_states_untouched,
                head_optimizer_state_matches_specification=head_state_matches_specification,
                head_optimizer_control=optimizer_control,
                head_adam_step_counter_reset=optimizer_control == 'clear_head_state',
                head_adam_second_moment_preserved=optimizer_control != 'clear_head_state',
                rng_preserved=rng_unchanged, train_monitor='Pre-update logits every training step',
                heldout_monitor='Every 10 local steps and at every train accuracy below 90%, including initial and final states',
                semantics='Readout robustness intervention with the explicitly stated head optimizer control. Hidden and embedding optimizer states are preserved. Adam states are not transformed covariantly, so this continuation is not a gauge-equivalence proof.',
                parameter_dtype='torch.float32', training_loss='torch.nn.functional.cross_entropy', ns_dtype='torch.float32',
                head_fitting='Original training rows only, no labels', device='cpu', threads=1,
                script_sha256=file_sha256(__file__))
    write_json(out / 'metadata.json', meta)
    torch.save(original.snapshot(model, optimizers, source['step'], source['seed']), out / 'start.pt')
    tr_x, tr_y, te_x, te_y = original.generate_modular_addition_data(seed=source['seed'])
    arrays = np.load(Path(metadata['source_feature_path']))
    reference = arrays['H0'] @ arrays['W0']
    reference_test = reference[arrays['heldout']]
    rows = []
    started = time.perf_counter()
    fields = ['local_step', 'step', 'train_accuracy', 'train_loss', 'heldout_measured', 'test_accuracy', 'test_cross_entropy', 'test_minimum_margin', 'test_centered_healthy_logit_relative_error', 'elapsed_seconds']
    with (out / 'trajectory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for local in range(steps + 1):
            def monitor(value):
                measured = local % 10 == 0 or value['train_accuracy'] < .9 or local == steps
                row = dict(local_step=local, step=source['step'] + local, **value, heldout_measured=measured,
                           test_accuracy=None, test_cross_entropy=None, test_minimum_margin=None,
                           test_centered_healthy_logit_relative_error=None, elapsed_seconds=time.perf_counter() - started)
                if measured:
                    training_mode = model.training
                    rng_before = torch.get_rng_state().clone()
                    with torch.no_grad():
                        model.eval()
                        z = torch.cat([model(te_x[i:i + 1024]) for i in range(0, len(te_x), 1024)]).double().numpy()
                        measured_metrics = metrics(z, te_y.numpy(), reference_test)
                    model.train(training_mode)
                    assert torch.equal(rng_before, torch.get_rng_state())
                    row.update(test_accuracy=measured_metrics['accuracy'], test_cross_entropy=measured_metrics['cross_entropy'],
                               test_minimum_margin=measured_metrics['minimum_margin'],
                               test_centered_healthy_logit_relative_error=measured_metrics['centered_healthy_logit_relative_error'])
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                if local % 100 == 0:
                    print(json.dumps(dict(event=metadata['event'], arm=arm, **row)), flush=True)
            if local == steps:
                with torch.no_grad():
                    model.train()
                    z = model(tr_x)
                    monitor(dict(train_accuracy=float((z.argmax(-1) == tr_y).double().mean()), train_loss=float(F.cross_entropy(z, tr_y))))
            else:
                monitored_step(model, optimizers, tr_x, tr_y, monitor)
    final = original.snapshot(model, optimizers, source['step'] + steps, source['seed'])
    torch.save(final, out / 'final.pt')
    measured = [r for r in rows if r['heldout_measured']]
    future = [r for r in measured if r['local_step'] > 0]
    recovered = next((r['local_step'] for r in future if r['train_accuracy'] >= .99 and r['test_accuracy'] >= .99), None)
    summary = dict(status='completed', **meta, initial=rows[0], final=rows[-1],
                   minimum_train_accuracy=float(min(r['train_accuracy'] for r in rows)),
                   minimum_test_accuracy=float(min(r['test_accuracy'] for r in measured)),
                   minimum_test_after_update=float(min(r['test_accuracy'] for r in future)) if future else None,
                   train_below_90_states=sum(r['train_accuracy'] < .9 for r in rows),
                   test_below_90_measured_states=sum(r['test_accuracy'] < .9 for r in measured),
                   heldout_measurement_count=len(measured), first_measured_joint_recovery_99_local_step=recovered,
                   mean_train_accuracy=float(np.mean([r['train_accuracy'] for r in rows])),
                   total_seconds=time.perf_counter() - started)
    write_json(out / 'summary.json', summary)
    print(json.dumps(dict(event=metadata['event'], arm=arm, status='completed', final_test=rows[-1]['test_accuracy'])), flush=True)


def run_suite(out, steps=500):
    configure_runtime(1)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), events=list(EVENTS),
                    continuation_events=['seed4_first', 'seed0_first'], continuation_arms=CONTINUATION_ARMS,
                    continuation_steps=steps, random_direction_seeds=[0, 1, 2], maximum_continuation_processes=2,
                    fitting='Original training rows only, labels unused',
                    event_selection='Previously captured exploratory first joint failure, plus the previously identified seed4 peak failure as a descriptive sensitivity check.',
                    interpretations='Inverse GL compensation tests the pure invertible-linear drift hypothesis. Head repair is an intervention on the representation-readout interface; specificity must be assessed against old-head and random norm-matched corrections.')
    write_json(out / 'protocol.json', protocol)
    replay_checks(out / 'verification.json')
    for event in EVENTS:
        prepare_event(event, out / event)
    jobs = [(event, arm) for event in ['seed4_first', 'seed0_first'] for arm in CONTINUATION_ARMS]
    running = []
    environment = dict(os.environ, OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    while jobs or running:
        while jobs and len(running) < 2:
            event, arm = jobs.pop(0)
            log = (out / f'{event}_{arm}.log').open('w')
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'continue', '--event-directory', str(out / event), '--arm', arm, '--steps', str(steps)], stdout=log, stderr=subprocess.STDOUT, env=environment)
            running.append((process, log, event, arm))
        for entry in list(running):
            process, log, event, arm = entry
            code = process.poll()
            if code is not None:
                log.close()
                running.remove(entry)
                if code != 0:
                    raise RuntimeError(f'{event} {arm} failed, see log')
                print(json.dumps(dict(event=event, arm=arm, status='completed')), flush=True)
        time.sleep(1)
    summaries = {event: {arm: json.loads((out / event / 'continuation' / arm / 'summary.json').read_text()) for arm in CONTINUATION_ARMS} for event in ['seed4_first', 'seed0_first']}
    write_json(out / 'continuation_summary.json', summaries)
    write_json(out / 'completion.json', dict(status='completed', events=3, instant_arms_per_event=10, continuation_arms=10, continuation_updates=steps * 10))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    suite = sub.add_parser('suite')
    suite.add_argument('--out', type=Path, required=True)
    suite.add_argument('--steps', type=int, default=500)
    branch = sub.add_parser('continue')
    branch.add_argument('--event-directory', type=Path, required=True)
    branch.add_argument('--arm', choices=CONTINUATION_ARMS, required=True)
    branch.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    if args.mode == 'suite':
        run_suite(args.out, args.steps)
    else:
        continue_arm(args.event_directory, args.arm, args.steps)


if __name__ == '__main__':
    main()
