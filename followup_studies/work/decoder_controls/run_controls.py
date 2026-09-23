"""Registered initialization and memorization-phase controls of existing probes.

No original artifact is modified. The prepare stage freezes new checkpoint and
feature hashes before the fit stage accepts them. See protocol.json.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
EXP = WORK / 'experiments'
sys.path.insert(0, str(EXP))
from run_collapse import make_model_optimizers, generate_modular_addition_data, snapshot, evaluate, residuals, train_step
from decoder_probe import fit_decoder

PAIRS = ['experiments/seed0_dense_event', 'basis_study/seed1_first_event',
         'basis_study/seed2_first_event', 'basis_study/seed3_peak_event',
         'experiments/seed4_peak_event']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def source_path(value):
    # Historical metadata paths start with work/, relative to followup_studies.
    return WORK / value.removeprefix('work/')


def record(path):
    p = Path(path)
    return {'path': str(p.relative_to(WORK)), 'sha256': sha(p), 'bytes': p.stat().st_size}


def check_record(r):
    p = WORK / r['path']
    if sha(p) != r['sha256']:
        raise RuntimeError(f'Frozen input changed: {p}')
    return p


def compare(a, b, path='root'):
    """Strict recursive equality including optimizer tensors and RNG bytes."""
    if torch.is_tensor(a):
        return [] if torch.equal(a, b) else [path]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [path + ':keys']
        return sum((compare(a[k], b[k], path + '.' + str(k)) for k in a), [])
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [path + ':length']
        return sum((compare(x, y, path + '.' + str(i)) for i, (x, y) in enumerate(zip(a, b))), [])
    return [] if a == b else [path]


def register():
    if (HERE / 'protocol.json').exists():
        raise FileExistsError('Protocol already registered')
    seeds = []
    for seed, pair in enumerate(PAIRS):
        base = WORK / ('experiments' if seed in (0, 4) else 'basis_study') / f"seed{seed}_{'original' if seed in (0, 4) else 'baseline'}"
        metadata = read(WORK / pair / 'metadata.json')
        rows = list(csv.DictReader((base / 'trajectory.csv').open()))
        scheduled = [r for r in rows if int(r['step']) % 100 == 0 and r['test_accuracy']]
        original_grokking = next(int(scheduled[i]['step']) for i in range(len(scheduled)-4)
                                  if all(float(r['test_accuracy']) >= .95 for r in scheduled[i:i+5]))
        selected_index = next(i for i in range(len(scheduled)-4)
                              if all(float(r['train_accuracy']) >= .999 for r in scheduled[i:i+5])
                              and float(scheduled[i]['test_accuracy']) < .95
                              and int(scheduled[i]['step']) < original_grokking)
        selected = scheduled[selected_index]
        checkpoint_roles = {'initialization': base / 'initial.pt',
                            'replay_validation_1000': base / 'step_001000.pt',
                            'healthy_reference': source_path(metadata['healthy_checkpoint']),
                            'failed': source_path(metadata['collapsed_checkpoint'])}
        decoder = read(WORK / pair / 'decoder_whitened.json')
        indices = decoder['cv_selected']['healthy']['split']['inner_validation_indices']
        assert indices == decoder['cv_selected']['collapsed']['split']['inner_validation_indices']
        train = np.arange(12769) < 3830
        expected = sorted(np.random.default_rng(0).permutation(np.flatnonzero(train))[:766].tolist())
        assert indices == expected
        seeds.append({'seed': seed, 'trajectory': record(base / 'trajectory.csv'),
            'pair': pair, 'pair_metadata': record(WORK / pair / 'metadata.json'),
            'existing_decoder': record(WORK / pair / 'decoder_whitened.json'),
            'existing_features': record(WORK / pair / 'features.npz'),
            'checkpoints': {k: record(v) for k, v in checkpoint_roles.items()},
            'memorization_step': int(selected['step']), 'selection_row': selected,
            'memorization_confirmation_step': int(scheduled[selected_index+4]['step']),
            'memorization_streak_rows': scheduled[selected_index:selected_index+5],
            'original_grokking_step': original_grokking,
            'healthy_step': metadata['healthy_step'], 'failed_step': metadata['collapsed_step'],
            'inner_validation_indices': indices})
    protocol = {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Initialization and memorization-phase controls for the five existing raw-feature decoder comparisons.',
        'selection_rule': 'Choose the first of five consecutive scheduled 100-step evaluations with native training accuracy at least 99.9%, requiring test accuracy below 95% at that checkpoint and a step before original grokking (the first of five scheduled test evaluations at or above 95%). This follows the paper\'s sustained-memorization definition. Selection uses native trajectory metrics and is frozen before decoder fitting.',
        'memorization_steps': [s['memorization_step'] for s in seeds],
        'replay': 'Restore the same saved initialization model, optimizer and torch CPU RNG states; replay the unchanged stock-CE full-batch CPU updates; save the selected step and verify exact model, optimizer and RNG equality to the existing step-1000 checkpoint.',
        'features': 'Actual input to unembedding captured by its forward pre-hook; float32 model forward, batch size 1024, cast to float64 for decoder fitting; no centering or Fourier projection.',
        'decoder': {'implementation': record(EXP / 'decoder_probe.py'), 'n_classes': 113,
            'initialization': 'zero', 'bias': False, 'dtype': 'float64',
            'feature_transform': 'uncentered full-dimensional SVD preconditioning',
            'relative_singular_floor': 1e-6, 'training_only_transform': True,
            'validation_fraction': .2, 'validation_seed': 0, 'validation_indices': 'Explicit frozen indices below, copied from existing reports and verified against the original procedure.',
            'regularization_grid': [0, 1e-8, 1e-6, 1e-4, 1e-2],
            'selection': 'minimum validation cross-entropy, exact ties select smaller penalty',
            'refit': 'independent zero initialization on all original training rows',
            'solver': 'SciPy L-BFGS-B', 'candidate_maxiter': 500, 'refit_maxiter': 1000,
            'maxfun': '3 * maxiter', 'maxls': 20, 'maxcor': 10, 'ftol': 1e-10, 'gtol': 1e-7,
            'unregularized_sensitivity': 'Reuse selected fit if lambda is zero; otherwise fit a separate lambda-zero candidate and refit with identical budgets.',
            'convergence': 'Retain actual success, status, message, iterations, evaluations, objective, gradient norms for every candidate and refit.'},
        'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'backend': 'CPU', 'torch_threads': 1, 'blas_threads': 1,
            'deterministic_algorithms': True, 'float32_matmul_precision': torch.get_float32_matmul_precision()},
        'source_files': [record(EXP / 'run_collapse.py'), record(EXP / 'run_dense_capture.py'), record(WORK / 'basis_study/train_capture.py'), record(EXP / 'alignment_core.py'), record(HERE / 'run_controls.py')],
        'seeds': seeds}
    dump(HERE / 'protocol.json', protocol)
    print(json.dumps({'registered': sha(HERE / 'protocol.json'), 'selected_steps': protocol['memorization_steps']}), flush=True)


def native(model, tx, ty, vx, vy):
    return {'train': evaluate(model, tx, ty), 'heldout': evaluate(model, vx, vy)}


def prepare(seed):
    protocol = read(HERE / 'protocol.json'); entry = protocol['seeds'][seed]
    out = HERE / f'seed{seed}'
    if (out / 'prepared.json').exists():
        raise FileExistsError('Seed already prepared')
    out.mkdir(exist_ok=True)
    for r in entry['checkpoints'].values(): check_record(r)
    source = torch.load(check_record(entry['checkpoints']['initialization']), map_location='cpu', weights_only=False)
    model, opts = make_model_optimizers(seed, checkpoint=source)
    torch.set_rng_state(source['torch_rng_state'])
    tx, ty, vx, vy = generate_modular_addition_data(seed=seed)
    inputs = torch.cat([tx, vx]); labels = torch.cat([ty, vy]).numpy()
    initial_features = residuals(model, inputs).double().numpy()
    initial_native = native(model, tx, ty, vx, vy)
    shutil.copy2(check_record(entry['checkpoints']['initialization']), out / 'initialization.pt')
    selected = entry['memorization_step']; verification_rows = []
    original_rows = {int(r['step']): r for r in csv.DictReader(check_record(entry['trajectory']).open())}
    started = time.monotonic()
    for step in range(1, 1001):
        train_step(model, opts, tx, ty)
        if step == selected:
            state = snapshot(model, opts, step, seed)
            torch.save(state, out / 'memorization.pt')
            memorization_features = residuals(model, inputs).double().numpy()
            memorization_native = native(model, tx, ty, vx, vy)
        if step % 100 == 0:
            metrics = native(model, tx, ty, vx, vy)
            original = original_rows[step]
            checks = {'train_accuracy_equal': metrics['train']['accuracy'] == float(original['train_accuracy']),
                      'test_accuracy_equal': metrics['heldout']['accuracy'] == float(original['test_accuracy']),
                      'test_loss_equal': metrics['heldout']['loss'] == float(original['test_loss'])}
            # Seeds 1--3 logged training loss from the full-batch training forward.
            verification_rows.append({'step': step, 'checks': checks, 'metrics': metrics})
            if not all(checks.values()):
                dump(out / 'replay_failure.json', verification_rows)
                raise RuntimeError(f'Native trajectory replay mismatch seed {seed} step {step}')
    actual = snapshot(model, opts, 1000, seed)
    expected = torch.load(check_record(entry['checkpoints']['replay_validation_1000']), map_location='cpu', weights_only=False)
    checks = {k: compare(actual[k], expected[k]) for k in ['model_state_dict', 'optimizer_state_dicts', 'torch_rng_state']}
    if any(checks.values()):
        dump(out / 'replay_failure.json', checks)
        raise RuntimeError('Final checkpoint exact equality failed')
    train = np.arange(len(inputs)) < len(tx); validation = np.zeros(len(inputs), bool)
    validation[entry['inner_validation_indices']] = True
    np.savez_compressed(out / 'features.npz', initialization=initial_features, memorization=memorization_features,
        inputs=inputs.numpy(), y=labels, train=train, heldout=~train, inner_validation=validation)
    previous = np.load(check_record(entry['existing_features']))
    assert np.array_equal(previous['y'], labels) and np.array_equal(previous['train'], train)
    audit = {'seed': seed, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(HERE / 'protocol.json'), 'replay_updates': 1000,
        'elapsed_seconds': time.monotonic() - started, 'exact_state_mismatches': checks,
        'scheduled_evaluations': verification_rows, 'original_feature_dataset_matches': True,
        'native': {'initialization': initial_native, 'memorization': memorization_native},
        'new_checkpoints': {k: record(out / (k + '.pt')) for k in ['initialization', 'memorization']},
        'features': record(out / 'features.npz')}
    dump(out / 'prepared.json', audit)
    print(json.dumps({'seed': seed, 'prepared': True, 'exact_replay': True, 'elapsed': audit['elapsed_seconds']}), flush=True)


def fit(seed):
    entry = read(HERE / 'protocol.json')['seeds'][seed]
    check_record(read(HERE / 'protocol.json')['decoder']['implementation'])
    out = HERE / f'seed{seed}'; audit = read(out / 'prepared.json')
    if audit['protocol_sha256'] != sha(HERE / 'protocol.json'): raise RuntimeError('Protocol changed')
    features = np.load(check_record(audit['features']))
    for role in ['initialization', 'memorization']:
        report_path = out / f'{role}_decoder.json'
        if report_path.exists():
            print(f'Skip complete {seed} {role}', flush=True); continue
        started = time.monotonic()
        kwargs = dict(n_classes=113, inner_validation_mask=features['inner_validation'],
                      feature_transform='whiten', whitening_relative_floor=1e-6,
                      native_feature_dtype='float32', max_iter=500, refit_max_iter=1000)
        result = fit_decoder(features[role], features['y'], features['train'], features['heldout'], **kwargs)
        if result.report['selected_regularization'] == 0:
            sensitivity = result
        else:
            sensitivity = fit_decoder(features[role], features['y'], features['train'], features['heldout'], regularization_grid=(0.,), **kwargs)
        assert result.report['split']['inner_validation_indices'] == entry['inner_validation_indices']
        report = {'seed': seed, 'role': role, 'checkpoint_step': 0 if role == 'initialization' else entry['memorization_step'],
            'protocol_sha256': sha(HERE / 'protocol.json'), 'prepared_sha256': sha(out / 'prepared.json'),
            'checkpoint': audit['new_checkpoints'][role], 'native': audit['native'][role],
            'cv_selected': result.report, 'unregularized_control': sensitivity.report,
            'elapsed_seconds': time.monotonic() - started}
        np.savez_compressed(out / f'{role}_decoder.npz', cv_readout=result.readout,
            cv_transformed_readout=result.scaled_readout, cv_feature_matrix=result.feature_matrix,
            unregularized_readout=sensitivity.readout, unregularized_transformed_readout=sensitivity.scaled_readout,
            unregularized_feature_matrix=sensitivity.feature_matrix)
        dump(report_path, report)
        print(json.dumps({'seed': seed, 'role': role, 'native': report['native']['heldout']['accuracy'],
            'decoder': result.report['classification']['heldout']['accuracy'], 'lambda': result.report['selected_regularization'],
            'converged': result.report['refit']['converged'], 'iterations': result.report['refit']['iterations'],
            'elapsed': report['elapsed_seconds']}), flush=True)


def summarize():
    protocol = read(HERE / 'protocol.json'); rows = []; sources = []
    for e in protocol['seeds']:
        seed = e['seed']; out = HERE / f'seed{seed}'
        previous = read(check_record(e['existing_decoder'])); meta = read(check_record(e['pair_metadata']))
        source_reports = []
        for role in ['initialization', 'memorization']:
            p = out / f'{role}_decoder.json'; r = read(p); sources.append(record(p))
            source_reports.append((role, r['checkpoint_step'], r['native'], r['cv_selected'], r['unregularized_control']))
        for role, key, step in [('healthy_reference', 'healthy', e['healthy_step']), ('failed', 'collapsed', e['failed_step'])]:
            source_reports.append((role, step, meta['native_' + key], previous['cv_selected'][key], previous['unregularized_control'][key]))
        for role, step, nat, cv, unreg in source_reports:
            assert cv['split']['inner_validation_indices'] == e['inner_validation_indices']
            rows.append({'seed': seed, 'role': role, 'step': step,
                'native_train_accuracy': nat['train']['accuracy'], 'native_test_accuracy': nat['heldout']['accuracy'],
                'decoder_train_accuracy': cv['classification']['train']['accuracy'],
                'decoder_test_accuracy': cv['classification']['heldout']['accuracy'],
                'selected_regularization': cv['selected_regularization'], 'refit_converged': cv['refit']['converged'],
                'refit_iterations': cv['refit']['iterations'], 'refit_message': cv['refit']['message'],
                'candidate_convergence': [c['optimization']['converged'] for c in cv['candidates']],
                'unregularized_test_accuracy': unreg['classification']['heldout']['accuracy'],
                'unregularized_refit_converged': unreg['refit']['converged'],
                'unregularized_refit_iterations': unreg['refit']['iterations'],
                'status': 'new control' if role in ['initialization', 'memorization'] else 'existing result reused'})
    summary = {'protocol_sha256': sha(HERE / 'protocol.json'), 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'all_seed_replays_exact_at_step_1000': True, 'all_validation_indices_preserved': True,
        'new_decoder_count': 10, 'existing_decoder_count': 10, 'rows': rows, 'new_report_manifest': sources}
    dump(HERE / 'summary.json', summary)
    with (HERE / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# Initialization and memorization decoder controls', '',
        'All ten new controls are complete. The existing ten reference/failure results are retained unchanged. Each early-state replay matches the original step-1,000 model, optimizer and RNG exactly. Validation indices match the existing decoders for every seed.', '',
        'The memorization checkpoint is the first of five consecutive scheduled 100-step evaluations with native training accuracy at least 99.9%, with test accuracy below 95% and before original grokking. The selected steps are 200, 400, 200, 100 and 200 for seeds 0–4. These thresholds and identities were registered before fitting. Accuracies below are percentages.', '',
        '| Seed | State | Step | Native train | Native test | Decoder test | CV refit converged | CV iterations | Unregularized test | Unregularized converged |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['role']} | {r['step']} | {r['native_train_accuracy']*100:.3f} | {r['native_test_accuracy']*100:.3f} | {r['decoder_test_accuracy']*100:.3f} | {r['refit_converged']} | {r['refit_iterations']} | {r['unregularized_test_accuracy']*100:.3f} | {r['unregularized_refit_converged']} |")
    lines += ['', 'Candidate and refit diagnostics are retained in the per-state JSON reports, including unsuccessful bounded fits. The initialization and memorization controls measure early linear accessibility under this decoder protocol. Strong decoding at a failed checkpoint establishes surviving linearly accessible task information; it does not alone identify when that information first became accessible, a unique internal algorithm, or exact preservation of the healthy representation.', '']
    (HERE / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'completed': True, 'rows': len(rows)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['register', 'prepare', 'fit', 'summarize'])
    parser.add_argument('--seed', type=int, choices=range(5)); args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        if args.stage == 'register': register()
        elif args.stage == 'prepare': prepare(args.seed)
        elif args.stage == 'fit': fit(args.seed)
        else: summarize()
