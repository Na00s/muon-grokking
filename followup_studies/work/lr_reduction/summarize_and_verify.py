"""Audit the completed matched learning-rate continuations without editing the paper.

The production mode requires all five prespecified 100,000-step endpoints.
An explicit --smoke mode can inspect shorter verification runs and writes only
to a separate directory. It never reports a completed scientific experiment.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'experiments'))
sys.path.insert(0, str(HERE.parent / 'causal_study'))
import run_collapse as original
from run_precision_branches import configure_runtime, make_arm, optimizer_hyperparameters
from run_dense_capture import evaluate_preserving_runtime
from run_dense_arithmetic import frozen_metrics


AUTHORIZED_STATEMENT = (
    'Generative AI tools were used for code completion suggestions and manuscript proofreading. '
    'Any generated content was verified, and the authors take full responsibility for the work and the final manuscript.'
)
PREVIOUS_STATEMENT = (
    'Generative AI tools were used for occasional code generation. All generated code was verified, '
    'and the authors take full responsibility for the work and the final manuscript.'
)
AUTHORIZED_REQUEST = AUTHORIZED_STATEMENT
AUTHORIZED_SCOPE = 'AI use statement only; scientific manuscript and training protocol unchanged.'
AUTHORIZED_PAPER_PATHS = frozenset({
    'paper/source/main.tex',
    'paper/Muon_Grokking_Revised.pdf',
    'paper/ICLR_Submission_Muon_Grokking_Revised_Source.zip',
    'paper/verification.json',
})


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_frozen_paper(plan, amendment_path, check):
    """Enforce original hashes, with one separately recorded author amendment.

    The protocol remains immutable. A permitted amendment names the exact
    author-requested disclosure replacement and its four possible derivatives.
    Reversing that replacement must reproduce the original main.tex bytes.
    """
    frozen = plan.get('frozen_paper_files', [])
    if isinstance(frozen, dict):
        frozen = [dict(path=path, sha256=digest) for path, digest in frozen.items()]
    baseline = {item['path']: item['sha256'] for item in frozen}
    check('frozen paper paths unique', len(baseline) == len(frozen))
    overrides, amendment = {}, None
    if amendment_path.exists():
        ledger = read_json(amendment_path)
        check('authorized paper ledger schema', isinstance(ledger, dict)
              and set(ledger) == {'schema_version', 'author_request', 'scope', 'entries'}
              and type(ledger['schema_version']) is int and ledger['schema_version'] == 1)
        check('authorized paper request exact', ledger['author_request'] == AUTHORIZED_REQUEST)
        check('authorized paper scope exact', ledger['scope'] == AUTHORIZED_SCOPE)
        check('authorized paper entries present', isinstance(ledger['entries'], list) and bool(ledger['entries']))
        for entry in ledger['entries']:
            check('authorized paper entry schema', isinstance(entry, dict)
                  and set(entry) == {'path', 'previous_sha256', 'authorized_sha256'})
            path = entry['path']
            check('authorized paper path allowed', isinstance(path, str) and path in AUTHORIZED_PAPER_PATHS,
                  path=path)
            check('authorized paper path unique', path not in overrides, path=path)
            check('authorized paper original hash', path in baseline and entry['previous_sha256'] == baseline[path],
                  path=path)
            digest = entry['authorized_sha256']
            check('authorized paper hash well formed', isinstance(digest, str) and len(digest) == 64
                  and all(character in '0123456789abcdef' for character in digest), path=path)
            check('authorized paper hash changed', digest != entry['previous_sha256'], path=path)
            overrides[path] = digest
        check('authorized paper main source included', 'paper/source/main.tex' in overrides)
        source = (REPOSITORY/'paper/source/main.tex').read_bytes()
        new, old = AUTHORIZED_STATEMENT.encode(), PREVIOUS_STATEMENT.encode()
        check('authorized paper statement occurs exactly once', source.count(new) == 1 and old not in source)
        restored = source.replace(new, old, 1)
        check('authorized paper sole source edit is disclosure',
              hashlib.sha256(restored).hexdigest() == baseline['paper/source/main.tex'])
        amendment = dict(path=str(amendment_path.relative_to(HERE)) if amendment_path.is_relative_to(HERE)
                         else amendment_path.name,
                         sha256=sha256(amendment_path), author_request=ledger['author_request'],
                         scope=ledger['scope'], entries=ledger['entries'],
                         disclosure_only_source_edit_verified=True)
    for item in frozen:
        path = Path(item['path'])
        if not path.is_absolute():
            path = REPOSITORY/path
        expected = overrides.get(item['path'], item['sha256'])
        check('frozen paper '+item['path'], sha256(path) == expected,
              author_authorized_update=item['path'] in overrides)
    return dict(frozen_paper_files_checked=len(frozen),
                original_frozen_hashes_enforced=len(frozen)-len(overrides),
                authorized_paper_update=amendment,
                scientific_manuscript_unchanged=True)


def equal_tree(left, right, path='root'):
    """Require exact nested tensor and scalar equality; return tensor count."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
        return 1
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        return sum(equal_tree(value, right[key], f'{path}.{key}') for key, value in left.items())
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right), path
        return sum(equal_tree(a, b, f'{path}[{i}]') for i, (a, b) in enumerate(zip(left, right)))
    assert left == right, (path, left, right)
    return 0


def state_core(state):
    return {key: state[key] for key in ('step', 'seed', 'model_config', 'model_state_dict',
            'optimizer_state_dicts', 'torch_rng_state', 'repository_commit')}


def optimizer_groups(state):
    return {name: [{key: value for key, value in group.items() if key != 'params'}
                  for group in optimizer['param_groups']]
            for name, optimizer in state['optimizer_state_dicts'].items()}


def parse_row(row):
    return dict(step=int(row['step']), local_step=int(row['local_step']),
                train_loss=float(row['train_loss']), train_accuracy=float(row['train_accuracy']),
                test_accuracy=float(row['test_accuracy']) if row['test_accuracy'] else None,
                test_loss64=float(row['test_loss64']) if row['test_loss64'] else None,
                joint_failure={'True': True, 'False': False}[row['joint_failure']],
                elapsed_seconds=float(row['elapsed_seconds']))


def derive(rows, start, end, eval_every=100):
    """Derive endpoints independently from the raw every-state trajectory."""
    assert [r['step'] for r in rows] == list(range(start, end + 1)), 'Noncontiguous training states'
    assert all(r['local_step'] == r['step'] - start for r in rows)
    measured = [r for r in rows if r['test_accuracy'] is not None]
    scheduled = {s for s in range(start, end + 1) if s % eval_every == 0 or s in (start, end)}
    expected = scheduled | {r['step'] for r in rows if r['train_accuracy'] < .9}
    assert {r['step'] for r in measured} == expected, 'Test monitoring schedule differs'
    for row in rows:
        assert math.isfinite(row['train_loss']) and row['train_loss'] >= 0
        assert math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds'] >= 0
        assert 0 <= row['train_accuracy'] <= 1
        if row['test_accuracy'] is not None:
            assert 0 <= row['test_accuracy'] <= 1
            assert row['test_loss64'] is not None and math.isfinite(row['test_loss64'])
            assert row['test_loss64'] >= 0
        else:
            assert row['test_loss64'] is None
        joint = row['train_accuracy'] < .9 and row['test_accuracy'] is not None and row['test_accuracy'] < .9
        assert row['joint_failure'] == joint
    assert all(b['elapsed_seconds'] >= a['elapsed_seconds'] for a, b in zip(rows, rows[1:]))
    joint = [r for r in measured if r['joint_failure']]
    minimum = min(measured, key=lambda r: r['test_accuracy'])
    return dict(start_step=start, end_step=end, updates=end-start, train_states=len(rows),
                test_evaluations=len(measured),
                first_joint_failure_step=joint[0]['step'] if joint else None,
                minimum_train_accuracy=min(r['train_accuracy'] for r in rows),
                minimum_measured_test_accuracy=minimum['test_accuracy'],
                train_below90_states=sum(r['train_accuracy'] < .9 for r in rows),
                test_below90_evaluations=sum(r['test_accuracy'] < .9 for r in measured),
                joint_below90_states=len(joint),
                test_below95_evaluations=sum(r['test_accuracy'] < .95 for r in measured),
                scheduled_test_below95_evaluations=sum(r['test_accuracy'] < .95 and r['step'] in scheduled for r in measured),
                off_grid_triggered_test_evaluations=len(expected-scheduled),
                minimum_test_state=minimum, final_state=rows[-1])


def check_diagnostics(records, start, end, interval=1000):
    expected = [s for s in range(start, end + 1) if s % interval == 0 or s in (start, end)]
    assert [r['step'] for r in records] == expected, 'Diagnostic sampling schedule differs'
    for record in records:
        for key in ('train_feature_mean_norm', 'full_grid_feature_mean_norm'):
            assert math.isfinite(record[key]) and record[key] >= 0
        assert set(record['metrics']) == {'stock32', 'accurate32', 'reference64'}
        for metric in record['metrics'].values():
            assert math.isfinite(metric['loss']) and metric['loss'] >= 0
            assert math.isfinite(metric['row_sum_norm']) and metric['row_sum_norm'] >= 0
            assert 0 <= metric['zero_target_fraction'] <= 1
            error, cosine = metric['relative_error'], metric['cosine']
            assert error is None or (math.isfinite(error) and error >= 0)
            assert cosine is None or (math.isfinite(cosine) and -1.000000000001 <= cosine <= 1.000000000001)
        assert record['metrics']['reference64']['relative_error'] in (0, None)
    return expected


def replay(path, row, data, diagnostic=None):
    """Replay accuracy and diagnostics, and inspect live optimizer assignment."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['step'] == row['step']
    model, optimizers, loss_fn = make_arm(state, 'original32')
    assert loss_fn is F.cross_entropy
    assert all(parameter.requires_grad for parameter in model.parameters())
    ids = [id(parameter) for optimizer in optimizers.values() for group in optimizer.param_groups for parameter in group['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}
    tx, ty, vx, vy = data
    assert len(tx) == 3830 and len(vx) == 8939
    model.train()
    features = []
    hook = model.unembedding.register_forward_pre_hook(lambda module, args: features.append(args[0].detach()))
    with torch.no_grad():
        logits = model(tx)
    hook.remove()
    loss = float(F.cross_entropy(logits, ty))
    accuracy = float((logits.argmax(-1) == ty).double().mean())
    assert loss == row['train_loss'] and accuracy == row['train_accuracy'], ('Train replay differs', path)
    if row['test_accuracy'] is not None:
        test = evaluate_preserving_runtime(model, vx, vy)
        assert test['accuracy'] == row['test_accuracy'] and test['loss64'] == row['test_loss64'], ('Test replay differs', path)
    if diagnostic is not None:
        train_features = features[0].double()
        test_features = original.residuals(model, vx).double()
        full_mean = (train_features.sum(0) + test_features.sum(0)) / (len(tx) + len(vx))
        assert float(train_features.mean(0).norm()) == diagnostic['train_feature_mean_norm']
        assert float(full_mean.norm()) == diagnostic['full_grid_feature_mean_norm']
        equal_tree(frozen_metrics(logits, ty), diagnostic['metrics'], 'diagnostic replay')
        # An independent softmax derivative checks the accurate reference.
        z = logits.detach().double().requires_grad_(True)
        from numerical_controls import accurate_cross_entropy
        reference, = torch.autograd.grad(accurate_cross_entropy(z, ty), z)
        with torch.no_grad():
            analytic = torch.softmax(logits.double(), dim=1)
            analytic.scatter_(1, ty[:, None], 0)
            analytic.scatter_(1, ty[:, None], -analytic.sum(1, keepdim=True))
            analytic /= len(ty)
            denominator = reference.norm()
            independent_error = float((analytic-reference).norm()/denominator) if denominator else 0.0
        assert independent_error <= 1e-12, ('Analytic derivative reference differs', independent_error)
    return dict(path=path.name, step=state['step'], training_accuracy=accuracy,
                test_accuracy=row['test_accuracy'], diagnostic_replayed=diagnostic is not None,
                all_parameters_trainable=True, optimizer_assignment_complete=True), state


def historical_binary():
    """Read established comparison outcomes without comparing event counts."""
    corrected_path = HERE.parent/'long_horizon'/'summary.json'
    corrected = [r for r in read_json(corrected_path)['records'] if r['operation'] == 'addition']
    assert sorted(r['seed'] for r in corrected) == list(range(5))
    assert all(r['arithmetic_start_step'] == 6000 and r['end_step'] == 100000 for r in corrected)
    assert sum(r['first_joint_failure_step'] is not None for r in corrected) == 0
    originals = []
    for seed in range(5):
        path = HERE.parent/'causal_study'/'diagnostics_results'/f'seed{seed}'/'event.json'
        event = read_json(path)
        stock = next(arm for arm in event['gradient_swap_arms'] if arm['accurate_groups'] == [])['metrics']
        assert stock['native_fullbatch_train_accuracy'] < .9 and stock['test_accuracy'] < .9
        originals.append(dict(seed=seed, joint_failure=True, evidence=str(path.relative_to(HERE.parent)), sha256=sha256(path)))
    return dict(original_stock=dict(seeds=5, seeds_with_joint_failure=5, evidence=originals),
                corrected_loss=dict(seeds=5, seeds_with_joint_failure=0,
                    evidence=str(corrected_path.relative_to(HERE.parent)), sha256=sha256(corrected_path)),
                comparison_scope='Binary per-seed joint failures only. Historical original monitoring differs; event counts and durations are not compared.')


def diagnostic_table(records):
    def number(value):
        return 'Undefined' if value is None else f'{value:.6g}'
    lines = ['| Seed | Full-grid mean norm, start | Final | Sampled maximum | Stock CE relative error, start | Final | Sampled maximum |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    keys = ('initial_full_grid_feature_mean_norm', 'final_full_grid_feature_mean_norm',
            'maximum_sampled_full_grid_feature_mean_norm', 'initial_ce_relative_error',
            'final_ce_relative_error', 'maximum_sampled_ce_relative_error')
    for record in records:
        lines.append('| ' + str(record['seed']) + ' | ' + ' | '.join(number(record[key]) for key in keys) + ' |')
    lines += ['', 'CE relative error is the stock float32 loss derivative error relative to the accurate float64 derivative of the same stored logits. Undefined denotes a zero reference norm. The full-grid feature mean covers all operand pairs. Training-set mean norms are retained in `summary.csv` as `initial_train_feature_mean_norm`, `final_train_feature_mean_norm`, and `maximum_sampled_train_feature_mean_norm`.']
    return lines


def audit(args):
    configure_runtime(1)
    plan = read_json(args.protocol)
    out = args.out or (HERE/'smoke_audit' if args.smoke else HERE)
    out.mkdir(parents=True, exist_ok=True)
    checks, records, checkpoints, replays = [], [], [], []
    verification = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                        scope='short verification runs' if args.smoke else 'five complete prespecified continuations',
                        auditor_sha256=sha256(__file__), protocol_sha256=sha256(args.protocol), checks=checks)

    def check(name, value, **details):
        checks.append(dict(name=name, passed=bool(value), **details))
        if not value:
            raise AssertionError((name, details))

    try:
        jobs = [j for j in plan['jobs'] if args.seeds is None or int(j['seed']) in args.seeds]
        if not args.smoke:
            check('five prespecified seeds', sorted(j['seed'] for j in jobs) == list(range(5)))
            check('470000 planned updates', sum(j['end_step']-j['start_step'] for j in jobs) == 470000)
        check('runner immutable', sha256(HERE/'run_lr_reduction.py') == plan['runner_sha256'])
        for job in jobs:
            name, seed = job['name'], int(job['seed'])
            root = args.runs_dir/name
            source_path = Path(job['checkpoint'])
            if not source_path.is_absolute():
                source_path = HERE.parent/source_path
            metadata, supplied = read_json(root/'metadata.json'), read_json(root/'summary.json')
            start, end = int(job['start_step']), int(supplied['end_step'] if args.smoke else job['end_step'])
            check(name+' complete horizon', supplied['status'] == 'completed' and supplied['end_step'] == end and start == 6000 and (args.smoke or end == 100000))
            check(name+' no runtime failure', not (root/'failure.json').exists() and not (root/'nonfinite.pt').exists())
            check(name+' provenance', sha256(source_path) == metadata['source_checkpoint_sha256'] == job['source_checkpoint_sha256']
                  and metadata['runner_sha256'] == plan['runner_sha256']
                  and (args.smoke or metadata['protocol_sha256'] == sha256(args.protocol)))
            check(name+' stock trainable configuration', metadata['seed'] == seed and metadata['operation'] == 'addition'
                  and metadata['normalization'] == 'none' and metadata['all_parameters_trainable']
                  and metadata['training_loss'] == 'torch.nn.functional.cross_entropy'
                  and metadata['hidden_learning_rate_before'] == .03 and metadata['hidden_learning_rate_after'] == .003)
            with (root/'trajectory.csv').open(newline='') as stream:
                rows = [parse_row(row) for row in csv.DictReader(stream)]
            eval_every = int(metadata['test_monitor'].split()[1]) if args.smoke else 100
            if not args.smoke:
                check(name+' prespecified diagnostic/checkpoint frequency', metadata['diagnostic_every'] == 1000 and metadata['checkpoint_grid'] == 10000)
            derived = derive(rows, start, end, eval_every=eval_every)
            for key, value in derived.items():
                check(name+' derived '+key, value == supplied[key])
            source = torch.load(source_path, map_location='cpu', weights_only=False)
            initial = torch.load(root/'start.pt', map_location='cpu', weights_only=False)
            expected = copy.deepcopy(state_core(source))
            for group in expected['optimizer_state_dicts']['muon']['param_groups']:
                check(name+' source hidden LR', group['learning_rate'] == .03)
                group['learning_rate'] = .003
            tensors = equal_tree(expected, state_core(initial))
            check(name+' complete initial state restored with only hidden LR changed', tensors > 0, tensor_count=tensors)
            groups = optimizer_groups(initial)
            check(name+' AdamW learning rates preserved', all(g['lr'] == .001 for g in groups['auxiliary_adamw'])
                  and all(g['lr'] == .00025 for g in groups['unembedding_adamw']))
            check(name+' metadata hyperparameters', json.loads(json.dumps(groups)) == metadata['optimizer_hyperparameters']
                  and json.loads(json.dumps(optimizer_groups(source))) == metadata['restored_optimizer_hyperparameters'])
            diagnostics = read_json(root/'gradient_metrics.json')
            diagnostic_steps = check_diagnostics(diagnostics, start, end, metadata['diagnostic_every'] if args.smoke else 1000)
            check(name+' diagnostic records', supplied['diagnostic_records'] == len(diagnostics), count=len(diagnostics))
            diagnostic_by_step = {r['step']: r for r in diagnostics}
            rows_by_step = {r['step']: r for r in rows}
            expected_grid = {f'step_{s:06d}.pt' for s in range(start, end+1) if s % metadata['checkpoint_grid'] == 0}
            check(name+' saved checkpoint grid', {p.name for p in root.glob('step_*.pt')} == expected_grid)
            first = derived['first_joint_failure_step']
            selected = {'start.pt', 'final.pt', 'minimum_test.pt'} | expected_grid
            if first is not None:
                selected |= {'collapse.pt', 'previous.pt'}
                check(name+' first event record', read_json(root/'event.json') == rows_by_step[first])
                healthy = [r for r in rows if r['step'] <= first and r['test_accuracy'] is not None
                           and r['train_accuracy'] >= .99 and r['test_accuracy'] >= .99]
                if healthy:
                    selected.add('healthy.pt')
                    check(name+' preceding healthy checkpoint step', torch.load(root/'healthy.pt', map_location='cpu', weights_only=False)['step'] == healthy[-1]['step'])
            else:
                check(name+' no event artifacts', not any((root/file).exists() for file in ('collapse.pt', 'previous.pt', 'healthy.pt', 'event.json')))
            data = original.generate_modular_addition_data(seed=seed, operation='addition')
            for filename in sorted(selected):
                path = root/filename
                state = torch.load(path, map_location='cpu', weights_only=False)
                step = int(state['step'])
                check(name+' '+filename+' metadata', state['seed'] == seed and state['arm'] == 'hidden_lr_0.003'
                      and state['arithmetic'] == 'stock' and state['normalization'] == 'none'
                      and state['operation'] == 'addition' and start <= step <= end)
                check(name+' '+filename+' fixed optimizer hyperparameters', optimizer_groups(state) == groups)
                for opt_name in ('auxiliary_adamw', 'unembedding_adamw'):
                    counters = [int(v['step']) for v in state['optimizer_state_dicts'][opt_name]['state'].values()]
                    check(name+' '+filename+' '+opt_name+' counters', len(counters) > 0 and all(s == step for s in counters))
                replay_record, _ = replay(path, rows_by_step[step], data, diagnostic_by_step.get(step))
                replays.append(dict(seed=seed, **replay_record))
                if filename == 'final.pt':
                    check(name+' final checkpoint hash', sha256(path) == supplied['final_checkpoint_sha256'])
                    check(name+' final checkpoint step', step == end)
                if filename == 'minimum_test.pt':
                    check(name+' minimum checkpoint selection', step == derived['minimum_test_state']['step'])
                if filename == 'collapse.pt':
                    check(name+' first collapse checkpoint selection', step == first)
                if filename == 'previous.pt':
                    check(name+' immediately preceding checkpoint selection', step == first-1)
            for path in sorted(root.glob('*.pt')):
                checkpoints.append(dict(path=str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path),
                                        bytes=path.stat().st_size, sha256=sha256(path)))
            record = {key: value for key, value in derived.items() if key not in ('minimum_test_state', 'final_state')}
            record.update(seed=seed, name=name, joint_failure=first is not None,
                          minimum_test_step=derived['minimum_test_state']['step'], final_train_accuracy=rows[-1]['train_accuracy'],
                          final_test_accuracy=rows[-1]['test_accuracy'], diagnostic_records=len(diagnostics),
                          initial_train_feature_mean_norm=diagnostics[0]['train_feature_mean_norm'],
                          final_train_feature_mean_norm=diagnostics[-1]['train_feature_mean_norm'],
                          initial_full_grid_feature_mean_norm=diagnostics[0]['full_grid_feature_mean_norm'],
                          final_full_grid_feature_mean_norm=diagnostics[-1]['full_grid_feature_mean_norm'],
                          maximum_sampled_train_feature_mean_norm=max(r['train_feature_mean_norm'] for r in diagnostics),
                          maximum_sampled_full_grid_feature_mean_norm=max(r['full_grid_feature_mean_norm'] for r in diagnostics),
                          initial_ce_relative_error=diagnostics[0]['metrics']['stock32']['relative_error'],
                          final_ce_relative_error=diagnostics[-1]['metrics']['stock32']['relative_error'],
                          maximum_sampled_ce_relative_error=max((r['metrics']['stock32']['relative_error'] for r in diagnostics if r['metrics']['stock32']['relative_error'] is not None), default=None),
                          trajectory_sha256=sha256(root/'trajectory.csv'), diagnostics_sha256=sha256(root/'gradient_metrics.json'))
            records.append(record)
            check(name+' source checkpoint remained unchanged', sha256(source_path) == job['source_checkpoint_sha256'])
            print(json.dumps(dict(audited=name, end_step=end, joint_failure=first is not None)), flush=True)
        paper_verification = verify_frozen_paper(plan, args.authorized_paper_updates, check)
        summary = dict(status='smoke_verified' if args.smoke else 'completed_and_verified',
                       seeds=len(records), seeds_with_joint_failure=sum(r['joint_failure'] for r in records),
                       total_updates=sum(r['updates'] for r in records), records=records,
                       scope=('Short verification trajectories only. No scientific endpoint conclusion.' if args.smoke else
                              'These five seeds and the fixed 100,000-step horizon. Test-only failures between scheduled evaluations may be missed.'),
                       interpretation='The hidden learning-rate reduction also scales hidden decoupled weight decay. This is a stabilization experiment; it does not identify the upstream mechanism.',
                       historical_comparison=historical_binary() if not args.smoke else None)
        verification.update(status='passed', checked_items=len(checks), checkpoint_replays=replays,
                            **paper_verification, source_checkpoints_preserved=True,
                            total_training_states=sum(r['train_states'] for r in records),
                            total_diagnostic_records=sum(r['diagnostic_records'] for r in records))
        write_json(out/'checkpoint_manifest.json', dict(files=checkpoints))
        write_json(out/'summary.json', summary)
        with (out/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        if not args.smoke:
            failures = summary['seeds_with_joint_failure']
            result = ('The reduction prevented the prespecified joint failure in all five tested continuations through step 100,000.'
                      if failures == 0 else f'The reduction was insufficient to prevent the prespecified joint failure: {failures}/5 seeds failed by step 100,000.')
            lines = [result, '', 'Each continuation restored the original step-6,000 model, all optimizer buffers, and RNG state. Hidden Muon learning rate changed from 0.03 to 0.003; embedding and readout AdamW learning rates stayed at 0.001 and 0.00025. All parameters remained trainable, with stock cross-entropy and unchanged weight-decay coefficients.', '',
                     '| Seed | Joint failure | First failure step | Minimum sampled test accuracy | Evaluations below 95% | Scheduled evaluations below 95% | Final test accuracy |',
                     '| --- | --- | --- | --- | --- | --- | --- |']
            for r in records:
                lines.append(f"| {r['seed']} | {'Yes' if r['joint_failure'] else 'No'} | {r['first_joint_failure_step'] if r['joint_failure'] else ''} | {100*r['minimum_measured_test_accuracy']:.4f}% | {r['test_below95_evaluations']} | {r['scheduled_test_below95_evaluations']} | {100*r['final_test_accuracy']:.4f}% |")
            lines += ['', f"Binary comparison: original stock runs 5/5 seeds with a joint failure; corrected-loss continuations 0/5; hidden-LR reduction {failures}/5. Historical original monitoring differs, so event counts and durations are not compared.", '',
                      'Training accuracy was checked at all 94,001 states per seed, including the source and final states. Test accuracy was evaluated every 100 updates and whenever training accuracy fell below 90%. Secondary evaluation counts describe this new experiment only. Feature-mean norms and loss-derivative error were recorded at 95 states per seed, every 1,000 updates including both endpoints.', '']
            lines += diagnostic_table(records)
            paper_note = ('The scientific manuscript remained unchanged. The author explicitly authorized replacement of the AI use statement while these runs were in progress; the separate amendment ledger records the four permitted artifact paths and preserves the original protocol hashes.'
                          if paper_verification['authorized_paper_update'] else 'The paper was left unchanged.')
            lines += ['', 'The intervention also reduces hidden decoupled weight decay per update. It tests stabilization under this particular tenfold reduction and cannot by itself establish the upstream failure mechanism. The conclusions are limited to these seeds, this architecture, and the fixed horizon. Test-only excursions between scheduled evaluations remain unobserved.', '',
                      f"Verification passed {len(checks)} explicit checks and {len(replays)} checkpoint metric replays. Model, optimizer, and RNG states at branch start match their source except for the intended hidden learning rate. All saved optimizer learning rates and weight-decay coefficients remained fixed. {paper_note}"]
            (out/'report.md').write_text('\n'.join(lines)+'\n')
        else:
            (out/'report.md').write_text('\n'.join(['Smoke verification only. These short runs do not establish scientific endpoint outcomes.', ''] + diagnostic_table(records))+'\n')
        write_json(out/'verification.json', verification)
        return summary
    except BaseException as error:
        verification.update(status='failed', error=repr(error), checked_items=len(checks), checkpoint_replays=replays)
        write_json(out/'verification.json', verification)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--runs-dir', type=Path, default=HERE/'runs')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--authorized-paper-updates', type=Path, default=HERE/'authorized_paper_updates.json',
                        help='Separate author-authorized disclosure amendment; original protocol hashes remain unchanged')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int)
    arguments = parser.parse_args()
    if arguments.smoke and arguments.out is not None and arguments.out.resolve() == HERE:
        parser.error('Smoke results must be written separately from production artifacts')
    if arguments.seeds is not None and not arguments.smoke:
        parser.error('Production audit requires all five seeds')
    audit(arguments)
