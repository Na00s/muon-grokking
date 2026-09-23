"""Reduce the hidden Muon learning rate after grokking, with stock cross-entropy.

Every training state is logged. Test accuracy is measured every 100 steps,
at both endpoints, and whenever training accuracy falls below 90%.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'experiments'))
sys.path.insert(0, str(HERE.parent / 'causal_study'))
import run_collapse as original
from run_precision_branches import configure_runtime, file_sha256, make_arm, optimizer_hyperparameters
from run_dense_capture import evaluate_preserving_runtime
from run_dense_arithmetic import frozen_metrics


def save(payload, path):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(payload, temporary)
    temporary.replace(path)


def write_json(path, payload):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(checkpoint, out, end_step, operation='addition', eval_every=100, save_every=10000, diagnostic_every=1000):
    configure_runtime(1)
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if int(source['step']) != 6000:
        raise ValueError('The intervention requires the original step-6000 state')
    if source.get('arm') not in (None, 'original32') or source.get('arithmetic') not in (None, 'stock'):
        raise ValueError('The intervention requires an original stock-CE source')
    if source.get('normalization') not in (None, 'none'):
        raise ValueError('This extension concerns the original unnormalized architecture')
    if source.get('operation', operation) != operation:
        raise ValueError('Source operation differs')
    model, optimizers, loss_fn = make_arm(source, 'original32')
    restored_hyperparameters = optimizer_hyperparameters(optimizers)
    for group in optimizers['muon'].param_groups:
        assert group['learning_rate'] == .03
        group['learning_rate'] = .003
    assert all(p.requires_grad for p in model.parameters())
    assert all(g['lr'] == .001 for g in optimizers['auxiliary_adamw'].param_groups)
    assert all(g['lr'] == .00025 for g in optimizers['unembedding_adamw'].param_groups)
    seed, start = int(source['seed']), int(source['step'])
    if end_step <= start:
        raise ValueError('The requested horizon must exceed the source step')
    tx, ty, vx, vy = original.generate_modular_addition_data(seed=seed, operation=operation)
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), seed=seed, operation=operation,
        arm='hidden_lr_0.003', normalization='none', training_loss='torch.nn.functional.cross_entropy',
        restored_optimizer_hyperparameters=restored_hyperparameters, hidden_learning_rate_before=.03, hidden_learning_rate_after=.003,
        start_step=start, end_step=end_step, planned_updates=end_step-start,
        source_checkpoint=str(Path(checkpoint).resolve()), source_checkpoint_sha256=file_sha256(checkpoint),
        source_repository_commit=source['repository_commit'], model_config=source['model_config'],
        optimizer_hyperparameters=optimizer_hyperparameters(optimizers),
        intervention_step=6000, all_parameters_trainable=True, diagnostic_every=diagnostic_every,
        interpretation='Stabilization experiment; the hidden LR reduction also scales decoupled weight decay.',
        train_monitor='Every pre-update state and the final state; all rows retained.',
        test_monitor=f'Every {eval_every} global steps, every training accuracy below 90%, start and final.',
        primary_endpoint='Any joint training/test accuracy below 90% between source and prespecified final step.',
        checkpoint_grid=save_every,
        extra_checkpoints='Start, final, first joint event and its immediately preceding state, most recent jointly >=99% state before the first event, and minimum measured test accuracy.',
        dtype='torch.float32', device='cpu', threads=1,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), torch_version=str(torch.__version__),
        runner_sha256=file_sha256(__file__), numeric_helper_sha256=file_sha256(HERE.parent/'experiments'/'numerical_controls.py'),
        source_runner_sha256=file_sha256(original.__file__),
        protocol_sha256=file_sha256(HERE/'protocol.json') if (HERE/'protocol.json').exists() else None)
    write_json(out/'metadata.json', metadata)

    def state(step):
        payload = original.snapshot(model, optimizers, step, seed)
        payload.update(arm='hidden_lr_0.003', operation=operation, normalization='none', arithmetic='stock',
                       branch_metadata=copy.deepcopy(metadata))
        return payload

    save(state(start), out/'start.pt')
    first_event = None
    previous = None
    healthy = None
    peak = None
    train_bad = joint_bad = test_bad = test_below95 = grid_below95 = eval_count = triggered_evals = 0
    min_train = min_test = 1.
    gradient_records = []
    started = time.perf_counter()
    fields = ['step','local_step','train_loss','train_accuracy','test_accuracy','test_loss64',
              'joint_failure','elapsed_seconds']
    try:
        with (out/'trajectory.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(start, end_step+1):
                model.train()
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                diagnostic_due = step in (start, end_step) or step % diagnostic_every == 0
                features = []
                handle = model.unembedding.register_forward_pre_hook(lambda mod, args: features.append(args[0].detach())) if diagnostic_due else None
                try:
                    logits = model(tx)
                finally:
                    if handle is not None:
                        handle.remove()
                loss = loss_fn(logits, ty)
                if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                    save(state(step), out/'nonfinite.pt')
                    raise FloatingPointError(f'Nonfinite training state at step {step}')
                ta = float((logits.detach().argmax(-1)==ty).double().mean())
                min_train = min(min_train, ta)
                te = None
                if step in (start,end_step) or step % eval_every == 0 or ta < .9:
                    te = evaluate_preserving_runtime(model, vx, vy)
                    eval_count += 1
                    min_test = min(min_test, te['accuracy'])
                    test_bad += te['accuracy'] < .9
                    test_below95 += te['accuracy'] < .95
                    on_grid = step in (start, end_step) or step % eval_every == 0
                    grid_below95 += on_grid and te['accuracy'] < .95
                    triggered_evals += not on_grid
                joint = bool(ta < .9 and te is not None and te['accuracy'] < .9)
                train_bad += ta < .9
                joint_bad += joint
                row = dict(step=step, local_step=step-start, train_loss=float(loss.detach()), train_accuracy=ta,
                           test_accuracy=te['accuracy'] if te else None, test_loss64=te['loss64'] if te else None,
                           joint_failure=joint, elapsed_seconds=time.perf_counter()-started)
                writer.writerow(row)
                if step % 100 == 0 or joint or step == end_step:
                    stream.flush()
                    write_json(out/'progress.json', row)
                current = None
                if step % save_every == 0 or joint or (te is not None and ta >= .99 and te['accuracy'] >= .99):
                    current = state(step)
                if te is not None and ta >= .99 and te['accuracy'] >= .99 and first_event is None:
                    healthy = current
                if step % save_every == 0:
                    save(current, out/f'step_{step:06d}.pt')
                if diagnostic_due:
                    was_training = model.training
                    rng = torch.get_rng_state().clone()
                    test_features = original.residuals(model, vx).double()
                    model.train(was_training)
                    assert torch.equal(rng, torch.get_rng_state())
                    train_features = features[0].double()
                    full_mean = (train_features.sum(0) + test_features.sum(0)) / (len(tx) + len(vx))
                    gradient_records.append(dict(step=step,
                        train_feature_mean_norm=float(train_features.mean(0).norm()),
                        full_grid_feature_mean_norm=float(full_mean.norm()),
                        metrics=frozen_metrics(logits,ty)))
                    write_json(out/'gradient_metrics.json', gradient_records)
                if step % 1000 == 0 or (joint and first_event is None) or step == end_step:
                    print(json.dumps(row), flush=True)
                if joint and first_event is None:
                    first_event = row
                    save(current, out/'collapse.pt')
                    if previous is not None:
                        save(previous, out/'previous.pt')
                    if healthy is not None:
                        save(healthy, out/'healthy.pt')
                    write_json(out/'event.json', row)
                if te is not None and (peak is None or te['accuracy'] < peak['test_accuracy']):
                    peak = row
                    save(current if current is not None else state(step), out/'minimum_test.pt')
                if step == end_step:
                    save(state(step), out/'final.pt')
                    break
                if first_event is None:
                    previous = current if current is not None else state(step)
                loss.backward()
                for optimizer in optimizers.values():
                    optimizer.step()
        summary = dict(status='completed', seed=seed, operation=operation, arm='hidden_lr_0.003', start_step=start,
            end_step=step, updates=step-start, train_states=step-start+1, test_evaluations=eval_count,
            first_joint_failure_step=first_event['step'] if first_event else None,
            minimum_train_accuracy=min_train, minimum_measured_test_accuracy=min_test,
            train_below90_states=train_bad, test_below90_evaluations=test_bad, joint_below90_states=joint_bad,
            test_below95_evaluations=test_below95, scheduled_test_below95_evaluations=grid_below95,
            off_grid_triggered_test_evaluations=triggered_evals, diagnostic_records=len(gradient_records),
            minimum_test_state=peak, final_state=row, elapsed_seconds=time.perf_counter()-started,
            source_checkpoint_sha256=metadata['source_checkpoint_sha256'],
            final_checkpoint_sha256=file_sha256(out/'final.pt'))
        write_json(out/'summary.json', summary)
        print(json.dumps(summary), flush=True)
        return summary
    except BaseException as error:
        write_json(out/'failure.json', dict(status='failed', error=repr(error), last_step=locals().get('step'),
            elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--end-step', type=int, required=True)
    parser.add_argument('--operation', choices=['addition','subtraction'], default='addition')
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=10000)
    parser.add_argument('--diagnostic-every', type=int, default=1000)
    run(**vars(parser.parse_args()))
