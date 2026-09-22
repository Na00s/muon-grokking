"""Exploratory one-step train-trigger search for brief joint accuracy failures.

Resume original32 exactly. Inspect already-computed full-batch training logits
before each update. Evaluate held-out data on a 100-step global grid or when
training accuracy is below 90%. Stop before the update at the first joint
train/test failure. This follow-up search differs from the initial 100-step
sampled capture protocol and is explicitly exploratory.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import csv
from datetime import datetime, timezone
import io
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

import run_collapse as original
from run_precision_branches import (
    branch_checkpoint, configure_runtime, file_sha256, make_arm,
    optimizer_hyperparameters,
)


TRAIN_LOG_EVERY = 10
TEST_EVERY = 100
SNAPSHOT_EVERY = 10
WINDOW_SIZE = 20
HEALTHY_THRESHOLD = .99
FAILURE_THRESHOLD = .90


def atomic_save(payload, path):
    temporary = path.with_name(path.stem + '.tmp.pt')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def evaluate_preserving_runtime(model, inputs, targets):
    """Evaluate using source API and verify the monitoring preserves RNG."""
    training = model.training
    rng_before = torch.get_rng_state().clone()
    try:
        metrics = original.evaluate(model, inputs, targets)
    finally:
        model.train(training)
    if not torch.equal(rng_before, torch.get_rng_state()):
        raise RuntimeError('Evaluation consumed RNG; exact replay cannot continue')
    return metrics


def run_capture(checkpoint_path, out, steps=5000, threads=1):
    if steps < 0:
        raise ValueError('steps must be nonnegative')
    configure_runtime(threads)
    checkpoint_path = Path(checkpoint_path).resolve()
    out = Path(out).resolve()
    # Read one immutable byte sequence so provenance also holds for latest.pt.
    checkpoint_bytes = checkpoint_path.read_bytes()
    source_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    source = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False)
    del checkpoint_bytes
    model, optimizers, loss_fn = make_arm(source, 'original32')
    seed = int(source['seed'])
    start_step = int(source['step'])
    train_x, train_y, test_x, test_y = original.generate_modular_addition_data(seed=seed)
    initial_train = evaluate_preserving_runtime(model, train_x, train_y)
    initial_test = evaluate_preserving_runtime(model, test_x, test_y)
    if initial_train['accuracy'] < HEALTHY_THRESHOLD or initial_test['accuracy'] < HEALTHY_THRESHOLD:
        raise ValueError('Dense capture requires initial train and test accuracy >= 0.99; '
                         f'observed train={initial_train["accuracy"]}, test={initial_test["accuracy"]}')
    metadata = dict(
        arm='original32', experiment='exploratory_dense_capture',
        selection_status='exploratory_followup',
        selection='First pre-update state with full-batch train accuracy below 0.90 and contemporaneous held-out accuracy below 0.90; train accuracy inspected every step.',
        protocol_context='Exploratory one-step train-trigger event search following the initially 100-step-sampled protocol; designed to detect brief joint failures between the original evaluation points.',
        source_checkpoint=str(checkpoint_path), source_checkpoint_sha256=source_hash,
        source_repository_commit=source['repository_commit'],
        start_step=start_step, seed=seed, requested_steps=steps,
        model_config=copy.deepcopy(source['model_config']),
        data_config=dict(modulus=113, train_fraction=.3, operation='addition', seed=seed),
        initial_train_metrics=initial_train, initial_test_metrics=initial_test,
        train_accuracy_source='Full-batch logits already computed for training, before the optimizer update',
        test_evaluation='Source evaluate() batching, every global step multiple of 100 or whenever contemporaneous training accuracy is below 0.90',
        train_log_every=TRAIN_LOG_EVERY, test_every=TEST_EVERY,
        snapshot_every=SNAPSHOT_EVERY, rolling_snapshot_count=WINDOW_SIZE,
        healthy_threshold=HEALTHY_THRESHOLD, failure_threshold=FAILURE_THRESHOLD,
        healthy_checkpoint_semantics='last_joint_healthy.pt requires contemporaneous train/test >= 0.99; last_train_healthy.pt uses train >= 0.99 and records whether held-out data was measured',
        optimizer_hyperparameters=optimizer_hyperparameters(optimizers),
        parameter_dtype='torch.float32', ns_compute_dtype='torch.float32',
        training_loss='torch.nn.functional.cross_entropy', device='cpu',
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        torch_version=str(torch.__version__), python=sys.version,
        created_utc=datetime.now(timezone.utc).isoformat(),
        runner_sha256=file_sha256(__file__),
        original_runner_sha256=file_sha256(original.__file__),
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')

    def capture_state(global_step, phase, row=None):
        state = branch_checkpoint(source, model, optimizers, global_step, metadata, phase)
        state['capture_metrics'] = copy.deepcopy(row)
        return state

    initial_state = capture_state(start_step, 'start', dict(
        step=start_step, train_accuracy=initial_train['accuracy'],
        test_accuracy=initial_test['accuracy'], heldout_measured=True,
        train_accuracy_source='Source evaluate() batching during initial validation'))
    torch.save(initial_state, out / 'start.pt')
    atomic_save(initial_state, out / 'latest.pt')
    atomic_save(initial_state, out / 'last_joint_healthy.pt')
    atomic_save(initial_state, out / 'last_train_healthy.pt')
    last_joint_healthy = initial_state
    last_train_healthy = initial_state
    latest_state = initial_state
    recent = deque(maxlen=WINDOW_SIZE)
    event = None
    last_row = None
    training_trigger_count = 0
    heldout_evaluation_count = 1
    updates_completed = 0
    started = time.perf_counter()
    fields = ['step', 'local_step', 'train_loss', 'train_accuracy',
              'heldout_measured', 'test_loss', 'test_loss64', 'test_accuracy',
              'train_below_0_90', 'joint_failure', 'elapsed_seconds']

    with (out / 'trajectory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for local_step in range(steps + 1):
            global_step = start_step + local_step
            # These operations and their ordering match original.train_step.
            model.train()
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            logits = model(train_x)
            loss = loss_fn(logits, train_y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            with torch.no_grad():
                train_accuracy = float((logits.argmax(-1) == train_y).double().mean())
                train_loss = float(loss.detach())
            trigger = train_accuracy < FAILURE_THRESHOLD
            training_trigger_count += int(trigger)
            test_metrics = None
            if local_step == 0:
                test_metrics = initial_test
            elif global_step % TEST_EVERY == 0 or trigger:
                test_metrics = evaluate_preserving_runtime(model, test_x, test_y)
                heldout_evaluation_count += 1
            joint_failure = bool(trigger and test_metrics is not None
                                 and test_metrics['accuracy'] < FAILURE_THRESHOLD)
            row = dict(
                step=global_step, local_step=local_step,
                train_loss=train_loss, train_accuracy=train_accuracy,
                heldout_measured=test_metrics is not None,
                test_loss=test_metrics['loss'] if test_metrics is not None else None,
                test_loss64=test_metrics['loss64'] if test_metrics is not None else None,
                test_accuracy=test_metrics['accuracy'] if test_metrics is not None else None,
                train_below_0_90=trigger, joint_failure=joint_failure,
                elapsed_seconds=time.perf_counter() - started,
            )
            last_row = row
            if global_step % TRAIN_LOG_EVERY == 0 or trigger or local_step in {0, steps}:
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row), flush=True)
            if joint_failure:
                # No backward or update is performed at this state.
                event = row
                event_state = capture_state(global_step, 'collapse', row)
                torch.save(event_state, out / 'collapse.pt')
                torch.save(event_state, out / 'final.pt')
                torch.save(latest_state, out / 'pre_event_latest.pt')
                torch.save(last_joint_healthy, out / 'pre_event_joint_healthy.pt')
                torch.save(last_train_healthy, out / 'pre_event_train_healthy.pt')
                event_directory = out / 'event_window'
                event_directory.mkdir()
                window_manifest = []
                for state in recent:
                    filename = f'step_{state["step"]:06d}.pt'
                    torch.save(state, event_directory / filename)
                    window_manifest.append(dict(step=state['step'], file=filename,
                                                metrics=state['capture_metrics']))
                event_filename = f'step_{global_step:06d}.pt'
                torch.save(event_state, event_directory / event_filename)
                (event_directory / 'manifest.json').write_text(json.dumps(dict(
                    grid_snapshots=window_manifest, event_step=global_step,
                    event_file=event_filename, snapshot_interval=SNAPSHOT_EVERY,
                    selection_status='exploratory_followup'), indent=2) + '\n')
                (out / 'event.json').write_text(json.dumps(dict(
                    **row, selection_status='exploratory_followup',
                    last_joint_healthy_step=last_joint_healthy['step'],
                    last_train_healthy_step=last_train_healthy['step'],
                    pre_event_latest_step=latest_state['step']), indent=2) + '\n')
                atomic_save(event_state, out / 'latest.pt')
                print('CAPTURED ' + json.dumps(row), flush=True)
                break
            # Store actual pre-update states on the global 10-step grid.
            state = None
            if global_step % SNAPSHOT_EVERY == 0:
                state = capture_state(global_step, 'rolling_grid', row)
                recent.append(state)
                latest_state = state
                atomic_save(state, out / 'latest.pt')
                if train_accuracy >= HEALTHY_THRESHOLD:
                    last_train_healthy = state
                    atomic_save(state, out / 'last_train_healthy.pt')
            if (test_metrics is not None and train_accuracy >= HEALTHY_THRESHOLD
                    and test_metrics['accuracy'] >= HEALTHY_THRESHOLD):
                if state is None:
                    state = capture_state(global_step, 'joint_healthy', row)
                last_joint_healthy = state
                atomic_save(state, out / 'last_joint_healthy.pt')
            if local_step == steps:
                final_state = capture_state(global_step, 'final', row)
                torch.save(final_state, out / 'final.pt')
                atomic_save(final_state, out / 'latest.pt')
                break
            loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()
            updates_completed += 1

    summary = dict(
        status='captured' if event is not None else 'horizon_completed_without_joint_failure',
        selection_status='exploratory_followup',
        source_checkpoint=str(checkpoint_path), source_checkpoint_sha256=source_hash,
        start_step=start_step, final_step=last_row['step'],
        requested_steps=steps, updates_completed=updates_completed,
        train_accuracy_inspections=local_step + 1,
        training_trigger_count=training_trigger_count,
        heldout_evaluation_count=heldout_evaluation_count,
        event_step=event['step'] if event is not None else None,
        final_metrics=last_row,
        last_joint_healthy_step=last_joint_healthy['step'],
        last_train_healthy_step=last_train_healthy['step'],
        elapsed_seconds=time.perf_counter() - started,
        interpretation='A captured event is selected by exploratory per-step train-trigger monitoring; held-out accuracy is otherwise measured only on a 100-step grid.',
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    run_capture(args.checkpoint, args.out, args.steps, args.threads)


if __name__ == '__main__':
    main()
