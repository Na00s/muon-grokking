"""Resume one numerical-control arm from a run_collapse.py checkpoint.

CPU only. Each invocation runs one explicitly selected arm and creates a new
output directory. The source research repository and checkpoint are read-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

import run_collapse as original
from numerical_controls import PrecisionMuon, accurate_cross_entropy


ARMS = {
    'original32': dict(dtype=torch.float32, accurate_ce=False, precision_muon=False),
    'stable32': dict(dtype=torch.float32, accurate_ce=True, precision_muon=False),
    'model64': dict(dtype=torch.float64, accurate_ce=False, precision_muon=False),
    'true64': dict(dtype=torch.float64, accurate_ce=True, precision_muon=True),
}
OPTIMIZER_KEYS = {'muon', 'auxiliary_adamw', 'unembedding_adamw'}
SOURCE_COMMIT = 'ANONYMIZED_SOURCE_REVISION'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configure_runtime(threads):
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def validate_checkpoint(checkpoint):
    required = {'step', 'seed', 'model_config', 'model_state_dict',
                'optimizer_state_dicts', 'torch_rng_state', 'repository_commit'}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f'Checkpoint is missing: {sorted(missing)}')
    if checkpoint['model_config'] != original.CONFIG:
        raise ValueError('Checkpoint configuration differs from run_collapse.CONFIG')
    if set(checkpoint['optimizer_state_dicts']) != OPTIMIZER_KEYS:
        raise ValueError('Expected all three run_collapse optimizer groups')
    if checkpoint['repository_commit'] != SOURCE_COMMIT:
        raise ValueError('Checkpoint comes from a different source repository commit')


def make_arm(checkpoint, arm):
    """Restore the original three optimizer groups with all saved hyperparameters.

    The original factory converts the model before loading optimizer states.
    The true64 arm then copies the restored Muon groups into PrecisionMuon.
    Other arms retain the exact original optimizer object implementation.
    """
    validate_checkpoint(checkpoint)
    settings = ARMS[arm]
    model, optimizers = original.make_model_optimizers(
        int(checkpoint['seed']), dtype=settings['dtype'], checkpoint=copy.deepcopy(checkpoint))
    if settings['precision_muon']:
        source_muon = optimizers['muon']
        groups = [{**group, 'params': list(group['params'])}
                  for group in source_muon.param_groups]
        constructor_keys = ('learning_rate', 'momentum', 'weight_decay',
                            'newton_schulz_steps', 'nesterov')
        precision_muon = PrecisionMuon(groups, **{
            key: source_muon.defaults[key] for key in constructor_keys})
        precision_muon.load_state_dict(copy.deepcopy(source_muon.state_dict()))
        optimizers['muon'] = precision_muon
    torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
    loss_fn = accurate_cross_entropy if settings['accurate_ce'] else F.cross_entropy
    return model, optimizers, loss_fn


@torch.no_grad()
def evaluate_branch(model, inputs, targets, loss_fn):
    """Match original evaluation batching and retain its loss fields exactly."""
    model.eval()
    logits = torch.cat([model(inputs[i:i + 1024]) for i in range(0, len(inputs), 1024)])
    selected = logits.gather(1, targets[:, None])[:, 0]
    other_maximum = logits.scatter(1, targets[:, None], float('-inf')).max(dim=1).values
    return dict(
        loss=float(F.cross_entropy(logits, targets)),
        loss64=float(F.cross_entropy(logits.double(), targets)),
        accurate_loss64=float(accurate_cross_entropy(logits.double(), targets)),
        objective_loss=float(loss_fn(logits, targets)),
        accuracy=float((logits.argmax(-1) == targets).double().mean()),
        max_abs_logit=float(logits.abs().max()),
        minimum_margin=float((selected - other_maximum).min()),
    )


def optimizer_hyperparameters(optimizers):
    return {name: [{key: value for key, value in group.items() if key != 'params'}
                   for group in optimizer.param_groups]
            for name, optimizer in optimizers.items()}


def branch_checkpoint(source, model, optimizers, global_step, metadata, phase):
    """Retain source metadata and replace state with the current branch state."""
    payload = copy.deepcopy(source)
    payload.update(original.snapshot(model, optimizers, global_step, int(source['seed'])))
    payload['source_checkpoint_metadata'] = {
        key: copy.deepcopy(value) for key, value in source.items()
        if key not in {'model_state_dict', 'optimizer_state_dicts', 'torch_rng_state'}
    }
    payload['source_checkpoint'] = metadata['source_checkpoint']
    payload['source_checkpoint_sha256'] = metadata['source_checkpoint_sha256']
    payload['arm'] = metadata['arm']
    payload['branch_start_step'] = metadata['start_step']
    payload['local_step'] = global_step - metadata['start_step']
    payload['branch_metadata'] = copy.deepcopy(metadata)
    payload['checkpoint_phase'] = phase
    return payload


def run_branch(checkpoint_path, out, arm, steps=1000, threads=1, eval_every=10, save_every=100):
    if steps < 0 or eval_every < 1 or save_every < 1:
        raise ValueError('steps must be nonnegative; eval_every and save_every must be positive')
    if arm not in ARMS:
        raise ValueError(f'Unknown arm: {arm}')
    configure_runtime(threads)
    checkpoint_path = Path(checkpoint_path).resolve()
    out = Path(out).resolve()
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model, optimizers, loss_fn = make_arm(source, arm)
    seed = int(source['seed'])
    train_x, train_y, test_x, test_y = original.generate_modular_addition_data(seed=seed)
    start_step = int(source['step'])
    settings = ARMS[arm]
    metadata = dict(
        arm=arm,
        source_checkpoint=str(checkpoint_path),
        source_checkpoint_sha256=file_sha256(checkpoint_path),
        source_checkpoint_step=start_step,
        source_repository_commit=source['repository_commit'],
        source_torch_version=str(source.get('torch_version', 'unrecorded')),
        start_step=start_step,
        requested_steps=steps,
        requested_end_step=start_step + steps,
        eval_every=eval_every,
        save_every=save_every,
        checkpoint_grid='Global step multiples, plus start.pt and final.pt',
        seed=seed,
        model_config=copy.deepcopy(source['model_config']),
        data_config=dict(modulus=113, train_fraction=.3, operation='addition', seed=seed),
        train_examples=len(train_x),
        test_examples=len(test_x),
        parameter_dtype=str(settings['dtype']),
        training_loss='accurate_cross_entropy' if settings['accurate_ce'] else 'torch.nn.functional.cross_entropy',
        ns_compute_dtype='torch.float64' if settings['precision_muon'] else 'torch.float32',
        muon_class=type(optimizers['muon']).__name__,
        optimizer_hyperparameters=optimizer_hyperparameters(optimizers),
        device='cpu',
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        torch_version=str(torch.__version__),
        python=sys.version,
        created_utc=datetime.now(timezone.utc).isoformat(),
        runner_sha256=file_sha256(__file__),
        original_runner_sha256=file_sha256(original.__file__),
        numeric_helper_sha256=file_sha256(Path(__file__).with_name('numerical_controls.py')),
        note='One arm resumed from the identical saved model, optimizer, and RNG state; all three saved optimizer groups retained.',
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    torch.save(branch_checkpoint(source, model, optimizers, start_step, metadata, 'start'), out / 'start.pt')
    started = time.perf_counter()
    history = []
    last_training_loss = None
    metric_fields = ['loss', 'loss64', 'accurate_loss64', 'objective_loss', 'accuracy', 'minimum_margin']
    fields = ['step', 'local_step', 'arm']
    fields += [f'{split}_{field}' for split in ['train', 'test'] for field in metric_fields]
    fields += ['max_abs_logit', 'last_training_loss', 'elapsed_seconds']
    with (out / 'trajectory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for local_step in range(steps + 1):
            global_step = start_step + local_step
            if local_step % eval_every == 0 or local_step == steps:
                train_metrics = evaluate_branch(model, train_x, train_y, loss_fn)
                test_metrics = evaluate_branch(model, test_x, test_y, loss_fn)
                row = dict(step=global_step, local_step=local_step, arm=arm)
                row.update({f'train_{field}': train_metrics[field] for field in metric_fields})
                row.update({f'test_{field}': test_metrics[field] for field in metric_fields})
                row.update(max_abs_logit=max(train_metrics['max_abs_logit'], test_metrics['max_abs_logit']),
                           last_training_loss=last_training_loss, elapsed_seconds=time.perf_counter() - started)
                writer.writerow(row)
                stream.flush()
                history.append(row)
                print(json.dumps(row), flush=True)
            if global_step % save_every == 0:
                torch.save(branch_checkpoint(source, model, optimizers, global_step, metadata, 'periodic'),
                           out / f'step_{global_step:06d}.pt')
            if local_step == steps:
                break
            last_training_loss = original.train_step(model, optimizers, train_x, train_y, loss_fn=loss_fn)
    torch.save(branch_checkpoint(source, model, optimizers, start_step + steps, metadata, 'final'), out / 'final.pt')
    summary = dict(
        status='completed', arm=arm, source_checkpoint=str(checkpoint_path),
        source_checkpoint_sha256=metadata['source_checkpoint_sha256'],
        start_step=start_step, end_step=start_step + steps, completed_steps=steps,
        evaluation_count=len(history), eval_every=eval_every, save_every=save_every,
        initial_metrics=history[0], final_metrics=history[-1],
        minimum_train_accuracy=min(row['train_accuracy'] for row in history),
        minimum_test_accuracy=min(row['test_accuracy'] for row in history),
        first_train_accuracy_below_0_90_step=next((row['step'] for row in history if row['train_accuracy'] < .90), None),
        first_test_accuracy_below_0_90_step=next((row['step'] for row in history if row['test_accuracy'] < .90), None),
        start_checkpoint=str(out / 'start.pt'), final_checkpoint=str(out / 'final.pt'),
        elapsed_seconds=time.perf_counter() - started,
        threshold_note='Threshold crossing fields summarize sampled evaluations and do not by themselves establish a collapse.',
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(ARMS), required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=100,
                        help='Save checkpoints at global step multiples of this interval')
    args = parser.parse_args()
    run_branch(args.checkpoint, args.out, args.arm, args.steps, args.threads, args.eval_every, args.save_every)


if __name__ == '__main__':
    main()
