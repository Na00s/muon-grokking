"""Supplementary corrections matched to the smaller orthogonal intervention.

This was specified after initial inverse-GL intervention norms were inspected.
It provides a control at the scale of the orthogonal intervention and does not
change or replace the primary protocol or its results.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

import interventions as study


def prepare(primary, out, continue_planned=True, context=None):
    out.mkdir(parents=True, exist_ok=False)
    study.write_json(out / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='Supplementary control specified after inspecting primary intervention norms',
        reason=context or 'Inverse-GL compensation changed head norm by over 6x in the first two events. Match control corrections to the smaller orthogonal intervention as a separate specificity test.',
        gaussian_direction_seeds=list(range(10)),
        deterministic_control='Opposite of the orthogonal correction from the same collapsed head',
        continuation=('Random orthogonal-size correction seed 0, 500 steps, both first failures. Choice fixed before control results are inspected.' if continue_planned else 'No continuation runs for these control fits.'),
        source=str(primary.resolve())))
    for event in study.EVENTS:
        original_dir = primary / event
        destination = out / event
        destination.mkdir()
        metadata = json.loads((original_dir / 'metadata.json').read_text())
        features = np.load(metadata['source_feature_path'])
        heads = np.load(original_dir / 'heads.npz')
        W1 = heads['original']
        correction = heads['orthogonal'] - W1
        norm = np.linalg.norm(correction)
        matched = {}
        for seed in range(10):
            noise = np.random.default_rng(seed).normal(size=W1.shape)
            matched[f'random_orthogonal_size_{seed}'] = W1 + noise * norm / np.linalg.norm(noise)
        matched['opposite_orthogonal_correction'] = W1 - correction
        np.savez_compressed(destination / 'heads.npz', **matched)
        metadata['supplement'] = 'Orthogonal-size specificity controls'
        study.write_json(destination / 'metadata.json', metadata)
        source = torch.load(metadata['collapsed_checkpoint'], map_location='cpu', weights_only=False)
        model, _, _ = study.make_arm(source, 'original32')
        train_x, train_y, test_x, test_y = study.original.generate_modular_addition_data(seed=source['seed'])
        x = torch.cat([train_x, test_x])
        labels = features['y']
        train, heldout = features['train'], features['heldout']
        reference = features['H0'] @ features['W0']
        report = {}
        for name, W in matched.items():
            with torch.no_grad():
                model.unembedding.weight.copy_(torch.from_numpy(W.T).float())
                z = torch.cat([model(x[i:i + 1024]) for i in range(0, len(x), 1024)]).double().numpy()
            actual = model.unembedding.weight.detach().double().numpy().T
            report[name] = dict(train=study.metrics(z[train], labels[train], reference[train]),
                                heldout=study.metrics(z[heldout], labels[heldout], reference[heldout]),
                                head_change_norm=float(np.linalg.norm(actual - W1)),
                                relative_head_change=float(np.linalg.norm(actual - W1) / np.linalg.norm(W1)))
        study.write_json(destination / 'instant.json', report)


def suite(primary, out):
    study.configure_runtime(1)
    prepare(primary, out)
    # Respect the primary suite's two-process training budget.
    while not (primary / 'completion.json').exists():
        time.sleep(1)
    running = []
    env = dict(os.environ, OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for event in ['seed4_first', 'seed0_first']:
        log = (out / f'{event}_continuation.log').open('w')
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'continue', '--out', str(out / event)], stdout=log, stderr=subprocess.STDOUT, env=env)
        running.append((process, log, event))
    for process, log, event in running:
        code = process.wait()
        log.close()
        if code:
            raise RuntimeError(f'{event} failed')
    study.write_json(out / 'completion.json', dict(status='completed', random_controls=30, opposite_direction_controls=3, continuation_updates=1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['suite', 'continue'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--primary', type=Path)
    args = parser.parse_args()
    if args.mode == 'suite':
        suite(args.primary, args.out)
    else:
        study.continue_arm(args.out, 'random_orthogonal_size_0', 500)
