"""Adaptive optimizer-memory controls after immediate repair recollapsed.

The three controls and 500-update horizon are fixed before any control result.
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

import interventions as study

ARMS = {
    'native_head_clear_state': ('original', 'clear_head_state'),
    'orthogonal_clear_state': ('orthogonal', 'clear_head_state'),
    'orthogonal_zero_first_moment': ('orthogonal', 'zero_head_first_moment'),
}


def suite(primary, out):
    out.mkdir(parents=True, exist_ok=False)
    study.write_json(out / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='Adaptive control, specified after observing immediate recollapse in every successful state-preserved repair',
        question='Does head optimizer memory explain immediate recollapse after one-time head compensation?',
        arms={name: dict(head=head, optimizer_control=control) for name, (head, control) in ARMS.items()},
        steps_per_arm=500, events=['seed4_first', 'seed0_first'], total_updates=3000,
        unchanged='All hidden and embedding parameters and optimizer states at intervention time; original float32 objective and update implementations',
        clear_state_semantics='Clear only the head AdamW state dictionary. This resets first and second moments and the head step counter. Learning rate, weight decay, betas, and other hyperparameters remain unchanged.',
        zero_first_moment_semantics='Set only head AdamW exp_avg tensors to zero; preserve exp_avg_sq, step counter, and every other state entry.',
        comparison='Primary native-head and orthogonal-head continuations with preserved moments already completed.',
        interpretation='This controls optimizer-memory sensitivity and does not implement a covariant Adam transformation.',
        source=str(primary.resolve())))
    for event in ['seed4_first', 'seed0_first']:
        destination = out / event
        destination.mkdir()
        metadata = json.loads((primary / event / 'metadata.json').read_text())
        metadata['supplement'] = 'Head optimizer memory controls'
        study.write_json(destination / 'metadata.json', metadata)
        heads = np.load(primary / event / 'heads.npz')
        np.savez_compressed(destination / 'heads.npz', **{name: heads[head] for name, (head, _) in ARMS.items()})
    jobs = [(event, arm) for event in ['seed4_first', 'seed0_first'] for arm in ARMS]
    running = []
    env = dict(os.environ, OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    while jobs or running:
        while jobs and len(running) < 2:
            event, arm = jobs.pop(0)
            log = (out / f'{event}_{arm}.log').open('w')
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'continue', '--out', str(out / event), '--arm', arm], stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((process, log, event, arm))
        for entry in list(running):
            process, log, event, arm = entry
            code = process.poll()
            if code is not None:
                log.close()
                running.remove(entry)
                if code:
                    raise RuntimeError(f'{event} {arm} failed, see log')
                print(json.dumps(dict(event=event, arm=arm, status='completed')), flush=True)
        time.sleep(1)
    study.write_json(out / 'completion.json', dict(status='completed', continuation_arms=6, continuation_updates=3000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['suite', 'continue'])
    parser.add_argument('--primary', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=list(ARMS))
    args = parser.parse_args()
    if args.mode == 'suite':
        suite(args.primary, args.out)
    else:
        study.continue_arm(args.out, args.arm, 500, optimizer_control=ARMS[args.arm][1])
