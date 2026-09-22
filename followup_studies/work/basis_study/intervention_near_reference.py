"""Seed0 reference-time sensitivity, retaining the original primary analysis."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import interventions as study


def main():
    study.configure_runtime(1)
    destination = study.HERE / 'intervention_near_reference'
    destination.mkdir(parents=True, exist_ok=False)
    feature_dir = destination / 'seed0_near_features'
    command = [sys.executable, str(study.EXPERIMENTS / 'run_alignment.py'),
               '--healthy', str(study.EXPERIMENTS / 'seed0_dense_capture/pre_event_train_healthy.pt'),
               '--collapsed', str(study.EXPERIMENTS / 'seed0_dense_capture/collapse.pt'),
               '--out', str(feature_dir)]
    study.write_json(destination / 'protocol.json', dict(
        status='Supplementary reference-time sensitivity, specified after examining primary head interventions',
        reason='Primary seed0 analysis used the last measured joint accuracy >=99% checkpoint at 17200. A later saved checkpoint had 100% training accuracy at 17490, four updates before the captured failure. Measure this reference to assess temporal-gap sensitivity.',
        selection='Use the already saved pre_event_train_healthy.pt, with heldout accuracy reported regardless of value',
        changes='Checkpoint reference only; same train-only fits, intervention definitions, and heldout data',
        feature_command=command))
    with (destination / 'feature_extraction.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    study.EVENTS['seed0_near'] = feature_dir
    study.prepare_event('seed0_near', destination / 'seed0_near')
    study.write_json(destination / 'completion.json', dict(status='completed', instant_arms=10))


if __name__ == '__main__':
    main()
