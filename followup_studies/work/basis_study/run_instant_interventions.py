"""Reusable train-only readout interventions for a prepared feature pair.

Input: features.npz and metadata.json in legacy run_alignment format, or the
adjacent-pair format with reference_role='preceding_step'. No training is run.
Output: <out>/<name>/instant.json and optional random_specificity/<name>/.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import interventions as study
import intervention_specificity as specificity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--skip-random-controls', action='store_true')
    args = parser.parse_args()
    study.configure_runtime(1)
    args.out.mkdir(parents=True, exist_ok=False)
    source_metadata = json.loads((args.pair_dir / 'metadata.json').read_text())
    study.write_json(args.out / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='Fixed intervention recipe applied to a newly supplied feature pair',
        name=args.name, source_pair=str(args.pair_dir.resolve()),
        reference_role=source_metadata.get('reference_role', 'selected_healthy_reference'),
        fitting='Original training rows only; labels unused',
        heads='Original, old reference head, orthogonal transport, inverse unregularized forward map, backward OLS transport, norm-matched old head, healthy-logit distillation, three GL-size random directions',
        specificity='Ten orthogonal-size Gaussian directions plus opposite orthogonal correction' if not args.skip_random_controls else 'Skipped by explicit flag',
        training='No continuation training'))
    study.EVENTS = {args.name: args.pair_dir.resolve()}
    study.prepare_event(args.name, args.out / args.name)
    if not args.skip_random_controls:
        specificity.prepare(args.out, args.out / 'random_specificity', continue_planned=False,
                            context='Apply the previously specified orthogonal correction-size specificity controls to the newly supplied feature pair.')
    study.write_json(args.out / 'completion.json', dict(status='completed', instant_evaluations=10 if args.skip_random_controls else 21))
    print(json.dumps(dict(status='completed', results=str(args.out / args.name / 'instant.json'))))


if __name__ == '__main__':
    main()
