"""Readout interventions for the two independently replayed adjacent pairs."""
from __future__ import annotations
from datetime import datetime, timezone
import interventions as study
import intervention_specificity as specificity


def main():
    study.configure_runtime(1)
    out = study.HERE / 'intervention_adjacent'
    out.mkdir(parents=True, exist_ok=False)
    study.write_json(out / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='Supplementary adjacent-step reference sensitivity',
        reference='The immediately preceding saved or exactly replayed checkpoint. Its accuracy is explicitly reported, regardless of whether it meets a health threshold.',
        pairs={'seed4_adjacent': [16055, 16056], 'seed0_adjacent': [17493, 17494]},
        continuation='No additional training',
        specificity='Ten orthogonal-size random corrections and the opposite orthogonal correction at each event',
        fitting='Original training pairs only; labels unused'))
    study.EVENTS = {'seed4_adjacent': study.HERE / 'adjacent_seed4', 'seed0_adjacent': study.HERE / 'adjacent_seed0'}
    for event in study.EVENTS:
        study.prepare_event(event, out / event)
    specificity.prepare(out, out / 'random_specificity', continue_planned=False,
                        context='Apply the same orthogonal correction-size specificity controls to immediately preceding references.')
    study.write_json(out / 'completion.json', dict(status='completed', primary_instant_arms=20, supplementary_instant_arms=22))


if __name__ == '__main__':
    main()
