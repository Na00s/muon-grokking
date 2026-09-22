"""Collect auditable readout intervention tables from completed local studies."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []
    groups = ['interventions', 'intervention_specificity', 'intervention_near_reference', 'intervention_adjacent', 'fresh_interventions']
    for group in groups:
        for instant_path in sorted((ROOT / group).rglob('instant.json')):
            metadata = json.loads((instant_path.parent / 'metadata.json').read_text())
            instant = json.loads(instant_path.read_text())
            for arm, results in instant.items():
                heldout = results['heldout']
                rows.append(dict(study_group=group, event=metadata['event'], reference_step=metadata['healthy_step'],
                                 current_step=metadata['collapsed_step'], reference_test_accuracy=metadata['native_healthy']['heldout']['accuracy'],
                                 arm=arm, train_accuracy=results['train']['accuracy'], heldout_accuracy=heldout['accuracy'],
                                 heldout_cross_entropy=heldout['cross_entropy'], heldout_minimum_margin=heldout['minimum_margin'],
                                 heldout_centered_logit_relative_error=heldout['centered_healthy_logit_relative_error'],
                                 head_change_norm=results['head_change_norm'], relative_head_change=results['relative_head_change'],
                                 source=str(instant_path)))
    write_csv(ROOT / 'intervention_instant_table.csv', rows)
    continuation_rows = []
    for group in ['interventions', 'intervention_specificity', 'optimizer_memory_results']:
        for path in sorted((ROOT / group).glob('*/continuation/*/summary.json')):
            summary = json.loads(path.read_text())
            trajectory = list(csv.DictReader((path.parent / 'trajectory.csv').open()))
            first_post_failure = next((int(r['local_step']) for r in trajectory if int(r['local_step']) > 0 and float(r['train_accuracy']) < .9), None)
            continuation_rows.append(dict(study_group=group, event=summary['event'], arm=summary['arm'],
                                          steps=summary['steps'], initial_test_accuracy=summary['initial']['test_accuracy'],
                                          final_test_accuracy=summary['final']['test_accuracy'], minimum_train_accuracy=summary['minimum_train_accuracy'],
                                          minimum_measured_test_accuracy=summary['minimum_test_accuracy'], first_post_update_train_below90=first_post_failure,
                                          train_below90_states=summary['train_below_90_states'], measured_test_below90_states=summary['test_below_90_measured_states'],
                                          measured_test_states=summary['heldout_measurement_count'],
                                          only_head_parameters_changed=summary['only_head_parameters_changed'],
                                          every_optimizer_state_preserved=summary['every_optimizer_state_preserved'],
                                          head_optimizer_control=summary.get('head_optimizer_control', 'preserve'),
                                          hidden_and_embedding_optimizer_states_preserved=summary.get('hidden_and_embedding_optimizer_states_preserved', summary['every_optimizer_state_preserved']),
                                          head_adam_step_counter_reset=summary.get('head_adam_step_counter_reset', False), source=str(path)))
    write_csv(ROOT / 'intervention_continuation_table.csv', continuation_rows)
    report = dict(instant_evaluations=len(rows), completed_continuations=len(continuation_rows),
                  completed_continuation_updates=sum(r['steps'] for r in continuation_rows),
                  original_protocol=str(ROOT / 'interventions/protocol.json'),
                  supplementary_specificity_protocol=str(ROOT / 'intervention_specificity/protocol.json'),
                  reference_sensitivity_protocol=str(ROOT / 'intervention_near_reference/protocol.json'),
                  optimizer_memory_protocol=str(ROOT / 'optimizer_memory_results/protocol.json'),
                  verification=json.loads((ROOT / 'interventions/verification.json').read_text()),
                  instant_table=str(ROOT / 'intervention_instant_table.csv'),
                  continuation_table=str(ROOT / 'intervention_continuation_table.csv'))
    (ROOT / 'intervention_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
