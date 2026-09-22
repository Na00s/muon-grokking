"""Consolidate the fixed instantaneous recipe on three fresh seed runs."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import interventions as study

ROOT = study.HERE


def main():
    root = ROOT / 'fresh_interventions'
    cases = ['seed1_first', 'seed1_adjacent', 'seed2_first', 'seed2_adjacent', 'seed3_first', 'seed3_peak', 'seed3_adjacent']
    records, summaries = [], []
    for name in cases:
        folder = root / name / name
        metadata = json.loads((folder / 'metadata.json').read_text())
        primary = json.loads((folder / 'instant.json').read_text())
        controls = json.loads((folder.parent / 'random_specificity' / name / 'instant.json').read_text())
        assert study.file_sha256(metadata['source_feature_path']) == metadata['source_feature_sha256']
        assert study.file_sha256(metadata['collapsed_checkpoint']) == metadata['collapsed_checkpoint_sha256']
        assert study.file_sha256(metadata['healthy_checkpoint']) == metadata['healthy_checkpoint_sha256']
        assert primary['original']['heldout']['accuracy'] == metadata['native_collapsed']['heldout']['accuracy']
        assert len(primary) == 10 and len(controls) == 11
        target_norm = primary['orthogonal']['head_change_norm']
        for arm, metrics in controls.items():
            assert abs(metrics['head_change_norm'] - target_norm) / target_norm < 2e-6
        for group, values in [('primary', primary), ('orthogonal_size_control', controls)]:
            for arm, metrics in values.items():
                h = metrics['heldout']
                records.append(dict(pair=name, seed=metadata['seed'], reference_role=metadata.get('reference_role', 'selected_healthy_reference'),
                                    reference_step=metadata['healthy_step'], current_step=metadata['collapsed_step'],
                                    reference_test_accuracy=metadata['native_healthy']['heldout']['accuracy'],
                                    arm_group=group, arm=arm, train_accuracy=metrics['train']['accuracy'],
                                    heldout_accuracy=h['accuracy'], heldout_cross_entropy=h['cross_entropy'],
                                    heldout_minimum_margin=h['minimum_margin'], heldout_centered_reference_logit_relative_error=h['centered_healthy_logit_relative_error'],
                                    relative_head_change=metrics['relative_head_change'], head_change_norm=metrics['head_change_norm']))
        random_values = [v['heldout']['accuracy'] for k, v in controls.items() if k.startswith('random')]
        summaries.append(dict(pair=name, reference_step=metadata['healthy_step'], current_step=metadata['collapsed_step'],
                              reference_accuracy=metadata['native_healthy']['heldout']['accuracy'],
                              original=primary['original']['heldout']['accuracy'], old_head=primary['old_head']['heldout']['accuracy'],
                              orthogonal=primary['orthogonal']['heldout']['accuracy'], inverse_gl=primary['inverse_gl']['heldout']['accuracy'],
                              old_head_relative_change=primary['old_head']['relative_head_change'],
                              random_orthogonal_size_minimum=min(random_values), random_orthogonal_size_maximum=max(random_values),
                              opposite_correction=controls['opposite_orthogonal_correction']['heldout']['accuracy'],
                              source_hashes_verified=True, native_baseline_reproduced=True, correction_norms_verified=True))
    for filename, rows in [('instant_table.csv', records), ('pair_summary.csv', summaries)]:
        with (root / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    study.write_json(root / 'summary.json', dict(status='completed', pairs=len(cases), instant_evaluations=len(records), continuation_updates=0,
                                              verification='All source hashes verified; each native baseline reproduced; all orthogonal-size controls match their target correction norms within 2e-6 relative error.',
                                              cases=summaries))
    lines = ['# Fresh-seed readout interventions', '',
             'The fixed intervention recipe was applied to seven pairs from three fresh runs, producing 147 instantaneous evaluations. Every fit used original training rows only, without labels. All outcomes below use actual float32 model forward passes after installing the fitted or control head. These experiments added no continuation training.', '',
             '| Pair | Reference to evaluated step | Reference test | Collapsed test | Old head | Orthogonal | Inverse GL | Random orthogonal-size range |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        values = [s[k] for k in ['reference_accuracy', 'original', 'old_head', 'orthogonal', 'inverse_gl']]
        lines.append(f"| {s['pair']} | {s['reference_step']} to {s['current_step']} | " + ' | '.join(f'{100*v:.3f}%' for v in values) + f" | {100*s['random_orthogonal_size_minimum']:.2f}% to {100*s['random_orthogonal_size_maximum']:.2f}% |")
    lines += ['',
              'Nearby old heads restore most or all of the preceding reference accuracy in every fresh seed. The preceding reference itself may already have degraded generalization; that value is shown explicitly. This repeats the original two-seed reference-time finding and supports a substantial readout contribution to the sudden failures.', '',
              'For the farther healthy references, inverse-GL compensation provides a strong rescue, but its accuracy can differ from orthogonal transport and from a nearby old-head swap. A fitted compensation that improves classification establishes linear decodability relative to that reference. Combined with old-head controls, these results do not isolate a unique representational-rotation cause.', '',
              'The random-direction controls match the magnitude of the orthogonal correction. Some small random perturbations improve a severely collapsed baseline, especially when chance accuracy is the starting point. The directed repairs remain substantially stronger; the complete random range is included rather than selecting a single comparison.', '',
              'Verification confirmed every source feature and checkpoint hash, every actual-model baseline accuracy, and the requested correction norms. Numerical transport orientation and exact monitoring were validated in the initial intervention suite. The reusable runner is run_instant_interventions.py. Detailed rows are in instant_table.csv, compact pair rows in pair_summary.csv, and provenance plus checks in summary.json.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='completed', pairs=7, instant_evaluations=len(records))))


if __name__ == '__main__':
    main()
