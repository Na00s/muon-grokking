"""Audit completed intervention files, saved states, counters, and metrics."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
import interventions as study


def first_step(optimizer_state):
    entries = list(optimizer_state['state'].values())
    return float(entries[0]['step']) if entries else 0.


def main():
    study.configure_runtime(1)
    manifest = json.loads((study.HERE / 'source_versions/manifest.json').read_text())
    for digest, path in manifest.items():
        assert study.file_sha256(path) == digest
    reports = []
    for group in ['interventions', 'intervention_specificity', 'optimizer_memory_results']:
        for path in sorted((study.HERE / group).glob('*/continuation/*/summary.json')):
            out = path.parent
            summary = json.loads(path.read_text())
            source = torch.load(summary['source_checkpoint'], map_location='cpu', weights_only=False)
            start = torch.load(out / 'start.pt', map_location='cpu', weights_only=False)
            final = torch.load(out / 'final.pt', map_location='cpu', weights_only=False)
            control = summary.get('head_optimizer_control', 'preserve')
            assert summary['script_sha256'] in manifest
            assert study.file_sha256(summary['source_checkpoint']) == summary['source_checkpoint_sha256']
            assert all(torch.equal(v, source['model_state_dict'][k]) for k, v in start['model_state_dict'].items() if k != 'unembedding.weight')
            assert all(study.state_equal(v, source['optimizer_state_dicts'][k]) for k, v in start['optimizer_state_dicts'].items() if k != 'unembedding_adamw')
            expected_head = source['optimizer_state_dicts']['unembedding_adamw']
            import copy
            expected_head = copy.deepcopy(expected_head)
            if control == 'clear_head_state':
                expected_head['state'] = {}
            elif control == 'zero_head_first_moment':
                for value in expected_head['state'].values():
                    value['exp_avg'].zero_()
            assert study.state_equal(start['optimizer_state_dicts']['unembedding_adamw'], expected_head)
            assert torch.equal(start['torch_rng_state'], source['torch_rng_state'])
            expected_step = 500 if control == 'clear_head_state' else first_step(source['optimizer_state_dicts']['unembedding_adamw']) + 500
            assert first_step(final['optimizer_state_dicts']['unembedding_adamw']) == expected_step
            assert final['step'] == source['step'] + 500
            assert all(torch.isfinite(t).all() for t in final['model_state_dict'].values())
            heads = np.load(out.parent.parent / 'heads.npz')
            assert torch.equal(start['model_state_dict']['unembedding.weight'], torch.from_numpy(heads[summary['arm']].T).float())
            trajectory = list(csv.DictReader((out / 'trajectory.csv').open()))
            assert len(trajectory) == 501
            assert [int(row['local_step']) for row in trajectory] == list(range(501))
            model, _, _ = study.make_arm(final, 'original32')
            _, _, test_x, test_y = study.original.generate_modular_addition_data(seed=source['seed'])
            actual = study.original.evaluate(model, test_x, test_y)
            assert actual['accuracy'] == summary['final']['test_accuracy']
            reports.append(dict(event=summary['event'], arm=summary['arm'], optimizer_control=control,
                                completed_updates=500, train_states=501, final_native_accuracy_verified=actual['accuracy'],
                                final_head_adam_step=expected_step, source_sha_and_runner_sha_verified=True,
                                start_state_intervention_exact=True, only_specified_optimizer_entries_changed=True))
    assert len(reports) == 18
    result = dict(status='passed', completed_continuations=len(reports), completed_updates=sum(r['completed_updates'] for r in reports),
                  runner_versions_verified=len(manifest), continuations=reports)
    study.write_json(study.HERE / 'intervention_final_audit.json', result)
    print(json.dumps(dict(status='passed', completed_continuations=len(reports), completed_updates=9000)))


if __name__ == '__main__':
    main()
