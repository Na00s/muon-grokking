"""Independently audit complete long-horizon trajectories and final states."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import torch

import run_long_horizon as runner
from verify_runner import compare

HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def resolve_source_path(job):
    recorded = Path(job['checkpoint'])
    if recorded.is_file():
        return recorded
    if job['operation'] == 'addition':
        seed = int(job['name'].removeprefix('addition_seed'))
        return HERE.parent/'causal_study'/'main_runs'/f'seed{seed}_accurate6000'/'final.pt'
    if job['operation'] == 'subtraction' and job['name'] == 'subtraction_seed0':
        return HERE.parent/'causal_study'/'generality'/'subtraction_accurate'/'final.pt'
    raise ValueError(f'No portable source path for {job["name"]}')


def main():
    runner.configure_runtime(1)
    plan = json.loads((HERE/'protocol.json').read_text())
    assert (HERE/'completion.json').is_file(), 'The suite has not completed'
    checks = []
    records = []
    artifacts = []
    def check(name, passed, **details):
        checks.append(dict(name=name,passed=bool(passed),**details))
        assert passed, (name,details)
    for job in plan['jobs']:
        name = job['name']
        root = HERE/'runs'/name
        metadata = json.loads((root/'metadata.json').read_text())
        summary = json.loads((root/'summary.json').read_text())
        rows = read_csv(root/'trajectory.csv')
        source_path = resolve_source_path(job)
        check(name+' source hash',runner.file_sha256(source_path)==job['source_checkpoint_sha256'])
        check(name+' runner hash',metadata['runner_sha256']==plan['runner_sha256']==runner.file_sha256(HERE/'run_long_horizon.py'))
        check(name+' protocol hash',metadata['protocol_sha256']==runner.file_sha256(HERE/'protocol.json'))
        check(name+' completion horizon',summary['status']=='completed' and summary['end_step']==job['end_step'])
        steps = [int(row['step']) for row in rows]
        check(name+' all training states',steps==list(range(job['start_step'],job['end_step']+1)),count=len(rows))
        measured = [row for row in rows if row['test_accuracy']]
        required_steps = {step for step in steps if step%100==0 or step in (steps[0],steps[-1])}
        required_steps |= {int(row['step']) for row in rows if float(row['train_accuracy'])<.9}
        check(name+' test schedule',required_steps=={int(row['step']) for row in measured},count=len(measured))
        train_bad = [row for row in rows if float(row['train_accuracy'])<.9]
        joint = [row for row in measured if float(row['train_accuracy'])<.9 and float(row['test_accuracy'])<.9]
        test_bad = [row for row in measured if float(row['test_accuracy'])<.9]
        check(name+' joint flags',all((row['joint_failure']=='True') == (float(row['train_accuracy'])<.9 and bool(row['test_accuracy']) and float(row['test_accuracy'])<.9) for row in rows))
        check(name+' failure counts',summary['train_below90_states']==len(train_bad) and summary['joint_below90_states']==len(joint) and summary['test_below90_evaluations']==len(test_bad))
        check(name+' minimum training accuracy',summary['minimum_train_accuracy']==min(float(row['train_accuracy']) for row in rows))
        check(name+' minimum measured test accuracy',summary['minimum_measured_test_accuracy']==min(float(row['test_accuracy']) for row in measured))
        check(name+' source retained',not (root/'failure.json').exists() and not (root/'nonfinite.pt').exists())
        source = torch.load(source_path,map_location='cpu',weights_only=False)
        start = torch.load(root/'start.pt',map_location='cpu',weights_only=False)
        tensor_count = compare(source,start)
        check(name+' exact restored start',tensor_count>0,tensors=tensor_count)
        final = torch.load(root/'final.pt',map_location='cpu',weights_only=False)
        check(name+' seed and operation',final['seed']==int(name.rsplit('seed',1)[1])
              and final['operation']==job['operation']==metadata['operation'])
        check(name+' final checkpoint horizon',final['step']==100000)
        check(name+' final hash',runner.file_sha256(root/'final.pt')==summary['final_checkpoint_sha256'])
        model,opts,loss_fn = runner.make_arm(final,'stable32')
        tx,ty,vx,vy = runner.original.generate_modular_addition_data(seed=final['seed'],operation=job['operation'])
        check(name+' split counts',len(tx)==3830 and len(vx)==8939)
        model.train()
        with torch.no_grad():
            logits = model(tx)
            train_loss = float(loss_fn(logits,ty))
            train_accuracy = float((logits.argmax(-1)==ty).double().mean())
        test_metrics = runner.evaluate_preserving_runtime(model,vx,vy)
        check(name+' final train replay',train_accuracy==float(rows[-1]['train_accuracy']) and train_loss==float(rows[-1]['train_loss']))
        check(name+' final test replay',test_metrics['accuracy']==float(rows[-1]['test_accuracy']) and test_metrics['loss64']==float(rows[-1]['test_loss64']))
        minimum = torch.load(root/'minimum_test.pt',map_location='cpu',weights_only=False)
        minimum_row = next(row for row in rows if int(row['step'])==minimum['step'])
        check(name+' minimum checkpoint step',minimum['step']==summary['minimum_test_state']['step'])
        minimum_model,_,_ = runner.make_arm(minimum,'stable32')
        minimum_metrics = runner.evaluate_preserving_runtime(minimum_model,vx,vy)
        check(name+' minimum checkpoint replay',minimum_metrics['accuracy']==summary['minimum_measured_test_accuracy']
              and minimum_metrics['accuracy']==float(minimum_row['test_accuracy'])
              and minimum_metrics['loss64']==float(minimum_row['test_loss64']))
        expected_grid = {f'step_{step:06d}.pt' for step in range(job['start_step'],job['end_step']+1) if step%10000==0}
        check(name+' checkpoint grid',expected_grid=={path.name for path in root.glob('step_*.pt')})
        grid_metadata_matches = True
        for filename in sorted(expected_grid):
            grid_state = torch.load(root/filename,map_location='cpu',weights_only=False)
            grid_metadata_matches &= (grid_state['step']==int(filename.removeprefix('step_').removesuffix('.pt'))
                                      and grid_state['seed']==final['seed'] and grid_state['operation']==job['operation'])
            if grid_state['step']==100000:
                final_grid_tensor_count = compare(grid_state,final)
        check(name+' grid snapshot metadata',grid_metadata_matches)
        check(name+' final equals grid snapshot',final_grid_tensor_count>0,tensors=final_grid_tensor_count)
        if joint:
            check(name+' first event',summary['first_joint_failure_step']==int(joint[0]['step']))
            for file in ['collapse.pt','previous.pt','event.json']:
                check(name+' retained '+file,(root/file).is_file())
        else:
            check(name+' no event',summary['first_joint_failure_step'] is None)
        for path in sorted(root.glob('*.pt')):
            artifacts.append(dict(path=str(path.relative_to(HERE)),bytes=path.stat().st_size,sha256=runner.file_sha256(path)))
        if job['operation']=='addition':
            prefix_path = HERE.parent/'causal_study'/'main_runs'/f'seed{final["seed"]}_accurate6000'
            prefix = json.loads((prefix_path/'summary.json').read_text())
            combined_start = prefix['start_step']
            total_train_states = prefix['updates']+1+summary['updates']
            prior_min_train = prefix['minimum_train_accuracy']
            prior_min_test = prefix['minimum_measured_test_accuracy']
            prefix_first = prefix['first_joint_failure_step']
            prefix_final = prefix['final_state']
            prefix_rows = read_csv(prefix_path/'trajectory.csv')
            total_test_evaluations = sum(bool(row['test_accuracy']) for row in prefix_rows)+len(measured)-1
        else:
            prefix_path = HERE.parent/'causal_study'/'generality'/'subtraction_accurate'
            prefix = json.loads((prefix_path/'completion.json').read_text())
            combined_start = prefix['grok_step']
            total_train_states = 100000-combined_start+1
            prior_min_train = prefix['post_grok_train_min']
            prior_min_test = prefix['post_grok_test_min']
            prefix_first = prefix['event_step']
            prefix_final = dict(train_accuracy=prefix['final_train_accuracy'],test_accuracy=prefix['final_test_accuracy'])
            prefix_rows = [row for row in read_csv(prefix_path/'trajectory.csv') if int(row['step'])>=combined_start]
            total_test_evaluations = sum(bool(row['test_accuracy']) for row in prefix_rows)+len(measured)-1
        check(name+' source accuracy boundary',float(rows[0]['train_accuracy'])==prefix_final['train_accuracy'] and float(rows[0]['test_accuracy'])==prefix_final['test_accuracy'])
        records.append(dict(name=name,seed=final['seed'],operation=job['operation'],arithmetic='accurate',
            arithmetic_start_step=6000 if job['operation']=='addition' else 0,
            monitoring_start_step=combined_start,end_step=100000,
            total_training_updates_from_initialization=100000,
            accurate_arithmetic_updates=94000 if job['operation']=='addition' else 100000,
            extension_updates=summary['updates'],
            monitored_train_states=total_train_states,test_evaluations=total_test_evaluations,
            minimum_train_accuracy=min(prior_min_train,summary['minimum_train_accuracy']),
            minimum_measured_test_accuracy=min(prior_min_test,summary['minimum_measured_test_accuracy']),
            final_train_accuracy=train_accuracy,final_test_accuracy=test_metrics['accuracy'],
            first_joint_failure_step=prefix_first if prefix_first is not None else summary['first_joint_failure_step'],
            extension_joint_failure_states=summary['joint_below90_states'],
            extension_test_below90_evaluations=len(test_bad),
            extension_test_below95_evaluations=sum(float(row['test_accuracy'])<.95 for row in measured),
            extension_minimum_test_accuracy=summary['minimum_measured_test_accuracy'],
            extension_seconds=summary['elapsed_seconds'],
            prefix_summary=str(prefix_path/('summary.json' if job['operation']=='addition' else 'completion.json')),
            extension_summary=str(root/'summary.json')))
    runner.write_json(HERE/'summary.json',dict(status='completed',records=records,
        scope='All outcomes concern these fixed seeds and a finite 100,000-step horizon. Test-only excursions between scheduled evaluations remain outside detection.',
        original_source_intervention='Addition is a matched branch from step 6,000; subtraction uses accurate arithmetic from initialization.'))
    with (HERE/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    runner.write_json(HERE/'verification.json',dict(status='passed',checks=checks,passed=sum(c['passed'] for c in checks),total=len(checks),
        runner_sha256=runner.file_sha256(HERE/'run_long_horizon.py'),protocol_sha256=runner.file_sha256(HERE/'protocol.json')))
    runner.write_json(HERE/'checkpoint_manifest.json',dict(
        scope='Retained scientific-run checkpoints under runs/. Verification fixtures are separate.',
        artifacts=artifacts,total_bytes=sum(a['bytes'] for a in artifacts)))
    print(json.dumps(dict(status='passed',checks=len(checks),records=records),indent=2))


if __name__ == '__main__': main()
