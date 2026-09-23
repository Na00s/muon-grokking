"""Audit final horizons, dense endpoint traces, and saved final model accuracy."""
import csv
import hashlib
import json
import re
from pathlib import Path
import torch
from aggregate import inspect_branch,read
from run_dense_arithmetic import original,configure_runtime,make_arm
import generality_runner as general

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs'/'causal_study'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(actual,expected):assert abs(actual-expected)<1e-12,(actual,expected)

def equal_state(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b

def source_path(recorded):
    """Prefer the local study tree when an archive is moved to another machine."""
    value=Path(recorded)
    if 'work' in value.parts:
        local=HERE.parent.joinpath(*value.parts[value.parts.index('work')+1:])
        if local.is_file():return local
    assert value.is_file(),value
    return value

def verify_source_qualification():
    manifest=read(HERE/'source_grokking_manifest.json')
    assert manifest['all_five_qualified'] and {r['seed'] for r in manifest['seeds']}==set(range(5))
    for record in manifest['seeds']:
        checkpoint=source_path(record['source_checkpoint'])
        trajectory=source_path(record['trajectory'])
        assert sha(checkpoint)==record['source_sha256']
        # Publication preserves numeric CSV content while normalizing newlines.
        allowed={record['trajectory_sha256']}
        if record.get('trajectory_normalized_sha256'):allowed.add(record['trajectory_normalized_sha256'])
        assert sha(trajectory) in allowed
        with trajectory.open() as stream:rows={int(r['step']):r for r in csv.DictReader(stream)}
        steps=record['six_evaluation_steps']
        assert len(steps)==6 and steps==list(range(steps[0],steps[0]+600,100)) and steps[-1]<=6000
        assert all(float(rows[step]['test_accuracy'])>=.95 for step in steps)
        assert [float(rows[step]['test_accuracy']) for step in steps]==record['six_test_accuracies']
    return dict(seeds=5,checkpoint_and_trajectory_hashes_verified=True,six_evaluation_qualification_verified=True)

@torch.no_grad()
def final_accuracy(model,seed,operation='addition'):
    x,y,vx,vy=original.generate_modular_addition_data(seed=seed,operation=operation)
    model.train()
    train=float((model(x).argmax(-1)==y).double().mean())
    test=original.evaluate(model,vx,vy)['accuracy']
    return dict(train_accuracy=train,test_accuracy=test)

def main():
    configure_runtime(1)
    results=read(HERE/'analysis'/'results.json');assert results['complete']
    qualification=verify_source_qualification()
    plan=read(HERE/'main_suite_plan.json')
    planned={f'main_runs/{job["name"]}':job for job in plan['jobs']}
    for seed in range(5):
        basis=next(j for j in plan['jobs'] if j['name']==f'seed{seed}_target_repair15000')
        planned[f'projection_runs/seed{seed}']={**basis,'arm':'row_projection'}
    assert {row['path'] for row in results['branches']}==set(planned)
    branch_checks=[]
    for row in results['branches']:
        directory=HERE/row['path'];again=inspect_branch(directory,row['end'])
        assert again==row
        job=planned[row['path']];metadata=read(directory/'metadata.json')
        assert row['seed']==job['seed'] and row['arm']==job['arm'] and row['end']==job['end']
        source=source_path(metadata['source_checkpoint'])
        assert sha(source)==metadata['source_checkpoint_sha256']==row['source_checkpoint_sha256']==job['checkpoint_sha256']
        inherited=torch.load(source,map_location='cpu',weights_only=False)
        start=torch.load(directory/'start.pt',map_location='cpu',weights_only=False)
        assert inherited['step']==start['step']==row['start']
        for key in ('model_state_dict','optimizer_state_dicts','torch_rng_state'):
            assert equal_state(inherited[key],start[key]),(row['path'],key,'source state changed at branching')
        state=torch.load(directory/'final.pt',map_location='cpu',weights_only=False)
        assert state['step']==row['end'] and state['seed']==row['seed']
        model,_,_=make_arm(state,'original32')
        acc=final_accuracy(model,row['seed'])
        close(acc['train_accuracy'],row['final_train_accuracy']);close(acc['test_accuracy'],row['final_test_accuracy'])
        branch_checks.append(dict(name=row['path'],end_step=row['end'],source_checkpoint_sha256=sha(source),
            start_model_optimizer_rng_match_source=True,final_checkpoint_sha256=sha(directory/'final.pt'),**acc,passed=True))
    # Seed 4 already reached the shared endpoint in the earlier published study.
    reused_directory=HERE.parent/'experiments'/'seed4_original'
    reused_path=reused_directory/'step_030000.pt'
    reused_state=torch.load(reused_path,map_location='cpu',weights_only=False)
    assert reused_state['step']==30000 and reused_state['seed']==4
    reused_model,_,_=make_arm(reused_state,'original32')
    reused_acc=final_accuracy(reused_model,4)
    with (reused_directory/'trajectory.csv').open() as stream:reused_rows=list(csv.DictReader(stream))
    reused_row=next(row for row in reused_rows if int(row['step'])==30000)
    close(reused_acc['train_accuracy'],float(reused_row['train_accuracy']))
    close(reused_acc['test_accuracy'],float(reused_row['test_accuracy']))
    reused_check=dict(seed=4,step=30000,checkpoint_sha256=sha(reused_path),**reused_acc,passed=True)
    general_checks=[]
    initial_states={}
    for name in ['subtraction_stock','subtraction_accurate','rms_stock','rms_accurate']:
        directory=HERE/'generality'/name
        summary=read(directory/'completion.json');assert summary['steps']==30000
        metadata=read(directory/'metadata.json')
        assert metadata['steps']==summary['steps'] and metadata['seed']==summary['seed']
        assert metadata['operation']==summary['operation'] and metadata['normalization']==summary['normalization']
        assert metadata['arithmetic']==summary['arithmetic']
        with (directory/'trajectory.csv').open() as stream:rows=list(csv.DictReader(stream))
        grid={int(x['step']):x for x in rows if int(x['step'])%10==0}
        assert set(grid)==set(range(0,30001,10))
        streak=0;qualified=None
        for step in range(0,30001,100):
            row=grid[step];assert row['test_accuracy']
            streak=streak+1 if float(row['test_accuracy'])>=.95 else 0
            if streak>=6 and qualified is None:qualified=step
        assert qualified==summary['grok_step']
        events=[x for x in rows if x['joint_failure']=='True']
        assert (int(events[0]['step']) if events else None)==summary['event_step']
        assert all(float(x['train_accuracy'])<.9 and float(x['test_accuracy'])<.9 for x in events)
        for row in rows:
            step=int(row['step'])
            assert (row['grok_confirmed']=='True')==(qualified is not None and step>=qualified)
            if qualified is not None and step>=qualified and float(row['train_accuracy'])<.9:
                assert row['test_accuracy'],(name,step,'missing test measurement at low-training state')
                assert (row['joint_failure']=='True')==(float(row['test_accuracy'])<.9)
        if events:
            audit=read(directory/'event_analysis.json')
            assert audit['exact_next_update_replay'] and audit['post_step']==summary['event_step']
            assert (directory/'adjacent_analysis.json').is_file()
        initial_states[name]=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
        assert initial_states[name]['step']==0 and initial_states[name]['seed']==summary['seed']
        state=torch.load(directory/'final.pt',map_location='cpu',weights_only=False)
        assert state['step']==30000 and state['seed']==summary['seed']
        assert state['normalization']==summary['normalization'] and state['operation']==summary['operation']
        model,_=general.make_model_optimizers(summary['seed'],summary['normalization'],checkpoint=state)
        acc=final_accuracy(model,summary['seed'],summary['operation'])
        close(acc['train_accuracy'],summary['final_train_accuracy']);close(acc['test_accuracy'],summary['final_test_accuracy'])
        general_checks.append(dict(name=name,normalization=summary['normalization'],operation=summary['operation'],
            grok_step=qualified,event_step=summary['event_step'],final_checkpoint_sha256=sha(directory/'final.pt'),
            runner_sha256=metadata['script_sha256'],**acc,passed=True))
    for condition in ('subtraction','rms'):
        a,b=[initial_states[f'{condition}_{arm}'] for arm in ('stock','accurate')]
        for key in ('model_state_dict','optimizer_state_dicts','torch_rng_state'):
            assert equal_state(a[key],b[key]),(condition,key,'generality initialization mismatch')
    assert len(branch_checks)==24 and len(general_checks)==4
    diagnostic_summary=read(HERE/'diagnostics_results'/'summary.json')
    assert diagnostic_summary['exact_model_optimizer_rng_replays']==5
    mean_errors=[read(HERE/'diagnostics_results'/f'seed{seed}'/'mean_mediation.json')['component_sum_relative_l2'] for seed in range(5)]
    assert max(mean_errors)<3e-16
    test_counts={}
    for label,name in [('numerical','tests.log'),('diagnostic','diagnostic_tests_final.log')]:
        log=(HERE/name).read_text()
        assert log.rstrip().endswith('OK'),name
        counts=re.findall(r'Ran (\d+) tests',log)
        assert len(counts)==1,name
        test_counts[label]=int(counts[0])
    backward_checks=read(HERE/'diagnostics_results'/'actual_backward_rule_verification.json')
    assert len(backward_checks)==4 and all(row['all_parameter_gradients_bitwise_equal'] for row in backward_checks)
    result=dict(complete=True,total_new_training_updates=results['new_main_and_projection_updates']+120000,
        main_and_projection_final_checks=branch_checks,generality_final_checks=general_checks,
        reused_original_seed4_final_check=reused_check,
        source_qualification=qualification,generality_pair_initial_model_optimizer_rng_match=True,
        diagnostic_counts=diagnostic_summary['counts'],maximum_mean_component_reconstruction_error=max(mean_errors),
        main_endpoint_monitoring='Every training state is recorded. Every training state below90% has a held-out measurement. Final states are explicit.',
        generality_monitoring='Training is checked every update by the verified runner; CSV retains every10th state and every joint failure. All100-step qualification evaluations and the final state are present.',
        generality_loader='RMS checkpoints require generality_runner.make_model_optimizers with normalization=rms; subtraction requires subtraction labels.',
        numerical_unit_tests=test_counts['numerical'],diagnostic_mathematical_tests=test_counts['diagnostic'],
        actual_backward_rule_bitwise_gradient_checks=len(backward_checks),
        exact_original_event_model_optimizer_rng_replays=5,
        exact_runner_three_update_checks=['original32','stable32','target_repair','row_projection'],
        metadata_erratum=results['metadata_erratum'])
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,main_checks=len(branch_checks),generality_checks=len(general_checks),updates=result['total_new_training_updates'])))

if __name__=='__main__':main()
