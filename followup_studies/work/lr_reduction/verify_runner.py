"""Independent bounded checks for the post-grokking LR continuation runner."""
from __future__ import annotations
import contextlib
import copy
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_lr_reduction as runner
original=runner.original


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tree_equal(left,right,path='root'):
    if isinstance(left,torch.Tensor):
        assert isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right),path
    elif isinstance(left,dict):
        assert isinstance(right,dict) and left.keys()==right.keys(),path
        for key in left:tree_equal(left[key],right[key],f'{path}/{key}')
    elif isinstance(left,(list,tuple)):
        assert type(left)==type(right) and len(left)==len(right),path
        for idx,(a,b) in enumerate(zip(left,right)):tree_equal(a,b,f'{path}/{idx}')
    else:assert left==right,(path,left,right)


def state_equal(left,right):
    for name in ['step','seed','model_config','model_state_dict','optimizer_state_dicts','torch_rng_state','repository_commit']:
        tree_equal(left[name],right[name],name)


def audit_rows(directory,end,eval_every):
    summary=json.loads((directory/'summary.json').read_text())
    rows=list(csv.DictReader((directory/'trajectory.csv').open()))
    assert [int(r['step']) for r in rows]==list(range(6000,end+1))
    measured=[r for r in rows if r['test_accuracy']!='']
    for r in rows:
        step=int(r['step']);train=float(r['train_accuracy'])
        expected=step in(6000,end) or step%eval_every==0 or train<.9
        assert bool(r['test_accuracy']!='')==expected,(step,expected)
        assert (r['joint_failure']=='True')==bool(train<.9 and r['test_accuracy'] and float(r['test_accuracy'])<.9)
    bad=[r for r in rows if float(r['train_accuracy'])<.9]
    joint=[r for r in rows if r['joint_failure']=='True']
    offgrid=[r for r in measured if int(r['step']) not in(6000,end) and int(r['step'])%eval_every!=0]
    expected=dict(updates=end-6000,train_states=end-6000+1,test_evaluations=len(measured),train_below90_states=len(bad),test_below90_evaluations=sum(float(r['test_accuracy'])<.9 for r in measured),joint_below90_states=len(joint),test_below95_evaluations=sum(float(r['test_accuracy'])<.95 for r in measured),scheduled_test_below95_evaluations=sum(float(r['test_accuracy'])<.95 and (int(r['step']) in(6000,end) or int(r['step'])%eval_every==0) for r in measured),off_grid_triggered_test_evaluations=len(offgrid),first_joint_failure_step=int(joint[0]['step']) if joint else None,minimum_train_accuracy=min(float(r['train_accuracy']) for r in rows),minimum_measured_test_accuracy=min(float(r['test_accuracy']) for r in measured))
    for key,value in expected.items():assert summary[key]==value,(key,summary[key],value)
    assert summary['end_step']==end and summary['final_state']['step']==end
    assert summary['final_state']['test_accuracy']==float(rows[-1]['test_accuracy'])
    assert summary['final_checkpoint_sha256']==sha(directory/'final.pt')
    return expected


def diagnostic_reference(model,tx,ty,vx):
    mode=model.training;rng=torch.get_rng_state().clone()
    captured=[]
    handle=model.unembedding.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().double()))
    with torch.no_grad():
        model.train()
        train_logits=model(tx)
        train_features=captured.pop()
        model.eval()
        for start in range(0,len(vx),1024):model(vx[start:start+1024])
    handle.remove();model.train(mode)
    assert torch.equal(rng,torch.get_rng_state())
    test_features=torch.cat(captured)
    # Float64 accumulation over the union of training and held-out examples.
    full_mean=(train_features.sum(0)+test_features.sum(0))/(len(tx)+len(vx))
    z=train_logits.detach().clone().requires_grad_(True)
    stock,=torch.autograd.grad(F.cross_entropy(z,ty),z)
    z64=z.detach().double(); shifted=z64-z64.max(-1,keepdim=True).values
    exp=shifted.exp();prob=exp/exp.sum(-1,keepdim=True)
    reference=prob.clone()
    wrong=prob.clone();wrong.scatter_(1,ty[:,None],0.)
    reference.scatter_(1,ty[:,None],-wrong.sum(-1,keepdim=True))
    reference/=len(ty)
    relative=float((stock.double()-reference).norm()/reference.norm())
    return dict(train_feature_mean_norm=float(train_features.mean(0).norm()),full_grid_feature_mean_norm=float(full_mean.norm()),stock32_relative_error=relative,stock32_zero_target_fraction=float((stock.gather(1,ty[:,None])==0).double().mean()),stock32_row_sum_norm=float(stock.double().sum(-1).norm()))


def main():
    runner.configure_runtime(1)
    out=HERE/'verification_runs'
    out.mkdir(exist_ok=False)
    hashes={}
    results=[]
    for seed in range(5):
        group='experiments' if seed in(0,4) else 'basis_study'
        label='original' if seed in(0,4) else 'baseline'
        source_path=HERE.parent/group/f'seed{seed}_{label}'/'step_006000.pt'
        hashes[str(source_path.relative_to(HERE.parent))]=sha(source_path)
        source=torch.load(source_path,map_location='cpu',weights_only=False)
        assert source['step']==6000 and source['seed']==seed
        model,opts,loss_fn=runner.make_arm(source,'original32')
        for g in opts['muon'].param_groups:g['learning_rate']=.003
        initial=original.snapshot(model,opts,6000,seed)
        expected=copy.deepcopy(source)
        for g in expected['optimizer_state_dicts']['muon']['param_groups']:g['learning_rate']=.003
        state_equal(initial,expected)
        assert all(p.requires_grad for p in model.parameters())
        tx,ty,vx,vy=original.generate_modular_addition_data(seed=seed)
        references={}
        manual_six=None
        for step in range(6000,6013):
            if step%3==0:references[step]=diagnostic_reference(model,tx,ty,vx)
            if step==6006:manual_six=original.snapshot(model,opts,step,seed)
            if step==6012:break
            original.train_step(model,opts,tx,ty)
        manual_final=original.snapshot(model,opts,6012,seed)
        destination=out/f'seed{seed}'
        with (out/f'seed{seed}.log').open('w') as stream,contextlib.redirect_stdout(stream):
            runner.run(source_path,destination,6012,eval_every=4,save_every=6,diagnostic_every=3)
        saved_start=torch.load(destination/'start.pt',map_location='cpu',weights_only=False)
        saved_six=torch.load(destination/'step_006006.pt',map_location='cpu',weights_only=False)
        saved_final=torch.load(destination/'final.pt',map_location='cpu',weights_only=False)
        state_equal(saved_start,expected)
        state_equal(saved_six,manual_six)
        state_equal(saved_final,manual_final)
        resumed,ropts,rloss=runner.make_arm(saved_six,'original32')
        for _ in range(6):original.train_step(resumed,ropts,tx,ty)
        resumed_final=original.snapshot(resumed,ropts,6012,seed)
        state_equal(resumed_final,manual_final)
        row_audit=audit_rows(destination,6012,4)
        diagnostics=json.loads((destination/'gradient_metrics.json').read_text())
        assert [r['step'] for r in diagnostics]==[6000,6003,6006,6009,6012]
        max_abs_error=0.
        for row in diagnostics:
            ref=references[row['step']]
            for key in ['train_feature_mean_norm','full_grid_feature_mean_norm']:
                assert row[key]==ref[key],(seed,row['step'],key,row[key],ref[key])
            for field,reference_name in [('relative_error','stock32_relative_error'),('zero_target_fraction','stock32_zero_target_fraction'),('row_sum_norm','stock32_row_sum_norm')]:
                difference=abs(row['metrics']['stock32'][field]-ref[reference_name])
                max_abs_error=max(max_abs_error,difference)
                assert difference<1e-12,(seed,row['step'],field,difference)
        assert all(float(s['step'])==6012 for name in('auxiliary_adamw','unembedding_adamw') for s in saved_final['optimizer_state_dicts'][name]['state'].values())
        assert all(g['learning_rate']==.003 and g['weight_decay']==.1 for g in saved_final['optimizer_state_dicts']['muon']['param_groups'])
        metadata=json.loads((destination/'metadata.json').read_text())
        assert metadata['restored_optimizer_hyperparameters']['muon'][0]['learning_rate']==.03
        assert metadata['optimizer_hyperparameters']['muon'][0]['learning_rate']==.003
        results.append(dict(seed=seed,source_sha256=sha(source_path),restored_states_equal_except_hidden_learning_rate=True,all_parameters_trainable=True,twelve_updates_model_optimizer_rng_bitwise_equal=True,six_plus_six_serialized_resume_bitwise_equal=True,diagnostic_feature_norms_exact=True,diagnostic_derivative_reference_maximum_absolute_difference=max_abs_error,training_update_counters_correct=True,monitoring=row_audit,diagnostic_steps=[r['step'] for r in diagnostics]))
        print(f'verified seed {seed}',flush=True)
    # A copied source with a zero readout supplies a known low-training state.
    # This verification-only case is explicitly segregated from the five scientific runs.
    source_path=HERE.parent/'experiments/seed0_original/step_006000.pt'
    injected=copy.deepcopy(torch.load(source_path,map_location='cpu',weights_only=False))
    injected['model_state_dict']['unembedding.weight'].zero_()
    injection=out/'verification_only_zero_readout.pt'
    torch.save(injected,injection)
    destination=out/'synthetic_train_trigger'
    with (out/'synthetic_train_trigger.log').open('w') as stream,contextlib.redirect_stdout(stream):
        runner.run(injection,destination,6003,eval_every=100,save_every=6,diagnostic_every=3)
    trigger=audit_rows(destination,6003,100)
    assert trigger['first_joint_failure_step']==6000
    assert trigger['off_grid_triggered_test_evaluations']>=1,trigger
    assert (destination/'collapse.pt').exists() and (destination/'event.json').exists()
    for path,expected_hash in hashes.items():assert sha(HERE.parent/path)==expected_hash,path
    result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),runner_sha256=sha(runner.__file__),verifier_sha256=sha(__file__),seeds=results,synthetic_trigger=dict(verification_only=True,construction='Copied seed0 step6000 state with readout weights set to zero. Original checkpoint unchanged.',monitoring=trigger),all_original_source_checkpoint_hashes_unchanged=True,scope='Only the new experiment directory was written. Paper frozen. These short verification trajectories are excluded from scientific results.')
    (HERE/'runner_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'seeds':len(results),'synthetic_offgrid_triggered_evaluations':trigger['off_grid_triggered_test_evaluations'],'output':str(HERE/'runner_verification.json')}))


if __name__=='__main__':main()
