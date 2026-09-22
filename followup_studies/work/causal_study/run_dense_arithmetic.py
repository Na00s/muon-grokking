"""Fixed-horizon arithmetic branch with every-update train-failure monitoring."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'experiments'))
import run_collapse as original
from run_precision_branches import make_arm, configure_runtime, file_sha256, optimizer_hyperparameters
from run_dense_capture import evaluate_preserving_runtime
from numerical_controls import accurate_cross_entropy
from diagnostics_repairs import ce_target_repair, ce_row_projection


def save(state,path):
    temporary=path.with_suffix('.tmp.pt');torch.save(state,temporary);temporary.replace(path)


def frozen_metrics(logits,y):
    result={};grads={}
    for name,dtype,fn in [('stock32',torch.float32,F.cross_entropy),('accurate32',torch.float32,accurate_cross_entropy),('reference64',torch.float64,accurate_cross_entropy)]:
        z=logits.detach().to(dtype).clone().requires_grad_(True)
        loss=fn(z,y);g,=torch.autograd.grad(loss,z);grads[name]=g.double()
        target=g.gather(1,y[:,None])[:,0]
        result[name]={'loss':float(loss.detach()),'zero_target_fraction':float((target==0).double().mean()),
                      'row_sum_norm':float(g.double().sum(-1).norm())}
    ref=grads['reference64'];norm=ref.norm()
    for name,g in grads.items():
        result[name]['relative_error']=float((g-ref).norm()/norm) if norm else None
        result[name]['cosine']=float(F.cosine_similarity(g.flatten()[None],ref.flatten()[None],eps=1e-300)) if norm and g.norm() else None
    return result


def run(checkpoint,out,arm,end_step,eval_every=100,save_every=1000,limit_updates=None):
    configure_runtime(1)
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model,opts,loss_fn=make_arm(source,'original32' if arm in ('target_repair','row_projection') else arm)
    if arm=='target_repair':loss_fn=ce_target_repair
    if arm=='row_projection':loss_fn=ce_row_projection
    tx,ty,vx,vy=original.generate_modular_addition_data(seed=source['seed'])
    start=int(source['step']);seed=int(source['seed'])
    if end_step<start:raise ValueError('End precedes source checkpoint')
    out.mkdir(parents=True,exist_ok=False)
    metadata=dict(seed=seed,arm=arm,start_step=start,end_step=end_step,
        source_checkpoint=str(Path(checkpoint).resolve()),source_checkpoint_sha256=file_sha256(checkpoint),
        paired_intervention_start=start,source_repository_commit=source['repository_commit'],
        current_repository_source_note='Model and optimizer code unchanged; publication-only commit is allowed.',
        model_config=source['model_config'],optimizer_hyperparameters=optimizer_hyperparameters(opts),
        train_monitor='Every state, before each update and at final state.',
        test_monitor=f'Every {eval_every} global updates, every train accuracy below90%, source and final states.',
        primary_endpoint='At least one joint training/test accuracy below90% by global step30000 after established grokking.',
        source_grokking='All source trajectories have already met six successive 100-grid test evaluations >=95%.',
        checkpoint_grid=save_every,torch_version=str(torch.__version__),threads=1,
        runner_sha256=file_sha256(__file__),numeric_helper_sha256=file_sha256(HERE.parent/'experiments'/'numerical_controls.py'))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def state(step):
        s=original.snapshot(model,opts,step,seed);s.update(arm=arm,branch_metadata=copy.deepcopy(metadata));return s
    save(state(start),out/'start.pt')
    first_event=None;previous=None;healthy=None;peak=None;min_train=1.;min_test=1.;train_bad=0;joint_bad=0
    started=time.perf_counter();gradient_records=[]
    fields=['step','local_step','train_loss','train_accuracy','test_accuracy','test_loss64','joint_failure','elapsed_seconds']
    with (out/'trajectory.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for step in range(start,end_step+1):
            model.train()
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            logits=model(tx);loss=loss_fn(logits,ty)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at{step}')
            ta=float((logits.detach().argmax(-1)==ty).double().mean());min_train=min(min_train,ta)
            te=None
            if step in [start,end_step] or step%eval_every==0 or ta<.9:
                te=evaluate_preserving_runtime(model,vx,vy);min_test=min(min_test,te['accuracy'])
            joint=bool(ta<.9 and te is not None and te['accuracy']<.9)
            train_bad+=ta<.9;joint_bad+=joint
            row=dict(step=step,local_step=step-start,train_loss=float(loss.detach()),train_accuracy=ta,
                     test_accuracy=te['accuracy'] if te else None,test_loss64=te['loss64'] if te else None,
                     joint_failure=joint,elapsed_seconds=time.perf_counter()-started)
            writer.writerow(row)
            if step%100==0 or joint or step==end_step:f.flush()
            s=None
            if step%save_every==0 or joint or (te is not None and ta>=.99 and te['accuracy']>=.99):s=state(step)
            if te is not None and ta>=.99 and te['accuracy']>=.99:healthy=s
            if step%save_every==0:
                save(s,out/f'step_{step:06d}.pt')
                print(json.dumps(row),flush=True)
            if step%1000==0 or joint and first_event is None:
                diagnostics=frozen_metrics(logits,ty)
                gradient_records.append(dict(step=step,metrics=diagnostics))
                (out/'gradient_metrics.json').write_text(json.dumps(gradient_records,indent=2)+'\n')
            if joint and first_event is None:
                first_event=row;save(s,out/'collapse.pt')
                if previous is not None:save(previous,out/'previous.pt')
                if healthy is not None:save(healthy,out/'healthy.pt')
                (out/'event.json').write_text(json.dumps(row,indent=2)+'\n')
                print('EVENT '+json.dumps(row),flush=True)
            if te is not None and (peak is None or te['accuracy']<peak['test_accuracy']):
                peak=row;save(s if s is not None else state(step),out/'minimum_test.pt')
            if step==end_step or (limit_updates is not None and step-start==limit_updates):
                save(state(step),out/'final.pt');break
            if first_event is None:previous=s if s is not None else state(step)
            loss.backward()
            for opt in opts.values():opt.step()
    summary=dict(status='completed',seed=seed,arm=arm,start_step=start,end_step=step,
        updates=step-start,first_joint_failure_step=first_event['step'] if first_event else None,
        minimum_train_accuracy=min_train,minimum_measured_test_accuracy=min_test,
        train_below90_states=train_bad,joint_below90_states=joint_bad,minimum_test_state=peak,
        final_state=row,elapsed_seconds=time.perf_counter()-started,
        event_count_scope='Counts refer only to this dense branch. Cached original prefixes use mixed monitoring and must not be compared by failure counts.',
        source_checkpoint_sha256=metadata['source_checkpoint_sha256'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=['original32','stable32','target_repair','row_projection'],required=True);p.add_argument('--end-step',type=int,default=30000)
    p.add_argument('--eval-every',type=int,default=100);p.add_argument('--save-every',type=int,default=1000)
    a=p.parse_args();run(a.checkpoint,a.out,a.arm,a.end_step,a.eval_every,a.save_every)
