"""Fresh baseline with every-step training inspection and bounded event follow-up."""
from collections import deque
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from run_collapse import make_model_optimizers, generate_modular_addition_data, snapshot, evaluate, CONFIG
from run_dense_capture import evaluate_preserving_runtime


def save(state,path):
    tmp=path.with_suffix('.tmp.pt');torch.save(state,tmp);os.replace(tmp,path)


def run(seed,out,max_steps=30000,follow_steps=200):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    model,opts=make_model_optimizers(seed)
    tx,ty,vx,vy=generate_modular_addition_data(seed=seed)
    out.mkdir(parents=True,exist_ok=False)
    metadata=dict(seed=seed,config=CONFIG,max_steps=max_steps,follow_steps=follow_steps,
        protocol='Six successive 100-step held-out evaluations >=95% establish grokking. Subsequently inspect train accuracy every step and held-out accuracy at every train<90% trigger. Capture first simultaneous <90% event and continue 200 steps.',
        selection='Additional seeds 1,2,3 chosen together before running; no condition changes.',
        torch_version=str(torch.__version__),threads=1,repository_commit='6d64a981af75f1300d9060109e81552d48a81360',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    save(snapshot(model,opts,0,seed),out/'initial.pt')
    recent=deque(maxlen=21);previous=None;last_joint_healthy=None
    streak=0;grokked=False;grok_confirmed=None;event=None;peak=None
    started=time.perf_counter()
    fields=['step','train_loss','train_accuracy','test_loss','test_accuracy','heldout_measured','post_grok','event_trigger','elapsed_seconds']
    with (out/'trajectory.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for step in range(max_steps+1):
            model.train()
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            logits=model(tx);loss=F.cross_entropy(logits,ty)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            train_acc=float((logits.detach().argmax(-1)==ty).double().mean())
            triggered=grokked and train_acc<.9
            te=None
            if step%100==0 or triggered or (event is not None and step%10==0):
                te=evaluate_preserving_runtime(model,vx,vy)
            if step%100==0:
                streak=streak+1 if te['accuracy']>=.95 else 0
                if streak>=6 and not grokked:grokked=True;grok_confirmed=step
            joint=bool(grokked and train_acc<.9 and te is not None and te['accuracy']<.9)
            row=dict(step=step,train_loss=float(loss.detach()),train_accuracy=train_acc,
                test_loss=te['loss'] if te else None,test_accuracy=te['accuracy'] if te else None,
                heldout_measured=te is not None,post_grok=grokked,event_trigger=joint,
                elapsed_seconds=time.perf_counter()-started)
            if step%10==0 or joint:
                writer.writerow(row);f.flush()
            state=None
            if step%10==0 or joint or (te is not None and train_acc>=.99 and te['accuracy']>=.99):
                state=snapshot(model,opts,step,seed);state['capture_metrics']=row
            if step%10==0:
                recent.append(state)
            if step%1000==0:
                save(state,out/f'step_{step:06d}.pt');print(json.dumps(row),flush=True)
            if step%100==0:save(state,out/'latest.pt')
            if te is not None and train_acc>=.99 and te['accuracy']>=.99:
                last_joint_healthy=state
            if joint and event is None:
                if last_joint_healthy is None:raise RuntimeError('No jointly healthy reference')
                event=row
                save(state,out/'collapse.pt');save(last_joint_healthy,out/'healthy.pt')
                if previous is not None:save(previous,out/'previous.pt')
                window=out/'event_window';window.mkdir()
                for s in recent:save(s,window/f'step_{s["step"]:06d}.pt')
                (out/'event.json').write_text(json.dumps(dict(**row,healthy_step=last_joint_healthy['step'],grok_confirmed_step=grok_confirmed),indent=2))
                print('CAPTURE '+json.dumps(row),flush=True)
            if event is not None and te is not None and (peak is None or te['accuracy']<peak['test_accuracy']):
                peak=row
                if state is None:state=snapshot(model,opts,step,seed)
                save(state,out/'peak.pt');(out/'peak.json').write_text(json.dumps(row,indent=2))
            if step==max_steps or (event is not None and step>=event['step']+follow_steps):
                save(snapshot(model,opts,step,seed),out/'final.pt');break
            # Keep the immediately preceding state, even between disk snapshots.
            if grokked and event is None:
                previous=state if state is not None else snapshot(model,opts,step,seed)
            loss.backward()
            for opt in opts.values():opt.step()
    completion=dict(seed=seed,last_step=step,event_step=event['step'] if event else None,
        peak_step=peak['step'] if peak else None,grok_confirmed_step=grok_confirmed,
        elapsed_seconds=time.perf_counter()-started)
    (out/'completion.json').write_text(json.dumps(completion,indent=2));print(json.dumps(completion),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-steps',type=int,default=30000);p.add_argument('--follow-steps',type=int,default=200)
    a=p.parse_args();run(a.seed,a.out,a.max_steps,a.follow_steps)
