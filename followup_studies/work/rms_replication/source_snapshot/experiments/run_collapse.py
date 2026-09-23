"""Reproduce the original Muon condition and capture an auditable collapse.

Imports the unchanged research repository. All new outputs are separate.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1] / 'muon-grokking'
sys.path.insert(0, str(ROOT))
from data import generate_modular_addition_data
from experiments.depth.train_depth_variant import DepthModularAdditionTransformer, split_parameter_groups
from optimizers.muon import Muon

CONFIG = dict(modulus=113, sequence_length=3, d_model=128, num_heads=4, d_mlp=512, num_layers=1)


def make_model_optimizers(seed, dtype=torch.float32, checkpoint=None):
    torch.manual_seed(seed)
    model = DepthModularAdditionTransformer(**CONFIG).to(dtype=dtype)
    hidden, aux, readout = split_parameter_groups(model)
    optimizers = {
        'muon': Muon(hidden, learning_rate=.03, momentum=.95, weight_decay=.1, newton_schulz_steps=5),
        'auxiliary_adamw': torch.optim.AdamW(aux, lr=.001, weight_decay=1., betas=(.9,.999)),
        'unembedding_adamw': torch.optim.AdamW(readout, lr=.00025, weight_decay=1., betas=(.9,.999)),
    }
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        for key,opt in optimizers.items():
            opt.load_state_dict(checkpoint['optimizer_state_dicts'][key])
    return model, optimizers


def snapshot(model, optimizers, step, seed):
    return copy.deepcopy(dict(step=step, seed=seed, model_config=CONFIG,
        model_state_dict=model.state_dict(),
        optimizer_state_dicts={k:o.state_dict() for k,o in optimizers.items()},
        torch_rng_state=torch.get_rng_state(), torch_version=torch.__version__,
        repository_commit='ANONYMIZED_SOURCE_REVISION'))


def train_step(model, optimizers, x, y, loss_fn=F.cross_entropy):
    model.train()
    for opt in optimizers.values(): opt.zero_grad(set_to_none=True)
    logits=model(x)
    loss=loss_fn(logits,y)
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    for opt in optimizers.values(): opt.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits=torch.cat([model(x[i:i+1024]) for i in range(0,len(x),1024)])
    return dict(loss=float(F.cross_entropy(logits,y)),
                loss64=float(F.cross_entropy(logits.double(),y)),
                accuracy=float((logits.argmax(-1)==y).double().mean()),
                max_abs_logit=float(logits.abs().max()))


@torch.no_grad()
def residuals(model, x):
    model.eval()
    values=[]
    hook=model.unembedding.register_forward_pre_hook(lambda mod,args: values.append(args[0].detach().cpu()))
    for i in range(0,len(x),1024): model(x[i:i+1024])
    hook.remove()
    return torch.cat(values)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=4)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--steps',type=int,default=25000)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--benchmark',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model,opts=make_model_optimizers(a.seed)
    tr_x,tr_y,te_x,te_y=generate_modular_addition_data(seed=a.seed)
    if a.benchmark:
        for _ in range(3): train_step(model,opts,tr_x,tr_y)
        t=time.perf_counter()
        for _ in range(20): train_step(model,opts,tr_x,tr_y)
        print(json.dumps(dict(threads=a.threads,ms_per_step=1000*(time.perf_counter()-t)/20)),flush=True)
        return
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'metadata.json').write_text(json.dumps(dict(seed=a.seed,config=CONFIG,steps=a.steps,
        threads=a.threads,torch_version=torch.__version__,python=sys.version,
        repository_commit='ANONYMIZED_SOURCE_REVISION',
        selection='First evaluation with train and test accuracy below 90% after six evaluations at or above 95% test; 100-step detection grid.',
        note='Fresh deterministic CPU replication; original saved initialization/checkpoints are absent from git.'),indent=2))
    torch.save(snapshot(model,opts,0,a.seed),a.out/'initial.pt')
    hist=[]; streak=0; grokked=False; event=None
    started=time.perf_counter()
    fields=['step','train_loss','train_loss64','train_accuracy','test_loss','test_loss64','test_accuracy','max_abs_logit','elapsed_seconds']
    with (a.out/'trajectory.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for step in range(a.steps+1):
            if step%100==0:
                tr=evaluate(model,tr_x,tr_y);te=evaluate(model,te_x,te_y)
                row=dict(step=step,train_loss=tr['loss'],train_loss64=tr['loss64'],train_accuracy=tr['accuracy'],
                    test_loss=te['loss'],test_loss64=te['loss64'],test_accuracy=te['accuracy'],
                    max_abs_logit=max(tr['max_abs_logit'],te['max_abs_logit']),elapsed_seconds=time.perf_counter()-started)
                writer.writerow(row);f.flush()
                streak=streak+1 if te['accuracy']>=.95 else 0
                if streak>=6: grokked=True
                state=snapshot(model,opts,step,a.seed)
                hist.append((row,state));hist=hist[-21:]
                if step%1000==0:
                    torch.save(state,a.out/f'step_{step:06d}.pt')
                    print(json.dumps(row),flush=True)
                torch.save(state,a.out/'latest.tmp.pt');os.replace(a.out/'latest.tmp.pt',a.out/'latest.pt')
                if grokked and event is None and tr['accuracy']<.90 and te['accuracy']<.90:
                    event=step
                    for r,s in hist: torch.save(s,a.out/f'event_window_{r["step"]:06d}.pt')
                    (a.out/'event.json').write_text(json.dumps(row,indent=2))
                    print('COLLAPSE '+json.dumps(row),flush=True)
                if event is not None:
                    torch.save(state,a.out/f'event_window_{step:06d}.pt')
                    if step>=event+500:
                        print('DONE capture',flush=True)
                        break
            if step==a.steps:break
            train_step(model,opts,tr_x,tr_y)
    (a.out/'completion.json').write_text(json.dumps(dict(last_step=step,collapse_step=event,elapsed_seconds=time.perf_counter()-started),indent=2))


if __name__=='__main__':main()
