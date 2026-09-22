"""Prespecified cross-task and normalized-architecture numerical controls.

Original architecture and parameter routing are imported unchanged. The RMS
variant changes only the forward pass with parameter-free pre-sublayer and
final RMS normalization. Its initial tensors match the original source.
"""
from __future__ import annotations
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

EXPERIMENTS = Path(__file__).resolve().parents[1] / 'experiments'
sys.path.insert(0, str(EXPERIMENTS))
from run_collapse import (CONFIG, DepthModularAdditionTransformer,
    split_parameter_groups, Muon, generate_modular_addition_data,
    make_model_optimizers as original_make, snapshot as original_snapshot,
    train_step as original_step)
from run_dense_capture import evaluate_preserving_runtime
from numerical_controls import accurate_cross_entropy


class RMSDepthTransformer(DepthModularAdditionTransformer):
    """Gain-free pre-RMS attention/MLP and final RMS, epsilon 1e-6."""
    @staticmethod
    def normalize(x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, token_ids):
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.transformer_blocks:
            hidden = hidden + block.attention(self.normalize(hidden))
            hidden = hidden + block.mlp(self.normalize(hidden))
        return self.unembedding(self.normalize(hidden[:, -1, :]))


def make_model_optimizers(seed, normalization, checkpoint=None):
    torch.manual_seed(seed)
    cls = DepthModularAdditionTransformer if normalization == 'none' else RMSDepthTransformer
    model = cls(**CONFIG)
    hidden, aux, readout = split_parameter_groups(model)
    opts = {
        'muon': Muon(hidden, learning_rate=.03, momentum=.95, weight_decay=.1, newton_schulz_steps=5),
        'auxiliary_adamw': torch.optim.AdamW(aux, lr=.001, weight_decay=1., betas=(.9,.999)),
        'unembedding_adamw': torch.optim.AdamW(readout, lr=.00025, weight_decay=1., betas=(.9,.999)),
    }
    if checkpoint is not None:
        cp = copy.deepcopy(checkpoint)
        model.load_state_dict(cp['model_state_dict'])
        for key, opt in opts.items(): opt.load_state_dict(cp['optimizer_state_dicts'][key])
        torch.set_rng_state(cp['torch_rng_state'])
    return model, opts


def snapshot(model, opts, step, seed, operation, normalization, arithmetic):
    result = original_snapshot(model, opts, step, seed)
    result.update(operation=operation, normalization=normalization, arithmetic=arithmetic,
                  rms_epsilon=1e-6 if normalization == 'rms' else None)
    return result


def configure_runtime():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run(seed, operation, normalization, arithmetic, out, steps):
    configure_runtime()
    out.mkdir(parents=True, exist_ok=False)
    model, opts = make_model_optimizers(seed, normalization)
    tx, ty, vx, vy = generate_modular_addition_data(seed=seed, operation=operation)
    loss_fn = F.cross_entropy if arithmetic == 'stock' else accurate_cross_entropy
    def state(step): return snapshot(model, opts, step, seed, operation, normalization, arithmetic)
    def save(s, name): torch.save(s, out/name)
    names = {id(value): name for name, value in model.named_parameters()}
    metadata = dict(seed=seed, operation=operation, normalization=normalization,
        arithmetic=arithmetic, steps=steps, config=CONFIG, rms_epsilon=1e-6 if normalization=='rms' else None,
        initial_tensor_sha256=tensor_digest(model), threads=1, torch_version=str(torch.__version__),
        parameter_groups={key:[[names[id(p)] for p in group['params']] for group in opt.param_groups]
                          for key,opt in opts.items()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        monitoring='Train every update; test every 100 and each post-grok train<90% state. Six consecutive grid test>=95% confirms grokking. First simultaneous train/test<90% is the joint event. Entire fixed horizon is completed.',
        train_examples=len(tx), test_examples=len(vx))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    save(state(0), 'initial.pt')
    streak=0; grokked=False; grok_step=None; event=None; peak=None; previous=None; healthy=None
    first_train_drop=None; min_train=1.; min_test=1.; failures=0; test_failures=0
    begin=time.perf_counter()
    fields=['step','objective','train_accuracy','test_loss','test_accuracy','grok_confirmed','joint_failure','elapsed_seconds']
    with (out/'trajectory.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for step in range(steps+1):
            model.train()
            for opt in opts.values(): opt.zero_grad(set_to_none=True)
            logits=model(tx); loss=loss_fn(logits,ty)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite training objective')
            train_accuracy=float((logits.detach().argmax(-1)==ty).double().mean())
            te=evaluate_preserving_runtime(model,vx,vy) if step%100==0 or (grokked and train_accuracy<.9) else None
            if step%100==0:
                streak=streak+1 if te['accuracy']>=.95 else 0
                if streak>=6 and not grokked: grokked=True; grok_step=step
            joint=bool(grokked and train_accuracy<.9 and te is not None and te['accuracy']<.9)
            row=dict(step=step,objective=float(loss.detach()),train_accuracy=train_accuracy,
                test_loss=te['loss'] if te else None,test_accuracy=te['accuracy'] if te else None,
                grok_confirmed=grokked,joint_failure=joint,elapsed_seconds=time.perf_counter()-begin)
            if step%10==0 or joint:
                writer.writerow(row); file.flush()
            if grokked:
                min_train=min(min_train,train_accuracy)
                if te: min_test=min(min_test,te['accuracy'])
                failures+=int(train_accuracy<.9); test_failures+=int(te is not None and te['accuracy']<.9)
                if train_accuracy<.9 and first_train_drop is None: first_train_drop=step
            current=None
            if te and train_accuracy>=.99 and te['accuracy']>=.99 and event is None:
                current=state(step); healthy=current
            if joint and event is None:
                event=dict(row, healthy_step=healthy['step'] if healthy else None, grok_step=grok_step)
                current=state(step); save(current,'collapse.pt')
                if previous: save(previous,'previous.pt')
                if healthy: save(healthy,'healthy.pt')
                (out/'event.json').write_text(json.dumps(event,indent=2))
                print('EVENT '+json.dumps(event),flush=True)
            if event is not None and te is not None and (peak is None or te['accuracy']<peak['test_accuracy']):
                peak=row; save(current or state(step),'peak.pt')
                (out/'peak.json').write_text(json.dumps(peak,indent=2))
            if step%1000==0:
                save(current or state(step),f'step_{step:06d}.pt')
                print(json.dumps(row),flush=True)
            if step%100==0:
                (out/'progress.json').write_text(json.dumps(row,indent=2))
            if step==steps:
                save(current or state(step),'final.pt'); break
            if grokked and event is None: previous=current or state(step)
            loss.backward()
            for opt in opts.values(): opt.step()
    result=dict(seed=seed,operation=operation,normalization=normalization,arithmetic=arithmetic,
        steps=steps,grok_step=grok_step,event_step=event['step'] if event else None,
        peak_step=peak['step'] if peak else None,first_train_drop_step=first_train_drop,
        post_grok_train_min=min_train if grokked else None,post_grok_test_min=min_test if grokked else None,
        post_grok_train_failure_states=failures,post_grok_test_failure_evaluations=test_failures,
        final_train_accuracy=train_accuracy,final_test_accuracy=te['accuracy'],
        elapsed_seconds=time.perf_counter()-begin)
    (out/'completion.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--operation',choices=['addition','subtraction'],required=True)
    parser.add_argument('--normalization',choices=['none','rms'],required=True)
    parser.add_argument('--arithmetic',choices=['stock','accurate'],required=True)
    parser.add_argument('--steps',type=int,default=30000)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); run(**vars(args))


if __name__=='__main__': main()
