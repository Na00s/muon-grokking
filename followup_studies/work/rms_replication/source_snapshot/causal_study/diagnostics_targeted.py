"""Targeted derivative controls at fixed checkpoints and terminal steps."""
import itertools
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from diagnostics import (WORK, make_arm, original, configure_runtime, gradient_bundle,
                         event_paths, compare, accurate_logit_gradient, apply_update, metrics)
from diagnostics_repairs import target_repair_gradient,row_project_gradient


def repaired_upstream(state,x,y):
    model,_,_=make_arm(state,'original32');model.train();z=model(x)
    params=list(model.named_parameters());names,values=zip(*params)
    stock,=torch.autograd.grad(F.cross_entropy(z,y),z,retain_graph=True)
    reference=accurate_logit_gradient(z,y)
    results={};grads={}
    for method,dlogit in [('target_repair',target_repair_gradient(stock,y)),('row_projection',row_project_gradient(stock))]:
        upstream=torch.autograd.grad(z,values,grad_outputs=dlogit,retain_graph=True)
        grads[method]={name:value.detach().clone() for name,value in zip(names,upstream)}
        results[method]=compare(dlogit,reference)
        target=dlogit.gather(1,y[:,None])[:,0]
        results[method].update(zero_target_fraction=float((target==0).double().mean()),
                              row_sum_relative_l2=float(dlogit.double().sum(1).norm()/reference.norm()))
    return results,grads


def run(seed,out):
    data=original.generate_modular_addition_data(seed=seed)
    baseline=WORK/(f'experiments/seed{seed}_original' if seed in [0,4] else f'basis_study/seed{seed}_baseline')
    paths=[baseline/f'step_{step:06d}.pt' for step in (6000,15000)]+[event_paths(seed)[0]]
    result=dict(seed=seed,checkpoints=[])
    for path in paths:
        state=torch.load(path,map_location='cpu',weights_only=False)
        info,base,groups=gradient_bundle(state,*data[:2])
        errors,gradients=repaired_upstream(state,*data[:2])
        item=dict(checkpoint=str(path),step=int(state['step']),frozen_logit_gradient_vs_reference=errors,upstream_vs_reference={},arms=[])
        for method,upstream in gradients.items():
            item['upstream_vs_reference'][method]={}
            for group,names in groups.items():
                actual=torch.cat([upstream[name].flatten() for name in names])
                reference=torch.cat([base['reference64_cast32'][name].flatten() for name in names])
                item['upstream_vs_reference'][method][group]=compare(actual,reference)
            if path==paths[-1]:
                replaced={**base,'accurate32':upstream}
                for bits in itertools.product([False,True],repeat=3):
                    selected=tuple(label for label,bit in zip(groups,bits) if bit)
                    model,_=apply_update(state,replaced,groups,selected)
                    item['arms'].append(dict(method=method,mask=''.join(str(int(bit)) for bit in bits),repaired_groups=selected,metrics=metrics(model,*data)))
        result['checkpoints'].append(item)
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    configure_runtime(1)
    for seed in range(5):
        run(seed,WORK/f'causal_study/diagnostics_results/seed{seed}/targeted.json')
        print('completed seed',seed,flush=True)
