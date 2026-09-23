"""Track a class-common numerical gradient error through the saved Adam map."""
import json
from pathlib import Path
import torch
from diagnostics import (WORK, HEAD, configure_runtime, gradient_bundle, event_paths, make_arm,
                         original, adam_decomposition, compare, apply_update)
from diagnostics_stepsize import logit_metrics


def components(value):
    value=value.double();common=value.mean(0,keepdim=True).expand_as(value);centered=value-common
    norm=float(value.norm())
    return dict(norm=norm,common_norm=float(common.norm()),centered_norm=float(centered.norm()),
                common_fraction=float(common.norm())/norm if norm else None,
                centered_fraction=float(centered.norm())/norm if norm else None)


def actual_head_update(state,gradient):
    model,optimizers,_=make_arm(state,'original32');head=model.unembedding.weight
    before=head.detach().clone();head.grad=gradient.float().clone();optimizers['unembedding_adamw'].step()
    return head.detach().clone(),head.detach().double()-before.double()


@torch.no_grad()
def head_metrics(h,head,y,ty):
    z=h@head.double().T
    return dict(train=logit_metrics(z[:len(y)],y),heldout=logit_metrics(z[len(y):],ty))


def run(seed,out):
    data=original.generate_modular_addition_data(seed=seed);x,y,tx,ty=data
    base=WORK/(f'experiments/seed{seed}_original' if seed in (0,4) else f'basis_study/seed{seed}_baseline')
    paths=[base/'step_015000.pt',event_paths(seed)[0]]
    result=dict(seed=seed,checkpoints=[])
    for path in paths:
        state=torch.load(path,map_location='cpu',weights_only=False)
        _,gradients,groups=gradient_bundle(state,x,y)
        g=gradients['reference64_cast32'][HEAD].double();stock=gradients['stock32'][HEAD].double()
        error=stock-g;common=error.mean(0,keepdim=True).expand_as(error);centered=error-common
        variants=dict(reference=g,stock=stock,reference_plus_common=g+common,reference_plus_centered=g+centered)
        model,optimizers,_=make_arm(state,'original32')
        _,reference_parts=adam_decomposition(model.unembedding.weight,optimizers['unembedding_adamw'],g)
        reference_head,reference_actual=actual_head_update(state,g)
        pre_h=original.residuals(model,torch.cat([x,tx])).double()
        post_model,_=apply_update(state,gradients,groups)
        post_h=original.residuals(post_model,torch.cat([x,tx])).double()
        entry=dict(step=int(state['step']),checkpoint=str(path),gradient_error=components(error),
                   reference_gradient_norm=float(g.norm()),variants={})
        for label,gradient in variants.items():
            _,parts=adam_decomposition(model.unembedding.weight,optimizers['unembedding_adamw'],gradient)
            head,actual=actual_head_update(state,gradient)
            delta_formula=parts['total']-reference_parts['total']
            delta_actual=actual-reference_actual
            entry['variants'][label]=dict(
                input_error_exact=components(gradient-g),input_error_realized32=components(gradient.float().double()-g),
                formula_update=components(parts['total']),formula_update_difference_from_reference=components(delta_formula),
                actual_update=components(actual),actual_update_difference_from_reference=components(delta_actual),
                formula_vs_actual_update=compare(parts['total'],actual),
                sgd_update_difference_from_reference=components(-optimizers['unembedding_adamw'].param_groups[0]['lr']*(gradient-g)),
                preceding_feature_metrics=head_metrics(pre_h,head,y,ty),following_stock_feature_metrics=head_metrics(post_h,head,y,ty))
        result['checkpoints'].append(entry)
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    configure_runtime(1)
    for seed in range(5):
        run(seed,WORK/f'causal_study/diagnostics_results/seed{seed}/adam_common.json')
        print('completed',seed,flush=True)
