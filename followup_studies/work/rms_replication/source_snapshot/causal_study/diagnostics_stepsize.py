"""Evaluation-only line curves along the exact captured readout update."""
import argparse
import json
from pathlib import Path
import torch
from diagnostics import event_paths, make_arm, original, accurate_logit_gradient, configure_runtime
from numerical_controls import accurate_cross_entropy

SCALES = [-1.,-.1,0.,.0001,.0003,.001,.003,.01,.03,.1,.3,.5,.75,1.,1.5,2.]


def logit_metrics(z,y):
    target=z.gather(1,y[:,None])[:,0]
    other=z.scatter(1,y[:,None],float('-inf')).max(1).values
    return dict(loss=float(accurate_cross_entropy(z,y)),accuracy=float((z.argmax(1)==y).double().mean()),
                minimum_margin=float((target-other).min()),mean_margin=float((target-other).mean()))


@torch.no_grad()
def run(seed,out):
    pre,post=[torch.load(p,map_location='cpu',weights_only=False) for p in event_paths(seed)]
    x,y,tx,ty=original.generate_modular_addition_data(seed=seed)
    all_x=torch.cat([x,tx]); w0=pre['model_state_dict']['unembedding.weight'].double().T
    w1=post['model_state_dict']['unembedding.weight'].double().T
    result=dict(seed=seed,pre_step=int(pre['step']),post_step=int(post['step']),feature_arms={})
    for label,state in [('pre_features',pre),('post_features',post)]:
        model,_,_=make_arm(state,'original32')
        h=original.residuals(model,all_x).double()
        z0=h@w0;direction=h@(w1-w0)
        probabilities=torch.softmax(z0[:len(y)],dim=1)
        derivative=float((accurate_logit_gradient(z0[:len(y)],y)*direction[:len(y)]).sum())
        derivative_per_example=(probabilities*direction[:len(y)]).sum(1)
        curvature=float((probabilities*(direction[:len(y)]-derivative_per_example[:,None]).square()).sum(1).mean())
        rows=[]
        for alpha in SCALES:
            z=z0+alpha*direction
            rows.append(dict(scale=alpha,train=logit_metrics(z[:len(y)],y),heldout=logit_metrics(z[len(y):],ty)))
        best=min((row for row in rows if row['scale']>=0),key=lambda row:row['train']['loss'])
        result['feature_arms'][label]=dict(
            feature_frobenius_norm=float(h.norm()),feature_row_norm_mean=float(h.norm(dim=1).mean()),
            feature_row_norm_max=float(h.norm(dim=1).max()),
            update_logit_rms=float(direction.square().mean().sqrt()),
            training_directional_derivative=derivative,training_directional_curvature=curvature,
            quadratic_optimum_scale=-derivative/curvature if curvature else None,
            best_training_loss_grid_point=best,curve=rows)
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();configure_runtime(1);run(a.seed,a.out)
