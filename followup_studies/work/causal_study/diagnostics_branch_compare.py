"""Frozen-gradient and feature-mean comparison of explicitly supplied matched states.

Each --checkpoint has the form label=/path/to/checkpoint.pt. Inputs are all
required to exist before this script runs. No missing checkpoint is skipped.
The saved state is read-only and its SHA256 is recorded in the result.
"""
import argparse
import json
from pathlib import Path
import torch
from diagnostics import (configure_runtime,gradient_bundle,make_arm,original,file_sha256,compare)
from diagnostics_mean_drift import feature_summary


def diagnose(path,label,objective=None):
    before=file_sha256(path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    seed=int(state['seed']);x,y,tx,ty=original.generate_modular_addition_data(seed=seed)
    gradient_report,gradient_values,groups=gradient_bundle(state,x,y)
    model,_,_=make_arm(state,'original32')
    h=original.residuals(model,torch.cat([x,tx])).double()
    w=model.unembedding.weight.detach().double();mean=w.mean(0)
    result=dict(label=label,checkpoint=str(path.resolve()),checkpoint_sha256=before,
                seed=seed,step=int(state['step']),gradient_diagnostics=gradient_report,
                native_metrics=dict(train=original.evaluate(model,x,y),heldout=original.evaluate(model,tx,ty)),
                head_norm=float(w.norm()),head_mean_norm=float(mean.norm()),
                head_centered_norm=float((w-mean).norm()),
                training_features=feature_summary(h[:len(y)],y,mean),
                full_grid_features=feature_summary(h,torch.cat([y,ty]),mean))
    if objective is not None:
        if objective in ['stock32','accurate32']:
            result['actual_training_gradient']=dict(objective=objective,
                logit_gradient=gradient_report['logit_gradients'][objective],
                parameter_gradients=gradient_report['parameter_gradients'][objective])
        elif objective in ['target_repair','row_projection']:
            from diagnostics_targeted import repaired_upstream
            logit_reports,repaired_gradients=repaired_upstream(state,x,y)
            values=repaired_gradients[objective];parameter_reports={}
            for group,names in groups.items():
                actual=torch.cat([values[name].reshape(-1) for name in names])
                reference=torch.cat([gradient_values['reference64_cast32'][name].reshape(-1) for name in names])
                parameter_reports[group]=compare(actual,reference)
            result['actual_training_gradient']=dict(objective=objective,
                logit_gradient=logit_reports[objective],parameter_gradients=parameter_reports)
        else:raise ValueError(f'Unknown actual objective: {objective}')
    assert before==file_sha256(path)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',action='append',required=True,help='label=/path/to/checkpoint.pt, repeat for each state')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--objective',choices=['stock32','accurate32','target_repair','row_projection'],default=None)
    a=p.parse_args();configure_runtime(1)
    inputs=[]
    for specification in a.checkpoint:
        label,value=specification.split('=',1);path=Path(value)
        if not path.is_file(): raise FileNotFoundError(path)
        inputs.append((label,path))
    results=[diagnose(path,label,a.objective) for label,path in inputs]
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(dict(comparisons=results,selection='Explicit supplied checkpoint list, all inputs required; fixed training labels for loss and feature-mean diagnostics, held-out data evaluation only.'),indent=2)+'\n')
    print(json.dumps(dict(status='complete',checkpoints=len(results),out=str(a.out))))
