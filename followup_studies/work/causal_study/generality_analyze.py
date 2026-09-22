"""Analyze completed prespecified generality arms without altering checkpoints."""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import generality_runner as g
from diagnostics import accurate_logit_gradient, compare, adam_decomposition, memberships
from run_collapse import residuals
from alignment_core import fit_orthogonal_forward, fit_ridge_map, classification_metrics, reconstruction_metrics


def load(path): return torch.load(path,map_location='cpu',weights_only=False)


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build(state):
    return g.make_model_optimizers(state['seed'],state['normalization'],state)


def model_metrics(model,data):
    tx,ty,vx,vy=data
    return dict(train=g.evaluate_preserving_runtime(model,tx,ty),
                heldout=g.evaluate_preserving_runtime(model,vx,vy))


def equal_tree(first,second):
    if isinstance(first,torch.Tensor): return torch.equal(first,second)
    if isinstance(first,dict): return first.keys()==second.keys() and all(equal_tree(first[k],second[k]) for k in first)
    if isinstance(first,(list,tuple)): return len(first)==len(second) and all(equal_tree(x,y) for x,y in zip(first,second))
    return first==second


def gradient_report(state,data):
    model,opts=build(state); model.train()
    features=[]
    hook=model.unembedding.register_forward_pre_hook(lambda module,args:features.append(args[0].detach()))
    x,y=data[:2]; logits=model(x)
    hook.remove()
    feature_norms=features[0].double().norm(dim=1)
    feature_mean=features[0].double().mean(dim=0)
    class_mean=model.unembedding.weight.detach().double().mean(dim=0)
    params=dict(model.named_parameters())
    groups=memberships(model,opts)
    reference=accurate_logit_gradient(logits,y)
    report=dict(step=int(state['step']),train_accuracy=float((logits.argmax(-1)==y).double().mean()),
                maximum_absolute_logit=float(logits.detach().abs().max()),
                readout_frobenius_norm=float(model.unembedding.weight.detach().double().norm()),
                residual_norm_mean=float(feature_norms.mean()),residual_norm_max=float(feature_norms.max()),
                residual_mean_norm=float(feature_mean.norm()),
                classifier_mean_feature_mean_cosine=float(F.cosine_similarity(class_mean[None],feature_mean[None])),
                losses={},logit_gradients={},parameter_gradients={})
    gradients={}
    for name,fn in [('stock',F.cross_entropy),('accurate',g.accurate_cross_entropy)]:
        loss=fn(logits,y);report['losses'][name]=float(loss.detach())
        dg,=torch.autograd.grad(loss,logits,retain_graph=True)
        raw=torch.autograd.grad(logits,tuple(params.values()),grad_outputs=dg,retain_graph=True)
        gradients[name]=dict(zip(params,raw))
        target=dg.gather(1,y[:,None])[:,0]
        report['logit_gradients'][name]=dict(**compare(dg,reference),
            zero_target_fraction=float((target==0).double().mean()),
            zero_target_with_nonzero_wrong_fraction=float(((target==0)&(dg.abs().sum(-1)>0)).double().mean()),
            row_sum_relative_l2=float(dg.double().sum(-1).norm()/reference.norm()))
    raw=torch.autograd.grad(logits,tuple(params.values()),grad_outputs=reference.to(logits.dtype))
    gradients['reference']=dict(zip(params,raw))
    for method,values in gradients.items():
        report['parameter_gradients'][method]={}
        for group,names in groups.items():
            flat=torch.cat([values[n].reshape(-1) for n in names])
            ref=torch.cat([gradients['reference'][n].reshape(-1) for n in names])
            report['parameter_gradients'][method][group]=compare(flat,ref)
    native=state['arithmetic']
    report['head_adam_decomposition']=adam_decomposition(model.unembedding.weight,opts['unembedding_adamw'],gradients[native]['unembedding.weight'])[0]
    return report


def pair_report(old,new,data,out):
    tx,ty,vx,vy=data
    oldmodel,oldopts=build(old);newmodel,newopts=build(new)
    x=torch.cat([tx,vx]);y=torch.cat([ty,vy]).numpy()
    h0=residuals(oldmodel,x).numpy().astype(np.float64)
    h1=residuals(newmodel,x).numpy().astype(np.float64)
    w0=oldmodel.unembedding.weight.detach().numpy().T.astype(np.float64)
    w1=newmodel.unembedding.weight.detach().numpy().T.astype(np.float64)
    train=np.arange(len(x))<len(tx); heldout=~train
    r=fit_orthogonal_forward(h0[train],h1[train])
    a=fit_ridge_map(h0[train],h1[train],0.)
    b=fit_ridge_map(h1[train],h0[train],0.)
    heads={'old':w0,'native':w1,'orthogonal':r.T@w0,'backward_gl':b@w0,'inverse_gl':np.linalg.solve(a,w0)}
    report=dict(old_step=int(old['step']),new_step=int(new['step']),native_old=model_metrics(oldmodel,data),native_new=model_metrics(newmodel,data),repairs={},geometry={},
        arithmetic_note='All alignment fits and feature-head diagnostic products use float64, with train-only fits. Native model and intervention metrics use the original float32 model.')
    for label,head in heads.items():
        report['repairs'][label]={split:classification_metrics((h1@head)[mask],y[mask]) for split,mask in [('train',train),('heldout',heldout)]}
    for label,pred,target in [('forward_gl',h0@a,h1),('backward_gl',h1@b,h0),('orthogonal',h0@r,h1)]:
        report['geometry'][label]=reconstruction_metrics(pred[heldout],target[heldout],prediction_train_mean=pred[train].mean(0),target_train_mean=target[train].mean(0))
    np.savez_compressed(out.with_suffix('.npz'),H0=h0,H1=h1,W0=w0,W1=w1,y=y,train=train,heldout=heldout,A=a,B=b,R=r)
    out.write_text(json.dumps(report,indent=2))
    return report


def event_report(path,data):
    pre,post=load(path/'previous.pt'),load(path/'collapse.pt')
    assert pre['step']+1==post['step']
    model,opts=build(pre)
    fn=F.cross_entropy if pre['arithmetic']=='stock' else g.accurate_cross_entropy
    g.original_step(model,opts,*data[:2],loss_fn=fn)
    reproduced=g.snapshot(model,opts,int(post['step']),int(post['seed']),post['operation'],post['normalization'],post['arithmetic'])
    for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state']:
        assert equal_tree(reproduced[key],post[key]),f'Exact replay failed: {key}'
    report=dict(exact_next_update_replay=True,pre_step=int(pre['step']),post_step=int(post['step']),
                checkpoint_sha256={'previous':sha256(path/'previous.pt'),'collapse':sha256(path/'collapse.pt')},
                pre_gradient=gradient_report(pre,data),parameter_hybrids=[],state_arms=[])
    groups=memberships(model,opts)
    for bits in itertools.product([False,True],repeat=3):
        model,opts=build(pre)
        state=copy.deepcopy(pre['model_state_dict'])
        for group,use_post in zip(groups,bits):
            if use_post:
                for name in groups[group]:state[name]=post['model_state_dict'][name].clone()
        model.load_state_dict(state)
        report['parameter_hybrids'].append(dict(mask=''.join(str(int(bit)) for bit in bits),
            updated_groups=[group for group,flag in zip(groups,bits) if flag],metrics=model_metrics(model,data)))
    for action in ['native','accurate','zero_head_m','reset_head_state','freeze_head']:
        model,opts=build(pre)
        if action=='zero_head_m':opts['unembedding_adamw'].state[model.unembedding.weight]['exp_avg'].zero_()
        if action=='reset_head_state':opts['unembedding_adamw'].state[model.unembedding.weight].clear()
        fn=g.accurate_cross_entropy if action=='accurate' or pre['arithmetic']=='accurate' else F.cross_entropy
        model.train()
        for opt in opts.values():opt.zero_grad(set_to_none=True)
        loss=fn(model(data[0]),data[1]);loss.backward()
        for name,opt in opts.items():
            if action=='freeze_head' and name=='unembedding_adamw':continue
            opt.step()
        report['state_arms'].append(dict(action=action,metrics=model_metrics(model,data)))
    (path/'event_analysis.json').write_text(json.dumps(report,indent=2))
    pair_report(pre,post,data,path/'adjacent_analysis.json')
    if (path/'healthy.pt').exists():pair_report(load(path/'healthy.pt'),post,data,path/'healthy_analysis.json')
    mean_mediation(path)
    return report


def mean_mediation(path):
    """Apply the independently observed original-seed mean-mediation diagnostic."""
    with np.load(path/'adjacent_analysis.npz') as arrays:
        h=arrays['H1'];w0=arrays['W0'];w1=arrays['W1'];y=arrays['y'];train=arrays['train'];heldout=arrays['heldout']
    mean=h[train].mean(0,keepdims=True);delta=w1-w0
    base=h@w0;direct=h@delta
    common=np.broadcast_to(mean@delta,direct.shape)
    centered=(h-mean)@delta
    np.testing.assert_allclose(common+centered,direct,rtol=1e-10,atol=1e-9)
    result=dict(training_mean_norm=float(np.linalg.norm(mean)),
        component_sum_relative_l2=float(np.linalg.norm(common+centered-direct)/np.linalg.norm(direct)),arms={},
        note='Evaluation-only decomposition using the training mean, updated features, and actual readout displacement. This follow-up applies the mean-mediation diagnostic established in the separate original-five-seed audit.')
    for label,logits in [('reference',base),('mean_only',base+common),('centered_only',base+centered),('full',base+common+centered)]:
        result['arms'][label]={}
        for split,mask in [('train',train),('heldout',heldout)]:
            value=classification_metrics(logits[mask],y[mask])
            hist=np.bincount(logits[mask].argmax(1),minlength=logits.shape[1])
            value.update(maximum_predicted_class_fraction=float(hist.max()/mask.sum()),most_predicted_class=int(hist.argmax()))
            result['arms'][label][split]=value
    (path/'mean_mediation.json').write_text(json.dumps(result,indent=2))
    return result


def main(path):
    metadata=json.loads((path/'metadata.json').read_text())
    completion=json.loads((path/'completion.json').read_text())
    data=g.generate_modular_addition_data(seed=metadata['seed'],operation=metadata['operation'])
    reports=[]
    for step in [1000,3000,6000,10000,15000,20000,25000,30000]:
        cp=path/f'step_{step:06d}.pt'
        if cp.exists():
            report=gradient_report(load(cp),data);report['checkpoint_sha256']=sha256(cp);reports.append(report)
    (path/'gradient_timecourse.json').write_text(json.dumps(reports,indent=2))
    if completion['event_step'] is not None:event_report(path,data)
    (path/'analysis_completion.json').write_text(json.dumps(dict(gradient_checkpoints=len(reports),event_analyzed=completion['event_step'] is not None,
        analysis_script_sha256=sha256(Path(__file__)),runner_script_sha256=sha256(Path(g.__file__))),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--path',type=Path,required=True)
    args=parser.parse_args();g.configure_runtime();main(args.path)
