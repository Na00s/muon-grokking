"""Apply matched replay, derivative, factorial swap and decoder analyses."""
from __future__ import annotations
import argparse,copy,itertools,json,sys,time
from pathlib import Path
import numpy as np
import torch
import runner as r
sys.path.insert(0,str(r.HERE/'source_snapshot/experiments'))
from decoder_probe import fit_decoder


def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def metric_logits(z,y):
    n=len(y);k=int((z.argmax(-1)==y).sum())
    return dict(correct=k,total=n,accuracy=k/n,accurate_ce32=float(r.source.accurate_cross_entropy(z,y)),accurate_ce64=float(r.source.accurate_cross_entropy(z.double(),y)))

@torch.no_grad()
def metrics(model,data):
    training=model.training;model.eval()
    tr=metric_logits(model(data[0]),data[1]);te=r.evaluate(model,*data[2:]);model.train(training)
    return dict(train=tr,test=te)

@torch.no_grad()
def features(model,data):
    # Match existing actual-readout-input hook and source1024 extraction batches.
    x=torch.cat([data[0],data[2]]);values=[]
    training=model.training;model.eval()
    hook=model.unembedding.register_forward_pre_hook(lambda module,args:values.append(args[0].detach().clone()))
    for i in range(0,len(x),1024):model(x[i:i+1024])
    hook.remove();model.train(training)
    return torch.cat(values).numpy()

def derivative(state,data,out,label):
    model,_=r.build(state['seed'],state);model.train();z=model(data[0]);y=data[1]
    loss=r.source.accurate_cross_entropy(z,y);g,=torch.autograd.grad(loss,z)
    zz=z.detach().double();exp=(zz-zz.max(1,keepdim=True).values).exp();normalizer=exp.sum(1,keepdim=True)
    ref=exp/normalizer;wrong_mass=exp.scatter(1,y[:,None],0).sum(1,keepdim=True)
    ref.scatter_(1,y[:,None],-wrong_mass/normalizer);ref/=len(y)
    d=g.detach().double()-ref;norm=float(ref.norm());rows=g.detach().double().sum(1)
    result=dict(step=int(state['step']),training=metric_logits(z.detach(),y),absolute_l2_error=float(d.norm()),maximum_absolute_error=float(d.abs().max()),relative_l2_error=float(d.norm())/norm if norm else None,reference_l2_norm=norm,accurate_gradient_l2_norm=float(g.detach().double().norm()),class_sum_residual_l2=float(rows.norm()),class_sum_residual_maximum_absolute=float(rows.abs().max()),class_sum_residual_relative_l2=float(rows.norm())/norm if norm else None,reference_class_sum_residual_l2=float(ref.sum(1).norm()),reference='Analytic cancellation-resistant float64 mean-CE derivative on identical stored float32 logits',train_feature_mean_definition='All3830training examples; forward pre-hook before readout after final RMS')
    h=features(model,data)[:len(y)].astype(np.float64);result['train_feature_mean_norm']=float(np.linalg.norm(h.mean(0)))
    np.savez_compressed(out/f'{label}_derivative_arrays.npz',logits=z.detach().numpy(),targets=y.numpy(),accurate32_gradient=g.detach().numpy(),reference64_gradient=ref.numpy())
    r.write_json(out/f'{label}_derivative.json',result);return result


def run(path,out,cohort):
    path=Path(path);out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    old,new=load(path/'previous.pt'),load(path/'collapse.pt');seed=int(old['seed']);assert old['step']+1==new['step']
    data=r.source.generate_modular_addition_data(seed=seed,operation='addition')
    # Record protocol and explicit validation identities before fitting.
    train=np.arange(12769)<3830;heldout=~train;validation=np.zeros(12769,dtype=bool);validation[np.random.default_rng(0).permutation(np.flatnonzero(train))[:766]]=True
    protocol=dict(cohort=cohort,seed=seed,previous_step=int(old['step']),failed_step=int(new['step']),checkpoints={label:dict(path=str(path/fn),sha256=r.sha(path/fn)) for label,fn in [('previous','previous.pt'),('failed','collapse.pt')]},study_protocol_sha256=r.sha(r.HERE/'protocol.json'),analysis_script_sha256=r.sha(__file__),validation_indices=np.flatnonzero(validation).tolist(),validation_operands=data[0][torch.from_numpy(validation[:3830])].tolist(),decoder_source_sha256=r.sha(r.HERE/'source_snapshot/experiments/decoder_probe.py'),decoder_settings=json.loads((r.HERE/'protocol.json').read_text())['event_analysis']['decoder'])
    r.write_json(out/'analysis_protocol.json',protocol)
    model,opts=r.build(seed,old);r.source.original_step(model,opts,*data[:2],loss_fn=r.source.accurate_cross_entropy)
    reproduced=r.snapshot(model,opts,int(new['step']),seed)
    keys=['model_state_dict','optimizer_state_dicts','torch_rng_state']+(['rng_states'] if 'rng_states' in new else [])
    for key in keys:assert r.equal(reproduced[key],new[key]),f'Exact event replay failed:{key}'
    replay_metrics=metrics(model,data);oldmodel,oldopts=r.build(seed,old);newmodel,newopts=r.build(seed,new)
    native_old,native_new=metrics(oldmodel,data),metrics(newmodel,data)
    assert replay_metrics==native_new
    event=json.loads((path/'event.json').read_text());assert native_new['train']['accuracy']==float(event['train_accuracy']);assert native_new['test']['accuracy']==float(event['test_accuracy'])
    replay=dict(passed=True,exact_state_keys=keys,from_step=int(old['step']),to_step=int(new['step']),metrics_exact=True,metrics=replay_metrics,backend='cpu',environment=json.loads((r.HERE/'runner_verification.json').read_text())['environment'])
    r.write_json(out/'replay_verification.json',replay)
    deriv={label:derivative(state,data,out,label) for label,state in [('previous',old),('failed',new)]}
    idnames={id(p):n for n,p in oldmodel.named_parameters()};routing={'embeddings':'auxiliary_adamw','hidden':'muon','readout':'unembedding_adamw'}
    groups={k:[idnames[id(p)] for group in oldopts[optname].param_groups for p in group['params']] for k,optname in routing.items()}
    swaps=[]
    for bits in itertools.product([0,1],repeat=3):
        model,_=r.build(seed,old);sd=copy.deepcopy(old['model_state_dict'])
        for group,bit in zip(groups,bits):
            if bit:
                for name in groups[group]:sd[name]=new['model_state_dict'][name].clone()
        model.load_state_dict(sd)
        swaps.append(dict(mask=''.join(map(str,bits)),group_order=list(groups),updated_groups=[g for g,b in zip(groups,bits) if b],metrics=metrics(model,data)))
    assert swaps[0]['metrics']==native_old and swaps[-1]['metrics']==native_new
    r.write_json(out/'parameter_swaps.json',dict(seed=seed,previous_step=int(old['step']),failed_step=int(new['step']),group_order=list(groups),parameter_memberships=groups,rows=swaps))
    h0,h1=features(oldmodel,data),features(newmodel,data);y=torch.cat([data[1],data[3]]).numpy()
    np.savez_compressed(out/'decoder_features.npz',H0=h0,H1=h1,y=y,train=train,heldout=heldout,validation=validation)
    # Verify actual input hook includes the final RMS rather than the raw residual.
    rms_checks={}
    for label,h,model in [('previous',h0,oldmodel),('failed',h1,newmodel)]:
        logits=h.astype(np.float64)@model.unembedding.weight.detach().numpy().T.astype(np.float64)
        rms_checks[label]=dict(maximum_feature_rms=float(np.sqrt(np.mean(h.astype(np.float64)**2,axis=1)).max()),mean_feature_rms=float(np.sqrt(np.mean(h.astype(np.float64)**2,axis=1)).mean()),native_feature_dtype=str(h.dtype),head_reconstruction_test_correct=int((logits[heldout].argmax(1)==y[heldout]).sum()))
    r.write_json(out/'feature_extraction_verification.json',dict(hook='unembedding forward_pre_hook, actual features entering readout after final RMS',checks=rms_checks,feature_file_sha256=r.sha(out/'decoder_features.npz')))
    reports=dict(cv_selected={},unregularized_control={});arrays={}
    for label,h in [('previous',h0),('failed',h1)]:
        settings=dict(n_classes=113,inner_validation_mask=validation,validation_seed=0,feature_transform='whiten',whitening_relative_floor=1e-6,native_feature_dtype='float32',max_iter=500,refit_max_iter=1000)
        cv=fit_decoder(h,y,train,heldout,regularization_grid=(0.,1e-8,1e-6,1e-4,1e-2),**settings)
        control=cv if cv.report['selected_regularization']==0 else fit_decoder(h,y,train,heldout,regularization_grid=(0.,),**settings)
        for kind,fit in [('cv_selected',cv),('unregularized_control',control)]:
            reports[kind][label]=fit.report
            arrays[f'{label}_{kind}_readout']=fit.readout;arrays[f'{label}_{kind}_scaled_readout']=fit.scaled_readout;arrays[f'{label}_{kind}_feature_matrix']=fit.feature_matrix
        r.write_json(out/'decoders.json',reports);np.savez_compressed(out/'decoder_weights.npz',**arrays)
        print(json.dumps(dict(seed=seed,checkpoint=label,step=int(old['step'] if label=='previous' else new['step']),cv=reports['cv_selected'][label]['classification'],cv_refit=reports['cv_selected'][label]['refit']['converged'],lambda_selected=cv.report['selected_regularization'],unregularized_refit=control.report['refit']['converged'])),flush=True)
    result=dict(seed=seed,cohort=cohort,previous_step=int(old['step']),failed_step=int(new['step']),native_previous=native_old,native_failed=native_new,exact_replay=True,derivatives=deriv,swaps=swaps,decoders={kind:{label:dict(classification=rep['classification'],selected_regularization=rep['selected_regularization'],refit_converged=rep['refit']['converged'],candidate_convergence=[dict(regularization=x['regularization'],converged=x['optimization']['converged']) for x in rep['candidates']]) for label,rep in values.items()} for kind,values in reports.items()},elapsed_seconds=time.monotonic()-start)
    r.write_json(out/'summary.json',result)
    manifest=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=r.sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    r.write_json(out/'artifact_manifest.json',manifest);r.write_json(out/'completion.json',dict(passed=True,seed=seed,cohort=cohort,elapsed_seconds=time.monotonic()-start))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--path',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cohort',required=True);a=p.parse_args();r.runtime();run(a.path,a.out,a.cohort)
