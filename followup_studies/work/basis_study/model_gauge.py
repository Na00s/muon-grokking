"""Exact residual-basis symmetry controls and cross-site checkpoint diagnostics.

Row activations transform H' = H A. Torch Linear weights have output-by-input
orientation. Input-facing matrices transform W' = W A^{-T}; residual-output
matrices transform W' = A^T W. Embedding row tables transform E' = E A.
Q, K, V, attention weights, and MLP preactivations therefore stay fixed.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
EXPERIMENTS=HERE.parent/'experiments'
sys.path.insert(0,str(EXPERIMENTS))
from run_collapse import DepthModularAdditionTransformer,generate_modular_addition_data
from alignment_core import classification_metrics,reconstruction_metrics,logit_comparison_metrics,spectrum_metrics,fit_ridge_map

PAIRS=[EXPERIMENTS/'seed4_dense_event',EXPERIMENTS/'seed4_peak_event',EXPERIMENTS/'seed0_dense_event',
       HERE/'intervention_near_reference'/'seed0_near_features',
       EXPERIMENTS/'seed4_healthy_temporal_16044_16050',EXPERIMENTS/'seed4_healthy_temporal_6000_7000']


def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def configure():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def load_model(path,dtype=torch.float32):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=DepthModularAdditionTransformer(**state['model_config']).to(dtype=dtype)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model,state


def gauge_model(model,A,*,compensate_readout=True):
    """Return a separate exactly gauge-transformed architecture, up to arithmetic."""
    result=copy.deepcopy(model)
    dtype=next(result.parameters()).dtype
    device=next(result.parameters()).device
    A=torch.as_tensor(A,dtype=dtype,device=device)
    if A.shape != (model.token_embedding.embedding_dim,)*2:
        raise ValueError('Gauge matrix dimensions must match the residual stream')
    inverse_transpose=torch.linalg.inv(A).T
    with torch.no_grad():
        result.token_embedding.weight.copy_(model.token_embedding.weight@A)
        result.position_embedding.weight.copy_(model.position_embedding.weight@A)
        for source,target in zip(model.transformer_blocks,result.transformer_blocks):
            target.attention.qkv_projection.weight.copy_(source.attention.qkv_projection.weight@inverse_transpose)
            target.attention.output_projection.weight.copy_(A.T@source.attention.output_projection.weight)
            target.mlp.input_projection.weight.copy_(source.mlp.input_projection.weight@inverse_transpose)
            target.mlp.output_projection.weight.copy_(A.T@source.mlp.output_projection.weight)
        if compensate_readout:
            result.unembedding.weight.copy_(model.unembedding.weight@inverse_transpose)
    return result


@torch.no_grad()
def capture_sites(model,inputs,batch_size=1024):
    """Execute the architecture's own modules, preserving its addition ordering."""
    model.eval();storage={};all_logits=[]
    def keep(name,value):storage.setdefault(name,[]).append(value.detach().cpu().numpy().copy())
    for start in range(0,len(inputs),batch_size):
        x=inputs[start:start+batch_size]
        positions=torch.arange(x.shape[1],device=x.device)
        H=model.token_embedding(x)+model.position_embedding(positions)
        keep('input',H)
        for k,block in enumerate(model.transformer_blocks):
            H=H+block.attention(H);keep(f'block{k}_after_attention',H)
            H=H+block.mlp(H);keep(f'block{k}_after_mlp',H)
        all_logits.append(model.unembedding(H[:,-1]).detach().cpu().numpy().copy())
    return {k:np.concatenate(v,axis=0) for k,v in storage.items()},np.concatenate(all_logits)


def final_key(sites):return list(sites)[-1]


def make_A(d,condition,seed=7201):
    rng=np.random.default_rng(seed)
    U,_=np.linalg.qr(rng.normal(size=(d,d)));V,_=np.linalg.qr(rng.normal(size=(d,d)))
    values=np.geomspace(condition**-.5,condition**.5,d)
    return U@np.diag(values)@V.T


def dataset(state):
    tx,ty,vx,vy=generate_modular_addition_data(seed=state['seed'],modulus=state['model_config']['modulus'])
    inputs=torch.cat([tx,vx]);y=torch.cat([ty,vy]).numpy()
    train=np.arange(len(y))<len(tx)
    return inputs,y,train,~train


def fit_common_maps(old,new,train):
    key=final_key(old)
    A=fit_ridge_map(old[key][train,-1],new[key][train,-1],0)
    source=[];target=[];scales={}
    # Normalize each site by its original training RMS. Apply the same scalar
    # to old and new, preserving the exact gauge relation if one is present.
    for site in old:
        X=np.asarray(old[site][train],dtype=np.float64).reshape(-1,old[site].shape[-1])
        Y=np.asarray(new[site][train],dtype=np.float64).reshape(X.shape)
        scale=np.sqrt(np.mean(X**2))
        if scale==0:scale=1.0
        scales[site]=float(scale)
        source.append(X/scale);target.append(Y/scale)
    J=fit_ridge_map(np.concatenate(source),np.concatenate(target),0)
    return {'final_token_ols':A,'joint_sites_normalized_ols':J},scales


def rec(pred,target,train,heldout):
    pred=np.asarray(pred,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    P=pred[heldout].reshape(-1,pred.shape[-1]);Y=target[heldout].reshape(P.shape)
    pm=pred[train].reshape(-1,pred.shape[-1]).mean(0)
    ym=target[train].reshape(-1,target.shape[-1]).mean(0)
    result=reconstruction_metrics(P,Y,prediction_train_mean=pm,target_train_mean=ym)
    if result['target_centered_frobenius_norm']<1e-10*max(result['target_frobenius_norm'],1):
        result['centered_relative_error']=None
        result['error_over_centered_target_norm']=None
        result['centered_frobenius_cosine']=None
        result['centering_note']='Target is effectively constant across this evaluation subset; centered ratios are undefined.'
    return result


def site_errors(old,new,A,train,heldout):
    results={}
    for site in old:
        pred=np.asarray(old[site],dtype=np.float64)@A
        results[site]={'all_tokens':rec(pred,new[site],train,heldout),'tokens':{
            str(t):rec(pred[:,t],new[site][:,t],train,heldout) for t in range(old[site].shape[1])}}
    return results


def parameter_gauge_mismatch(old_model,new_model,A):
    # Compute comparison in float64. This is a consistency diagnostic for a
    # model-wide gauge, not a requirement of a final-interface-only hypothesis.
    transformed=gauge_model(old_model.double(),A,compensate_readout=False)
    original=dict(old_model.named_parameters());current=dict(new_model.named_parameters())
    total_error=total_target=total_actual_change=0.;rows={}
    for name,pred in transformed.named_parameters():
        if name=='unembedding.weight':continue
        target=current[name].detach().double().numpy();value=pred.detach().double().numpy()
        previous=original[name].detach().double().numpy()
        error=float(np.sum((value-target)**2));norm=float(np.sum(target**2))
        actual_change=float(np.sum((target-previous)**2))
        total_error+=error;total_target+=norm;total_actual_change+=actual_change
        rows[name]=dict(relative_error=float(np.sqrt(error/norm)),
            observed_relative_parameter_change=float(np.sqrt(actual_change/norm)))
    return dict(aggregate_relative_error=float(np.sqrt(total_error/total_target)),
        observed_aggregate_relative_parameter_change=float(np.sqrt(total_actual_change/total_target)),
        readout_excluded=True,parameters=rows,
        scope='Tests a shared gauge of the entire model. Final-interface transport alone does not require these internal parameter relations.')


def map_report(old,new,W0,y,train,heldout,A):
    H1=np.asarray(new[final_key(new)][:,-1],dtype=np.float64)
    head=np.linalg.solve(A,W0)
    return dict(map_spectrum=spectrum_metrics(A),sites=site_errors(old,new,A,train,heldout),
        compensated_readout_classification=classification_metrics((H1@head)[heldout],y[heldout]))


def plant_controls(healthy_path,out):
    out.mkdir(parents=True,exist_ok=True);reports=[]
    for name,dtype in [('float32',torch.float32),('float64',torch.float64)]:
        model,state=load_model(healthy_path,dtype=dtype)
        inputs,y,train,heldout=dataset(state)
        old,logits0=capture_sites(model,inputs)
        W0=model.unembedding.weight.detach().T.double().numpy().copy()
        for condition in [1,3,10,100]:
            start=time.perf_counter();A=make_A(W0.shape[0],condition)
            native_A=A.astype(np.float32 if dtype==torch.float32 else np.float64).astype(np.float64)
            changed=gauge_model(model,A,compensate_readout=True)
            new,compensated=capture_sites(changed,inputs)
            # Leaving the original readout in place is the deliberate
            # representation-readout mismatch positive control.
            with torch.no_grad():
                old_head_logits=torch.from_numpy(new[final_key(new)][:,-1])@model.unembedding.weight.T.cpu()
            old_head_logits=old_head_logits.numpy()
            fitted,scales=fit_common_maps(old,new,train)
            item=dict(dtype=name,planted_condition=condition,actual_condition=float(np.linalg.cond(native_A)),
                healthy_checkpoint=str(healthy_path),healthy_step=state['step'],seed=state['seed'],
                native_original=classification_metrics(logits0[heldout],y[heldout]),
                native_compensated=classification_metrics(compensated[heldout],y[heldout]),
                native_old_readout=classification_metrics(old_head_logits[heldout],y[heldout]),
                native_compensated_logit_error=logit_comparison_metrics(compensated[heldout],logits0[heldout]),
                known_A_sites=site_errors(old,new,native_A,train,heldout),
                fits={},joint_normalization_scales=scales)
            for fitname,Af in fitted.items():
                item['fits'][fitname]=map_report(old,new,W0,y,train,heldout,Af)
                item['fits'][fitname]['relative_error_to_known_A']=float(np.linalg.norm(Af-native_A)/np.linalg.norm(native_A))
            item['elapsed_seconds']=time.perf_counter()-start
            reports.append(item)
            stem=f'{name}_cond{condition}'
            write_json(out/f'{stem}.json',item)
            np.savez_compressed(out/f'{stem}_maps.npz',known_A=native_A,**fitted)
            print(json.dumps(dict(kind='planted',dtype=name,condition=condition,
                old_head=item['native_old_readout']['accuracy'],compensated=item['native_compensated']['accuracy'],
                class_centered_logit_error=item['native_compensated_logit_error']['class_centered_relative_error'],
                final_error=item['known_A_sites'][final_key(old)]['tokens']['2']['centered_relative_error'],
                recovered_accuracy=item['fits']['final_token_ols']['compensated_readout_classification']['accuracy'])),flush=True)
            del changed,new
    write_json(out/'summary.json',reports)
    return reports


def actual_pairs(out,pairs=PAIRS):
    out.mkdir(parents=True,exist_ok=True);reports=[]
    for source in pairs:
        if not source.exists():continue
        metadata=json.loads((source/'metadata.json').read_text())
        old_model,old_state=load_model(metadata['healthy_checkpoint'])
        new_model,new_state=load_model(metadata['collapsed_checkpoint'])
        inputs,y,train,heldout=dataset(old_state)
        digest=hashlib.sha256(inputs.numpy().tobytes()+y.tobytes()).hexdigest()
        if 'dataset_sha256' in metadata and digest!=metadata['dataset_sha256']:
            raise ValueError('Input reconstruction failed saved dataset hash')
        old,l0=capture_sites(old_model,inputs);new,l1=capture_sites(new_model,inputs)
        W0=old_model.unembedding.weight.detach().T.double().numpy().copy()
        fitted,scales=fit_common_maps(old,new,train)
        item=dict(source=str(source),metadata=metadata,dataset_sha256=digest,
            healthy_native_classification=classification_metrics(l0[heldout],y[heldout]),
            current_native_classification=classification_metrics(l1[heldout],y[heldout]),
            joint_normalization_scales=scales,maps={})
        for name,A in fitted.items():
            item['maps'][name]=map_report(old,new,W0,y,train,heldout,A)
            item['maps'][name]['nonreadout_parameter_gauge_mismatch']=parameter_gauge_mismatch(copy.deepcopy(old_model),new_model,A)
            # Calibrate numerical error at this exact fitted A, including its
            # orientation and condition number, by planting it in the same model.
            matched_model=gauge_model(old_model,A,compensate_readout=True)
            matched_sites,matched_logits=capture_sites(matched_model,inputs)
            native_A=A.astype(np.float32).astype(np.float64)
            item['maps'][name]['same_A_network_positive_control']=dict(
                native_compensated_classification=classification_metrics(matched_logits[heldout],y[heldout]),
                native_compensated_logit_error=logit_comparison_metrics(matched_logits[heldout],l0[heldout]),
                known_A_sites=site_errors(old,matched_sites,native_A,train,heldout),
                purpose='Native float32 numerical floor for this same learned matrix, planted as an exact full-network residual gauge.')
            del matched_model,matched_sites
        reports.append(item)
        write_json(out/f'{source.name}.json',item)
        np.savez_compressed(out/f'{source.name}_maps.npz',**fitted)
        print(json.dumps(dict(kind='actual',pair=source.name,healthy_step=old_state['step'],current_step=new_state['step'],
            fits={key:dict(input_error=val['sites']['input']['all_tokens']['centered_relative_error'],
                attn_error=val['sites']['block0_after_attention']['all_tokens']['centered_relative_error'],
                final_equals_error=val['sites']['block0_after_mlp']['tokens']['2']['centered_relative_error'],
                compensated_accuracy=val['compensated_readout_classification']['accuracy'],
                parameter_gauge_error=val['nonreadout_parameter_gauge_mismatch']['aggregate_relative_error']) for key,val in item['maps'].items()})),flush=True)
        del old,new
    aggregate=[json.loads(path.read_text()) for path in sorted(out.glob('*.json')) if path.name!='summary.json']
    write_json(out/'summary.json',aggregate)
    return reports


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=HERE/'model_gauge_results')
    parser.add_argument('--skip-planted',action='store_true')
    parser.add_argument('--pair',type=Path,action='append')
    args=parser.parse_args();configure();args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(scope='Exact network residual gauge positive controls, then stronger model-wide gauge consistency diagnostics for actual checkpoints.',
        algebra='Hprime=H A; embedding Eprime=E A; qkv/MLP-input/readout torch weights Wprime=W A^{-T}; attention-output/MLP-output Wprime=A^T W.',
        planting='One fixed random pair of orthogonal factors, singular values geometrically spaced from condition^-0.5 to condition^0.5; conditions 1,3,10,100. Float32 and float64 execution. Float64 model is a cast of saved float32 weights.',
        fitting='OLS in float64, original training examples only, no labels used. Final-token map versus shared map across all residual sites and all token positions; joint sites normalized by original training RMS.',
        matched_controls='For each actual fitted A, plant that exact matrix into its healthy model in float32 and measure the numerical floor with compensation; matches condition number and orientation.',
        interpretation='A single global model-wide gauge is stronger than a final representation-readout basis hypothesis. Internal mismatch rejects only the stronger restriction. Readout is excluded from parameter comparisons because its failure to compensate is allowed by the collapse mechanism.')
    write_json(args.out/'protocol.json',protocol)
    if not args.skip_planted:
        healthy=Path(json.loads((EXPERIMENTS/'seed4_dense_event'/'metadata.json').read_text())['healthy_checkpoint'])
        plant_controls(healthy,args.out/'planted')
    actual_pairs(args.out/'actual',args.pair or PAIRS)
    write_json(args.out/'completion.json',dict(complete=True))


if __name__=='__main__':main()
