"""Independent RMS-only whole-network direction and training-only line panel."""
import csv
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from diagnostics import (WORK,configure_runtime,accurate_logit_gradient,compare,
                         memberships,file_sha256,assert_identical_tree)
from numerical_controls import accurate_cross_entropy
from generality_runner import (make_model_optimizers,RMSDepthTransformer,
                               generate_modular_addition_data)

SCALES=[0.,.001,.01,.1,1.]


def rms_gradients(state,x,y):
    assert state['normalization']=='rms'
    model,optimizers=make_model_optimizers(state['seed'],normalization='rms',checkpoint=state)
    assert isinstance(model,RMSDepthTransformer)
    model.train();logits=model(x)
    names,parameters=zip(*model.named_parameters())
    gradients={};dlogits={}
    for label,loss_fn in [('stock32',F.cross_entropy),('accurate32',accurate_cross_entropy)]:
        derivative,=torch.autograd.grad(loss_fn(logits,y),logits,retain_graph=True)
        upstream=torch.autograd.grad(logits,parameters,grad_outputs=derivative,retain_graph=True)
        gradients[label]={name:g.detach().clone() for name,g in zip(names,upstream)}
        dlogits[label]=derivative.detach()
    reference=accurate_logit_gradient(logits,y)
    upstream=torch.autograd.grad(logits,parameters,grad_outputs=reference.float())
    gradients['reference64_cast32']={name:g.detach().clone() for name,g in zip(names,upstream)}
    dlogits['reference64']=reference
    return model,gradients,memberships(model,optimizers),{key:compare(value,reference) for key,value in dlogits.items()}


def run(arithmetic):
    directory=WORK/f'causal_study/generality/rms_{arithmetic}'
    paths=[directory/'previous.pt',directory/'collapse.pt']
    hashes=[file_sha256(path) for path in paths]
    pre,post=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    assert pre['normalization']==post['normalization']=='rms'
    assert pre['arithmetic']==post['arithmetic']==arithmetic
    assert pre['step']+1==post['step'] and pre['seed']==post['seed']
    x,y,_,_=generate_modular_addition_data(seed=pre['seed'],operation=pre['operation'])
    model,gradients,groups,logit_comparison=rms_gradients(pre,x,y)
    delta={name:post['model_state_dict'][name].double()-before.double() for name,before in pre['model_state_dict'].items()}
    result=dict(arithmetic=arithmetic,seed=int(pre['seed']),operation=pre['operation'],normalization='rms',
                model_class=type(model).__name__,rms_epsilon=pre['rms_epsilon'],
                rms_factory_sha256=file_sha256(Path(__file__).with_name('generality_runner.py')),
                diagnostic_script_sha256=file_sha256(__file__),
                pre_step=int(pre['step']),post_step=int(post['step']),
                input_checkpoints=[dict(path=str(path),sha256=sha) for path,sha in zip(paths,hashes)],
                logit_gradient_comparison=logit_comparison,directional_derivatives={},interpolation=[],endpoint_parameters_exact={})
    for method,values in gradients.items():
        components={label:sum(float((values[name].double()*delta[name]).sum()) for name in names) for label,names in groups.items()}
        result['directional_derivatives'][method]=dict(by_group=components,total=sum(components.values()))
    for alpha in SCALES:
        interpolated={name:(before.double()+alpha*delta[name]).float() for name,before in pre['model_state_dict'].items()}
        model.load_state_dict(interpolated);model.train()
        if alpha in [0.,1.]:
            target=pre if alpha==0. else post
            assert_identical_tree(model.state_dict(),target['model_state_dict'],'RMS_interpolation_endpoint')
            result['endpoint_parameters_exact'][str(alpha)]=True
        with torch.no_grad():
            logits=model(x)
            result['interpolation'].append(dict(scale=alpha,
                train_accurate_loss32=float(accurate_cross_entropy(logits,y)),
                train_accurate_loss64=float(accurate_cross_entropy(logits.double(),y)),
                train_accuracy=float((logits.argmax(1)==y).double().mean())))
    assert hashes==[file_sha256(path) for path in paths]
    result['input_hashes_unchanged']=True
    return result


def main():
    configure_runtime(1);events=[run(arithmetic) for arithmetic in ['stock','accurate']]
    out=WORK/'causal_study/diagnostics_results'
    (out/'rms_joint_direction.json').write_text(json.dumps(dict(scales=SCALES,events=events,
        note='RMSDepthTransformer factory used for every forward pass. No optimizer step, held-out evaluation, or parameter fitting. First events from each complete trajectory; this panel is separate from the unnormalized five-seed analysis.'),indent=2)+'\n')
    rows=[]
    for event in events:
        d=event['directional_derivatives']['reference64_cast32']
        rows.append(dict(arithmetic=event['arithmetic'],seed=event['seed'],pre_step=event['pre_step'],post_step=event['post_step'],
                         model_class=event['model_class'],total_directional_derivative=d['total'],
                         **{label+'_directional_derivative':value for label,value in d['by_group'].items()},
                         **{f'loss_alpha_{r["scale"]}':r['train_accurate_loss64'] for r in event['interpolation']},
                         endpoints_exact=all(event['endpoint_parameters_exact'].values())))
    with (out/'rms_joint_direction.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Independent RMS event direction panel','',
          'This analysis uses the actual gain-free pre-sublayer and final RMS architecture through `generality_runner.make_model_optimizers(..., normalization="rms")`. Both events are the first every-update joint failures in the completed trajectories. Only training examples are evaluated.','',
          '| Arithmetic | Step | Hidden derivative | Embedding derivative | Readout derivative | Complete derivative |',
          '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['arithmetic']} | {row['pre_step']} → {row['post_step']} | {row['hidden_directional_derivative']:.7g} | {row['embeddings_directional_derivative']:.7g} | {row['readout_directional_derivative']:.7g} | {row['total_directional_derivative']:.7g} |")
    text+=['','| Arithmetic | Loss at 0 | Loss at 0.001 | Loss at 0.01 | Loss at 0.1 | Loss at 1 |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append('| '+row['arithmetic']+' | '+' | '.join(f"{row['loss_alpha_'+str(alpha)]:.7g}" for alpha in SCALES)+' |')
    text+=['','The primary derivative evaluates the accurate float64 CE derivative on the same float32 logits, then casts that derivative to float32 for the RMS network backward. Accurate-float32 and stock-float32 derivative variants are retained in the JSON. Finite-grid losses evaluate accurate float64 CE on each interpolated float32 network’s logits.','',
           'All four interpolation endpoints exactly match the saved preceding/following parameter tensors. All four checkpoint hashes are unchanged. This is an independent local-direction and finite-step test, with no optimizer updates. It does not assume the normalized and unnormalized events have the same causal mechanism.','',
           'Protocol: `../diagnostics_rms_joint_protocol.md`. Raw measurements and hashes: `rms_joint_direction.json`.']
    (out/'rms_joint_direction.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
