"""Whole-network directional derivatives and a fixed training-only line grid."""
import csv
import json
import torch
from diagnostics import (WORK,configure_runtime,event_paths,gradient_bundle,make_arm,original,
                         accurate_cross_entropy,file_sha256,assert_identical_tree)

SCALES=[0.,.001,.01,.1,1.]


def run(seed):
    paths=event_paths(seed)
    pre,post=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    x,y,_,_=original.generate_modular_addition_data(seed=seed)
    _,gradients,groups=gradient_bundle(pre,x,y)
    delta={name:post['model_state_dict'][name].double()-value.double() for name,value in pre['model_state_dict'].items()}
    result=dict(seed=seed,pre_step=int(pre['step']),post_step=int(post['step']),
                input_checkpoints=[dict(path=str(path),sha256=file_sha256(path)) for path in paths],
                directional_derivatives={},interpolation=[],endpoint_parameters_exact={})
    for method,by_name in gradients.items():
        contributions={label:sum(float((by_name[name].double()*delta[name]).sum()) for name in names) for label,names in groups.items()}
        result['directional_derivatives'][method]=dict(by_group=contributions,total=sum(contributions.values()))
    model,_,_=make_arm(pre,'original32');model.train()
    for alpha in SCALES:
        interpolated={name:(before.double()+alpha*delta[name]).float() for name,before in pre['model_state_dict'].items()}
        model.load_state_dict(interpolated)
        if alpha in [0.,1.]:
            target=pre if alpha==0. else post
            assert_identical_tree(model.state_dict(),target['model_state_dict'],'interpolation_endpoint')
            result['endpoint_parameters_exact'][str(alpha)]=True
        with torch.no_grad():
            z=model(x)
            result['interpolation'].append(dict(scale=alpha,
                training_accurate_loss32=float(accurate_cross_entropy(z,y)),
                training_accurate_loss64=float(accurate_cross_entropy(z.double(),y)),
                training_accuracy=float((z.argmax(1)==y).double().mean())))
    return result


def main():
    configure_runtime(1);results=[run(seed) for seed in range(5)]
    out=WORK/'causal_study/diagnostics_results'
    (out/'joint_direction.json').write_text(json.dumps(dict(scales=SCALES,events=results,
        note='Training-only evaluation of the entire actual parameter displacement. Local analytic derivative and finite interpolation are distinct tests. No optimizer steps or fitting.'),indent=2)+'\n')
    rows=[]
    for r in results:
        primary=r['directional_derivatives']['reference64_cast32']
        rows.append(dict(seed=r['seed'],pre_step=r['pre_step'],post_step=r['post_step'],
            total_directional_derivative=primary['total'],**{label+'_directional_derivative':value for label,value in primary['by_group'].items()},
            **{f'loss_alpha_{x["scale"]}':x['training_accurate_loss64'] for x in r['interpolation']},
            endpoints_exact=all(r['endpoint_parameters_exact'].values())))
    with (out/'joint_direction.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Full-network direction of the captured updates','',
          'The same-logit accurate derivative is backpropagated through the original float32 network. Its dot product with the actual complete parameter displacement gives the local direction. Loss values use accurate float64 CE on each interpolated float32 network’s logits, evaluated on training examples only.','',
          '| Seed | Hidden derivative | Embedding derivative | Readout derivative | Total derivative |',
          '|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['seed']} | {r['hidden_directional_derivative']:.6g} | {r['embeddings_directional_derivative']:.6g} | {r['readout_directional_derivative']:.6g} | {r['total_directional_derivative']:.6g} |")
    text+=['','| Seed | Loss at 0 | Loss at 0.001 | Loss at 0.01 | Loss at 0.1 | Loss at 1 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append('| '+str(r['seed'])+' | '+' | '.join(f"{r['loss_alpha_'+str(alpha)]:.6g}" for alpha in SCALES)+' |')
    text+=['','All five interpolation endpoints exactly match the saved preceding and following model parameters. The infinitesimal derivative, finite-step loss increase, and group contributions are reported separately. These are the actual captured full-network directions; conclusions about them should not be generalized to every Muon update or every training configuration.','',
           'Protocol: `../diagnostics_joint_direction_protocol.md`. Raw derivative variants, training accuracies, hashes, and reconstruction checks: `joint_direction.json`.']
    (out/'joint_direction.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
