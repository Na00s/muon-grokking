"""Exact mean/centered-feature decomposition of each captured head logit change."""
import json
import torch
from diagnostics import WORK,HEAD,configure_runtime,event_paths,make_arm,original,compare
from diagnostics_stepsize import logit_metrics


def class_center(z):return z-z.mean(1,keepdim=True)


@torch.no_grad()
def run(seed,out):
    pre,post=[torch.load(path,map_location='cpu',weights_only=False) for path in event_paths(seed)]
    x,y,tx,ty=original.generate_modular_addition_data(seed=seed)
    model,_,_=make_arm(post,'original32')
    h=original.residuals(model,torch.cat([x,tx])).double()
    w0=pre['model_state_dict'][HEAD].double().T;w1=post['model_state_dict'][HEAD].double().T
    mean=h[:len(y)].mean(0,keepdim=True);delta=w1-w0
    reference=h@w0;direct=h@delta
    mean_component=(mean@delta).expand_as(direct)
    centered_component=(h-mean)@delta
    residual=mean_component+centered_component-direct
    torch.testing.assert_close(mean_component+centered_component,direct,rtol=1e-10,atol=1e-9)
    logits=dict(reference=reference,mean_only=reference+mean_component,
                centered_only=reference+centered_component,full=reference+mean_component+centered_component)
    result=dict(seed=seed,pre_step=int(pre['step']),post_step=int(post['step']),
                training_mean_norm=float(mean.norm()),
                component_sum_relative_l2=float(residual.norm()/direct.norm()),
                full_vs_direct_post_logits=compare(logits['full'],h@w1),arms={},component_norms={})
    for label,z in logits.items():
        result['arms'][label]=dict(train=logit_metrics(z[:len(y)],y),heldout=logit_metrics(z[len(y):],ty))
        for split,subset in [('train',z[:len(y)]),('heldout',z[len(y):])]:
            histogram=torch.bincount(subset.argmax(1),minlength=z.shape[1])
            result['arms'][label][split]['predicted_class_histogram']=histogram.tolist()
            result['arms'][label][split]['maximum_predicted_class_fraction']=float(histogram.max())/len(subset)
            result['arms'][label][split]['most_predicted_class']=int(histogram.argmax())
    for label,z in [('mean',mean_component),('centered',centered_component),('total',direct)]:
        result['component_norms'][label]={}
        for split,subset in [('training',z[:len(y)]),('heldout',z[len(y):])]:
            result['component_norms'][label][split]=dict(raw_norm=float(subset.norm()),class_centered_norm=float(class_center(subset).norm()))
    result['interpretation']='Training-mean-derived evaluation-only logit intervention, using actual following features and actual head displacement. Norms need not add on held-out examples.'
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    configure_runtime(1)
    for seed in range(5):
        run(seed,WORK/f'causal_study/diagnostics_results/seed{seed}/mean_mediation.json')
        print('completed',seed,flush=True)
