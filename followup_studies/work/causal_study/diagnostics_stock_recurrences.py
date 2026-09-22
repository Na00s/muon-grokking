"""Two independent late-recurrence frozen panels, with one exact replay each."""
import copy
import csv
import itertools
import json
from pathlib import Path
import torch
from diagnostics import (WORK,HEAD,configure_runtime,gradient_bundle,apply_update,make_arm,
                         original,metrics,assert_identical_tree,file_sha256)
from diagnostics_stepsize import logit_metrics


@torch.no_grad()
def mean_mediation(pre,post,data):
    x,y,tx,ty=data
    model,_,_=make_arm(post,'original32')
    h=original.residuals(model,torch.cat([x,tx])).double()
    mean=h[:len(y)].mean(0,keepdim=True)
    w0=pre['model_state_dict'][HEAD].double().T
    dw=post['model_state_dict'][HEAD].double().T-w0
    base=h@w0;full=h@dw;common=(mean@dw).expand_as(full);centered=(h-mean)@dw
    torch.testing.assert_close(common+centered,full,rtol=1e-10,atol=1e-9)
    result=dict(training_mean_norm=float(mean.norm()),component_sum_relative_l2=float((common+centered-full).norm()/full.norm()),arms={})
    for label,z in [('prior_head',base),('mean_only',base+common),('centered_only',base+centered),('full',base+full)]:
        test=z[len(y):]
        histogram=torch.bincount(test.argmax(1),minlength=test.shape[1])
        result['arms'][label]=dict(train=logit_metrics(z[:len(y)],y),heldout=logit_metrics(test,ty),
                                  heldout_maximum_predicted_class_fraction=float(histogram.max())/len(test),
                                  heldout_most_predicted_class=int(histogram.argmax()))
    return result


def run(seed,expected_step):
    directory=WORK/f'causal_study/main_runs/seed{seed}_stock_extension'
    paths=[directory/'previous.pt',directory/'collapse.pt']
    hashes=[file_sha256(path) for path in paths]
    pre,post=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    assert pre['step']+1==post['step']==expected_step
    assert pre['seed']==post['seed']==seed
    assert pre.get('normalization','none')=='none' and post.get('normalization','none')=='none'
    data=original.generate_modular_addition_data(seed=seed)
    frozen,gradients,groups=gradient_bundle(pre,*data[:2])
    replay_model,replay_opts=apply_update(pre,gradients,groups)
    replay=original.snapshot(replay_model,replay_opts,post['step'],seed)
    for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state']:
        assert_identical_tree(replay[key],post[key],key)
    derivatives={}
    for method,values in gradients.items():
        contributions={label:sum(float((values[name].double()*(post['model_state_dict'][name].double()-pre['model_state_dict'][name].double())).sum()) for name in names) for label,names in groups.items()}
        derivatives[method]=dict(by_group=contributions,total=sum(contributions.values()))
    result=dict(seed=seed,pre_step=pre['step'],post_step=post['step'],
                input_checkpoints=[dict(path=str(path),sha256=sha) for path,sha in zip(paths,hashes)],
                exact_model_optimizer_rng_replay=True,frozen_gradient_diagnostics=frozen,
                complete_parameter_direction=derivatives,parameter_hybrids=[],
                hybrid_mask_order=list(groups),script_sha256=file_sha256(__file__))
    for bits in itertools.product([False,True],repeat=3):
        model,_,_=make_arm(pre,'original32')
        mixed=copy.deepcopy(pre['model_state_dict'])
        for label,bit in zip(groups,bits):
            if bit:
                for name in groups[label]:mixed[name]=post['model_state_dict'][name].clone()
        model.load_state_dict(mixed)
        result['parameter_hybrids'].append(dict(mask=''.join(str(int(bit)) for bit in bits),metrics=metrics(model,*data)))
    result['mean_mediation']=mean_mediation(pre,post,data)
    assert hashes==[file_sha256(path) for path in paths]
    assert_identical_tree(pre,torch.load(paths[0],map_location='cpu',weights_only=False),'unchanged_source')
    result['input_hashes_and_source_state_unchanged']=True
    return result


def main():
    configure_runtime(1);events=[run(1,29549),run(3,27748)]
    out=WORK/'causal_study/diagnostics_results'
    (out/'stock_recurrences.json').write_text(json.dumps(dict(events=events,
        scope='Two late recurrence cases, separate from the original five-seed panel. One exact replay each; sixteen parameter hybrids and eight mean-mediated logit arms. No further training.'),indent=2)+'\n')
    rows=[]
    for e in events:
        for hybrid in e['parameter_hybrids']:
            rows.append(dict(seed=e['seed'],pre_step=e['pre_step'],post_step=e['post_step'],mask=hybrid['mask'],**hybrid['metrics']))
    with (out/'stock_recurrences.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Two late original-stock recurrences','',
          'Each captured update reproduces all model parameters, optimizer buffers, and RNG state exactly. These events are reported separately from the original five-seed counts. The mask order is hidden, embeddings, readout; zero uses preceding parameters and one uses following parameters.','',
          '| Seed | Event step | Hybrid | Training accuracy | Held-out accuracy |',
          '|---|---:|---|---:|---:|']
    for row in rows:text.append(f"| {row['seed']} | {row['post_step']} | {row['mask']} | {row['train_accuracy']*100:.4f}% | {row['test_accuracy']*100:.4f}% |")
    text+=['','| Seed | Stock logit-gradient error | Accurate logit-gradient error | Accurate complete-update derivative |',
           '|---|---:|---:|---:|']
    for e in events:
        g=e['frozen_gradient_diagnostics']['logit_gradients'];d=e['complete_parameter_direction']['reference64_cast32']['total']
        text.append(f"| {e['seed']} | {g['stock32']['relative_l2_error']:.5g} | {g['accurate32']['relative_l2_error']:.5g} | {d:.7g} |")
    text+=['','| Seed | Following features / prior head | Mean term alone | Centered term alone | Complete head displacement |',
           '|---|---:|---:|---:|---:|']
    for e in events:
        arms=e['mean_mediation']['arms']
        text.append('| '+str(e['seed'])+' | '+' | '.join(f"{arms[label]['heldout']['accuracy']*100:.4f}%" for label in ['prior_head','mean_only','centered_only','full'])+' |')
    text+=['','The feature mean is computed using training examples only and then applied unchanged to held-out examples. This logit decomposition is evaluation-only; it does not specify a training intervention. Parameter hybrid effects identify the captured step and do not determine why later recovery leaves a generalization gap. Full gradient comparisons, group derivatives, hashes, source-independence checks, and predicted-class concentration are retained in the JSON.','',
           'Protocol: `../diagnostics_stock_recurrences_protocol.md`.']
    (out/'stock_recurrences.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(events=[dict(seed=e['seed'],step=e['post_step'],exact_replay=True) for e in events]),indent=2))


if __name__=='__main__':main()
