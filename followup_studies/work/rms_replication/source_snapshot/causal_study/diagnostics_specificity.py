"""Matched twenty-state step-20000 endpoint diagnostics, in independent panels."""
import argparse
import csv
import json
from pathlib import Path
from diagnostics import WORK,configure_runtime
from diagnostics_branch_compare import diagnose


def paths(panel):
    result=[]
    for seed in range(5):
        if panel=='stock':
            path=(WORK/'experiments/seed4_original/step_020000.pt' if seed==4
                  else WORK/f'causal_study/main_runs/seed{seed}_stock_extension/step_020000.pt')
            result.append((seed,'stock32',path))
        else:
            result.extend([(seed,'accurate32',WORK/f'causal_study/main_runs/seed{seed}_stable3215000/final.pt'),
                           (seed,'target_repair',WORK/f'causal_study/main_runs/seed{seed}_target_repair15000/final.pt'),
                           (seed,'row_projection',WORK/f'causal_study/projection_runs/seed{seed}/final.pt')])
    return result


def summarize(records,panel):
    out=WORK/'causal_study/diagnostics_results'
    stem='matched_stock20000' if panel=='stock' else 'matched_specificity_final'
    assert all(r['step']==20000 for r in records)
    assert len(records)==(5 if panel=='stock' else 15)
    rows=[]
    for r in records:
        f=r['full_grid_features'];g=r['gradient_diagnostics'];actual=r['actual_training_gradient']
        rows.append(dict(seed=r['seed'],step=r['step'],objective=actual['objective'],checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],
            train_accuracy=r['native_metrics']['train']['accuracy'],heldout_accuracy=r['native_metrics']['heldout']['accuracy'],
            feature_mean_norm=f['mean_norm'],head_mean_norm=r['head_mean_norm'],mean_cosine=f['classifier_feature_mean_cosine'],
            global_mean_feature_power=f['global_mean_power_fraction'],within_between_energy=f['within_to_between_class_energy'],
            actual_logit_gradient_relative_error=actual['logit_gradient']['relative_l2_error'],
            actual_readout_gradient_relative_error=actual['parameter_gradients']['readout']['relative_l2_error'],
            actual_logit_row_sum_relative_l2=actual['logit_gradient']['row_sum_relative_l2'],
            actual_zero_target_fraction=actual['logit_gradient']['zero_target_fraction'],
            hypothetical_stock_logit_gradient_relative_error=g['logit_gradients']['stock32']['relative_l2_error'],
            hypothetical_stock_readout_gradient_relative_error=g['parameter_gradients']['stock32']['readout']['relative_l2_error'],
            hypothetical_stock_zero_target_fraction=g['logit_gradients']['stock32']['zero_target_fraction']))
    (out/(stem+'.json')).write_text(json.dumps(dict(comparisons=records,
        selection='Fixed all-five-seed step-20000 '+panel+' panel; every step verified and checkpoint hashed. Actual training backward rule is evaluated separately from diagnostic stock CE.'),indent=2)+'\n')
    with (out/(stem+'.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Matched step-20000 '+panel+' states','',
          'Every checkpoint records step 20000 and has an input SHA256 hash. The measured training backward rule is identified explicitly. Its error and the error produced by hypothetical stock CE at the same frozen model are kept separate. The independently recorded dense trajectories determine intervention success through the full interval.','',
          '| Seed | Backward rule | Held-out accuracy | Feature mean norm | Head mean norm | Mean cosine | Global mean power |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['seed']} | {r['objective']} | {r['heldout_accuracy']*100:.4f}% | {r['feature_mean_norm']:,.2f} | {r['head_mean_norm']:.6f} | {r['mean_cosine']:.4f} | {r['global_mean_feature_power']*100:.2f}% |")
    text+=['','| Seed | Backward rule | Actual logit-gradient error | Hypothetical stock error | Actual readout-gradient error | Actual zero-sum residual |',
           '|---|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['seed']} | {r['objective']} | {r['actual_logit_gradient_relative_error']:.4g} | {r['hypothetical_stock_logit_gradient_relative_error']:.4g} | {r['actual_readout_gradient_relative_error']:.4g} | {r['actual_logit_row_sum_relative_l2']:.4g} |")
    text+=['','Errors are relative L2 norms against the same-logit accurate derivative. The zero-sum residual is the norm of per-example sums across output classes divided by the reference logit-gradient norm. Row projection restores that invariant while retaining a different derivative from accurate CE; targeted repair reconstructs the correct-class derivative while preserving stock wrong-class entries.','',
           'These state comparisons describe downstream effects and do not identify a unique mediator of all earlier updates. Raw JSON retains every parameter-group comparison, feature statistic, and input hash.']
    (out/(stem+'.md')).write_text('\n'.join(text)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--panel',choices=['stock','specificity'],required=True)
    args=parser.parse_args();configure_runtime(1)
    inputs=paths(args.panel)
    for _,_,path in inputs:
        if not path.is_file():raise FileNotFoundError(path)
    records=[]
    for seed,objective,path in inputs:
        result=diagnose(path,f'seed{seed}_{objective}20000',objective=objective)
        assert result['step']==20000 and result['seed']==seed
        records.append(result);print('completed',seed,objective,flush=True)
    summarize(records,args.panel)
