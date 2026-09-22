"""Fixed fifteen-checkpoint early-arithmetic comparison and quantitative summary."""
import csv
import json
from pathlib import Path
from diagnostics import WORK,configure_runtime
from diagnostics_branch_compare import diagnose


def run():
    inputs=[]
    for seed in range(5):
        stock=WORK/(f'experiments/seed{seed}_original' if seed in (0,4) else f'basis_study/seed{seed}_baseline')/'step_015000.pt'
        accurate=WORK/f'causal_study/main_runs/seed{seed}_accurate6000'
        inputs += [(f'seed{seed}_stock15000',stock),
                   (f'seed{seed}_accurate15000',accurate/'step_015000.pt'),
                   (f'seed{seed}_accurate20000',accurate/'step_020000.pt')]
    for _,path in inputs:
        if not path.is_file():raise FileNotFoundError(path)
    records=[]
    for label,path in inputs:
        records.append(diagnose(path,label));print('completed',label,flush=True)
    out=WORK/'causal_study/diagnostics_results'
    (out/'matched_early_arithmetic.json').write_text(json.dumps(dict(comparisons=records,
        selection='All five fixed original-stock step-15000 states and accurate-from-6000 states at steps 15000 and 20000. Same saved step-6000 model and optimizer state within each seed; held-out labels used only for evaluation.'),indent=2)+'\n')
    rows=[]
    for r in records:
        f=r['full_grid_features'];g=r['gradient_diagnostics']
        rows.append(dict(seed=r['seed'],step=r['step'],label=r['label'],checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],
            train_accuracy=r['native_metrics']['train']['accuracy'],heldout_accuracy=r['native_metrics']['heldout']['accuracy'],
            feature_mean_norm=f['mean_norm'],head_mean_norm=r['head_mean_norm'],mean_cosine=f['classifier_feature_mean_cosine'],
            feature_global_mean_power=f['global_mean_power_fraction'],within_to_between_class_energy=f['within_to_between_class_energy'],
            stock_logit_gradient_relative_error=g['logit_gradients']['stock32']['relative_l2_error'],
            stock_readout_gradient_relative_error=g['parameter_gradients']['stock32']['readout']['relative_l2_error'],
            accurate_logit_gradient_relative_error=g['logit_gradients']['accurate32']['relative_l2_error'],
            accurate_readout_gradient_relative_error=g['parameter_gradients']['accurate32']['readout']['relative_l2_error'],
            stock_zero_target_fraction=g['logit_gradients']['stock32']['zero_target_fraction']))
    with (out/'matched_early_arithmetic.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=['# Matched early-arithmetic state comparison','',
            'All five seeds compare the original trajectory with an accurate-cross-entropy continuation from the same saved step-6000 model, optimizer, and RNG state. Checkpoint arithmetic is the experimental intervention; the following state properties are descriptive outcomes.','',
            'At step 15000, the stock trajectories have feature-mean norms 2.22–3.83 times those of the accurate trajectories. Their global-mean feature power is 60.52–75.38%, compared with 17.75–24.64% under accurate training. Classifier/feature means are also more antiparallel under stock training. At step 20000, all five sampled accurate states retain 100% held-out accuracy and feature-mean norms of 401–754.','',
            '| Seed | Step | Arithmetic since 6000 | Feature mean norm | Head mean norm | Mean cosine | Global mean feature power | Held-out accuracy |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        arithmetic='Stock' if 'stock' in row['label'] else 'Accurate'
        report.append(f"| {row['seed']} | {row['step']} | {arithmetic} | {row['feature_mean_norm']:,.2f} | {row['head_mean_norm']:.6f} | {row['mean_cosine']:.4f} | {row['feature_global_mean_power']*100:.2f}% | {row['heldout_accuracy']*100:.4f}% |")
    report+=['','At the common step 15000:','',
             '| Seed | Stock / accurate-trained feature mean norm | Stock / accurate-trained head mean norm | Stock CE logit error on stock / accurate-trained states | Stock CE readout error on stock / accurate-trained states |',
             '|---|---:|---:|---:|---:|']
    for seed in range(5):
        stock=next(r for r in rows if r['label']==f'seed{seed}_stock15000')
        accurate=next(r for r in rows if r['label']==f'seed{seed}_accurate15000')
        report.append(f"| {seed} | {stock['feature_mean_norm']/accurate['feature_mean_norm']:.3f} | {stock['head_mean_norm']/accurate['head_mean_norm']:.3f} | {stock['stock_logit_gradient_relative_error']*100:.2f}% / {accurate['stock_logit_gradient_relative_error']*100:.2f}% | {stock['stock_readout_gradient_relative_error']*100:.2f}% / {accurate['stock_readout_gradient_relative_error']*100:.2f}% |")
    report+=['','Gradient errors in this table evaluate stock CE on each frozen model state, including states trained with accurate CE. They measure that state’s sensitivity to the stock loss derivative; they are not the derivative errors used during accurate training. The accurate-loss derivative errors are retained separately in the CSV and raw JSON.','',
             'At step 20000, evaluating stock CE diagnostically on the accurately trained states would zero 84.07–100% of target derivatives and produce 63.64–95.40% relative logit-gradient error. The accurate derivative actually used in training stays within 2.8e-7 relative error of the same-logit reference. Small or zero stock-reported losses alone therefore do not establish which derivatives drove the trajectory.','',
             'The fifteen checkpoint SHA256 values and all full diagnostics are in `matched_early_arithmetic.json`. This comparison describes state changes caused by the arithmetic intervention and does not identify a unique mediator of the complete training history. These sampled accuracies do not replace the dense trajectory results. Original-stock step-20000 comparisons and the complete step-30000 controls are reported separately.']
    (out/'matched_early_arithmetic.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':configure_runtime(1);run()
