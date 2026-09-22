"""Validate and summarize the five fixed accurate-CE step-30000 endpoints."""
import csv
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'diagnostics_results'


def main():
    result=json.loads((OUT/'matched_accurate_final.json').read_text())
    records=result['comparisons']
    assert len(records)==5 and {r['seed'] for r in records}==set(range(5))
    assert all(r['step']==30000 for r in records)
    assert all(len(r['checkpoint_sha256'])==64 for r in records)
    rows=[]
    for r in sorted(records,key=lambda r:r['seed']):
        f=r['full_grid_features'];g=r['gradient_diagnostics']
        rows.append(dict(seed=r['seed'],step=r['step'],checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],
                         train_accuracy=r['native_metrics']['train']['accuracy'],heldout_accuracy=r['native_metrics']['heldout']['accuracy'],
                         feature_mean_norm=f['mean_norm'],head_mean_norm=r['head_mean_norm'],mean_cosine=f['classifier_feature_mean_cosine'],
                         global_mean_feature_power=f['global_mean_power_fraction'],within_between_energy=f['within_to_between_class_energy'],
                         actual_accurate_logit_gradient_relative_error=g['logit_gradients']['accurate32']['relative_l2_error'],
                         actual_accurate_readout_gradient_relative_error=g['parameter_gradients']['accurate32']['readout']['relative_l2_error'],
                         counterfactual_stock_logit_gradient_relative_error=g['logit_gradients']['stock32']['relative_l2_error'],
                         counterfactual_stock_zero_target_fraction=g['logit_gradients']['stock32']['zero_target_fraction']))
    with (OUT/'matched_accurate_final.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# Accurate-cross-entropy step-30000 endpoints','',
          'All five fixed final checkpoints exist, record step 30000, and have SHA256 hashes retained in the full JSON. They continue from the corresponding original step-6000 model, optimizer, and RNG state. This panel describes the final states; the separate dense trajectories determine whether intermediate failures occurred.','',
          '| Seed | Train accuracy | Held-out accuracy | Feature mean norm | Head mean norm | Mean cosine | Global mean feature power |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['seed']} | {row['train_accuracy']*100:.4f}% | {row['heldout_accuracy']*100:.4f}% | {row['feature_mean_norm']:,.2f} | {row['head_mean_norm']:.6f} | {row['mean_cosine']:.4f} | {row['global_mean_feature_power']*100:.2f}% |")
    errors=[r['actual_accurate_logit_gradient_relative_error'] for r in rows]
    head_errors=[r['actual_accurate_readout_gradient_relative_error'] for r in rows]
    stock_errors=[r['counterfactual_stock_logit_gradient_relative_error'] for r in rows]
    zeros=[r['counterfactual_stock_zero_target_fraction'] for r in rows]
    text += ['',f"At these final states, the accurate logit derivative used by the training objective matches the same-logit analytic reference with relative L2 error {min(errors):.3g}–{max(errors):.3g}. Its downstream readout-gradient error is {min(head_errors):.3g}–{max(head_errors):.3g}. Applying stock CE diagnostically to the same states would yield {min(stock_errors)*100:.2f}–{max(stock_errors)*100:.2f}% logit-gradient error and zero {min(zeros)*100:.2f}–{max(zeros)*100:.2f}% of target derivatives.",'',
             'The stock-loss measurements are counterfactual diagnostics of each frozen final model. The actual continuation used accurate cross-entropy. Feature/classifier mean measurements describe the intervention’s endpoint and do not establish unique mediation of its full history.','',
             'Raw data: `matched_accurate_final.json`. Compact table: `matched_accurate_final.csv`. The five input hashes are recorded in both.']
    (OUT/'matched_accurate_final.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='complete',verified_seeds=5,verified_step=30000,rows=rows),indent=2))


if __name__=='__main__':main()
