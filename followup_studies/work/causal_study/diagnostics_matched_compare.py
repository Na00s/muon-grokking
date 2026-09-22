"""Summarize matched corrected and recovered stock endpoints at step 20000."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'diagnostics_results'
MARKER='\n## Matched step-20000 intervention endpoints\n'


def append_report():
    stock=json.loads((OUT/'matched_stock20000.json').read_text())['comparisons']
    specificity=json.loads((OUT/'matched_specificity_final.json').read_text())['comparisons']
    all_rows=stock+specificity
    assert len(all_rows)==20 and all(r['step']==20000 for r in all_rows)
    groups={}
    for objective in ['stock32','accurate32','target_repair','row_projection']:
        rows=[r for r in all_rows if r['actual_training_gradient']['objective']==objective]
        assert len(rows)==5 and {r['seed'] for r in rows}==set(range(5))
        measurements=dict(heldout_accuracy=[r['native_metrics']['heldout']['accuracy'] for r in rows],
                          feature_mean_norm=[r['full_grid_features']['mean_norm'] for r in rows],
                          head_mean_norm=[r['head_mean_norm'] for r in rows],
                          mean_cosine=[r['full_grid_features']['classifier_feature_mean_cosine'] for r in rows],
                          mean_power=[r['full_grid_features']['global_mean_power_fraction'] for r in rows],
                          actual_logit_error=[r['actual_training_gradient']['logit_gradient']['relative_l2_error'] for r in rows],
                          actual_zero_sum_residual=[r['actual_training_gradient']['logit_gradient']['row_sum_relative_l2'] for r in rows],
                          hypothetical_stock_logit_error=[r['gradient_diagnostics']['logit_gradients']['stock32']['relative_l2_error'] for r in rows])
        groups[objective]={key:dict(min=min(values),max=max(values)) for key,values in measurements.items()}
    (OUT/'matched_step20000_ranges.json').write_text(json.dumps(groups,indent=2)+'\n')
    lines=[MARKER,'All twenty fixed checkpoints verify step 20000 and retain input hashes. Original-stock endpoints have recovered high accuracy after their captured failures. Each corrected endpoint has 100% held-out accuracy. The dense trajectory results determine whether an intervention prevents failures during the interval.','',
           '| Training backward rule | Held-out accuracy | Feature mean norm | Head mean norm | Mean cosine | Global mean feature power |',
           '|---|---:|---:|---:|---:|---:|']
    for objective,stats in groups.items():
        def bounds(key,scale=1,decimals=2):
            v=stats[key];return f'{v["min"]*scale:.{decimals}f}–{v["max"]*scale:.{decimals}f}'
        lines.append(f"| {objective} | {bounds('heldout_accuracy',100)}% | {bounds('feature_mean_norm')} | {bounds('head_mean_norm',decimals=5)} | {bounds('mean_cosine',decimals=3)} | {bounds('mean_power',100)}% |")
    lines += ['',
        'At these matched endpoints, the actual accurate-CE and targeted-repair derivatives match the same-logit reference within 3.2e-7 relative L2 error. The actual projected derivative retains 52.94–86.07% relative error, while its per-example zero-sum residual is only 1.4e-8–2.1e-8 relative to the reference gradient norm. This separates restoration of the zero-sum invariant from complete derivative accuracy. Hypothetical stock-CE errors at those same corrected states are reported in separate columns in the detailed panel and were not the gradients used during corrected training.','',
        'After recovery, the stock endpoint derivative errors are smaller than their pre-event values; its feature means still carry 54.92–67.07% of feature power, compared with 16.75–41.90% for accurate/targeted continuations and 3.06–20.09% for projection. These are downstream state differences and do not identify a unique mediator of all earlier updates.','',
        'Detailed results: `matched_specificity_final.md`, `matched_stock20000.md`, and the corresponding JSON/CSV files. Five earlier-intervention step-30000 endpoints are reported in `matched_accurate_final.md`; all five have 100% train and held-out accuracy at that endpoint.']
    block='\n'.join(lines)+'\n'
    (OUT/'matched_step20000_comparison.md').write_text(block.lstrip().replace('## Matched','# Matched',1))
    path=OUT/'report.md'
    original=path.read_text().split(MARKER)[0]
    path.write_text(original.rstrip()+'\n'+block)


if __name__=='__main__':append_report()
