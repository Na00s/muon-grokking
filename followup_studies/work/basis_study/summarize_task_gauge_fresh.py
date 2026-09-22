"""Summarize fresh-seed diagnostics without selecting among maps by test score."""
import json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent

def pct(x):return f'{100*x:.2f}%'
def rng_str(values,unit='pp'):
    return f'{min(values):+.2f} to {max(values):+.2f} {unit}'

def main():
    tasks=[];gauges=[]
    for path in sorted((HERE/'task_subspace_results').glob('seed[123]_*_event/report.json')):
        r=json.loads(path.read_text());meta=r['source']['metadata'];name=path.parent.name
        if not re.fullmatch('seed[123]_(first|adjacent|peak)_event',name):continue
        by={}
        for z in r['restricted_map_tests']:
            if z['method']=='ols' and z['split'].startswith('output_class'):
                by.setdefault(z['split'],{})[z['fit']]=z['results']['heldout_excluded']['classification']['accuracy']
        gaps=[100*(v['restricted']-v['random_matched_size']) for v in by.values()]
        tasks.append(dict(name=name,healthy_step=meta['healthy_step'],current_step=meta['collapsed_step'],
            reference_accuracy=r['decomposition']['healthy']['raw_classification']['accuracy'],
            current_accuracy=r['decomposition']['collapsed']['raw_classification']['accuracy'],
            backward_transport_accuracy=r['mapping']['global_train_selected_backward']['raw_classification']['accuracy'],
            task_invertible_transport_accuracy=r['mapping']['task_invertible_completion']['raw_classification']['accuracy'],
            class_exclusion_gaps_pp=gaps,source=str(path)))
    for path in sorted((HERE/'model_gauge_fresh'/'actual').glob('seed[123]_*_event.json')):
        r=json.loads(path.read_text());rows=[]
        for name,m in r['maps'].items():
            actual=m['sites']['block0_after_mlp']['tokens']['2']['centered_relative_error']
            floor=m['same_A_network_positive_control']['known_A_sites']['block0_after_mlp']['tokens']['2']['centered_relative_error']
            rows.append(dict(map=name,centered_final_error=actual,same_A_positive_floor=floor,error_over_floor=actual/floor,
                inverse_forward_rescue_accuracy=m['compensated_readout_classification']['accuracy'],
                centered_input_error=m['sites']['input']['all_tokens']['centered_relative_error'],
                centered_attention_error=m['sites']['block0_after_attention']['all_tokens']['centered_relative_error']))
        gauges.append(dict(name=path.stem,maps=rows,source=str(path)))
    lines=['# Fresh-seed task and network gauge diagnostics','',
        'These analyses separate a long reference interval from the final one-update drop. All maps use original training inputs only; the three class-exclusion groups and random controls are fixed. An adjacent checkpoint can already have reduced accuracy, which is shown explicitly below. These checkpoints share their training trajectories and are dependent observations.','',
        '## Task-subspace and output-class transfer','',
        '| Comparison | Steps | Reference | Later model | Global backward transport | Task-derived invertible transport | Class-exclusion effect versus random control |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in tasks:
        lines.append(f"| {r['name']} | {r['healthy_step']} to {r['current_step']} | {pct(r['reference_accuracy'])} | {pct(r['current_accuracy'])} | {pct(r['backward_transport_accuracy'])} | {pct(r['task_invertible_transport_accuracy'])} | {rng_str(r['class_exclusion_gaps_pp'])} |")
    lines += ['',
        'The last column is the range of signed, paired accuracy differences across three predefined excluded output-class groups, in percentage points. A negative value means fitting without those classes transfers worse than a random training subset of identical size, evaluated on exactly the same classes. This is a fixed split sensitivity analysis, not a confidence interval.',
        '',
        'The task-derived map aligns the full label-conditioned Fourier coefficient matrices. Their full row rank guarantees the existence of an invertible completion. Exact matching of this component is therefore an identifiability limitation rather than independent proof of a historical basis transformation. The table reports deployment on raw residuals, which is a valid held-out prediction test. Label-informed task projection metrics remain separately identified in the JSON files.',
        '', '## Model-wide residual gauge','',
        '| Comparison | Fit | Centered final residual error | Same-matrix network floor | Compensated readout accuracy |',
        '|---|---|---:|---:|---:|']
    for r in gauges:
        for m in r['maps']:
            label='Final-token OLS' if m['map']=='final_token_ols' else 'Shared-site OLS'
            lines.append(f"| {r['name']} | {label} | {m['centered_final_error']:.4f} | {m['same_A_positive_floor']:.2e} | {pct(m['inverse_forward_rescue_accuracy'])} |")
    lines += ['',
        'The shared-site map must describe the input residual, post-attention residual, and post-MLP residual at every token position. Each site is normalized by its reference training RMS before fitting. The final-token map uses only the equals-token residual. Both are forward maps; this table uses their inverse for readout compensation, whereas the first table uses the separately fit backward map.',
        '',
        'For every learned matrix, its exact residual gauge is planted into the reference network in float32. This matches matrix orientation, conditioning, architecture, and source checkpoint when measuring the numerical floor. Those compensated positive controls preserve source accuracy. Actual representation errors exceed their corresponding floors substantially.',
        '',
        'The weaker final-interface account allows internal computations to change. It therefore does not require the stronger common-map constraints. Successful acute transport and poor long-interval transfer can coexist. Ordinary healthy training also departs from an exact gauge, as the earlier temporal controls demonstrate.',
        '',
        'Full per-class, per-site, margin, and matrix-spectrum results are in task_subspace_results/ and model_gauge_fresh/actual/. All task coefficients and maps are fit using original training examples only. The task projection explicitly uses labels as a diagnostic; global geometric map fitting uses no labels.',
        '']
    (HERE/'fresh_task_and_gauge_memo.md').write_text('\n'.join(lines))
    (HERE/'fresh_task_and_gauge_summary.json').write_text(json.dumps(dict(tasks=tasks,gauges=gauges),indent=2)+'\n')
    print(json.dumps(dict(task_pairs=len(tasks),gauge_pairs=len(gauges))))

if __name__=='__main__':main()
