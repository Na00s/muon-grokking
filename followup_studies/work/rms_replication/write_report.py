"""Generate the study's tables and concise claim record from verified summaries."""
from pathlib import Path
import csv,json
HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/'summary.json').read_text());verification=json.loads((HERE/'verification.json').read_text());assert verification['complete']
pct=lambda x:f'{100*x:.4f}%'
lines=['# Accurate-CE RMS replication and localization','','The prospective cohort contains seeds 1–4 on the same CPU backend and float32 update implementation as the historical seed 0 pilot. The pilot remains a separately identified observation. Runs stop at the first joint train/test failure after six scheduled test evaluations at or above 95%, or at 100,000 updates.','',f"Prospective outcome: {summary['prospective_failure_count']}/{summary['prospective_seed_count']} seeds captured a joint failure. All recorded outcomes and event analyses passed the completion audit.",'','| Cohort | Seed | Grokking confirmation | First captured joint failure / endpoint | Outcome |','| --- | ---: | ---: | ---: | --- |','| Historical CPU pilot |0|19,900|28,495|Confirmed grokking with captured failure|']
for row in summary['runs']:lines.append(f"|Prospective CPU|{row['seed']}|{row['confirmation_step']}|{row['terminal_step']}|{row['status']}|")
lines+=['','Historical seed 0 completed its original 30,000-update horizon, so its later states remain available. This replication analyzes its first captured failure using the same event procedures as the prospective cohort. Original every-update monitoring and event checkpoints identify that transition; its historical CSV records ordinary states every 10 updates. Prospective training counts are logged at every state.','','## Adjacent update and fresh decoder results','','| Cohort | Seed | Steps | Native test before | Native test after | Fresh decoder before | Fresh decoder after | CV refit convergence before/after |','| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
flat=[];swaps=[];derivs=[];decoder_rows=[];solver_rows=[];run_rows=[]
for e in summary['event_summaries']:
    d=e['decoders']['cv_selected'];u=e['decoders']['unregularized_control'];row=dict(cohort=e['cohort'],seed=e['seed'],previous_step=e['previous_step'],failed_step=e['failed_step'],native_before=e['native_previous']['test']['accuracy'],native_after=e['native_failed']['test']['accuracy'],decoder_before=d['previous']['classification']['heldout']['accuracy'],decoder_after=d['failed']['classification']['heldout']['accuracy'],decoder_before_converged=d['previous']['refit_converged'],decoder_after_converged=d['failed']['refit_converged'],unregularized_before=u['previous']['classification']['heldout']['accuracy'],unregularized_after=u['failed']['classification']['heldout']['accuracy'],unregularized_before_converged=u['previous']['refit_converged'],unregularized_after_converged=u['failed']['refit_converged'])
    for state,source_key in [('before','native_previous'),('after','native_failed')]:
        for split,m in e[source_key].items():
            for key,value in m.items():row[f'native_{state}_{split}_{key}']=value
    for method,values in e['decoders'].items():
        for state,rep in values.items():
            row[f'{method}_{state}_lambda']=rep['selected_regularization']
    flat.append(row)
    folder=HERE/'analyses'/('historical_seed0' if e['cohort']=='historical_cpu_pilot' else f"prospective_seed{e['seed']}")
    full_decoders=json.loads((folder/'decoders.json').read_text())
    for method,values in full_decoders.items():
        for state,rep in values.items():
            decoder_row=dict(cohort=e['cohort'],seed=e['seed'],checkpoint=state,step=e['previous_step'] if state=='previous' else e['failed_step'],method=method,selected_regularization=rep['selected_regularization'],refit_converged=rep['refit']['converged'],refit_status=rep['refit']['status'],refit_iterations=rep['refit']['iterations'],refit_function_evaluations=rep['refit']['function_evaluations'],refit_gradient_infinity_norm=rep['refit']['gradient_infinity_norm'],refit_gradient_l2_norm=rep['refit']['gradient_l2_norm'])
            for split,m in rep['classification'].items():
                for key in ['n','correct_count','accuracy','cross_entropy']:decoder_row[f'{split}_{key}']=m[key]
            decoder_rows.append(decoder_row)
            fits=[('candidate',candidate['regularization'],candidate['optimization'],candidate['validation']['cross_entropy']) for candidate in rep['candidates']]+[('refit',rep['selected_regularization'],rep['refit'],None)]
            for fit_kind,penalty,fit,val_ce in fits:
                solver_rows.append(dict(cohort=e['cohort'],seed=e['seed'],checkpoint=state,method=method,fit=fit_kind,regularization=penalty,validation_cross_entropy=val_ce,**{key:fit[key] for key in ['converged','status','message','iterations','function_evaluations','iteration_limit','function_evaluation_limit','initial_objective','final_objective','gradient_infinity_norm','gradient_l2_norm']}))
    lines.append(f"|{e['cohort']}|{e['seed']}|{e['previous_step']}→{e['failed_step']}|{pct(row['native_before'])}|{pct(row['native_after'])}|{pct(row['decoder_before'])}|{pct(row['decoder_after'])}|{row['decoder_before_converged']}/{row['decoder_after_converged']}|")
    for swap in e['swaps']:
        record=dict(cohort=e['cohort'],seed=e['seed'],mask=swap['mask'],updated_groups=','.join(swap['updated_groups']))
        for split,m in swap['metrics'].items():
            for key,value in m.items():record[f'{split}_{key}']=value
        swaps.append(record)
    for label,dg in e['derivatives'].items():derivs.append(dict(cohort=e['cohort'],seed=e['seed'],checkpoint=label,step=dg['step'],relative_l2_error=dg['relative_l2_error'],absolute_l2_error=dg['absolute_l2_error'],reference_l2_norm=dg['reference_l2_norm'],class_sum_residual_l2=dg['class_sum_residual_l2'],class_sum_residual_relative_l2=dg['class_sum_residual_relative_l2'],maximum_absolute_error=dg['maximum_absolute_error'],accurate_gradient_l2_norm=dg['accurate_gradient_l2_norm'],class_sum_residual_maximum_absolute=dg['class_sum_residual_maximum_absolute'],reference_class_sum_residual_l2=dg['reference_class_sum_residual_l2'],train_feature_mean_norm=dg['train_feature_mean_norm']))
lines+=['','Decoder inputs are the actual unembedding inputs after final RMS. Each transform and coefficient fit uses training examples only, with fixed 766-example internal validation subsets and validation-cross-entropy selection. The reports retain every candidate and refit status. Low accuracy from these bounded fits does not establish that no linear decoder exists.','','## All parameter-group swaps','','Mask order is embeddings, hidden matrices, readout; 1 uses the new state. Accuracies use float32 model evaluation.','','| Cohort | Seed | Mask | Train | Test |','| --- | ---: | --- | ---: | ---: |']
for row in swaps:lines.append(f"|{row['cohort']}|{row['seed']}|{row['mask']}|{pct(row['train_accuracy'])}|{pct(row['test_accuracy'])}|")
lines+=['','## Accurate derivative diagnostics','','Each comparison uses identical stored float32 logits and an analytic float64 reference. Errors below summarize the complete mean-CE logit derivative.','','| Cohort | Seed | State | Relative L2 error | Absolute L2 error | Reference norm | Class-sum L2 residual |','| --- | ---: | --- | ---: | ---: | ---: | ---: |']
for row in derivs:lines.append(f"|{row['cohort']}|{row['seed']}|{row['checkpoint']}|{row['relative_l2_error']:.6g}|{row['absolute_l2_error']:.6g}|{row['reference_l2_norm']:.6g}|{row['class_sum_residual_l2']:.6g}|")
lines+=['','## Scope','','These interventions localize the captured transitions within each RMS trajectory. The swaps distinguish the contributions of changed embeddings, hidden matrices and readout, including their interactions. They do not identify the complete upstream reason that the optimizer produced those changes. The RMS findings qualify claims about both numerical prevention and retained linear accessibility; claims about the unnormalized readout failures retain their separate supporting evidence.','','## Reproducibility','','`protocol.json` was frozen before launching seeds 1–4. `source_manifest.json` records the exact source snapshots. `launch.json` records each execution command, environment overrides and process identity. `runner_verification.json` checks original source initialization, monitored/manual updates and the historical event replay. Event subdirectories contain full replay verification, derivative arrays, all 8 swaps, actual readout-input features, decoder weights and candidate/refit diagnostics. `verification.json` recomputes trajectory endpoints, scheduled/triggered evaluation coverage, derivative errors and decoder predictions from raw records.','']
lines+=['Prospective minimum sampled test accuracy and below-95% evaluation counts include the grokking-confirmation evaluation and end at the event or horizon. Historical replay verifies model, optimizer and torch RNG equality. Prospective replay additionally verifies Python and NumPy RNG states.','']
for run in summary['runs']:
    record={k:v for k,v in run.items() if k!='terminal'}
    record['post_confirmation_count_includes_confirmation']=True
    for split,m in run['terminal'].items():
        for key,value in m.items():record[f'terminal_{split}_{key}']=value
    run_rows.append(record)
(HERE/'report.md').write_text('\n'.join(lines))
for name,rows in [('event_summary',flat),('parameter_swaps',swaps),('derivative_summary',derivs),('decoder_summary',decoder_rows),('decoder_optimization',solver_rows),('run_summary',run_rows)]:
    with (HERE/f'{name}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(HERE/'report.md')
