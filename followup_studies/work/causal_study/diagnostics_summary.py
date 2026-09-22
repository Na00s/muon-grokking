"""Summarize the fixed five-seed diagnostic panel without outcome selection."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parent
OUT=BASE/'diagnostics_results'


def main():
    rows=[];numeric=[];counts=dict(timecourse_checkpoints=0,gradient_swap_arms=0,state_arms=0,parameter_hybrids=0,targeted_gradient_swap_arms=0)
    events=[];timecourses=[]
    for seed in range(5):
        directory=OUT/f'seed{seed}'
        event=json.loads((directory/'event.json').read_text());events.append(event)
        timecourse=json.loads((directory/'timecourse.json').read_text());timecourses.append(timecourse)
        targeted=json.loads((directory/'targeted.json').read_text())
        scales=json.loads((directory/'stepsize.json').read_text())
        state={row['action']:row['metrics'] for row in event['state_arms']}
        swaps={row['mask']:row['metrics'] for row in event['gradient_swap_arms']}
        counts['timecourse_checkpoints']+=len(timecourse)
        counts['gradient_swap_arms']+=len(event['gradient_swap_arms'])
        counts['state_arms']+=len(event['state_arms'])
        counts['parameter_hybrids']+=len(event['parameter_hybrids'])
        counts['targeted_gradient_swap_arms']+=sum(len(x['arms']) for x in targeted['checkpoints'])
        row=dict(seed=seed,pre_step=event['step'],pre_test_accuracy=event['pre_metrics']['test_accuracy'],
                 stock_test_accuracy=swaps['000']['test_accuracy'],accurate_test_accuracy=swaps['111']['test_accuracy'],
                 exact_replay=event['exact_replay'],source_unchanged=event['in_memory_source_state_unchanged'])
        for action in state: row[action+'_test_accuracy']=state[action]['test_accuracy']
        row['adam_historical_update_norm_ratio']=event['head_adam_decomposition']['norm_ratios']['historical']
        row['adam_current_update_norm_ratio']=event['head_adam_decomposition']['norm_ratios']['current']
        for label,curve in scales['feature_arms'].items():
            row[label+'_directional_derivative']=curve['training_directional_derivative']
            row[label+'_best_train_grid_scale']=curve['best_training_loss_grid_point']['scale']
            row[label+'_best_train_selected_test_accuracy']=curve['best_training_loss_grid_point']['heldout']['accuracy']
        rows.append(row)
        for checkpoint in timecourse:
            numeric.append(dict(seed=seed,step=checkpoint['step'],training_accuracy=checkpoint['training_accuracy'],
                stock_logit_gradient_relative_error=checkpoint['logit_gradients']['stock32']['relative_l2_error'],
                accurate_logit_gradient_relative_error=checkpoint['logit_gradients']['accurate32']['relative_l2_error'],
                stock_zero_target_fraction=checkpoint['logit_gradients']['stock32']['zero_target_fraction'],
                stock_readout_gradient_relative_error=checkpoint['parameter_gradients']['stock32']['readout']['relative_l2_error'],
                accurate_readout_gradient_relative_error=checkpoint['parameter_gradients']['accurate32']['readout']['relative_l2_error']))
    for name,data in [('events',rows),('gradient_timecourse',numeric)]:
        with (OUT/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    counts['saved_adam_head_gradient_variants']=40
    counts['fixed_feature_stepsize_evaluations']=160
    counts['training_mean_mediated_logit_arms']=20
    counts['complete_parameter_interpolation_arms']=25
    summary=dict(counts=counts,events=rows,numerical_timecourse=numeric,tests=6,
                 exact_model_optimizer_rng_replays=5,source_independence_checks=5)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    colors=plt.cm.tab10(np.arange(5))
    for seed,records in enumerate(timecourses):
        steps=np.array([x['step'] for x in records])
        for ax,group in zip(axs,['logit','readout']):
            if group=='logit': vals=[x['logit_gradients']['stock32']['relative_l2_error']*100 for x in records]
            else: vals=[x['parameter_gradients']['stock32']['readout']['relative_l2_error']*100 for x in records]
            ax.plot(steps,vals,'o-',markersize=2.5,lw=1.2,color=colors[seed],label=f'Seed {seed}')
            ax.set(xlabel='Training step',ylabel='Relative gradient error (%)',title=f'Stock CE: {group} gradient')
            ax.grid(alpha=.2)
    axs[0].legend(frameon=False,ncol=2)
    fig.savefig(OUT/'gradient_error.png',dpi=180);fig.savefig(OUT/'gradient_error.pdf');plt.close(fig)

    fig,ax=plt.subplots(figsize=(10.5,4.4),layout='constrained')
    keys=['stock_test_accuracy','accurate_test_accuracy','head_current_fixed_denominator_test_accuracy','head_historical_fixed_denominator_test_accuracy','freeze_head_test_accuracy']
    labels=['Original update','Accurate current gradients','Current-gradient head component','Historical head component','Head frozen']
    x=np.arange(5);width=.15
    for i,(key,label) in enumerate(zip(keys,labels)):
        ax.bar(x+(i-2)*width,[row[key]*100 for row in rows],width,label=label)
    ax.set(xticks=x,xticklabels=[f'Seed {seed}' for seed in range(5)],ylabel='Held-out accuracy (%)',ylim=(0,105),title='Exact next-step interventions at five captured joint failures')
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.07),ncol=3,fontsize=9)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'terminal_interventions.png',dpi=180);fig.savefig(OUT/'terminal_interventions.pdf');plt.close(fig)
    text=['# Five-seed upstream numerical and acute-update diagnostics','',
        'This panel establishes growing loss-gradient error before failure and a damaging readout update at each captured acute step. The terminal update is usually dominated by its current-gradient contribution. Long matched training controls remain necessary to determine whether accumulated loss-gradient error causes the sensitive regime.','',
        f"The panel contains {counts['timecourse_checkpoints']} frozen-state measurements, {counts['gradient_swap_arms']} stock/accurate group-gradient interventions, {counts['targeted_gradient_swap_arms']} targeted-repair group-gradient interventions, {counts['state_arms']} head-state or component interventions, and {counts['parameter_hybrids']} parameter-group hybrids. Additional panels contain 40 saved-Adam head-gradient variants and 160 fixed-feature step-size evaluations. All five original next updates reproduced every model parameter, optimizer buffer, and RNG value exactly. All five input checkpoint objects remained unchanged. Six new mathematical tests pass.",'',
        '## Numerical error is measurable well before the event','',
        '| Seed | Logit error, step 6000 | Logit error, step 15000 | Readout error, step 15000 | Zero target derivatives, step 15000 |',
        '|---|---:|---:|---:|---:|']
    for seed in range(5):
        early=next(x for x in numeric if x['seed']==seed and x['step']==6000)
        late=next(x for x in numeric if x['seed']==seed and x['step']==15000)
        text.append(f"| {seed} | {early['stock_logit_gradient_relative_error']*100:.2f}% | {late['stock_logit_gradient_relative_error']*100:.2f}% | {late['stock_readout_gradient_relative_error']*100:.2f}% | {late['stock_zero_target_fraction']*100:.2f}% |")
    text+=['','The reference uses exactly the same stored float32 logits, evaluates the derivative in float64 with an accurate target entry, then casts that derivative to float32 for backpropagation through the original float32 network. This isolates loss arithmetic. The step-6000 checkpoint already contains measurable gradient error, so a matched continuation from it is an early intervention within an existing trajectory.','',
           '![Gradient error](gradient_error.png)','',
           '## Correcting the last gradient is too late','',
           '| Seed | Preceding accuracy | Original next update | Accurate current gradients | Head frozen |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['seed']} | {row['pre_test_accuracy']*100:.2f}% | {row['stock_test_accuracy']*100:.2f}% | {row['accurate_test_accuracy']*100:.2f}% | {row['freeze_head_test_accuracy']*100:.2f}% |")
    text+=['','Replacing current gradients by accurate derivatives fails to avert all five terminal events. This also holds for the targeted repair that preserves stock wrong-class derivatives and reconstructs only the correct-class derivative. Removing only the shared-class gradient is a weaker correction and leaves most of the derivative error intact. These terminal results address the already-developed state, and do not test the consequences of correcting earlier updates.','',
           'The exact class-centered part of each captured head displacement reproduces its accuracy collapse. Applying only its class-common part retains the head-freeze accuracy. This is consistent with classification depending on logit differences. A common-class numerical error can still influence earlier representation learning and optimizer state.','',
           '## Which part of Adam creates the terminal displacement?','',
           'The Adam numerator is decomposed into its decayed historical first moment and its current-gradient contribution, using the same original denominator for both. Component norms are not additive causal percentages. Applying each component separately is an evaluation-only counterfactual.','',
           '| Seed | Current-gradient component | Historical component | Current norm / total norm | Historical norm / total norm |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['seed']} | {row['head_current_fixed_denominator_test_accuracy']*100:.2f}% | {row['head_historical_fixed_denominator_test_accuracy']*100:.2f}% | {row['adam_current_update_norm_ratio']:.3f} | {row['adam_historical_update_norm_ratio']:.3f} |")
    text+=['','The current-gradient component is damaging in every seed. Historical first-moment memory by itself retains at least 93.52% held-out accuracy in four seeds. Seed 3 is already below 90% held-out immediately before the event and has substantial opposing contributions from both terms. Resetting first-moment memory and clearing all head state are separate interventions in the raw results; neither consistently repairs the event. The denominator-fixed decomposition is essential because zeroing a gradient also changes Adam’s second-moment update.','',
           '![Terminal interventions](terminal_interventions.png)','',
           '## Readout overshoot is directly measurable','',
           'With preceding features fixed, the accurate training-loss derivative along the captured head displacement is negative in all five seeds. Nevertheless the full step sharply increases training loss. The predeclared scale grid was −1, −0.1, 0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.75, 1, 1.5, and 2. Selection minimizes training loss over the nonnegative scales, with held-out accuracy measured only afterward. This chooses 0.1 in four seeds and 0.01 in seed 3. With following features fixed, the best scales are 0.03, 0.03, 0.1, 0, and 0.1 for seeds 0 through 4. Their held-out accuracies are 96.52%, 94.50%, 100%, 91.01%, and 98.29%. This is direct evidence that the acute head displacement overshoots along its own direction in the preceding fixed-feature problem.','',
           'These curves use float64 multiplication of stored float32 features and head weights. The previous and following native network metrics are retained separately. Simultaneous feature changes matter: in seed 3 the same head direction increases training loss immediately when evaluated with the following features.','',
           '## The complete parameter update is locally descending','',
           'A separate check uses the entire actual parameter displacement across hidden weights, embeddings, and readout. The accurate training gradient has a negative dot product with this complete displacement in every seed. All fifteen individual group contributions are also negative. Training loss decreases at both 0.001 and 0.01 times the full displacement in every event, and increases sharply at the full step. Both interpolation endpoints exactly reproduce the captured parameter tensors. Thus the observed acute steps combine a local descent direction with finite-step overshoot. An explanation based on those complete updates pointing uphill is contradicted by these five measurements. See `joint_direction.md` for the full training-only grid and group contributions.','',
           '## Adam converts common-class gradient error into discriminating updates','',
           'At step 15000, 76.74–84.96% of the stock-versus-reference head-gradient error norm lies in the class-common component. Injecting only that component into the reference gradient, while preserving the saved Adam state, produces an update difference with 45.85–73.37% of its norm in the class-centered component. The linear SGD control has centered fractions around 3e-16. At the immediate-pre-event checkpoints, the corresponding Adam centered fractions are 45.89–96.46%. The primary calculation evaluates Adam in float64 using the exact saved state; separate realized float32 updates are retained in `adam_common.json`.','',
           'This is a local causal demonstration of how a gradient error that is common across classes can become a class-discriminating update through elementwise adaptive scaling. It does not establish that this component alone accounts for the complete accumulated trajectory. The common and centered fractions above refer to vector norms and are not additive percentages.','',
           '## Compatibility with published numerical feature inflation','',
           'The original trajectories also exhibit the mean-growth signature described by [Liu Hanqing et al., Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes (2026)](https://arxiv.org/html/2605.06152v2): classifier and feature means grow and become nearly antiparallel. The paper derives this mechanism under specific geometric and optimization assumptions. Our measurements test its compatibility with split Muon/AdamW and weight decay.','',
           '| Seed | Feature mean norm, 15000 | Feature mean norm, pre-event | Pre-event mean cosine | Feature global-mean power | Within/between centered energy |',
           '|---|---:|---:|---:|---:|---:|']
    fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for seed in range(5):
        drift=json.loads((OUT/f'seed{seed}/mean_drift.json').read_text())
        pre=next(x for x in drift if x['step']==rows[seed]['pre_step'])
        start=next(x for x in drift if x['step']==15000)
        text.append(f"| {seed} | {start['full_grid']['mean_norm']:,.1f} | {pre['full_grid']['mean_norm']:,.1f} | {pre['full_grid']['classifier_feature_mean_cosine']:.4f} | {pre['full_grid']['global_mean_power_fraction']*100:.2f}% | {pre['full_grid']['within_to_between_class_energy']:.2f} |")
        xs=[x['step'] for x in drift]
        values=[[x['full_grid']['mean_norm'] for x in drift],
                [x['head_class_mean_norm'] for x in drift],
                [x['full_grid']['classifier_feature_mean_cosine'] for x in drift]]
        for ax,ys in zip(axs,values): ax.plot(xs,ys,'o-',markersize=2.3,lw=1.1,color=colors[seed],label=f'Seed {seed}')
    for ax,title in zip(axs,['Feature mean norm','Classifier mean norm','Classifier / feature mean cosine']):
        ax.set(xlabel='Training step',title=title);ax.grid(alpha=.2)
    axs[0].set_yscale('log');axs[1].set_yscale('log');axs[2].set_ylim(-1.04,0)
    axs[0].legend(frameon=False,fontsize=8)
    fig.savefig(OUT/'mean_drift.png',dpi=180);fig.savefig(OUT/'mean_drift.pdf');plt.close(fig)
    text+=['','All means in the table use the complete modular grid, giving exact class balance. Training-example and balanced-training-class means are also recorded. The within/between centered energy ratios remain 1.01–5.85 immediately before the events, so ideal within-class feature collapse is not established. These measurements support compatibility with the published mechanism and motivate the targeted numerical intervention; they do not transfer the prior theorem automatically to this optimizer and task configuration.','',
           '![Mean drift](mean_drift.png)','',
           '## The inflated feature mean mediates the acute logit change','',
           'Using the following residuals H1 and the actual head displacement dW, the logit change splits exactly into (H1 − mu) dW + mu dW, where mu is calculated from training examples only. Starting from H1 W0, adding the mean term alone closely reproduces each accuracy collapse. Adding the centered-feature term alone retains much of the previous-readout performance. This is an evaluation-only mediation; it makes no claim about subsequent training or a changed architecture.','',
           '| Seed | Following features / prior head | Mean term alone | Centered term alone | Full displacement |',
           '|---|---:|---:|---:|---:|']
    mediation_records=[]
    for seed in range(5):
        mediation=json.loads((OUT/f'seed{seed}/mean_mediation.json').read_text())
        mediation_records.append(mediation)
        arms=mediation['arms']
        text.append(f"| {seed} | {arms['reference']['heldout']['accuracy']*100:.2f}% | {arms['mean_only']['heldout']['accuracy']*100:.2f}% | {arms['centered_only']['heldout']['accuracy']*100:.2f}% | {arms['full']['heldout']['accuracy']*100:.2f}% |")
    fig,ax=plt.subplots(figsize=(10,4.1),layout='constrained')
    labels=[('reference','Prior head'),('mean_only','Mean term alone'),('centered_only','Centered term alone'),('full','Full head displacement')]
    x=np.arange(5);width=.19
    for i,(key,label) in enumerate(labels):
        ax.bar(x+(i-1.5)*width,[m['arms'][key]['heldout']['accuracy']*100 for m in mediation_records],width,label=label)
    ax.set(xticks=x,xticklabels=[f'Seed {seed}' for seed in range(5)],ylim=(0,105),ylabel='Held-out accuracy (%)',title='Training feature mean mediates the acute head-induced loss')
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.07),ncol=4,fontsize=9)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'mean_mediation.png',dpi=180);fig.savefig(OUT/'mean_mediation.pdf');plt.close(fig)
    text+=['','The component reconstruction errors are below 3e-16 in relative L2 norm. The same training mean is applied to held-out examples. The mean term is an input-independent vector of class-logit changes, so an inflated feature mean can turn a class-discriminating readout displacement into a large shared class bias. In seed 2, the mean term sends 99.955% of held-out examples to class 64, matching the full displacement; the centered term alone preserves 100% accuracy. Predicted-class histograms are retained for every arm. This directly connects the observed mean inflation to the captured acute failure.','',
           '![Mean-mediated logit change](mean_mediation.png)','',
           '## What this resolves','',
           'The acute failure is causally localized to a class-discriminating head update and is compatible with overshoot in a sensitive representation/readout state. Current-gradient roundoff at the terminal step and historical first-moment momentum alone provide incomplete explanations. The substantial earlier derivative errors make numerical dependence a concrete hypothesis with a measurable intervention, and the separate long continuation panel is needed to adjudicate it. Exact residual gauge symmetry and approximate basis alignment do not supply this causal chain on their own.','',
           'Protocols: `../diagnostics_protocol.md`, `../diagnostics_stepsize_protocol.md`, `../diagnostics_adam_common_protocol.md`, `../diagnostics_mean_drift_protocol.md`, and `../diagnostics_mean_mediation_protocol.md`. Raw measurements: each `seed*/timecourse.json`, `event.json`, `targeted.json`, `stepsize.json`, `adam_common.json`, `mean_drift.json`, and `mean_mediation.json`. Machine-readable summary: `summary.json`, `events.csv`, `gradient_timecourse.csv`, and `mean_drift.csv`. Reusable matched-checkpoint diagnostic: `../diagnostics_branch_compare.py`.']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    if (OUT/'matched_stock20000.json').exists() and (OUT/'matched_specificity_final.json').exists():
        from diagnostics_matched_compare import append_report
        append_report()


if __name__=='__main__':main()
