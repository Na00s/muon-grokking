"""Write the final results and manuscript replacements from audited measurements."""
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs'/'causal_study'
def read(p):return json.loads(p.read_text())
def pct(v):return 'Unevaluated' if v is None else f'{100*v:.3f}%'

def main():
    data=read(HERE/'analysis'/'results.json');verification=read(OUT/'verification.json')
    assert data['complete'] and verification['complete']
    general=read(HERE/'generality'/'summary.json')
    diagnostic_counts=verification['diagnostic_counts']
    intervention_count=sum(diagnostic_counts[key] for key in ['gradient_swap_arms','state_arms','parameter_hybrids','targeted_gradient_swap_arms','saved_adam_head_gradient_variants'])
    events=sum(x['accurate']['event'] for x in data['primary'])
    corrected=[x for x in data['branches'] if x['arm']!='original32' and x['event']]
    if corrected:
        audit=read(HERE/'corrected_event_audit.json')
        assert all(x['path'] in audit['completed_paths'] for x in corrected)
    minima=min(x['accurate']['minimum_measured_test_accuracy'] for x in data['primary'])
    finals=[x['accurate']['final_test_accuracy'] for x in data['primary']]
    final_sentence=(f'All five final accuracies are {pct(finals[0])}.' if len(set(finals))==1 else
                    f'Final accuracies range from {pct(min(finals))} to {pct(max(finals))}.')
    counts={arm:sum(x[arm]['event'] for x in data['specificity']) for arm in ['accurate','target_repair','row_projection']}
    primary_sentence=(f'All five original trajectories have verified joint training/test failures by update 30,000. '
        f'Accurate cross-entropy, introduced from the same saved post-grokking state at update 6,000, produces {events}/5 such failures through update 30,000. '
        f'Its lowest measured held-out accuracy across the complete branches is {pct(minima)}, including their starting states. '+final_sentence)
    specificity_sentence=(f'In the common update 15,000 to 20,000 window, failure counts are {counts["accurate"]}/5 for accurate CE, '
        f'{counts["target_repair"]}/5 for correct-target-derivative repair, and {counts["row_projection"]}/5 for per-example zero-sum projection. '
        'Each corresponding original trajectory has a verified failure inside that window. The three corrections retain the original float32 model and original Muon arithmetic.')
    if counts['row_projection']==0:
        specificity_interpretation=('Removing the class-common gradient component prevents the registered joint failure in all five projection branches through update 20,000, '
            'while leaving most fixed-logit derivative error in the measured example. This intervention supports a consequential role for the broken class-sum identity in this window. ')
    else:
        specificity_interpretation=(f'Zero-sum projection avoids the registered joint failure in {5-counts["row_projection"]}/5 branches through update 20,000. '
            'Its outcome measures the effect of removing the class-common gradient component while retaining class-relative error. ')
    if counts['target_repair']==0 and counts['accurate']==0:
        specificity_interpretation+='Agreement with target-only repair and accurate CE strengthens the arithmetic-specific interpretation. '
    specificity_interpretation+='These are finite, intervention-dependent results; they do not identify a unique mediator of the entire training history.'
    g={f'{x["condition"]}_{x["arithmetic"]}':x for x in general}
    sub_stock=g['subtraction_stock'];sub_accurate=g['subtraction_accurate']
    assert sub_stock['grok_step'] is not None and sub_accurate['grok_step'] is not None
    assert sub_stock['event_step'] is not None
    general_sentence=(f'The matched subtraction pair also attained sustained grokking. The original-CE arm first failed at update {sub_stock["event_step"]}, '
        f'while the accurate-CE arm {"had no joint failure" if sub_accurate["event_step"] is None else "first failed at update "+str(sub_accurate["event_step"])} through update 30,000. '
        f'Final subtraction test accuracies were {pct(sub_stock["final_test_accuracy"])} and {pct(sub_accurate["final_test_accuracy"])}. ')
    for arm in ['stock','accurate']:
        r=g['rms_'+arm]
        general_sentence += (f'The {arm}-CE RMS-normalized addition arm '+
            (f'qualified for grokking at update {r["grok_step"]}' if r['grok_step'] is not None else 'did not attain sustained grokking')+
            f' and ended at {pct(r["final_test_accuracy"])} held-out accuracy. ')
        if r['grok_step'] is not None:
            general_sentence += ('It had no joint failure through update 30,000. ' if r['event_step'] is None else
                                 f'Its first joint failure occurred at update {r["event_step"]}. ')
    if any(g['rms_'+arm]['grok_step'] is None for arm in ['stock','accurate']):
        general_sentence+='An arm that never groks provides no post-grokking stability evidence.'
    general_sentence=general_sentence.strip()
    lines=['# Completed causal study: arithmetic, feature inflation, and readout failure','',
        '**The paper requires a revised explanation.** The completed basis tests reject an exact global change of basis for the measured checkpoint pairs. In the original unnormalized configuration, the causal study identifies a numerical contribution and an acute readout failure acting on inflated feature means. Substantial task information remains recoverable in those cases. The normalized extension supplies a separate embedding-update failure even with accurate loss arithmetic.','',
        f'This study completed **{verification["total_new_training_updates"]:,} additional optimizer updates**: five branches of 24,000 updates starting at step 6,000, fifteen specificity branches of 5,000 updates, four original-trajectory extensions totaling 49,302 updates, and four generality runs of 30,000 updates. Verification updates are excluded. '
        f'The original-trajectory diagnostic panel contains {diagnostic_counts["timecourse_checkpoints"]} frozen-state measurements, {intervention_count} gradient, optimizer-state, and parameter interventions, '
        f'{diagnostic_counts["fixed_feature_stepsize_evaluations"]} readout-scale evaluations, and {diagnostic_counts["training_mean_mediated_logit_arms"]} mean-component logit interventions. These panel counts include reference and identity-control arms. '
        'A separate early arithmetic panel contains 15 matched checkpoint diagnostics. Final-state arithmetic comparisons and generality-event diagnostics are reported separately in the accompanying artifacts.','',
        '## Matched main result','',primary_sentence,'',
        '| Seed | Captured original joint failure | Original test accuracy at that event | Accurate-CE joint failure | Accurate minimum measured test | Accurate final test |',
        '|---|---:|---:|---|---:|---:|']
    for row in data['primary']:
        a=row['accurate'];event='None through 30,000' if not a['event'] else str(a['first_joint_failure_step'])
        lines.append(f'| {row["seed"]} | {row["stock_event_step"]} | {pct(row["stock_event_test_accuracy"])} | {event} | {pct(a["minimum_measured_test_accuracy"])} | {pct(a["final_test_accuracy"])} |')
    lines += ['', 'The intervention starts from identical model parameters, all optimizer buffers, and RNG state within each seed. All five sources satisfy six consecutive 100-step-grid evaluations at or above 95% test accuracy by update 6,000. Measurable gradient error already exists at that source, so these are interventions on an existing post-grokking trajectory. The primary event requires both train and test accuracy below 90%. The corrected branches record every training state and evaluate test at every possible joint-failure trigger, every 100 updates, and the explicit final state. This gives exhaustive detection of the specified joint event. Test-only excursions between scheduled measurements remain outside that guarantee.','',
        'Original prefixes contain densely verified positive events and mixed historical evaluation grids. They establish binary incidence. The listed events are captured failures; mixed historical monitoring can miss earlier brief failures. Their event counts, durations, and sampled historical minima are excluded from comparisons with the uniformly monitored corrected branches. The five selected trajectories support a matched mechanism result; they do not estimate prevalence across all possible seeds.','',
        'The uniformly monitored original extensions capture additional joint failures at update 29,549 in seed 1 and 27,748 in seed 3. Seed 1 ends update 30,000 at 80.703% test accuracy despite recovering to 100% training accuracy; the corrected branch ends at 100% on both splits. The complete extension traces and endpoint measurements are retained in branches.csv.','',
        'Both recurrent updates replay exactly and show the same local readout pattern. Following features with the preceding head score 99.262% and 99.978% for seeds 1 and 3; the full updated models score 58.027% and 1.824%. The mean contribution alone gives 57.982% and 1.824%, while the centered contribution gives 99.195% and 99.978%. These diagnostics are separate from the original five-event panel. Seed 1 already has 87.493% held-out accuracy immediately before its recurrence, so this intervention attributes the captured step and does not explain its entire preceding or final generalization gap.','',
        '![Matched trajectories](figures/matched_trajectories.png)','',
        '## Targeted derivative controls','',specificity_sentence,'',
        '| Seed | Accurate CE minimum test | Target repair minimum test | Zero-sum projection minimum test |',
        '|---|---:|---:|---:|']
    for row in data['specificity']:
        lines.append(f'| {row["seed"]} | {pct(row["accurate"]["minimum_measured_test_accuracy"])} | {pct(row["target_repair"]["minimum_measured_test_accuracy"])} | {pct(row["row_projection"]["minimum_measured_test_accuracy"])} |')
    lines += ['', 'Exact CE satisfies a zero class sum for each example\'s logit gradient. Target repair retains the original reported forward loss and every wrong-class derivative, reconstructing only the correct-class entry from the negative sum of the remaining entries. Zero-sum projection subtracts the class mean of the stock logit gradient. In the fixed seed 4 update 16,000 audit, target repair reduces relative gradient error from 32.02% to about 4.19e-7 (0.0000419%), while projection leaves 31.88%. The two interventions test different aspects of the observed error.','',
        specificity_interpretation,'',
        'At their common update-20,000 endpoint, all fifteen specificity branches achieve 100% held-out accuracy. Accurate CE and target repair match the analytic logit derivative within 3.2e-7 relative error. The actual projected derivative retains 52.94–86.07% relative error, while its zero-sum residual is only 1.4e-8–2.1e-8 relative to the reference-gradient norm. Projection branches have feature-mean norms of 125–386 and mean power of 3.06–20.09%. These endpoint measurements further separate the restored derivative identity from complete derivative accuracy.','',
        'The original models have recovered to 99.91–100% test accuracy at update 20,000, illustrating why the outcome records failure during the interval. Their global-mean feature power remains 54.92–67.07%, compared with 16.75–41.90% for accurate CE and target repair, and 3.06–20.09% for projection. All twenty matched endpoint checkpoints have verified step numbers and input hashes.','',
        '![Specificity and overshoot](figures/specificity_and_overshoot.png)','',
        '## What produces the acute failure','',
        'Fixed-logit audits isolate substantial loss-gradient error before collapse. At update 15,000, stock logit-gradient relative errors range from 10.56% to 21.86%, and readout-gradient errors range from 18.95% to 43.42%. Using the same saved Adam state, a class-common component of gradient error creates a class-centered update difference. Its centered norm fraction is 45.85–73.37% at those checkpoints; scalar-SGD controls remain near 3e-16. These norm ratios are not additive causal percentages.','',
        'The original feature-mean norms grow to 10,374–12,203 before failure and become almost antiparallel to classifier means. At the common update 15,000, original feature means are 2.22–3.83 times those in the accurate branches started at 6,000. The latter retain much smaller global-mean feature power. This documents how the arithmetic intervention changes the state preceding failure.','',
        'At update 30,000, all five accurate-CE branches retain 100% train and test accuracy, with feature-mean norms of 150–390 and global-mean feature power of 10.25–25.00%. The loss derivative actually used by these branches has relative error below 2.5e-7 against the analytic reference. Applying stock CE to those same frozen states would round every correct-class derivative to zero. These counterfactual stock-loss diagnostics are distinct from the arithmetic used in the successful runs.','',
        'Every original acute update reproduces model, optimizer, and RNG state exactly. Keeping the preceding readout with updated features preserves or improves prior test accuracy in all five seeds. Replacing only the terminal gradients with accurate ones leaves the failures, consistent with a sensitive state that has already developed. With preceding features fixed, the actual head displacement initially descends the accurate training objective in all five cases, but its full scale overshoots.','',
        'A separate complete-network check confirms the same conclusion for the actual joint parameter displacement. In every seed, its accurate training-gradient dot product is negative, including each hidden, embedding, and readout contribution. Scaling the whole displacement to 0.001 or 0.01 decreases training loss in every case; the full step sharply increases it. All five interpolation endpoints exactly reproduce their saved model parameters. This directly establishes local descent followed by finite-step overshoot for the complete captured updates.','',
        'Let H1 be the following representation, W0 the preceding head, and dW the captured head displacement. With mu estimated from training examples, H1 dW = (H1 − mu) dW + mu dW. Starting from H1 W0, the mean term alone gives test accuracies of 32.11%, 9.42%, 0.93%, 53.34%, and 66.07%, closely matching the full displacements at 31.87%, 9.09%, 0.93%, 53.60%, and 66.33%. The centered term alone retains 95.26%, 92.48%, 100%, 87.20%, and 98.46%. In seed 2, the mean term sends 99.955% of held-out examples to one class. The two components reconstruct the head-induced logit change within 3e-16 relative error.','',
        'This is an evaluation-only intervention on the head-induced logit change. Its training-estimated mean is used unchanged on held-out inputs. It establishes local component sufficiency. The large feature mean and the class-discriminating head displacement act jointly through their product; this decomposition does not identify a unique mediator of the accumulated optimization trajectory.','',
        '![Mean mediation](figures/mean_mediation.png)','',
        '## Operation and architecture controls','',general_sentence,'',
        '| Condition | Arithmetic | Grokking confirmation | First joint failure | Minimum measured post-grok test | Final train | Final test |',
        '|---|---|---:|---|---:|---:|---:|']
    for r in general:
        event=str(r['event_step']) if r['event_step'] is not None else ('None through 30,000' if r['grok_step'] is not None else 'Unevaluated')
        lines.append(f'| {r["condition"]} | {r["arithmetic"]} | {r["grok_step"] or "Not attained"} | {event} | {pct(r["post_grok_test_min"])} | {pct(r["final_train_accuracy"])} | {pct(r["final_test_accuracy"])} |')
    lines += ['', 'The subtraction failure reproduces the acute mechanism: actual test accuracy falls to 5.85%; the preceding head with updated features gives 98.81%; the head update alone gives 6.28%. Its mean contribution alone gives 5.84%, while the centered contribution retains 98.76%. Exact next-update replay passes. Each generality condition has one seed. Hyperparameters and horizons were fixed before these runs.','',
        'The RMS architecture supplies a decisive scope counterexample. In the accurate-CE arm, test accuracy falls from 96.208% to 0.884% at update 28,495. The captured accurate logit derivative has relative error 9.48e-8 against the analytic reference, and no correct-class derivative vanishes. Updating only the readout retains 96.208%; updating only embeddings gives 0.940%; updating hidden matrices and readout together retains 96.409%. Freezing the readout still gives 0.895%. Exact model, optimizer, and RNG replay passes. The acute destructive component is the embedding update in this event.','',
        'The independent full-network directional check also passes for both RMS events using the RMS forward pass. Each complete displacement and every parameter-group contribution is locally descending. At scales 0.001 and 0.01, both updates reduce accurate training loss. The full stock-RMS step increases it from 0.000813 to 0.951; the full accurate-RMS step increases it from 0.342 to 6.811. All interpolation endpoints exactly reproduce saved parameters. Finite-step overshoot survives this architecture and arithmetic change, with the embedding update supplying the acute destructive component.','',
        'Both RMS runs recover to 100% final accuracy after their recorded failures. Interim progress snapshots concealed these brief events, so an early running status incorrectly described RMS as failure-free. The every-update event detector retained the failures, and the completed audit corrects that interim statement. Accurate CE prevents the observed failures in the five original matched trajectories and subtraction run; the normalized counterexample rules out a universal sufficiency claim.','',
        '![Generality controls](figures/generality_controls.png)','',
        '## Claim decisions','',
        '| Claim | Final decision |',
        '|---|---|',
        '| Substantial generalizing task information survives the selected unnormalized failures | Supported: prior training-only decoders recover 98.20–100% held-out accuracy. |',
        '| The acute destructive component is localized to the readout in the unnormalized model | Supported for all five original addition transitions and the new subtraction event. The feature-mean contribution nearly reproduces the acute damage. |',
        '| Loss arithmetic causally contributes to the unnormalized instability | Supported for the completed original addition and subtraction contrasts, with derivative-specific interventions in addition. |',
        '| The observed checkpoint pairs follow an exact global change of basis | Requires revision: measured existence/geometry tests reject this account against planted numerical controls. |',
        '| There is no numerical pathology | Requires revision: fixed-state errors and matched arithmetic interventions directly contradict it. |',
        '| The captured failure moves along a loss-underdetermined direction | Requires revision: accurate directional derivatives of the complete parameter displacements are negative; small scaled steps lower loss and the full steps overshoot. |',
        '| Dominant Fourier-frequency rankings are GL-invariant | False: only exact zero/nonzero support has general invertible invariance; orthogonal maps preserve power. |',
        '| A one-time decoder repair gives sustained stability | Requires its measured qualification: tested one-time repairs rapidly relapse under continuing states. Earlier arithmetic interventions have separate outcomes. |',
        '| Accurate loss arithmetic prevents collapse across architectures | Requires revision: the accurate-CE RMS run collapses through an embedding update with an accurate loss derivative. |',
        '| The detailed readout mechanism applies to normalized architectures or every seed/horizon | Requires revision: the RMS event has a different acute locus. Scope the readout/feature-mean account to the measured unnormalized cases. |','',
        'The revised contribution is retained information during severe native failure, causal readout localization and the feature-mean logit intervention, a direct test of the basis hypothesis, and a measured numerical pathway in the regularized split-Muon/AdamW setting. The CE cancellation account builds on [Prieto et al.](https://arxiv.org/html/2501.04697v2); the numerical feature-inflation account and zero-sum projection are credited to [Liu Hanqing et al.](https://arxiv.org/html/2605.06152v2).','',
        '## Reproduction and completion','',
        'The fixed endpoints are complete. Final saved-model predictions, monitoring traces, source qualification, exact update checks, loss-gradient tests, and decomposition identities are audited. `verification.json` records the 28 new-run final-state checks, the reused seed-4 original endpoint check, and total planned training updates. `results.json` and `branches.csv` contain the main and specificity measurements. Full diagnostics, source hashes, every retained checkpoint, optimizer states, RNG states, and generality analyses are included with the accompanying code and artifacts.','',
        'One metadata wording correction is explicit: the shared runner\'s generic primary_endpoint string says 30,000 even for the specificity windows. Their written protocols, actual end_step, saved trajectories, and aggregation all use 20,000. The data and execution used the intended endpoints. RMS checkpoints must be loaded with the RMS architecture and subtraction checkpoints with subtraction labels.','',
        'The claims above have reached the stated decision criteria. The required manuscript changes are supplied in [paper_revision.md](paper_revision.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    template=(HERE/'manuscript_revision_template.md').read_text()
    import re
    replacements={'PRIMARY_ARITHMETIC_ENDPOINT_SENTENCE':primary_sentence,'PRIMARY_ARITHMETIC_RESULTS':primary_sentence,'SPECIFICITY_RESULTS':specificity_sentence,'GENERALITY_RESULTS':general_sentence}
    for key,text in replacements.items():template=re.sub(r'\*\*\['+key+r':[^\]]*\]\*\*',text,template)
    assert not re.search(r'\[(PRIMARY_|SPECIFICITY_|GENERALITY_)',template)
    template=template.replace('# Manuscript revision template','# Completed manuscript revisions')
    template=template.replace('Bracketed fields are reserved exclusively for ongoing arithmetic and generality endpoints. Replace those fields with completed measurements before submission.','All arithmetic and generality endpoints below are completed measurements.')
    (OUT/'paper_revision.md').write_text(template)
    for name in ['results.json','branches.csv']:shutil.copy2(HERE/'analysis'/name,OUT/name)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    for base in [HERE/'figures',HERE/'diagnostics_results',HERE/'generality']:
        for p in base.glob('*'):
            if p.suffix in ('.png','.pdf'):shutil.copy2(p,figures/p.name)
    print(OUT/'report.md')

if __name__=='__main__':main()
