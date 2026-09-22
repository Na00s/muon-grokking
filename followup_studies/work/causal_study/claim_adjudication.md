# Decision criteria for the final causal study

Written September 22, 2026, before inspecting the new causal-study training results. This is a prospective decision rule for this additional block of experiments. Earlier collapse, basis, intervention, and numerical-control outcomes are already known and are the reason for the study. The five seeds and initial hyperparameters were selected in earlier work; this is a matched follow-up of those cases.

## Question and finite endpoint

The central question is whether the observed post-grokking failures require revising the proposed explanation, and which explanation the completed evidence supports within the tested setting. A finite experiment cannot establish stability for every possible future training step. We will decide the claims about the five recorded initializations, the specified task and architecture, and a common 30,000-update horizon.

The primary contrast starts from the same saved model, all optimizer states, and RNG state at update 6,000. One branch uses the original float32 cross-entropy computation; the other computes the same mathematical objective with accurate small-loss and correct-class derivative arithmetic. Original Muon, its float32 Newton-Schulz computation, the architecture, data split, hyperparameters, and optimizer states remain unchanged. Step 6,000 is an early post-grokking intervention. It already has measurable numerical contamination in the previously examined seed 4, so it must not be described as preceding all numerical error.

The primary binary endpoint is an observed simultaneous training and test accuracy below 90% after sustained grokking and by update 30,000. Sustained grokking follows the original six consecutive 100-step evaluations at or above 95% test accuracy. Record training accuracy every update and test accuracy on a fixed dense grid, plus every possible joint-collapse trigger identified by training accuracy below 90%. A branch without a joint event has a valid no-event result for this endpoint if every low-training-accuracy state is checked on test. Test-only excursions between scheduled evaluations remain outside this guarantee. Also report any failure to maintain the original 95% test-accuracy stability threshold on the stated evaluation grid.

Validated stock trajectories can be reused. Their densely recorded positive events suffice for the binary endpoint even when the surrounding historical observations use different grids. Exact resumed-checkpoint provenance, replay checks, and treatment of every saved optimizer buffer must be retained. Mixed historical monitoring prevents a fair comparison of event counts, duration below threshold, or historical minimum accuracy; those comparisons are excluded from the primary analysis.

## Decisions already settled by completed evidence

1. **Exact global basis equivalence of the measured collapse pairs requires revision.** The full-domain least-squares obstruction, held-out fits, and planted full-network gauge controls already distinguish the observed representations from one global invertible change of coordinates at the measured numerical scale. Additional successful approximate alignments cannot turn these measured pairs into exact gauge orbits. The architecture's joint residual gauge symmetry remains correct.
2. **Universal invariance of dominant-frequency rankings under GL(d) is false.** Invertible maps preserve exact zero/nonzero Fourier support. Orthogonal maps preserve power. A diagonal invertible rescaling can reverse the ranking of two nonzero Fourier vectors. This explicit counterexample settles the claim without more training.
3. **Substantial linear task information survives the selected failures.** Train-only probes recover 98.20 to 100% held-out accuracy at the five studied collapse checkpoints. The existing data support this quantitative statement. A stronger statement that every bit of task information survives every failure exceeds the observations and is unnecessary for the central result.
4. **The readout update is sufficient for most acute loss in the five examined adjacent transitions.** The four representation/readout swaps establish this event-level result. They also show that the updated representation remains useful with the preceding readout. This resolves the acute locus within these transitions; it does not identify the earlier dynamics that generated the harmful update.
5. **One-time head repair under the tested continuing optimizer states has a limited duration.** The completed continuation experiments already establish rapid relapse, including state-reset controls. These intervention outcomes should be reported as measured. Adam's coordinatewise state generally changes the dynamics under arbitrary residual basis transformations, so immediate relapse is not an independent proof against architecture symmetry.

These claims have reached their stopping criteria. More runs directed at proving their stronger original versions would repeat a question that already has a counterexample or a supported finite answer.

## Decisions for the new numerical experiment

| New result by the common horizon | Required conclusion |
| --- | --- |
| All five stock branches collapse and all five accurate-loss branches avoid the endpoint | The arithmetic change prevents the observed class of joint collapse in all five matched continuations through 30,000. Together with frozen-gradient error measurements, this establishes a causal contribution of CE arithmetic in this configuration. Revise the assertion of no numerical pathology. Retain the finite horizon and the prior contaminated state at branching. |
| Accurate-loss branches also collapse in one or more seeds | CE cancellation is not necessary for every observed failure. Analyze the accurate-loss event with the same adjacent head swaps and derivative checks. Report seed-specific effects on incidence and timing. A failure of the accurately evaluated objective does not restore the false claim that the stock trajectory lacks numerical error. |
| A stock branch does not reproduce its already saved dense event under exact replay | Stop scientific interpretation for that branch and resolve the implementation or checkpoint-state discrepancy. A missing reproduction is an execution failure, not evidence of stability. |
| The arithmetic change delays collapse beyond the tested stock event but another event appears before 30,000 | Report delayed or altered dynamics with the measured new failure. The endpoint rule prevents early successful windows from being called permanent prevention. |
| Arithmetic correction prevents failures in the main configuration but a generality case fails | Scope the rescue to the settings where it occurs. Inspect the corrected failure to determine whether the same acute readout mechanism still holds. One counterexample resolves a universal arithmetic-sufficiency claim. |

The numerical claim will be adjudicated after all five matched branches finish, all observed discrepancies are resolved, and the correction-specificity study below finishes if the primary rescue occurs. There is no requirement to search indefinitely for another seed or another horizon after these registered comparisons.

## Most decisive additional control: repair one derivative identity

For each example, exact softmax cross-entropy obeys

    sum_j dL/dz_j = 0
    dL/dz_y = -sum_(j != y) dL/dz_j.

At a fixed logit matrix, use PyTorch's original forward loss and retain its wrong-class derivatives. Replace only the correct-class derivative with the negative sum of the wrong-class derivatives, using a float64 sum and casting the result to the model dtype. This target-derivative repair changes the same cancellation-damaged entry identified by the fixed-logit audit. It keeps the objective's reported forward value and the remaining model/optimizer arithmetic unchanged. Explicit custom-backward derivatives must respect the loss reduction and any incoming scalar gradient.

Run this targeted repair from update 15,000 through 20,000 for each seed, with the same retained optimizer states and monitoring. Compare the inherited stock event with the accurate-CE and targeted-repair branches over this shared window. Verify on fixed logits that the repaired gradient closely agrees with an accurate float64 reference across confident, uncertain, and incorrect predictions.

- If the targeted repair recovers the accurate-CE outcome across the tested seeds, cancellation in the correct-class derivative becomes a specifically supported upstream contributor. The paper can report this controlled arithmetic mechanism together with the readout-dominated acute failure.
- If accurate CE rescues a case that targeted repair fails to rescue, retain the broader arithmetic-sensitivity result. Inspect the disagreement in fixed-logit derivatives and trajectory before assigning it to another CE operation.
- A supplementary row-mean projection subtracts the stock gradient's class mean from every class. It removes the shift-violating component but retains class-relative derivative error. Its result distinguishes consequences of the common component from the remaining error. Failure of this projection is compatible with success of targeted repair.

Frozen-state update diagnostics must separate raw gradient errors, readout moments, the applied parameter update, and class-centered logit changes. A class-common raw readout gradient directly changes only a shared logit shift under a scalar update. Adam's class-dependent coordinate scaling and optimizer history can convert that component into class-relative parameter changes. Demonstrate that conversion with the actual saved state before attributing the damaging update to common-mode leakage.

An accurate-loss rescue plus fixed-state diagnostics identifies an intervention-sensitive causal pathway. A unique complete microscopic explanation of every parameter's long history is a stronger claim. The paper can finish this study with a quantitative mechanism and clearly specified counterfactuals.

## Generality and final stopping rule

The additional subtraction and gain-free RMS-normalized addition comparisons test named extensions, each with matched stock and accurate arithmetic. Report the operation, normalization, seed, horizon, and endpoint for each case. If a variant does not grok by the horizon, label its post-grokking collapse question unevaluated and report its training result; do not search hyperparameters solely to produce the desired event.

The study is complete when: (1) the five main matched endpoints and the declared extension endpoints are available; (2) the targeted derivative controls, if triggered by primary rescue, have completed; (3) every new corrected-arithmetic failure has had a fixed-state derivative check and adjacent head swap; (4) execution/replay and loss-gradient checks pass; and (5) the manuscript claims are rewritten to match these outcomes. Counterexamples require revision immediately. Supported finite claims are accepted with their tested scope. Universal permanence, exact gauge motion of already incompatible matrices, and a mathematical invariance with an explicit counterexample will not be held open as demands for further experimentation.
