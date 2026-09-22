# Final report structure and required claim decisions

This outline is written while the fixed-horizon experiments run. Numerical outcomes belong in the final report only after the corresponding branch has completed and passed the dense-monitoring audit.

## Main result

State the five original and five corrected-arithmetic binary endpoints through update 30000. Give each original event step and native held-out accuracy, each corrected branch's lowest measured held-out accuracy, and final accuracy. The unit is a training seed. The seeds are five selected original trajectories; they are a matched mechanism panel and cannot estimate population prevalence.

## Specificity

Report the common15000-to20000 interval for accurate CE, correct-target-derivative repair, and row-mean projection. All15 arms inherit identical per-seed parameters, all optimizer buffers, data split, and RNG. Target repair retains the original forward loss and wrong-class derivatives. Projection is the prior literature's zero-sum-gradient intervention and retains most of the fixed-logit derivative error. Report every failure as well as every rescue. The shared stock comparator has a verified positive event in each window.

## Measured causal chain

1. Fixed-logit target-gradient error is already measurable at update6000 and grows before failure.
2. Compare original and intervention feature/classifier means at common checkpoints. Retain the prior numerical-feature-inflation attribution and state which parts are descriptive compatibility.
3. Under actual saved Adam state, a class-common gradient perturbation can induce a class-centered parameter-update difference. Give the SGD cancellation control.
4. At the captured event, parameter hybrids and old/new readout swaps identify the acute damaging component. Exact replay must pass.
5. The actual head displacement is an initially descending direction with preceding features fixed. Its full scale overshoots. Report training-selected scale controls and the following-feature seed3 qualification.
6. The feature-mean decomposition tests how much of the head-induced logit error comes from an input-independent class shift. Its evaluation-only scope must remain explicit.

## Generality

Report all four fixed30000-update runs, including failed grokking if any. The two configurations are subtraction without normalization and addition with gain-free RMS normalization. Each has one matched seed. Keep the earlier broad sweep and these specific arithmetic controls distinct. Analyze every corrected-arithmetic joint failure before final adjudication.

## Already resolved questions

- Selected collapsed representations retain substantial generalizing linear information: prior training-only probes recover98.20–100%.
- The measured representation pairs fail the exact global basis-map hypothesis against planted numerical controls. Architecture symmetry remains valid.
- Exact Fourier support is GL-invariant. Dominant-frequency rankings require correction because a general invertible map can change norms. Orthogonal maps preserve power.
- One-time head repairs have their measured continuation limits. Long arithmetic interventions test a different, earlier causal handle.

## Manuscript edits

Provide replacement abstract, central-mechanism paragraph, numerical-related-work paragraph, Fourier-invariance statement, and limitations. Revise statements contradicted by the measurements. Credit earlier numerical-collapse and feature-inflation studies. The final claim set must have completed supporting experiments and explicit finite scope.

## Reproducibility

Publish code, written protocols, trajectories, full checkpoints including all optimizer and RNG states, exact-update checks, source and artifact hashes, figures, aggregate audit, and release download instructions. Preserve previously published studies and the original implementation. Record that runner metadata's generic30000 endpoint label is overridden by the actual end_step and20k specificity protocols. Archive the original reports as historical assessments and mark the new report as the current conclusion.
