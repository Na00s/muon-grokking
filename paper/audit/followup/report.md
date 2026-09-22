# Follow-up evidence audit

The reported follow-up accuracies, errors, counts, steps, and horizons agree with the saved experimental evidence. The independent ledger contains 604 checks covering all four new tables, every point or bar in both new figures, the numerical statements in the follow-up appendix, and repeated main-text quantities. The principal correction is the relationship between the original and follow-up grokking definitions: the new runs require six evaluations, while the original sweep requires five. A universal 500-update offset between those definitions is therefore unjustified.

## What was recomputed

- All 20 adjacent-swap accuracies were recomputed by multiplying the saved residual arrays by each saved readout.
- All 10 reference/collapsed fresh-decoder accuracies were recomputed from saved coefficient matrices and held-out features. All 10 refits and all 50 inner-validation candidate fits converge. Training and test masks have 3,830 and 8,939 examples. The 20% validation split, five-value regularization grid, full 128-dimensional uncentered preconditioner, and 1e-6 singular-value floor match the code and records. Validation CE selects the recorded penalty in every case.
- All 15 reconstruction errors in the geometry table were independently recomputed by solving the unregularized training-only least-squares problem and evaluating train-mean-centered residuals on held-out examples. All paired feature matrices have full column rank 128. Healthy intervals match each seed's reference-to-failure interval.
- All 24 causal branch CSVs were scanned directly. Their training state coverage is complete, and every state below 90% training accuracy includes a test evaluation. The five early accurate-loss branches have no joint failures through 30,000 and remain at 100% training accuracy. The 15 specificity branches have no joint failures through 20,000. Their final test accuracies are all 100%.
- All four generality trajectories were rescanned for six-check confirmation, first joint failure, measured post-grokking minimum, and final accuracy. The runner examines training accuracy every update and saves every tenth state plus every joint failure. The sparse CSV alone does not document every training-only excursion; the source loop establishes that monitoring behavior.
- All five mean and centered-only interventions were independently reconstructed from raw feature arrays and actual readout displacements. Their reported accuracies and decomposition-error bound agree.
- All 24 matched arithmetic/stock branch starts were compared to their source checkpoints. Parameters, optimizer states, and RNG states agree bitwise. The four generality initializations also agree bitwise across all three state categories.
- All seven whole-network direction/interpolation panels were re-evaluated from checkpoints. Every directional derivative, interpolated loss, and endpoint check matches the recorded diagnostics exactly. The readout-gradient reference uses accurate float64 derivatives on identical float32 logits, then casts those derivatives to float32 for the network backward pass.
- Both new figures regenerated successfully using the repaired default paths in a simulated repository. Every figure input hash matches the published results tree, and all six original PDFs remain unchanged.

The update totals are correct: 52,698 fresh baseline updates; 18 continuations of 500 updates, totaling 9,000; and 364,302 causal-study updates. The last fresh-baseline checkpoints are essential for counting updates because their CSV logs are thinned and omit the final non-grid step.

## Wording and reproducibility corrections sent to the main audit

1. Describe the follow-up six-check confirmation separately from the original five-check streak-start definition. All five step-6,000 branch sources independently meet the new six-check criterion.
2. Specify that decoder regularization applies to coefficients in the preconditioned coordinates. Whitening changes the implied penalty in raw coordinates; the recorded unregularized controls distinguish that penalty from the numerical conditioning change.
3. Identify the 4.62% task-only-map example as seed 4's step-16,000-to-17,000 pair. The number is correct. It is distinct from the preceding seed-0 example and from the adjacent failure pairs.
4. Scalar-SGD centered update fractions span 2.81e-16 to 4.06e-16. Describing this as float64 roundoff avoids excessive precision in the earlier approximate 3e-16 wording.
5. The target repair and row-mean projection both perform their reductions in float64 and cast the resulting derivative back to float32. The trainable model and optimizer computations remain float32.
6. Phrase the RMS result at the scale actually tested: the two RMS trajectories use one seed, and their embedding-only interventions reproduce the acute accuracy loss. The existing universal-prevention counterexample is logically valid. Population prevalence is unmeasured.
7. The plotting script's default data root and output directory were incorrect. In `work/submission_audit/revised/figures/build_followup_figures.py`, defaults now select the repository's `followup_studies` directory and the containing paper source. Outside that repository layout the script requests an explicit `--workspace` argument. The standalone README already gives the correct explicit command.

## Claims and prose review

The core conclusions are appropriately bounded: substantial linear task information survives in all five selected collapsed representations; the measured feature pairs fail exact global basis equivalence; arithmetic controls prevent joint failures within the stated horizons; the recorded updates locally descend the accurate training objective and overshoot at full scale; the RMS event supplies a counterexample to universal prevention.

The mathematical GL statements are correct. Two full-column-rank feature matrices admit an invertible feature-coordinate map exactly when their column spaces agree. Two full-row-rank 113-by-128 task coefficient matrices always admit an invertible alignment, because their independent rows can be completed to bases of R^128. Exact task-only alignment therefore supplies weak evidence about a global feature transformation. The separate full-domain residual tests are needed for the exact-global claim.

The mean decomposition is an identity, and the evaluation interventions establish local sufficiency of the mean term under the chosen baseline. They do not establish unique historical mediation. The manuscript correctly retains that limitation. In seed 3 the centered-only term also reduces accuracy below 90%, so a necessity claim would be unsupported.

The linear-decoder result establishes a constructive lower bound on retained decodability. It does not establish identical representation quality: three seeds have small reference-to-collapse gaps. The text already acknowledges those gaps and the selected-checkpoint scope.

The appendix is readable technical methods prose. Keep direct verbs and explicit experimental subjects. Sentences such as “In the RMS runs, the embedding update causes the acute loss of accuracy” are clearer than the broad phrase “RMS normalization changes the acute locus.” The new paragraphs do not need a wider rewrite. Claims of permanent prevention, unique mediation, universal Muon-specific failure, or general population incidence would exceed the evidence; none is needed for the current contribution.

## Audit artifacts

- `audit_followup.py`: reproducible scalar/array/trajectory audit.
- `claim_ledger.json`: 604 claim-level entries with reported values, recomputed values, source locations, and evidence paths.
- `recomputed_datasets.json`: intermediate adjacent-swap, decoder, branch, generality, and continuation data.
- `checkpoint_replays.py`: independent saved-state comparison and re-evaluation of the seven direction panels.
- `checkpoint_replay_audit.json`: fresh replay results.
- `plot_smoke/paper/source/figures/followup_figure_provenance.json`: regeneration provenance from the default-path test.

Run from the original workspace using the research Python environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/homebrew/Caskroom/miniforge/base/envs/research/bin/python work/submission_audit/followup/checkpoint_replays.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/homebrew/Caskroom/miniforge/base/envs/research/bin/python work/submission_audit/followup/audit_followup.py
```

No experiment source, stored measurement, or original paper figure was altered by the audit.
