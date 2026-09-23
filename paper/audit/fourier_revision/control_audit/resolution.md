# Final resolution

The audit passes after one numerical tie correction and one report rounding correction.

- Independent primary, mathematical, checkpoint donor, and mean-update audit: 1,877 checks passed.
- Independent explicit Fourier synthesis of every secondary prefix and phase/relocation accuracy: 4,020 checks passed. This covers all 600 stochastic perturbations, both evaluation contexts, and each train/held-out/full-grid denominator. No integer correct count differs from the regenerated secondary records. The largest minimum-margin difference is 9.98e-15.
- The control implementation restores the exact answer-zero orbit identity under frequency relocation before subtracting the original template. This makes the resulting all-zero held-out logits respect the declared smallest-index tie rule. The fix and its reason are recorded in `work/fourier_control/corrections.md`. The original protocol documents were preserved. All isolated-family metrics are unchanged.
- The seed-1 raw post-event donor-table entry is now 9.09 percent, consistent with 813/8,939. The manuscript donor table was checked after correction.
- The updated Fourier-control appendix includes the normalized single-pair counterexample, explicit ablation denominators, phase means with replication scope, and the correct limitation on held-out donor averaging. These statements agree with the independent computations.

The independent scripts import neither control implementation. Source hashes at review time are saved in `audit_manifest.json`. The mean/full-update maximum gap is independently confirmed as 0.3244210761830182 percentage points, with source paths and hashes in `independent_verification.json`.

No numerical discrepancy remains in the audited new controls. The mathematical result establishes that the tested Fourier interventions can acquire predictive structure through an externally supplied answer-orbit relation. Original measurements, raw-feature decoder results, and readout-swap findings keep their separate evidential scope. The simple memorizer's full-grid ablation is 29.9945 percent, so it does not reproduce the original trained-model near-chance full-grid ablation. Held-out-donor leave-one-out remains a task-informed diagnostic using other held-out forward outputs.

The narrow wording recommendation “with corrected CE” for the arithmetic intervention was sent to root. The new Fourier appendix is scientifically scoped as written.
