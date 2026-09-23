# Fourier interpretation and original-horizon revision

The revision addresses the task structure supplied by the Fourier filter, promotes the acute mean-dependent update result, extends corrected-loss branches to the original horizon, and centers the abstract and introduction on the failure mechanism and surviving information. The original experiments and reported results are retained.

## Fourier controls and interpretation

The complete addition family with DC equals averaging over pairs `(a+t, b-t)` with the same answer. A linear readout commutes with the average. The subtraction identity uses `(a+t, b+t)`.

On all five actual addition splits at p=113, a lookup predictor that memorizes training answers and returns zero on unseen inputs scores 0.761–0.906% on the held-out split before filtering and 100% afterward. The raw correct counts are 79, 68, 79, 81, and 77 out of 8,939. The zero logits use smallest-index argmax. A training-count-normalized lookup scores 100% after retaining any one of the 56 diagonal conjugate pairs plus DC, verified in 280 cases. Sparse sufficiency and phase sensitivity therefore also require care in algorithm identification.

The ablation denominators are explicit: the lookup retains 100% training accuracy, gives 0% held-out accuracy, and gives 29.9945% over the full grid. This control does not reproduce the trained models' near-chance full-grid ablation results. Donor-restricted averages at ten existing endpoints add information about unseen-input logits while still using task-defined answer orbits.

The paper preserves the early 95.14% filtered result and reframes its interpretation. Sections 5 and 6 now discuss task-informed projections and interference. Appendix K gives the proof, controls, exact counts, sparse counterexample, perturbation results, and donor analysis. Figure 9 shows the memorizer and sparse-control results. The original six figure PDFs and numerical cells in the 17 original appendix table files are retained.

## Acute mean-dependent readout failure

Section 4.4 now contains the decomposition of the actual head-induced logit change into a training-mean term and a centered term. Main-text Table 3 reports every seed. The mean term alone nearly reproduces each failing test accuracy; the largest observed difference is approximately 0.324421 percentage points. The centered term retains 87.20–100% test accuracy and at least 99.09% training accuracy.

This is an evaluation-only intervention using the post-update representation and its training mean. It establishes local sufficiency of a class-dependent offset shared across examples. Accuracy agreement does not establish prediction-by-prediction equivalence or a unique historical mediator. The raw-feature decoders, adjacent swaps, and numerical interventions provide separate evidence.

## Corrected loss through the original horizon

All five corrected addition branches and the corrected subtraction run completed step 100,000. Each continued for 70,000 updates from its preserved step-30,000 state, totaling 420,000 new updates. None recorded a joint train/test failure below 90%. The five addition branches finish at 100% test accuracy; their minimum measured test accuracy over steps 6,000–100,000 is 98.859%. The subtraction run finishes at 99.955%; its post-confirmation minimum is 96.879%.

Every training state was checked. Test evaluation occurred every 100 steps and whenever training accuracy fell below 90%, detecting every joint event. Final and minimum-accuracy states were replayed. The extension confirms suppression through the original depth-1 budget. Behavior after step 100,000 and brief test-only excursions between scheduled evaluations retain their stated finite-measurement limits.

## Positioning and prose

The abstract and contribution list lead with acute failure localization, surviving information, and the numerical mechanism's scope. Liu et al.'s numerical feature-inflation mechanism and normalization studies, Chou et al.'s representation/readout probe diagnostic, and Wang's Muon speed/stability results are credited explicitly. The added evidence concerns actual captured displacements, matched long-horizon interventions, and the acute failure site under RMS normalization.

The four requested wording changes are applied: nearly constant Muon updates at small gradient norms; stabilization of tested depth-1 runs by freezing embeddings and readout; 100% accuracy of the filtered representation over the full grid; and a concrete comparison of numerical error and the failing parameter group between architectures.

## Submission format

The scientific main text occupies nine pages, meeting the initial-submission limit in the [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines). The reproducibility and required AI use statements appear separately and are excluded from that limit. The compiled `sec:main-end` marker measures the scientific main text independently of the references' starting page.

The layout adjustment moves the quiet-window figure to Appendix C and the Fourier-grid figure to Appendix G, retaining their original PDF bytes and results. Four redundant sentences added during revision were removed. The causal follow-up figure remains in the main text. All six original figure PDFs and the numerical cells of the 17 original appendix table files are preserved. The AI use statement describes assistance with experiments, verification, derivations, literature comparison, figures, and revision.

The included style, bibliography style, natbib, and fancyhdr files are byte-identical to their counterparts in the [official ICLR 2027 template](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip). Existing filenames are retained; `audit/fourier_revision/control_audit/official_template_verification.json` records the comparisons and hashes.

## Verification and reproduction

The Fourier implementation passes 1,552 checks. An independent implementation passes 5,897 checks, including explicit Fourier synthesis of all 600 secondary perturbations. The audit corrected exact-zero tie handling in one auxiliary relocation control and a rounding error in the report; the primary scientific results are unchanged. A published-path reproduction independently matches the synthetic results and figure pixels. The earlier 604-check audit remains available with its immutable input snapshot.

Long-horizon verification checks all 420,006 extension states, test schedules, original hashes, restored model/optimizer/PyTorch random state, checkpoint grids, endpoint metrics, and minimum-accuracy metrics. The initial restoration and 12-update continuation checks are bitwise exact in all six branches. The source figure scripts and final publication checks record their input and output hashes.

Reproduction code and raw results live in `followup_studies/work/fourier_control` and `followup_studies/work/long_horizon`. The new checkpoint release uses `FOURIER_HORIZON_ARTIFACTS.json`; earlier source checkpoints remain in `CAUSAL_ARTIFACTS.json`. `paper/audit/fourier_revision` contains the independent review, and `paper/verification.json` records final compilation, preservation, and visual checks.
