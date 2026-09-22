# Muon grokking follow-up experiments

22 September 2026

## What the experiments establish

The central empirical claim is supported: **generalizing, linearly accessible task information survives a joint training/test accuracy collapse.** Fresh readouts trained on the original training split recover 100% held-out accuracy in seed 4 and 98.72% in seed 0.

The cause needs a more qualified account. **The implementation is numerically sensitive.** In a matched continuation, changing only the arithmetic used to evaluate cross-entropy keeps test accuracy at 100%, while the original calculation falls to 18.13%. Both float64 controls also remain at 100% over that window. These results warrant revising the draft's assertion that numerical pathology is absent.

All findings below are new experiments unless explicitly marked as an archived-log analysis. The original repository was kept unchanged.

## 1. Information survives the collapse

All reported accuracies in this table use the same 8,939 held-out examples for each seed. A readout is a bias-free linear map applied to the raw final residuals. No Fourier filtering or held-out fitting is used.

| Held-out evaluation | Seed 4, first failure | Seed 4, four steps later | Seed 0, first failure |
|---|---:|---:|---:|
| Healthy → evaluated step | 16050 → 16056 | 16050 → 16060 | 17200 → 17494 |
| Original healthy model | 100.00% | 100.00% | 99.12% |
| Original failed model | **66.33%** | **18.13%** | **31.87%** |
| Healthy readout on failed residuals | 99.81% | 99.62% | 14.05% |
| Orthogonally transported healthy readout | 99.99% | 99.94% | 95.51% |
| Backward linear transport | 99.99% | 99.98% | 95.40% |
| Fresh whitened linear readout | **100.00%** | **100.00%** | **98.72%** |

The corresponding native training accuracies at the three failed states are 70.08%, 21.78%, and 48.25%. The two seed-4 columns describe the same episode, rather than independent replications. Step 16060 is the worst held-out observation on the 10-step grid in the matched 15000–18000 continuation; it is a secondary severity check.

The healthy fresh-readout comparators achieve 100% for seed 4 and 99.51% for seed 0. Thus the seed-0 representation retains almost all of the healthy comparator's accessible generalization, with a residual difference of about 0.78 percentage points.

**Decoder optimization mattered.** The initial RMS-scaled decoder on seed 4's later step-17000 state stopped without convergence at 88.04% test accuracy. An uncentered SVD whitening control converged and reached 99.96%. Both reports are retained. Every reported whitened fit in the table converged, selected zero regularization using training-only validation, and retained all 128 dimensions without flooring any singular values. At seed 4's first failure, both healthy and collapsed decoders converged in 19 iterations.

The seed-4 event strongly localizes the immediate failure to the readout's compatibility: restoring the older readout almost completely repairs predictions. Seed 0 requires substantial alignment or refitting, showing that the interface changes differently across the two trajectories.

## 2. Numerical arithmetic changes the observed failure

These four continuations begin with the identical seed-4 model, all three optimizer states, and RNG state at step 15000. They run through step 18000 and evaluate every 10 steps.

| Continuation | Minimum train accuracy | Minimum test accuracy |
|---|---:|---:|
| Original float32 model and cross-entropy | 21.78% | **18.13%** |
| Float32 model, accurately evaluated cross-entropy | 100.00% | **100.00%** |
| Float64 model, stock CE, original float32 Muon orthogonalization | 100.00% | **100.00%** |
| Float64 model and orthogonalization, accurate CE | 100.00% | **100.00%** |

The resumed original arm matches the baseline's model, all optimizer states, and RNG **bitwise after 1,000 updates**, through step 16000. This verifies the original-arm replay; the other arms change arithmetic starting from the common step-15000 state.

An additional accurate-CE float32 continuation, specified before observing the new failure, starts at step 6000 and runs to step 20000. Training accuracy remains 100%; the lowest sampled test accuracy is its starting value, 99.8434%, and it ends at 100%. It has no sampled test-accuracy drop below 99% on its 100-step grid.

For a correctly predicted example, the accurate calculation evaluates the same cross-entropy objective as

\[
L=\log\!\left(1+\sum_{j\ne y}\exp(z_j-z_y)\right),
\]

using `log1p` and preserving the small correct-class derivative. Incorrect predictions use ordinary cross-entropy with safe handling of the unused expression. This is an arithmetic control for the existing objective. It is separate from the paper's frozen-auxiliary Stable Muon intervention.

The original Muon implementation casts the Newton–Schulz input to float32 even for float64 parameters. The two float64 arms distinguish model precision from orthogonalization precision. The draft's existing float64 experiments concern the no-Newton–Schulz ablation, so they do not settle numerical sensitivity in this original Muon condition.

**Scope:** these are finite continuations around one observed seed-4 episode, with one longer accurate-CE continuation. They establish numerical sensitivity of that trajectory. They do not establish permanent prevention, the frequency of failures across seeds, or a unique cause of every collapse.

## 3. Numerical contamination is measurable before failure

The fixed-logit check compares derivatives on exactly the same float32 logits, promoting those logits to float64 for the reference calculation.

| Checkpoint | Stock-CE logit-gradient relative L2 error | Examples with zero stock loss and zero correct-class gradient |
|---|---:|---:|
| Seed 4, step 6000 | 1.75% | 0.00% |
| Seed 4, step 12000 | 6.52% | 0.47% |
| Seed 4, step 16000 | **32.02%** | **26.53%** |

At step 16000, those zero-target-gradient examples still have nonzero wrong-class gradients. The class-common component comprises 70.63% of the stock unembedding-gradient norm, versus about 0.0000495% with accurate float32 CE. Exact cross-entropy is invariant to a shared shift of all class logits, so this common component should vanish in exact arithmetic. It measures derivative error; direct changes in predictions require class-relative logit changes.

This pattern is directly relevant to Prieto et al.'s softmax-collapse analysis, which allows continuing wrong-class updates after the correct-class gradient becomes zero. Their Appendix I also documents sensitivity to the implementation of `log_softmax`. See [the official ICLR 2025 paper, §3.1 and Appendix I](https://proceedings.iclr.cc/paper_files/paper/2025/file/c9e6ac15e689e06139d7b39e1667b165-Paper-Conference.pdf).

## 4. The final failing update is localized to the readout

An exact replay reconstructs step 16055 and then reproduces the saved step-16056 failure, including all optimizer states. At step 16055 the model has already begun degrading: 98.77% training and 91.34% test accuracy.

Hybrid models apply selected groups' actual 16055→16056 parameter changes to the preceding state, with no further training:

| Post-step parameter groups applied | Train accuracy | Test accuracy |
|---|---:|---:|
| None | 98.77% | 91.34% |
| Readout only | 69.03% | 66.75% |
| Hidden matrices and embeddings | 99.06% | 98.66% |
| All groups | 70.08% | 66.33% |

Across all eight combinations, all four containing the updated readout jointly fail the 90% threshold; all four retaining the preceding readout avoid it. The readout changes by 9.63% of its previous Frobenius norm in this single update; its class-centered change is 9.04%. Embeddings change by 14.50%, and hidden matrices by 0.33%.

Accurate CE applied **only at this final step** yields the same collapsed accuracies. The immediate failure is localized to the readout update, while the preceding trajectory and optimizer state remain important to the explanation. This is consistent with the stronger effect of correcting arithmetic from step 15000.

## 5. How much does this support a change-of-basis mechanism?

The fitted maps provide effective repairs, but representation reconstruction is approximate. For seed 4's six-step event, orthogonal reconstruction has about 2.36% raw relative error and 31.54% error after centering; backward linear reconstruction has about 1.71% raw error and 25.60% centered error. Large common components make the raw errors look much smaller.

A matched healthy six-step interval, 16044→16050, has only about 1.51% centered orthogonal error and 1.18% centered backward-linear error. Both checkpoints and all swapped/transported readouts retain 100% accuracy there. Planted rotations and shuffled-correspondence controls also behave as expected.

The supported claim is that **task information remains linearly accessible while the learned readout becomes incompatible with it**. A global invertible basis change remains a proposed mechanism. Effective transport, numerical rank, and low uncentered reconstruction error alone do not establish it. Rank sensitivity at float32-related thresholds is saved alongside the fits.

## Methods, selection, and verification

- Source: [Na00s/muon-grokking](https://github.com/Na00s/muon-grokking), commit `6d64a981af75f1300d9060109e81552d48a81360`.
- Task and architecture: modular addition, p=113, 30% training data, one transformer block, width 128, four heads, MLP width 512, no normalization or biases, full batch.
- Optimizers: hidden Muon learning rate .03, momentum .95, weight decay .1, five NS iterations; embedding AdamW learning rate .001 and decay 1; readout AdamW learning rate .00025 and decay 1.
- Runtime: deterministic single-thread CPU, PyTorch 2.13.0, NumPy 2.5.2, SciPy 1.18.0. These are fresh replications; the original serialized checkpoints and complete runtime provenance were absent from Git.
- Seed 4 was selected for an early failure in the archived logs; seed 0 was added as a second initialization. These selections support mechanism tests and do not estimate population failure rates.
- The initial baseline protocol evaluates every 100 steps, with a 30000-step cap and stopping 500 steps after a qualifying joint failure. A qualifying event follows six consecutive evaluations at or above 95% test accuracy. Dense exploratory replays start at seed-4 step 15000 and seed-0 step 16000 and inspect training accuracy every step, checking test accuracy whenever training accuracy falls below 90%. They capture failures at 16056 and 17494. Seed 4's brief joint failures are missed by the 100-step grid; seed 0's coarse-grid event is at 17500.
- All map fitting and decoder training use the original 3,830 training examples. Regularization selection uses 3,064 inner-fit and 766 inner-validation examples; the 8,939 test examples remain excluded from fitting and hyperparameter selection. Event and healthy-checkpoint selection are explicitly retrospective diagnostic choices.
- The initial model and five updates match the source training loop bitwise. Checkpoint continuation, numerical helpers, planted transformations, destroyed-feature controls, held-out independence, decoder derivatives, and convergence reporting were checked. Exact replay checks also cover 1,000 resumed updates and the six updates leading to the captured collapse.

An archived-log reanalysis separately counts 49 sampled post-grokking test-below-90% episodes across the paper's five original seeds, containing 665 affected evaluations. Only four sampled evaluations also have training accuracy below 90%. The new dense replay demonstrates why these sparse observations cannot measure the full frequency or duration of training failures. The five archived frozen-auxiliary arms have no sampled post-grokking test drop below 95%.

## Manuscript implications

1. Add the train-only fresh-readout and transport results as direct evidence that the task information survives failure.
2. Add the matched arithmetic controls and revise the statement that numerical pathology is absent. Preserved information and numerical sensitivity can coexist.
3. Use the one-step group interventions to describe the immediate readout failure, while qualifying the proposed basis-change mechanism and reporting centered reconstruction errors.
4. Distinguish observations from independent episodes and seeds, and increase temporal resolution around failures. Retain the no-normalization and modular-arithmetic scope.

The bundle includes all reported scripts, raw arrays, selected checkpoints, trajectories, original and improved decoder fits, verification records, and a SHA-256 manifest. The research report and figures are a summary of that evidence.
