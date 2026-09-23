# Control, event, speed and feature-mean provenance audit

This audit preserves all historical results. It recomputes their recorded summaries and distinguishes what the retained records certify from what the original execution records no longer establish. Relative paths below are within the research repository unless prefixed `work/`.

## Table 2 controls

| Suite / run identity | First sampled training accuracy <90% | Minimum sampled test accuracy | Final test accuracy | Maximum sampled auxiliary gradient norm | Samples and horizon |
|---|---:|---:|---:|---:|---|
| Group branches: `branch_control_from_44000` | 44,700 | 16.7692139745% | 100% | 913.072381521 | 201 samples, every 10 updates, 44,000 to 46,000 |
| Component branches: `aux_control_from_44000` | 44,710 | 43.5731053352% | 99.9328792095% | 329.285370934 | 201 samples, every 10 updates, 44,000 to 46,000 |

The two minima, final accuracies and maximum gradient norms reproduce the published values. Each control's minimum occurs at its first sampled training failure. These are distinct trajectories. At step 44,000 their recorded train loss, test loss and accuracies match exactly: 1.8678194635413092e-7, 3.3246786188101396e-5, 1.0 and 1.0. The same four values match the original `runs/muon_lr_0p01_seed_0_full.csv` row at step 44,000. At step 44,010 the two controls already have different losses, gradient norms and displacements.

The source path and tensors of the historical step-44,000 checkpoint were not retained in the inspected available tree. Matching scalar values support the reported common source but do not establish bitwise model/optimizer equivalence. The exact original execution backend, torch version, launch command and numerical settings likewise lack a retained runtime manifest. The source device resolver supports CUDA and MPS and prefers CUDA in auto mode; this is a source-code property, not independent certification of the actual backend.

Both branch implementations reconstruct the same single-block model and seeded 30% split, load the model state and load both Muon and auxiliary AdamW state dictionaries. The earlier branch design uses one auxiliary AdamW group for embeddings and readout; it differs from the selected-configuration follow-up's separate head optimizer. The historical run name identifies hidden LR 0.01; exact checkpoint-defined hyperparameters cannot be independently recovered from the available checkpoint record. The code reseeds Python and torch with the checkpoint seed, including CUDA when available, rather than restoring saved RNG. The branch checkpoint writers save model and optimizer states and the seed, but omit RNG and backend records. The full-batch model has no dropout or random operations in its forward pass.

The component summary script defines first collapse using the first `collapse_detected==1` row and takes column minima, maxima and the final row. `collapse_detected` is training accuracy below 90%, with `has_memorized=True` from the source. Training and test metrics are inspected only at the branch's evaluation grid. These are training-only failure summaries rather than the later joint-failure endpoint.

Evidence locations: `analysis/interventions/branch_collapse.py:133` (device), `:159` (seeding), `:458` (optimizer restore), `:583` (model restore), `:750` (loop), `:783` (criterion), `:867` (checkpoint payload); `analysis/interventions/branch_auxiliary_components.py:133`, `:161`, `:533`, `:673`, `:787`, `:820`, `:889` respectively. Summary calculation: `analysis/plots/plot_auxiliary_component_branches.py:43` and `:177`. The uninstrumented group implementation is retained from Git commit `ANONYMIZED_SOURCE_REVISION` in `historical_branch_collapse.py`.

Suggested concise caption/prose:

> The group and component suites have separate control continuations. Their logged losses and accuracies agree at step 44,000 and first differ at 44,010. Each suite is summarized on its own ten-update evaluation grid through step 46,000. The retained logs verify the reported outcomes; the original checkpoint and runtime manifests are unavailable for a bitwise cross-suite reconstruction.

A controlled CPU implementation replay is documented separately in `parity_protocol.json` and `parity_verification.json`. Its source is an explicitly adapted saved seed-0 step-16,000 state, so it evaluates implementation parity without claiming to reconstruct either historical control. All saved moments and step counters are retained; the head is merged into the common auxiliary group with its learning rate explicitly set to 0.001. Each program completed 2,000 updates, with the original ten-update metric schedule. All 201 shared metric rows agree exactly. All three terminal model and optimizer states are bitwise equal, and their terminal torch RNG states agree. This controlled test finds no implementation-induced difference on its CPU execution; historical cross-suite divergence remains unresolved.

## Distinct observations around 44,700

| Observation | Run identity | Sampling | Associated analysis |
|---|---|---|---|
| 44,700 train/test failure, 18.5379% / 16.7692% | `runs/branch_control_from_44000.csv` | Every 10 updates, through 46,000 | Table 2 group control |
| 44,702 train/test failure, 24.3081% / 22.1166% | `runs/branch_control_from_44000_instrumented.csv` | Every update, through 46,000 | Quiet-window applied-step and decay diagnostics; last train-healthy sample 44,701 |
| 44,710 train/test failure, 47.7024% / 43.5731% | `runs/aux_control_from_44000.csv` | Every 10 updates, through 46,000 | Table 2 component control |
| 44,710 train/test failure, 21.1227% / 19.0402% | `runs/collapse_spectral_replay_from_44000.csv` | Every 10 updates, through 45,000 | Spectral figure and quiet-window scalar table; last sampled healthy state 44,700 |

The spectral reference is the 44,700 state of the spectral replay. Its 44,710 accuracy drop and its spectrum are never paired with the 44,700 failure of the group-control run. The repeated timestamp 44,710 identifies two distinct recorded failures. Ten-update observations identify first sampled failures, with possible unobserved excursions between measurements.

## Operational definitions

- **Original grokking:** the first of five consecutive scheduled test evaluations at or above 95%. With a 100-update grid, those observations span 400 updates. The source implementation returns the first row of the qualifying window (`scripts/training/train.py:775`).
- **Follow-up confirmation:** the sixth consecutive scheduled test evaluation at or above 95%. With a 100-update grid, the qualifying observations span 500 updates, and the reported confirmation step is the last row.
- **Strict stability:** every subsequent sampled test accuracy remains at least 95%. State the post-grokking window and terminal horizon. It establishes sampled stability.
- **Joint failure:** contemporaneous training and test accuracy both fall below 90% after the required grokking confirmation. Training every update plus test on every training-below-90 trigger exhaustively detects this endpoint.
- **Table 2 failure:** training accuracy below 90% on its ten-update grid after memorization. Its reported failure criterion does not require the test threshold.
- **First detected failure:** earliest observation satisfying the run's monitored endpoint. State the actual schedule and eligibility window.
- **First captured failure:** earliest qualifying saved state obtained under the capture protocol. An exploratory dense replay can capture an earlier transition than a prior coarse log detected; the replay must retain its distinct run identity.
- **Minimum sampled accuracy:** minimum over the specified observed evaluations. It need not equal the trajectory's unsampled minimum.
- **Final accuracy:** accuracy explicitly evaluated at the run's stated terminal step.
- **Evaluation count:** number of logged evaluations meeting a threshold, including triggered off-grid evaluations when specified. Counts concern dependent observations and depend on the monitoring schedule.
- **Seed-level incidence:** number of seeds with at least one qualifying event divided by the number of evaluated seeds. Multiple events or samples within a seed count once.

The first captured joint failures in the fresh five-seed stock cohort are 17,494, 17,721, 18,069, 16,308 and 16,056 for seeds 0 through 4. Seeds 0 and 4 use exploratory dense replays from steps 16,000 and 15,000. Seed 0's prior coarse-grid detection was 17,500; seed 4's brief early joint failures were missed by the 100-step grid. Seeds 1 through 3 were specified together prospectively and inspect training every update after the six-evaluation confirmation. Selected decoder snapshots can differ: the existing peak seed-4 probe uses 16,060, while its first captured event is 16,056. The selected seed-3 peak probe uses 16,309 rather than its first captured 16,308 event. Preserve exact snapshot identities in decoder tables and figure captions.

## Speed and learning-rate scope

`speed_audit.json` reproduces:

| Quantity | AdamW | Muon | Ratio |
|---|---:|---:|---:|
| Selected-configuration median, five seeds | 8,300 | 5,400 | 1.537037 = 1.54x |
| Fastest successful single-seed sweep entry | 6,300 | 5,400 | 1.166667 = 1.17x |
| Success-conditional sweep mean | 30,485.7143 (7/11 successes) | 13,011.1111 (9/9 successes) | 2.343052 = 2.34x |

The five-seed AdamW grokking steps are 8,800, 8,300, 4,000, 11,700 and 6,400. Muon gives 5,400, 5,300, 5,400, 5,400 and 5,400. Each is the start of a five-evaluation streak.

Configuration selection is asymmetric. The selected AdamW baseline is the fastest strictly stable entry in its seed-0 sweep, LR 0.001 and weight decay 3. Muon is selected for speed among nine entries that all show at least one subsequent sampled test accuracy below 95%, with selected hidden LR 0.03 and decay 0.1. The sweep grids differ: AdamW uses eleven LR/decay pairs with LR in {0.0003,0.001,0.003,0.01} and decay in {0.1,1,3}, omitting the (0.003,3) pair; Muon uses all nine pairs from LR {0.003,0.01,0.03} and hidden decay {0.03,0.1,0.3}, with fixed interface settings. Four unsuccessful AdamW configurations are excluded from the mean. The horizon is 100,000 updates. The fastest AdamW entry is unstable and distinct from the selected stability-screened reference.

Suggested section heading: “Earlier grokking in the tested settings.”

The completed learning-rate continuation result remains 5/5 original stock, 5/5 hidden-LR/10 and 0/5 accurate-CE seed-level joint failures. The ordinary reduction changes hidden LR from 0.03 to 0.003 at step 6,000 while retaining all weight-decay coefficients and allowing embeddings/readout to train. Therefore the per-update hidden decoupled decay factor also changes from 0.997 to 0.9997. This establishes the insufficiency of that particular reduction through 100,000. Compare binary incidence to historical original runs; event timings, counts and durations have different monitoring and event-selection provenance.

## Matched depth-4 freeze

`depth4_audit.json` verifies the shared state at step 126,200, a 300,000 endpoint and 173,800 subsequent updates on the recorded MPS backend. Each branch has 1,739 rows on a 100-update grid including the common initial row. The summary excludes the shared row and uses 1,738 post-branch evaluations. Mean test accuracy is 48.0696% in control versus 97.5019% frozen; final accuracy is 36.8833% versus 98.8701%. Counts below 90% are 1,353 versus 66. The metadata records zero maximum model-parameter difference at the branch and independent storage. This finite-horizon result includes residual failures in the frozen branch.

## Chance notation

`chance_notation_audit.json` searches all current canonical manuscript TeX. Correct chance accuracy is 1/113, approximately 0.885%; 100/113 is the numerical percentage before appending a percent sign, not the accuracy fraction. The audited current source contains no incorrect 100/113 notation.

## Feature-mean timeline

`feature_mean_timeline.csv` derives the actual final-residual mean norm at 49 Muon and 39 selected AdamW archived checkpoints. Its definition is the Euclidean norm of the feature vector averaged uniformly over the full 113x113 operand grid. The archived orthonormal Fourier summaries retain total spectral power and the DC fraction, so the norm is `sqrt(total_spectral_power * dc_power_fraction_total) / 113`. This follows from `FFT_ortho(h)[0,0] = 113 * mean(h)`; no filtering enters this measurement.

`feature_mean_timeline.pdf` shows every observed checkpoint with a point, uses a shared logarithmic norm axis, and separates the two selected configurations. The lower event rasters show the start-of-five grokking steps and all recorded joint failures in their associated 100-step historical logs. These are the original archived seed-study trajectories, distinct from the fresh densely captured trajectories underlying the 5/5 failure comparison and adjacent-update analysis. The archived Muon logs contain four joint-failure evaluations across three seeds: seed 1 at 89,500; seed 2 at 89,100; seed 4 at 15,500 and 85,900. The archived AdamW logs contain three: seed 1 at 99,700; seed 2 at 22,100; seed 3 at 99,200. A missing cross provides no evidence that a trajectory remained event-free between its scheduled observations. The plot reports the archived seed study's CPU training designation and CPU Fourier evaluation. Original runtime/software manifests remain unavailable. Sparse selected checkpoints and joining lines are descriptive; intermediate means and unseen failures are not inferred. Fresh deterministic CPU follow-up trajectories have separate identifiers and are not merged into these curves.

Suggested caption:

> Readout-input feature means in five archived Muon and five selected AdamW trajectories. Circles show every available sampled checkpoint (49 and 39, respectively); linear segments connect sampled checkpoints. Norms are measured over the full operand grid, recovered from the archived orthonormal-DFT constant-mode power. Lower rasters mark original grokking and joint training/test failures observed on the 100-update grid (four Muon observations across three seeds and three AdamW observations across three seeds). These original archived trajectories are distinct from the fresh five-seed densely captured cohort; absence of a cross does not establish event-free training. Configuration selection and backend provenance are as described in the text. The larger recorded Muon feature means are descriptive evidence about these trajectories.

## Exact audit commands

```sh
python work/control_provenance/audit_records.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python work/control_provenance/replay_control_implementations.py
MPLCONFIGDIR=work/control_provenance/mplcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python work/control_provenance/build_feature_mean_timeline.py
```

The replay has a non-overwriting output directory and will refuse to replace existing checkpoint folders. All historical input files remain unchanged. The report and machine-readable audit records are new artifacts only.

The independent feature-mean identity check gives a relative discrepancy of 5.49e-16 on a saved seed-0 step-6,000 feature matrix. All 88 plotted source rows independently recompute correctly; the alternative total-minus-nonconstant-power calculation differs by at most 1.43e-7 relative, consistent with stored float32 summaries.
