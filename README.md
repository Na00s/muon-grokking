# Post-grokking collapse at the representation–readout interface


## Revised submission

The revised manuscript centers on acute readout failures and surviving information. It adds Fourier memorizer controls and continues all five corrected addition branches and the subtraction control through the original 100,000-step horizon, with no joint train/test failure. The scientific main text is nine pages, with the exempt reproducibility and required AI use statements separate. All six original figure PDFs and numerical cells in the 17 original appendix tables are preserved; the quiet-window and Fourier-grid figures appear in Appendices C and G. The mean-dependent update is now in main-text Table 3; Appendix K and Figure 9 establish the task structure supplied by Fourier filtering.

- [Revised paper](paper/Muon_Grokking_Revised.pdf)
- [Complete LaTeX source ZIP](paper/ICLR_Submission_Muon_Grokking_Revised_Source.zip)
- [Fourier controls and 100,000-step results](paper/revision_addendum.md)
- [New-experiment audit and claim review](paper/audit_report.md)
- [Reproducible audit ledger and scripts](paper/audit/)
- [Section-by-section revision notes](paper/revision_notes.md)
- [Exact changes to original source files](paper/targeted_revision.patch)
- [Compilation and preservation checks](paper/verification.json)
- [Figure values and input hashes](paper/source/figures/followup_figure_provenance.json)

Code and run artifacts for a study of what happens after a model groks,
under split-optimizer routing that gives Muon the hidden weight matrices and leaves
embeddings and the output head with AdamW.

The task is modular arithmetic on a decoder-only transformer with no normalization layers.
Runs cover two operations, two moduli, two widths, two training fractions, depths 1, 2, and
4, and five seeds.

---

## Follow-up experiments, September 22, 2026

The [Fourier controls](followup_studies/work/fourier_control/report.md) show that complete-family filtering is answer-orbit averaging: a lookup memorizer reaches 100% filtered accuracy on every seed. A training-count-normalized lookup also succeeds with any single diagonal frequency pair. The [long-horizon study](followup_studies/work/long_horizon/README.md) adds 420,000 updates and verifies all six continuations through step 100,000. Reproduction data are available in the [Fourier and horizon release](https://github.com/Na00s/muon-grokking/releases/tag/fourier-horizon-study-2026-09-22).


The [completed causal study](followup_studies/causal_study/report.md) adds 364,302 optimizer updates, five-seed arithmetic controls through update 30,000, targeted derivative repairs, and operation/architecture controls. All five original addition trajectories collapse; 0 of five accurate-arithmetic continuations collapse within the matched horizon. The study directly tests how numerical error, feature-mean inflation, and a damaging readout update interact. The normalized controls also expose an embedding-update collapse under accurate loss arithmetic. The [manuscript revisions](followup_studies/causal_study/paper_revision.md) replace the unsupported pure-basis and no-numerical-pathology claims.

The [earlier basis study](followup_studies/README.md) includes training-only decoder recovery, exact and approximate basis tests, and continuation interventions. Every retained binary is available across the [basis-study release](https://github.com/Na00s/muon-grokking/releases/tag/basis-study-2026-09-22) and [causal-study release](https://github.com/Na00s/muon-grokking/releases/tag/causal-study-2026-09-22).

## Layout

```
data.py                     operand grid, train/test split, operation selection
model.py                    transformer, no normalization, bias-free projections
metrics.py                  applied-update decomposition, gradient versus decay
optimizers/muon.py          Newton–Schulz orthogonalized momentum
optimizers/muon_no_ns.py    the same optimizer with the orthogonalization removed
scripts/training/           single-run entry points
scripts/sweeps/             hyperparameter and generality sweeps
experiments/depth/          depth variants and layerwise Fourier analysis
analysis/                   Fourier decomposition, interventions, spectral replay
analysis/interventions/     matched-branch experiments
analysis/plots/             figure generation
runs/                       all run and analysis outputs
```

`runs/` holds every artifact the paper reports: 191 CSVs and 20 HTML summaries. Model
checkpoints are gitignored, so the analysis pipeline must be re-run against locally
regenerated checkpoints rather than against a fresh clone.

## Environment

PyTorch with an MPS, CUDA, or CPU backend. Fourier transforms run on CPU regardless, since
prime-length transforms are not deterministically supported on every accelerator backend.

Trajectories are not bitwise reproducible across processes on MPS, and two runs of the same
configuration can reach sustained generalization at substantially different steps. CPU
execution is bitwise deterministic.

Anything that compares trajectories should therefore run on CPU with `--device cpu`, where
two runs sharing a seed are identical until they are deliberately made to differ, so a
matched pair needs no in-process fork and its pairing can be verified by diffing the logs.
On MPS, seed variance and backend nondeterminism are confounded. The depth-4 matched
comparisons predate this and are instead constructed by branching within a single process,
with zero parameter difference at the branch point.

`--dtype float64` is available on CPU as a numerical control. MPS does not support it.

## Running

Single run at depth 1:

```bash
python scripts/training/train.py --optimizer muon --operation addition
```

Depth variants, which carry the locked depth-1 selections as defaults and write the
checkpoint schema the Fourier pipeline reads:

```bash
python experiments/depth/train_depth_variant.py --num-layers 1 --operation subtraction
```

Sweeps:

```bash
python scripts/sweeps/run_adamw_sweep.py
python scripts/sweeps/run_muon_sweep.py
python scripts/sweeps/run_generality_suite.py
```

One arm of a seed replication. Each seed needs its own depth-one initial state, and the seed
sets both the initialization and the train/test split:

```bash
python experiments/depth/train_depth_variant.py \
  --regime stable_muon --num-layers 1 --seed 3 --device cpu \
  --run-name seedstudy_stable_muon_seed_3 \
  --initial-state-path checkpoints/initial_states/depth_nested_v2_1_seed_3.pt \
  --depth-one-initial-state-path checkpoints/initial_states/model_seed_3_sweep_compatible.pt
```

Muon and stable Muon at the same seed on CPU are the same trajectory until the freeze fires,
so running both gives a matched pair without an in-process fork. Threading scales poorly
here; many single-threaded jobs under `OMP_NUM_THREADS=1` outperform a few wide ones.

## Configurations

The selected configurations, used wherever the paper reports a comparison:

| Group | Optimizer | Settings |
|---|---|---|
| Hidden matrices | Muon | lr 0.03, wd 0.1, momentum 0.95, Newton–Schulz steps 5 |
| Token and position embeddings | AdamW | lr 1e-3, wd 1.0 |
| Unembedding | AdamW | lr 2.5e-4, wd 1.0 |
| Baseline, all parameters | AdamW | lr 1e-3, wd 3.0 |

Stable Muon is the Muon configuration with the embeddings and unembedding held constant
after sustained generalization. It is the same run as Muon before the freeze.

`train_depth_variant.py` selects between these with `--regime`: `adamw`, `muon`,
`stable_muon`, and `muon_no_ns`. The Muon and auxiliary defaults are the locked depth-one
selections above, so a depth variant needs no hyperparameter arguments to reproduce them.
The unstable AdamW comparison is `--regime adamw --adamw-lr 1e-2 --adamw-weight-decay 1.0`.

## Definitions

Evaluations occur every 100 steps, so sustained criteria require a threshold to hold across
a 500-step window.

| Term | Definition |
|---|---|
| Sustained 95% test | First evaluation followed by five evaluations at or above 95% test accuracy |
| Strictly stable | No post-grokking evaluation below 95% test accuracy |
| Collapse | Training accuracy below 90% after the model has memorized |

## Which artifacts back which claims

| Claim | Artifacts |
|---|---|
| Speed and instability sweeps | `adamw_sweep_*`, `muon_sweep_*`, `*_summary.csv` |
| Generality across modulus, width, training fraction | `generality_*` |
| Matched branch localization | `branch_control_*`, `branch_freeze_*`, `auxiliary_component_branches_*` |
| Long-run prevention | `muon_freeze_all_auxiliary_*`, `stable_muon_*`, `generality_*_stable_muon_*` |
| Collapse spectral replay | `collapse_spectral_replay_*`, `collapse_spectral_summary.csv` |
| Fourier circuit and interventions | `depth_fourier_family_interventions_*`, `depth_fourier_frequency_controls_*`, `depth_fourier_phase_controls_*`, `depth_fourier_cross_readout_*` |
| Two collapse modes | `depth4_matched_freeze_*`, `depth4_matched_fourier_*` |
| Layerwise circuit construction | `depth_fourier_layerwise_causal_v2_*`, `depth_fourier_mode_layer_summary_*` |
| Depth | `depth_sweep_v2_*`, `depth_sweep_v3_*` |
| Modular subtraction | `subtraction_*`, `addition_depth1_*` |
| Applied-update decomposition | `branch_control_from_44000_instrumented.csv` |
| Newton–Schulz ablation | `no_ns_depth_1_lr_*`, `no_ns_fourier_*` |
| Rescaling the task-aligned component | `alpha_scaling_curve_*`, `alpha_scaling_margin_*` |
| Seed replication, five seeds, four conditions | `seedstudy_*` |

## Fourier circuit analysis

Two stages. `depth_fourier_mode_identification.py` partitions the two-dimensional Fourier
modes over the operand grid into families — addition `(k,k)`, subtraction `(k,-k)`, a-only,
b-only, constant, and generic interaction — and writes a model summary. `depth_fourier_hypothesis_tests.py`
consumes that summary and runs sufficiency and ablation per family, plus the frequency
relocation, phase, and cross-readout controls.

```bash
python experiments/depth/depth_fourier_mode_identification.py \
  --manifest runs/<manifest>.csv --operation addition

python experiments/depth/depth_fourier_hypothesis_tests.py \
  --model-summary runs/<summary>.csv --operation addition
```

A manifest is one row per run: `label,depth,regime,csv_path,checkpoint_directories`. Depths 1,
2, and 4 are accepted, and checkpoints must carry `model_config` with `num_layers`, which the
depth-variant trainer writes and `scripts/training/train.py` does not.

`--operation` selects the labels accuracy is scored against and does not affect the family
partition. Family interventions loop over all non-DC families and are operation-general. The
frequency, phase, and cross-readout controls are written against `(k,k)` specifically and are
meaningful for addition-trained models only.

## Mechanism experiments

Three instruments isolating what the split-optimizer account attributes to what.

**Applied updates.** `muon_applied_update_norm` records the orthogonalized step before weight
decay, which is not the tensor subtracted from the parameters. `metrics.py` measures the
difference across an optimizer step and splits it into a gradient-driven and a decay
component. The split is exact for Muon, the Newton–Schulz ablation, and decoupled AdamW
alike, since all three apply `parameter <- parameter * (1 - lr * wd) - lr * update`. The
snapshot is taken in double precision, because the applied update is a difference of two
nearly equal numbers. `analysis/interventions/branch_collapse.py` and
`experiments/depth/train_depth_variant.py` log it per parameter group.

**Newton–Schulz ablation.** `optimizers/muon_no_ns.py` is identical to `muon.py` except that
the update is the momentum buffer rather than its approximate zeroth power, so the step
carries the gradient's magnitude instead of a magnitude set by the matrix shape. The learning
rate is therefore not transferable from Muon, and a run is only interpretable against a
learning-rate sweep or an explicit calibration. Because Muon couples decay to the learning
rate through `1 - lr * wd`, a sweep should hold `lr * wd` fixed so it varies step size rather
than decay. Select with `--regime muon_no_ns`.

**Rescaling the task-aligned component.** `experiments/depth/alpha_scaling.py` splits the
final residual into the task-aligned family and the remainder, rescales the first across a
sweep of factors, and records full-model accuracy and the resulting family power share at
each. It also decomposes the correct-class margin through the unembedding, which is exact
because that readout is a bias-free linear map.

## Empty modules

`config.py`, `analysis/fourier.py`, `analysis/restricted_loss.py`, and
`analysis/update_spectra.py` are placeholders. The last two correspond to planned
experiments: Nanda's restricted loss across a collapse, and direct measurement of Muon's
update singular values.

`metrics.py` and `optimizers/muon_no_ns.py` were placeholders and are now implemented; both
are described under Mechanism experiments.
