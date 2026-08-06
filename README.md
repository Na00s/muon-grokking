# Post-grokking collapse at the representation–readout interface

Code and run artifacts for a study of what happens to a grokked circuit after it forms,
under split-optimizer routing that gives Muon the hidden weight matrices and leaves
embeddings and the output head with AdamW.

The task is modular arithmetic on a decoder-only transformer with no normalization layers.
Runs cover two operations, two moduli, two widths, two training fractions, and depths 1, 2,
and 4.

---

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

`runs/` holds every artifact the paper reports: 160 CSVs and 18 HTML summaries. Model
checkpoints are gitignored.

## Environment

PyTorch with an MPS, CUDA, or CPU backend. Fourier transforms run on CPU regardless, since
prime-length transforms are not deterministically supported on every accelerator backend.

Trajectories are not bitwise reproducible across processes on MPS. Two runs of the depth-4
Muon configuration reach sustained generalization at 52,600 and 101,300 steps. Every matched
comparison in the paper is therefore constructed by branching within a single process, with
zero parameter difference at the branch point.

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

## Configurations

The selected configurations, used wherever the paper reports a comparison:

| Group | Optimizer | Settings |
|---|---|---|
| Hidden matrices | Muon | lr 0.03, wd 0.1, momentum 0.95, Newton–Schulz steps 5 |
| Token and position embeddings | AdamW | lr 1e-3, wd 1.0 |
| Unembedding | AdamW | lr 2.5e-4, wd 1.0 |
| Baseline, all parameters | AdamW | lr 1e-3, wd 3.0 |

Stable Muon is the Muon configuration with the embeddings and unembedding held constant
once the circuit has formed. It is the same run as Muon before the freeze.

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

## Reproducing the family result

The central mechanistic claim is that the model computes the task through one family of
two-dimensional Fourier modes over the operand grid, and that the task selects the family.
Across eleven solved addition checkpoints, projection onto the `(k,k)` family gives 100%
accuracy and ablating it leaves 1.34%, while every other family in isolation sits at the
chance rate of 0.88%. On `(a-b) mod 113` the two families exchange roles exactly.

```bash
python experiments/depth/depth_fourier_hypothesis_tests.py --help
```

Family interventions loop over all non-DC families and are operation-general. The frequency,
phase, and cross-readout controls are written against `(k,k)` and report on addition models
only.

## Mechanism experiments

Three experiments testing what the split-optimizer account actually attributes to what.

**Applied updates.** The logged `muon_applied_update_norm` is the orthogonalized step before
weight decay, not the tensor subtracted from the parameters. `metrics.py` decomposes the
difference across a step into its gradient-driven and decay components, which for Muon, the
ablation, and decoupled AdamW alike is exact, since all three apply
`parameter <- parameter * (1 - lr * wd) - lr * update`.

In the matched branch the hidden gradient and decay components are 0.1499 and 0.1475, so the
net applied update is 0.0462, smaller by a factor of 3.2 than the quoted figure, and it grows
64% across the quiet window rather than holding constant. The competing explanation, that the
per-parameter separation is the learning-rate ratio rather than the update rule, is refuted:
Muon's elasticity of log update on log gradient is -0.026 against AdamW's 1.51 and 1.47. Adam
is scale-invariant under a constant rescaling of the gradient, not under gradient magnitude
drifting over time, because its two moment horizons differ by two orders of magnitude.

**Newton–Schulz ablation.** `optimizers/muon_no_ns.py` is identical to `muon.py` except that
the update is the momentum buffer rather than its approximate zeroth power. It separates two
phenomena the account treats as one.

Orthogonalization causes the distributed code. The effective non-DC pair count at solved
checkpoints is 326.09 with it, 4.11 without, and 4.95 for AdamW. The family itself is
unaffected: `(k,k)` sufficiency is 100.00% at every solved ablation checkpoint.

Orthogonalization does not cause the instability, it contains a worse one. Across a
decay-matched learning-rate sweep spanning 0.03 to 10, every run that learned went on to
diverge to a non-finite loss, six of six, including in double precision on CPU. Muon degrades
gradually and recovers instead. Death steps span 22,889 to 38,955 and one run died at a flat
hidden norm, so the timing is chaotic and no norm threshold governs it.

**Rescaling the task-aligned component.** `alpha_scaling.py` splits the final residual into
the task-aligned family and the remainder and rescales the first. Masked-branch accuracy rises
monotonically and crosses 95% at a family share of 0.6247, below the frozen branch's native
0.9069, reaching 0.9991 with no retraining and no change to the readout.

The margin decomposition through the unembedding, exact because that readout is bias-free and
linear, shows what power share alone cannot. In both branches the task-aligned component
yields a positive margin on 100% of examples, including every example the masked model gets
wrong, and the remainder is adversarial rather than neutral in both. The frozen branch wins on
amplitude, +19.28 against -7.67; the masked branch loses it, +5.72 against -6.43. Masking is a
competition the task-aligned component loses on amplitude, not a loss of task information.

## Empty modules

`config.py`, `analysis/fourier.py`, `analysis/restricted_loss.py`, and
`analysis/update_spectra.py` are placeholders. The last two correspond to planned
experiments: Nanda's restricted loss across a collapse, and direct measurement of Muon's
update singular values.

`metrics.py` and `optimizers/muon_no_ns.py` were placeholders and are now implemented. The
Newton–Schulz ablation they were reserved for is reported above.
