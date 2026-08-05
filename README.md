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
optimizers/muon.py          Newton–Schulz orthogonalized momentum
scripts/training/           single-run entry points
scripts/sweeps/             hyperparameter and generality sweeps
experiments/depth/          depth variants and layerwise Fourier analysis
analysis/                   Fourier decomposition, interventions, spectral replay
analysis/interventions/     matched-branch experiments
analysis/plots/             figure generation
runs/                       all run and analysis outputs
```

`runs/` holds every artifact the paper reports: 136 CSVs and 16 HTML summaries. Model
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

## Empty modules

`config.py`, `metrics.py`, `analysis/fourier.py`, `analysis/restricted_loss.py`,
`analysis/update_spectra.py`, and `optimizers/muon_no_ns.py` are placeholders. The last three
correspond to planned experiments: Nanda's restricted loss across a collapse, direct
measurement of Muon's update singular values, and a Newton–Schulz ablation.
