# Tests of a shared linear basis transformation

These endpoint tests distinguish two claims: exact representation-wide basis change, and approximate linear/readout drift sufficient to reproduce the accuracy loss. The existing events support the second claim substantially more strongly than the first. They comprise two seeds, with several correlated comparisons from seed 4.

## Exact representation-wide hypothesis

Rows index input examples. An exact basis-only explanation requires a single invertible matrix A such that H1 = H0 A. Equal column spaces are a necessary condition. We compute an unrestricted least-squares optimum using stable float64 SVD, forward and backward, and compare predictions on untouched original test inputs. We also compute all-example column-space projection residuals. These latter quantities are explicitly transductive geometric lower bounds, never held-out prediction evidence.

The errors below use the backward map from later residuals to healthy residuals. Centered errors remove the respective original-training means. This matters because a large constant component makes raw reconstruction appear much closer than the variation among examples.

| Pair | Later accuracy | Held-out raw error | Held-out centered error | All-example oracle raw error | Median / maximum principal angle |
|---|---:|---:|---:|---:|---:|
| Seed 4, six-update collapse onset | 66.33% | 1.713% | 25.60% | 1.677% | 11.27 / 41.73 degrees |
| Seed 4, collapse peak | 18.13% | 2.253% | 33.67% | 2.211% | 17.56 / 50.11 degrees |
| Seed 0, collapse | 31.87% | 7.779% | 43.63% | 7.597% | 24.35 / 59.05 degrees |
| Seed 4, matched six healthy updates | 100% | 0.0820% | 1.176% | 0.0801% | 0.47 / 2.22 degrees |
| Seed 4, 1,000 healthy updates | 100% | 19.17% | 41.43% | 18.56% | 20.94 / 59.37 degrees |

The short matched comparison shows substantially larger non-linear-map residual around collapse. The long healthy comparison shows that substantial non-basis drift also happens during successful training. Geometric mismatch alone does not establish a cause of failure.

Native float32 planted transformations with condition number 100 produce only 0.000292% raw backward reconstruction error for the seed-4 collapse starting features, versus 1.713% at onset. The corresponding centered control error is 0.00437%, versus 25.60%. Identity controls have errors around 1e-15. These are measured computational floors, not universal error bounds for every possible ill-conditioned transformation.

Source ranks remain 128 under every tolerance through 128 times float32 epsilon for the event pairs. Under larger truncation tolerances, reconstruction errors grow modestly while the main functional repair survives. Allowing an affine intercept changes the onset centered error from 25.60% to 25.52%, the peak from 33.67% to 33.29%, and seed 0 from 43.63% to 43.39%. A missing translation explains little of these errors.

## Functional decomposition

For the original-train-only forward OLS fit A, write E = H1 - H0 A. The logit identity is

H1 W1 - H0 W0 = H0(A W1 - W0) + E W1.

The first term includes both the fitted representation map and the observed readout change. It cannot by itself distinguish a representation rotation from harmful readout adaptation. The second is the contribution of the feature residual beyond this fitted map. Their squared norms have a cross term, so treating them as additive explained-variance percentages would be incorrect.

| Pair | Actual accuracy | Basis-predicted H0 A W1 | Residual-only H0 W0 + E W1 | Readout-change-only H0 W1 | Backward-map repaired accuracy |
|---|---:|---:|---:|---:|---:|
| Seed 4, onset | 66.33% | 67.55% | 99.99% | 68.25% | 99.99% |
| Seed 4, peak | 18.13% | 18.69% | 99.32% | 12.54% | 99.98% |
| Seed 0 | 31.87% | 33.11% | 95.51% | 8.68% | 95.40% |

At the seed-4 peak, the basis component has 99.94% of the total class-centered logit-change norm, and the residual has 4.03%. Squared norm ratios sum with a cross term of -0.000490 to one. The basis-predicted model captures 98.41% of the actual flips away from healthy predictions, while its exact predicted-class agreement with the actual collapsed model is 69.55%. Similar accuracy does not mean identical predictions.

Seed 0 has a larger residual contribution: 28.84% of the total change norm. The basis-predicted counterfactual still reproduces most of the accuracy loss. Its actual-flip recall is 92.33%, and exact class agreement is 74.43%.

These findings make an approximate linear/readout mismatch a useful functional account of these endpoint failures. A precise global invertible symmetry, its optimizer dynamics, and causal responsibility for the whole training trajectory require additional evidence. The observed readout-alone failure also warrants particular attention.

## Reproducibility and checks

`geometry.py` accepts an existing NPZ with H0, H1, W0, W1, y, train, heldout and emits `geometry.json` plus `geometry_maps.npz`. `analyze_pair` exposes the same functionality as an import. Maps use original training inputs only; none use test labels or select hyperparameters on test data. `summarize_geometry.py` regenerates the compact JSON and CSV summaries.

Ten deterministic mathematical tests pass. They cover planted invertible maps, held-out invariance of all fitted maps, affine translation, non-linear obstruction, oracle agreement with explicit least squares, dominant constant components, rank deficiency, float32 planted controls, a negative correspondence control, and the exact functional identity including its cross term.

Commands use `/opt/homebrew/Caskroom/miniforge/base/envs/research/bin/python` with `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Example:

```sh
python work/basis_study/geometry.py --features work/experiments/seed4_dense_event/features.npz --out work/basis_study/geometry_seed4_dense_event --controls
python -m unittest discover -s work/basis_study -p test_geometry.py -v
```
