# Completed geometry and endpoint-attribution suite

The suite contains 20 geometric pair comparisons and 21 exact bilinear comparisons. These include five distinct training seeds, repeated endpoints from some seeds, healthy temporal controls, and an identity control. The pair count is not an independent-sample count. Seeds 1, 2, and 3 were specified before their fresh training runs; seeds 0 and 4 came from the preceding replication work. All experiments here use the same one-layer, no-normalization, modulus-113 addition condition.

## The adjacent update consistently localizes the acute failure

Percent test accuracy on the original held-out examples:

| Seed | Preceding model | New features, old head | Old features, new head | Actual later model |
|---|---:|---:|---:|---:|
| 0 | 95.49 | 96.39 | 31.86 | 31.87 |
| 1 | 94.14 | 94.18 | 13.56 | 9.09 |
| 2 | 99.99 | 100.00 | 1.01 | 0.93 |
| 3 | 89.54 | 91.01 | 50.80 | 53.60 |
| 4 | 91.34 | 98.66 | 66.75 | 66.33 |

Across all five observed adjacent transitions, the actual feature update paired with the preceding readout retains or improves accuracy. The actual readout update paired with preceding features reproduces a severe loss. This is the clearest supported local component-level conclusion. It does not identify the earlier optimization dynamics that created the harmful readout update.

In the exact identity ΔZ = H0 ΔW + ΔH W0 + ΔH ΔW, the readout term's class-centered norm is 0.980 to 1.045 times the total change norm across these five adjacent transitions. The feature term is 0.0446 to 0.0839 times total, and the interaction is 0.00251 to 0.0552 times total. Every report retains all three cross terms. These norm ratios are not causal percentages or additive explained-variance shares.

## Exact global basis change and approximate prediction are distinct

| Seed | Adjacent backward centered error | Longer first-event centered error | Matched healthy centered error |
|---|---:|---:|---:|
| 0 | 1.877% | 43.63% over 294 updates | 32.66% over 294 updates |
| 1 | 4.632% | 43.05% over 421 updates | 10.82% over 421 updates |
| 2 | 1.103% | 9.052% over 69 updates | 2.761% over 69 updates |
| 3 | 10.66% | 45.73% over 508 updates | 14.30% over 508 updates |
| 4 | 25.63% | 25.60% over 6 updates | 1.176% over 6 updates |

All predictive maps fit only the original training examples. The original held-out inputs remain unused during fitting. All-example oracle projections are reported separately as transductive geometric diagnostics, never held-out prediction evidence.

Every adjacent event has a nonzero all-example minimum reconstruction residual above the measured native-rounding planted-map controls. Thus the observed full activation matrices do not follow an exact common invertible change of basis at the tested precision. Native float32 planted maps with conditions 1, 10, and 100 verify that the numerical procedure can detect such maps when present. These controls are empirical rounding benchmarks, not universal bounds for every arbitrarily ill-conditioned transformation.

Healthy training also produces appreciable departure from an exact common map. The matched healthy controls select an earlier, eligible interval with equal duration and >=99% endpoint test accuracy; their selection is recorded in metadata. They do not match every stage of optimization or every model norm. Geometry residual size therefore provides a descriptive comparison, not an isolated causal treatment.

A fitted forward map improves prediction of the later logits, including wrong predictions after collapse. However, the linear-map predictor H0 A W1 contains the actual later readout. The simpler identity predictor H0 W1 already captures most acute failure. Predictive improvement after fitting A supports useful approximation of representation changes; it does not show that basis drift caused the collapse. An exact joint transformation H1=H0 A, W1=A^-1 W0 would preserve the logits identically.

## Checks and reproducibility

- Ten geometry tests and five bilinear tests passed. These include exact planted maps, native float32 rounding, affine translations, nonlinear obstructions, rank deficiency, dominant constant components, held-out invariance, permutation controls, exact component identities, and cancellation under joint basis transformations.
- The seed 0 adjacent checkpoint replay reproduces the stored collapse model tensors bitwise after the next update.
- `geometry_summary.csv/json` includes all 20 geometry pairs. `bilinear/summary.csv/json` includes all 21 bilinear pairs; the extra bilinear pair is the earlier four-update seed 0 near-reference comparison.
- `geometry.py --features PATH --out DIRECTORY --controls` analyzes an arbitrary compatible feature NPZ. `bilinear_diagnostic.py --features PATH1 PATH2 ... --out DIRECTORY` supports a batch of pairs. Source directory basenames must be unique within a bilinear output directory.
- Scripts contain no hardcoded machine-specific absolute paths. They locate the shared `experiments` helpers relative to the sibling directory. Preserve sibling `basis_study/` and `experiments/` directories when copying the reproduction code.
- Existing feature files, metadata, and earlier alignment results were preserved. Each geometry run adds `geometry.json` and `geometry_maps.npz`; bilinear results are separate.
