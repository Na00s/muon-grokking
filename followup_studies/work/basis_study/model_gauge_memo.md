# Full-network residual gauge controls

An exact residual change of basis can cause the same qualitative phenomenon as the reported collapse. I transformed the healthy seed 4 network through a known invertible residual map while leaving its old readout in place. Accuracy fell from 100% to 0.60-0.92%. Exact compensation of the readout and compensation inferred from original training residuals both restored 100% in every positive control.

The actual checkpoints depart substantially from a single exact residual gauge. Final-token maps that recover prediction accuracy can fail badly at the input and post-attention residual sites. These results support useful final-interface alignment while limiting the stronger account that the entire model simply changed residual coordinates.

## Exact symmetry of this architecture

Use row activations H and write Hprime = H A. A is any invertible 128 by 128 matrix. In PyTorch's output-by-input Linear weight convention, transform:

- Token and position embedding tables: Eprime = E A.
- Attention QKV, MLP input, and final readout weights: Wprime = W inverse(A).transpose.
- Attention output and MLP output weights: Wprime = A.transpose W.

The Q, K, and V activations remain unchanged because H A multiplied by the transformed input-facing weights equals the original product. The same holds for MLP preactivations, so ReLU preserves the calculation. Residual-output projections map each contribution by A, and residual addition commutes with the transformation. These identities apply to every block and every token position. The readout compensation preserves logits.

`gauge_model` implements this transformation on a separate model copy. A test on a two-block network checks every residual site and all logits in float64; additional tests check float32 behavior, equality of the site-capture path with the ordinary forward method, immutability of the source model, and the deliberately uncompensated readout. All five tests pass.

## Positive controls using the trained network

The source is seed 4 at step 16050. For each condition number 1, 3, 10, and 100, use fixed random orthogonal factors and geometrically spaced singular values from condition^-0.5 to condition^0.5. Execute the transformed network on all 12,769 inputs, in float32 and float64. Float64 execution uses the stored float32 checkpoint weights cast to double; it is an arithmetic control rather than a separately trained model.

| Condition number | Accuracy with old readout | Accuracy with exact compensation | Accuracy with train-fit compensation | Float32 centered final-residual error under known A | Float64 error |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.92% | 100% | 100% | 9.10e-6 | 1.70e-14 |
| 3 | 0.60% | 100% | 100% | 9.34e-6 | 1.76e-14 |
| 10 | 0.84% | 100% | 100% | 1.15e-5 | 2.46e-14 |
| 100 | 0.84% | 100% | 100% | 3.47e-5 | 5.67e-14 |

Readout compensation inferred from the final-token map and from the joint-site map both achieves 100%. The final-token fit recovers the planted matrix to relative error 1.30e-4 to 4.99e-4 in float32, while the joint-site fit gives 1.40e-6 to 5.71e-6. In float64 these ranges are approximately 2.1e-13 to 7.4e-13 and 3.1e-14 to 3.3e-14. The joint-site observations constrain weak directions more effectively.

The positive controls show that the pipeline detects and corrects known basis-induced failures in the actual nonlinear network. They also supply a numerical floor for exact gauges through the full forward computation.

## One common map across residual sites

For the observed checkpoint pairs, extract the input residual, post-attention residual, and post-MLP residual at all three token positions. Fit either:

1. A forward ordinary-least-squares map using only the final equals-token residual from original training examples.
2. One forward map using all residual sites and token positions from original training examples. Normalize each site by its healthy training RMS, applying the same scalar to both checkpoints, so sites contribute comparably while an exact gauge remains exact.

Fits are float64 and use no labels. Evaluate on the original held-out examples. Readout compensation is inverse(A) times the healthy readout. Centered errors remove each representation's own training mean. Raw errors, mean mismatch, and errors divided by the centered target norm are also retained in the JSON reports.

| Comparison | Final-token map error at input site | Final-token map error at attention site | Final-token map error at final equals token | Readout rescue |
|---|---:|---:|---:|---:|
| Seed 4, 16050 to 16056 | 3.834 | 6.864 | 0.251 | 99.99% |
| Seed 4, 16050 to 16060 | 6.178 | 10.125 | 0.329 | 99.98% |
| Seed 0, 17200 to 17494 | 7.026 | 6.084 | 0.414 | 96.23% |
| Seed 0, nearby 17490 to 17494 | 0.348 | 0.347 | 0.0191 | 96.45% |
| Seed 4, healthy 16044 to 16050 | 0.160 | 0.318 | 0.0118 | 100% |
| Seed 4, healthy 6000 to 7000 | 6.568 | 7.706 | 0.440 | 99.99% |

The input and attention entries pool all token positions; the final entry uses the equals token, matching the fitted interface.

| Comparison | Joint-map error at input site | Joint-map error at attention site | Joint-map error at final equals token | Readout rescue |
|---|---:|---:|---:|---:|
| Seed 4, 16050 to 16056 | 0.0360 | 0.0542 | 0.361 | 100% |
| Seed 4, 16050 to 16060 | 0.0986 | 0.1398 | 0.517 | 50.53% |
| Seed 0, 17200 to 17494 | 0.1181 | 0.1816 | 0.674 | 94.61% |
| Seed 0, nearby 17490 to 17494 | 0.00382 | 0.00679 | 0.0278 | 96.44% |
| Seed 4, healthy 16044 to 16050 | 0.00310 | 0.00420 | 0.0241 | 100% |
| Seed 4, healthy 6000 to 7000 | 0.1059 | 0.2258 | 0.532 | 100% |

The nearby seed 0 reference already has 96.49% test accuracy, whereas the earlier seed 0 reference has 99.12%. Its four-step interval has modest representation drift, comparable in size to the healthy seed 4 six-step control, alongside a substantial drop to 31.87%. This cautions against treating the larger 294-step comparison as a measurement of the acute transition alone.

The healthy 1000-step interval also lacks a shared exact gauge. Thus failure of the exact model-wide restriction is present during ordinary training; it is not a unique signature of collapse. On the matched six-step seed 4 intervals, acute onset has approximately 21 times the final-token centered error of the healthy control.

## Floors matched to the actual inferred matrices

Some inferred final-token maps have condition numbers above 100. To control for both conditioning and orientation, I planted each inferred matrix into its own healthy network using the exact gauge construction, then executed that network in float32. This adds 12 matched network controls, one for each map and comparison.

For final-token fits, the measured actual errors and same-matrix exact-gauge floors are:

| Comparison | Actual centered final error | Exact-gauge floor for the same A | Ratio |
|---|---:|---:|---:|
| Seed 4 onset | 0.251 | 5.94e-5 | 4,228 |
| Seed 4 peak | 0.329 | 6.80e-5 | 4,838 |
| Seed 0 earlier reference | 0.414 | 8.89e-5 | 4,661 |
| Seed 0 nearby reference | 0.0191 | 3.56e-6 | 5,364 |
| Healthy seed 4 six-step interval | 0.0118 | 3.96e-6 | 2,989 |
| Healthy seed 4 1000-step interval | 0.440 | 6.78e-5 | 6,490 |

All matched controls retain their source model's accuracy after exact compensation. The discrepancy between actual and planted residuals is far larger than the floating-point floor of the corresponding full-network gauge. Healthy training also changes representations beyond that floor, which remains essential to interpreting this comparison.

## Parameter consistency and scope

The reports additionally compare actual non-readout parameters with parameters predicted by applying each fitted gauge to the healthy network. The readout is excluded because an uncompensated readout is allowed to cause collapse. For the joint map, aggregate parameter mismatch is 0.224 at seed 4 onset and 2.410 at its peak, compared with 0.0215 for the healthy six-step interval. Observed aggregate parameter motion is 0.00690 at onset and 0.01511 at peak.

This parameter check describes the explicit gauge construction used here. Attention-head symmetries and MLP permutations introduce further parameter nonidentifiability, so parameter mismatch alone cannot reject equivalent residual computations. The cross-site representation measurements provide the more direct test of one shared residual map.

A final-interface hypothesis can allow internal representations and weights to change in more complicated ways. It therefore does not require the cross-site or parameter identities of a model-wide gauge. The evidence supports effective linear compensation at the final interface, with checkpoint-dependent limits. The stronger claim that the observed model simply follows a pure residual-coordinate transformation is unsupported by these measurements.

## Reproduction

- `model_gauge.py`: exact parameter transformation, all-site extraction, planted controls, fitted maps, and matched arithmetic floors.
- `test_model_gauge.py`: five architecture and algebra tests.
- `model_gauge_results/planted/`: eight fixed-condition positive controls and recovered maps.
- `model_gauge_results/actual/`: six checkpoint comparisons, twelve fitted maps, and twelve same-matrix controls.
- `model_gauge_run.log` and `model_gauge_actual_run.log`: execution summaries.

All site values are re-extracted from saved checkpoints. Original input order is regenerated from the seed and checked against the saved dataset hash. Additional pairs can be analyzed with `--skip-planted --pair /path/to/features-directory`; each directory must contain its checkpoint metadata.
