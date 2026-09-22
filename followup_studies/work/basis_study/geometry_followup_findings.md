# Adjacent steps, matched healthy drift, and event timecourses

The acute failure is clearer when its immediately preceding step serves as the reference. Both preceding checkpoints already have test errors, so they are labeled preceding steps throughout the pair metadata.

## Adjacent transitions

| Pair | Preceding test accuracy | Later test accuracy | Backward centered reconstruction error | Basis-predicted accuracy | Residual-only accuracy | Readout-change-only accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Seed 4, 16055 to 16056 | 91.34% | 66.33% | 25.63% | 67.43% | 89.20% | 66.75% |
| Seed 0, 17493 to 17494 | 95.49% | 31.87% | 1.877% | 31.65% | 95.50% | 31.86% |

Here the forward map A fits H0 A to H1 on original training examples. Basis-predicted logits are H0 A W1; residual-only logits are H0 W0 + (H1 - H0 A) W1; readout-change-only logits are H0 W1. These are endpoint counterfactuals. Their implications concern the sufficiency of each changed component at these endpoints.

For seed 0's final update, applying only the new readout to the preceding features closely reproduces the observed accuracy loss. The centered geometry residual is much smaller than the 43.63% measured across the longer 17200-to-17494 interval. The fitted backward map condition number is 5.37 for this adjacent step, compared with 2,720.61 across the longer interval. Thus the original long interval bundled substantial earlier representation evolution with the acute failure.

The seed 4 adjacent update changes feature geometry more substantially. Its residual contribution alone also lowers the already imperfect preceding test accuracy from 91.34% to 89.20%, while the readout-only counterfactual reaches 66.75%. The acute readout change still accounts for most of the accuracy loss under this endpoint decomposition.

## Matched seed 0 control

The healthy control 16906 to 17200 spans the same 294 updates as the original seed 0 event comparison. Both endpoints pass the predefined >=99% test criterion: 99.91% and 99.12%. Its backward centered error is 32.66%, compared with 43.63% in the 17200-to-17494 event pair. Its raw all-example oracle residual is 10.59%, compared with 7.60% in the event pair. Raw errors across these intervals use differently sized target norms, so their ordering is not a causal measure.

Substantial failure of an exact shared linear map occurs during successful training. A large long-interval reconstruction residual is consequently insufficient evidence for the cause of collapse.

## Fixed-reference timecourses

The seed 4 timecourse contains 23 checkpoints from 15860 through 16060, aligned to fixed reference 16050. At 16055, accuracy has already fallen to 91.34%, while the backward centered residual is only 1.189%. The large geometry jump occurs at 16056, where the residual reaches 25.60%. At the peak endpoint 16060 it reaches 33.67%. The fixed-reference old head, orthogonal transport, and backward OLS all retain 100% test accuracy at 16055.

The seed 0 timecourse contains 46 checkpoints from the original coarse and dense event windows, plus the fixed 17200 reference and adjacent 17493 checkpoint. Its final failure occurs with almost unchanged long-reference geometry error:

| Step | Test accuracy | Centered reconstruction error relative to 17200 |
|---|---:|---:|
| 17490 | 96.49% | 43.41% |
| 17493 | 95.49% | 43.72% |
| 17494 | 31.87% | 43.63% |

The old 17200 readout already has only 13.98% accuracy when applied to 17490 features, despite the actual model retaining 96.49%. That long-reference mismatch reflects successful prior coadaptation. The immediately preceding readout is a more informative reference for the acute failure.

## Files and verification

- `adjacent_seed4/` and `adjacent_seed0/`: features, checkpoint metadata, full geometry and functional decomposition, maps, and identity/planted/permutation controls.
- `seed0_matched_healthy_16906_17200/`: the matched control with the same outputs.
- `timecourse_seed4/trajectory.csv` and `timecourse_seed0/trajectory.csv`: native train/test accuracy and loss, forward/backward map conditioning, backward centered reconstruction error, orthogonal/backward/old-head repair accuracy, and exact functional decomposition summaries.
- Each timecourse also has `functional_decompositions.json` with full per-step train and held-out metrics, flip counts, norm ratios, and the necessary cross term.
- `geometry_followups.py` reproduces the short checkpoint replays, pair extraction, geometry analyses, and timecourses. It uses one CPU thread and no test-selected tuning parameters.

The seed 0 preceding checkpoint was regenerated with three original float32 updates from 17490. A fourth update reproduced every model-state tensor in the stored 17494 collapse checkpoint bitwise. The control's 16906 checkpoint required six original updates from 16900. No long training runs were introduced for these comparisons.
