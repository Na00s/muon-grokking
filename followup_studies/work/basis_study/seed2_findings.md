# Fresh seed 2: near-chance collapse in one update

Seed 2 was declared before training alongside fresh seeds 1 and 3. Its first captured event occurs at step 18069 and is also the minimum test accuracy within the following 200 updates. The latest measured joint >=99% checkpoint is step 18000, a 69-update interval. Its immediately preceding checkpoint is 18068 and retains 99.9888% test accuracy.

| Adjacent counterfactual | Test accuracy |
|---|---:|
| Preceding features and preceding head | 99.9888% |
| Later features and preceding head | 100% |
| Preceding features and later head | 1.0068% |
| Actual later model | 0.9285% |

This is a particularly clear acute readout failure. The actual feature update paired with the preceding head reaches 100% accuracy, while the actual readout update on preceding features reaches approximately the final failed accuracy.

The readout, feature, and interaction component norms are 0.980, 0.0446, and 0.00251 times the total class-centered logit-change norm. The full reports retain the three cross terms. The identity predictor already matches 99.9217% of actual later predicted classes. A map fitted on original training examples increases exact-class agreement to 100% and predicts the same 0.9285% test accuracy. Its class-centered logit error is 0.7528%, compared with 4.1777% for the identity predictor.

Adjacent backward linear reconstruction has 1.103% centered held-out error and 0.08524% raw error. The fitted map condition number is 3.13. The all-example oracle raw residual is 0.08316%; median and maximum principal angles are 0.448 and 1.933 degrees. The transported preceding readout restores 99.9888% accuracy. These results show that a catastrophic accuracy loss can accompany comparatively small representation geometry change.

Across the longer 18000-to-18069 interval, centered reconstruction error is 9.052%, condition number is 90.71, and backward transport restores 100% accuracy. The long-reference preceding head on later features has 78.07% accuracy. This again shows why immediate and longer references answer different questions about coadaptation.

Full geometry, fitted maps, and planted native rounding controls are in `seed2_first_event/` and `seed2_adjacent_event/`. Separate exact bilinear outputs are under `bilinear/`. No duplicate peak pair is counted.
