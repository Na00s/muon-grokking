# Fresh seed 1: severe collapse with little adjacent feature change

Seed 1 was specified before training with fresh seeds 2 and 3. Its first captured collapse is step 17721 and is also the lowest test accuracy within the following 200 updates; no duplicate peak feature pair is counted. The latest measured joint >=99% reference is step 17300, spanning 421 updates. Its immediate preceding reference is 17720, whose test accuracy is already 94.14%.

| Adjacent counterfactual | Test accuracy |
|---|---:|
| Preceding features and preceding head | 94.14% |
| Later features and preceding head | 94.18% |
| Preceding features and later head | 13.56% |
| Actual later model | 9.09% |

The observed readout update alone produces most of the acute loss. The feature update has little effect when paired with the preceding head. The interaction matters for the remaining accuracy loss, so readout-only and full updates should not be described as identical.

The readout, feature, and interaction component norms are 1.012, 0.0839, and 0.00985 times total class-centered logit change. Cross terms are retained. The identity predictor H0 W1 agrees with 95.50% of actual later predictions. Fitting the train-only feature map increases agreement to 99.15% and predicts 9.04% accuracy. Its class-centered logit error is 3.852%, compared with the identity predictor's 7.537%.

Adjacent backward linear reconstruction has 4.632% centered held-out error and 0.671% raw error. The map condition number is 19.56. A transported preceding readout reaches 94.33%. The all-example oracle raw residual is 0.657%; median and maximum principal angles are 2.36 and 10.47 degrees. These are small compared with longer checkpoint drift, yet substantially above the explicit planted-transform numerical floors.

Across the longer 17300-to-17721 interval, centered reconstruction error is 43.05%, map condition number is 3,911, and the backward transported readout reaches 93.04%. This comparison includes prior coadaptation. The long-reference linear-map counterfactual predicts 8.64% accuracy, while the residual-only counterfactual reaches 92.84%.

Full geometry and native rounding controls are in `seed1_first_event/` and `seed1_adjacent_event/`. Exact bilinear reports are in the corresponding `bilinear/` subdirectories. Existing feature arrays, metadata, and original alignment outputs were preserved.
