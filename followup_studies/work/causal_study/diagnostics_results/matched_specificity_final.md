# Matched step-20000 specificity states

Every checkpoint records step 20000 and has an input SHA256 hash. The measured training backward rule is identified explicitly. Its error and the error produced by hypothetical stock CE at the same frozen model are kept separate. The independently recorded dense trajectories determine intervention success through the full interval.

| Seed | Backward rule | Held-out accuracy | Feature mean norm | Head mean norm | Mean cosine | Global mean power |
|---|---|---:|---:|---:|---:|---:|
| 0 | accurate32 | 100.0000% | 516.36 | 0.006736 | -0.2254 | 25.21% |
| 0 | target_repair | 100.0000% | 521.76 | 0.006737 | -0.2234 | 25.55% |
| 0 | row_projection | 100.0000% | 234.15 | 0.006451 | -0.0307 | 8.93% |
| 1 | accurate32 | 100.0000% | 725.60 | 0.007781 | -0.1528 | 36.61% |
| 1 | target_repair | 100.0000% | 722.78 | 0.007777 | -0.1532 | 36.44% |
| 1 | row_projection | 100.0000% | 386.30 | 0.007411 | -0.0169 | 20.09% |
| 2 | accurate32 | 100.0000% | 614.68 | 0.008112 | -0.1508 | 36.54% |
| 2 | target_repair | 100.0000% | 614.26 | 0.008110 | -0.1507 | 36.49% |
| 2 | row_projection | 100.0000% | 295.17 | 0.007883 | -0.1222 | 15.17% |
| 3 | accurate32 | 100.0000% | 415.77 | 0.007140 | -0.2329 | 16.75% |
| 3 | target_repair | 100.0000% | 419.51 | 0.007129 | -0.2250 | 16.93% |
| 3 | row_projection | 100.0000% | 125.37 | 0.006633 | 0.0607 | 3.06% |
| 4 | accurate32 | 100.0000% | 692.23 | 0.005484 | -0.0965 | 41.90% |
| 4 | target_repair | 100.0000% | 686.61 | 0.005488 | -0.0986 | 41.65% |
| 4 | row_projection | 100.0000% | 216.23 | 0.005274 | 0.1320 | 10.98% |

| Seed | Backward rule | Actual logit-gradient error | Hypothetical stock error | Actual readout-gradient error | Actual zero-sum residual |
|---|---|---:|---:|---:|---:|
| 0 | accurate32 | 2.553e-07 | 0.8162 | 1.911e-07 | 4.476e-08 |
| 0 | target_repair | 2.874e-07 | 0.8002 | 2.115e-07 | 2.532e-08 |
| 0 | row_projection | 0.6648 | 0.6677 | 0.67 | 1.849e-08 |
| 1 | accurate32 | 2.606e-07 | 0.713 | 2.023e-07 | 4.575e-08 |
| 1 | target_repair | 3.118e-07 | 0.7021 | 2.231e-07 | 2.57e-08 |
| 1 | row_projection | 0.582 | 0.5846 | 0.6132 | 1.947e-08 |
| 2 | accurate32 | 2.314e-07 | 0.638 | 1.948e-07 | 4.356e-08 |
| 2 | target_repair | 2.926e-07 | 0.6363 | 2.604e-07 | 2.466e-08 |
| 2 | row_projection | 0.5294 | 0.5317 | 0.5308 | 2.088e-08 |
| 3 | accurate32 | 2.518e-07 | 0.954 | 1.702e-07 | 4.502e-08 |
| 3 | target_repair | 2.52e-07 | 0.9536 | 1.742e-07 | 2.501e-08 |
| 3 | row_projection | 0.8441 | 0.8479 | 0.809 | 1.515e-08 |
| 4 | accurate32 | 2.744e-07 | 0.9458 | 1.872e-07 | 4.481e-08 |
| 4 | target_repair | 2.78e-07 | 0.9463 | 1.885e-07 | 2.398e-08 |
| 4 | row_projection | 0.8607 | 0.8646 | 0.8277 | 1.408e-08 |

Errors are relative L2 norms against the same-logit accurate derivative. The zero-sum residual is the norm of per-example sums across output classes divided by the reference logit-gradient norm. Row projection restores that invariant while retaining a different derivative from accurate CE; targeted repair reconstructs the correct-class derivative while preserving stock wrong-class entries.

These state comparisons describe downstream effects and do not identify a unique mediator of all earlier updates. Raw JSON retains every parameter-group comparison, feature statistic, and input hash.
