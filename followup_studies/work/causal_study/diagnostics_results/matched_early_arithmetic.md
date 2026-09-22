# Matched early-arithmetic state comparison

All five seeds compare the original trajectory with an accurate-cross-entropy continuation from the same saved step-6000 model, optimizer, and RNG state. Checkpoint arithmetic is the experimental intervention; the following state properties are descriptive outcomes.

At step 15000, the stock trajectories have feature-mean norms 2.22–3.83 times those of the accurate trajectories. Their global-mean feature power is 60.52–75.38%, compared with 17.75–24.64% under accurate training. Classifier/feature means are also more antiparallel under stock training. At step 20000, all five sampled accurate states retain 100% held-out accuracy and feature-mean norms of 401–754.

| Seed | Step | Arithmetic since 6000 | Feature mean norm | Head mean norm | Mean cosine | Global mean feature power | Held-out accuracy |
|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 15000 | Stock | 1,146.40 | 0.006605 | -0.5729 | 64.93% | 100.0000% |
| 0 | 15000 | Accurate | 444.02 | 0.004410 | -0.2988 | 21.69% | 99.9776% |
| 0 | 20000 | Accurate | 546.72 | 0.006062 | -0.1830 | 28.12% | 100.0000% |
| 1 | 15000 | Stock | 1,128.73 | 0.007164 | -0.5288 | 63.07% | 100.0000% |
| 1 | 15000 | Accurate | 413.27 | 0.005052 | -0.2880 | 18.30% | 100.0000% |
| 1 | 20000 | Accurate | 753.69 | 0.006928 | -0.1334 | 39.02% | 100.0000% |
| 2 | 15000 | Stock | 1,092.78 | 0.005200 | -0.6756 | 60.52% | 100.0000% |
| 2 | 15000 | Accurate | 484.83 | 0.003982 | -0.3073 | 22.57% | 100.0000% |
| 2 | 20000 | Accurate | 621.41 | 0.007849 | -0.1292 | 37.06% | 100.0000% |
| 3 | 15000 | Stock | 1,486.31 | 0.010922 | -0.8257 | 75.38% | 100.0000% |
| 3 | 15000 | Accurate | 387.74 | 0.004042 | -0.3182 | 17.75% | 100.0000% |
| 3 | 20000 | Accurate | 401.10 | 0.004812 | -0.2631 | 15.83% | 100.0000% |
| 4 | 15000 | Stock | 1,086.88 | 0.007491 | -0.7393 | 62.63% | 100.0000% |
| 4 | 15000 | Accurate | 490.66 | 0.004201 | -0.2953 | 24.64% | 100.0000% |
| 4 | 20000 | Accurate | 690.47 | 0.004698 | -0.0982 | 41.97% | 100.0000% |

At the common step 15000:

| Seed | Stock / accurate-trained feature mean norm | Stock / accurate-trained head mean norm | Stock CE logit error on stock / accurate-trained states | Stock CE readout error on stock / accurate-trained states |
|---|---:|---:|---:|---:|
| 0 | 2.582 | 1.498 | 14.95% / 13.99% | 26.49% / 15.60% |
| 1 | 2.731 | 1.418 | 14.18% / 13.51% | 26.21% / 15.30% |
| 2 | 2.254 | 1.306 | 10.56% / 9.12% | 18.95% / 9.54% |
| 3 | 3.833 | 2.702 | 21.86% / 19.23% | 43.42% / 23.46% |
| 4 | 2.215 | 1.783 | 20.86% / 18.81% | 38.64% / 23.22% |

Gradient errors in this table evaluate stock CE on each frozen model state, including states trained with accurate CE. They measure that state’s sensitivity to the stock loss derivative; they are not the derivative errors used during accurate training. The accurate-loss derivative errors are retained separately in the CSV and raw JSON.

At step 20000, evaluating stock CE diagnostically on the accurately trained states would zero 84.07–100% of target derivatives and produce 63.64–95.40% relative logit-gradient error. The accurate derivative actually used in training stays within 2.8e-7 relative error of the same-logit reference. Small or zero stock-reported losses alone therefore do not establish which derivatives drove the trajectory.

The fifteen checkpoint SHA256 values and all full diagnostics are in `matched_early_arithmetic.json`. This comparison describes state changes caused by the arithmetic intervention and does not identify a unique mediator of the complete training history. These sampled accuracies do not replace the dense trajectory results. Original-stock step-20000 comparisons and the complete step-30000 controls are reported separately.
