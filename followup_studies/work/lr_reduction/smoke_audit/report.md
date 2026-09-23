Smoke verification only. These short runs do not establish scientific endpoint outcomes.

| Seed | Full-grid mean norm, start | Final | Sampled maximum | Stock CE relative error, start | Final | Sampled maximum |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 310.837 | 310.629 | 310.837 | 0.0143953 | 0.013835 | 0.0143953 |
| 1 | 459.464 | 459.408 | 459.464 | 0.0150046 | 0.0145783 | 0.0150046 |
| 2 | 355.11 | 354.315 | 355.11 | 0.0096736 | 0.00923378 | 0.0096736 |
| 3 | 411.861 | 412.081 | 412.081 | 0.0222844 | 0.0221865 | 0.0222844 |
| 4 | 434.116 | 434.967 | 434.967 | 0.0174846 | 0.0174337 | 0.0175924 |

CE relative error is the stock float32 loss derivative error relative to the accurate float64 derivative of the same stored logits. Undefined denotes a zero reference norm. The full-grid feature mean covers all operand pairs. Training-set mean norms are retained in `summary.csv` as `initial_train_feature_mean_norm`, `final_train_feature_mean_norm`, and `maximum_sampled_train_feature_mean_norm`.
