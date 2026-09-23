The reduction was insufficient to prevent the prespecified joint failure: 5/5 seeds failed by step 100,000.

Each continuation restored the original step-6,000 model, all optimizer buffers, and RNG state. Hidden Muon learning rate changed from 0.03 to 0.003; embedding and readout AdamW learning rates stayed at 0.001 and 0.00025. All parameters remained trainable, with stock cross-entropy and unchanged weight-decay coefficients.

| Seed | Joint failure | First failure step | Minimum sampled test accuracy | Evaluations below 95% | Scheduled evaluations below 95% | Final test accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Yes | 18767 | 0.8614% | 105 | 13 | 100.0000% |
| 1 | Yes | 18476 | 4.7209% | 88 | 20 | 100.0000% |
| 2 | Yes | 19439 | 0.8950% | 135 | 14 | 99.6644% |
| 3 | Yes | 17041 | 4.8999% | 52 | 22 | 99.8881% |
| 4 | Yes | 17578 | 0.8166% | 103 | 13 | 100.0000% |

Binary comparison: original stock runs 5/5 seeds with a joint failure; corrected-loss continuations 0/5; hidden-LR reduction 5/5. Historical original monitoring differs, so event counts and durations are not compared.

Training accuracy was checked at all 94,001 states per seed, including the source and final states. Test accuracy was evaluated every 100 updates and whenever training accuracy fell below 90%. Secondary evaluation counts describe this new experiment only. Feature-mean norms and loss-derivative error were recorded at 95 states per seed, every 1,000 updates including both endpoints.

| Seed | Full-grid mean norm, start | Final | Sampled maximum | Stock CE relative error, start | Final | Sampled maximum |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 310.837 | 292.583 | 4282.48 | 0.0143953 | 0.000302195 | 0.38558 |
| 1 | 459.464 | 505.286 | 8783.76 | 0.0150046 | 0.00179877 | 0.462735 |
| 2 | 355.11 | 1096.47 | 8100.36 | 0.0096736 | 0.228367 | 0.365848 |
| 3 | 411.861 | 955.815 | 8878.48 | 0.0222844 | 0.054083 | 0.484311 |
| 4 | 434.116 | 445.098 | 3596 | 0.0174846 | 0.000268405 | 0.636452 |

CE relative error is the stock float32 loss derivative error relative to the accurate float64 derivative of the same stored logits. Undefined denotes a zero reference norm. The full-grid feature mean covers all operand pairs. Training-set mean norms are retained in `summary.csv` as `initial_train_feature_mean_norm`, `final_train_feature_mean_norm`, and `maximum_sampled_train_feature_mean_norm`.

The intervention also reduces hidden decoupled weight decay per update. It tests stabilization under this particular tenfold reduction and cannot by itself establish the upstream failure mechanism. The conclusions are limited to these seeds, this architecture, and the fixed horizon. Test-only excursions between scheduled evaluations remain unobserved.

Verification passed 682 explicit checks and 80 checkpoint metric replays. Model, optimizer, and RNG states at branch start match their source except for the intended hidden learning rate. All saved optimizer learning rates and weight-decay coefficients remained fixed. The scientific manuscript remained unchanged. The author explicitly authorized replacement of the AI use statement while these runs were in progress; the separate amendment ledger records the four permitted artifact paths and preserves the original protocol hashes.
