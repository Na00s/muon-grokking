# Matched step-20000 stock states

Every checkpoint records step 20000 and has an input SHA256 hash. The measured training backward rule is identified explicitly. Its error and the error produced by hypothetical stock CE at the same frozen model are kept separate. The independently recorded dense trajectories determine intervention success through the full interval.

| Seed | Backward rule | Held-out accuracy | Feature mean norm | Head mean norm | Mean cosine | Global mean power |
|---|---|---:|---:|---:|---:|---:|
| 0 | stock32 | 99.9888% | 635.96 | 0.033501 | -0.8473 | 66.38% |
| 1 | stock32 | 99.9553% | 649.66 | 0.036533 | -0.8500 | 67.07% |
| 2 | stock32 | 99.9105% | 923.12 | 0.043074 | -0.5061 | 58.00% |
| 3 | stock32 | 100.0000% | 620.61 | 0.026888 | -0.7772 | 60.11% |
| 4 | stock32 | 99.9329% | 800.21 | 0.017463 | -0.3716 | 54.92% |

| Seed | Backward rule | Actual logit-gradient error | Hypothetical stock error | Actual readout-gradient error | Actual zero-sum residual |
|---|---|---:|---:|---:|---:|
| 0 | stock32 | 0.001872 | 0.001872 | 0.002601 | 0.001872 |
| 1 | stock32 | 0.001352 | 0.001352 | 0.001652 | 0.001352 |
| 2 | stock32 | 0.001019 | 0.001019 | 0.001358 | 0.001019 |
| 3 | stock32 | 0.00285 | 0.00285 | 0.003899 | 0.002851 |
| 4 | stock32 | 0.001152 | 0.001152 | 0.002142 | 0.001152 |

Errors are relative L2 norms against the same-logit accurate derivative. The zero-sum residual is the norm of per-example sums across output classes divided by the reference logit-gradient norm. Row projection restores that invariant while retaining a different derivative from accurate CE; targeted repair reconstructs the correct-class derivative while preserving stock wrong-class entries.

These state comparisons describe downstream effects and do not identify a unique mediator of all earlier updates. Raw JSON retains every parameter-group comparison, feature statistic, and input hash.
