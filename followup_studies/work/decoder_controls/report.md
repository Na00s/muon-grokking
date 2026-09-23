# Initialization and memorization decoder controls

All ten new controls are complete. The existing ten reference/failure results are retained unchanged. Each early-state replay matches the original step-1,000 model, optimizer and RNG exactly. Validation indices match the existing decoders for every seed.

The memorization checkpoint is the first of five consecutive scheduled 100-step evaluations with native training accuracy at least 99.9%, with test accuracy below 95% and before original grokking. The selected steps are 200, 400, 200, 100 and 200 for seeds 0–4. These thresholds and identities were registered before fitting. Accuracies below are percentages.

| Seed | State | Step | Native train | Native test | Decoder test | CV refit converged | CV iterations | Unregularized test | Unregularized converged |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| 0 | initialization | 0 | 0.757 | 0.839 | 0.123 | True | 10 | 0.447 | False |
| 0 | memorization | 200 | 99.948 | 0.078 | 0.022 | True | 36 | 0.034 | True |
| 0 | healthy_reference | 17200 | 100.000 | 99.116 | 99.508 | True | 19 | 99.508 | True |
| 0 | failed | 17494 | 48.251 | 31.872 | 98.725 | True | 18 | 98.725 | True |
| 1 | initialization | 0 | 0.809 | 0.917 | 0.123 | True | 10 | 0.268 | False |
| 1 | memorization | 400 | 99.974 | 0.224 | 0.246 | True | 33 | 0.324 | True |
| 1 | healthy_reference | 17300 | 100.000 | 99.195 | 99.564 | True | 19 | 99.564 | True |
| 1 | failed | 17721 | 15.666 | 9.095 | 98.199 | True | 17 | 98.199 | True |
| 2 | initialization | 0 | 0.731 | 0.895 | 0.067 | True | 10 | 0.336 | False |
| 2 | memorization | 200 | 99.948 | 0.492 | 0.548 | True | 34 | 0.626 | True |
| 2 | healthy_reference | 18000 | 100.000 | 100.000 | 100.000 | True | 19 | 100.000 | True |
| 2 | failed | 18069 | 0.992 | 0.929 | 100.000 | True | 18 | 100.000 | True |
| 3 | initialization | 0 | 0.862 | 0.906 | 0.235 | True | 11 | 0.615 | False |
| 3 | memorization | 100 | 100.000 | 0.089 | 0.067 | True | 37 | 0.067 | True |
| 3 | healthy_reference | 15800 | 100.000 | 99.071 | 99.396 | True | 19 | 99.396 | True |
| 3 | failed | 16309 | 42.637 | 32.118 | 98.769 | True | 18 | 98.769 | True |
| 4 | initialization | 0 | 1.044 | 0.828 | 0.045 | True | 10 | 0.179 | False |
| 4 | memorization | 200 | 99.948 | 1.018 | 1.119 | True | 33 | 1.119 | True |
| 4 | healthy_reference | 16050 | 100.000 | 100.000 | 100.000 | True | 19 | 100.000 | True |
| 4 | failed | 16060 | 21.775 | 18.134 | 100.000 | True | 17 | 100.000 | True |

Candidate and refit diagnostics are retained in the per-state JSON reports, including unsuccessful bounded fits. The initialization and memorization controls measure early linear accessibility under this decoder protocol. Strong decoding at a failed checkpoint establishes surviving linearly accessible task information; it does not alone identify when that information first became accessible, a unique internal algorithm, or exact preservation of the healthy representation.
