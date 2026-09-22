# Two late original-stock recurrences

Each captured update reproduces all model parameters, optimizer buffers, and RNG state exactly. These events are reported separately from the original five-seed counts. The mask order is hidden, embeddings, readout; zero uses preceding parameters and one uses following parameters.

| Seed | Event step | Hybrid | Training accuracy | Held-out accuracy |
|---|---:|---|---:|---:|
| 1 | 29549 | 000 | 99.9478% | 87.4930% |
| 1 | 29549 | 001 | 75.1958% | 66.3161% |
| 1 | 29549 | 010 | 99.9739% | 94.9883% |
| 1 | 29549 | 011 | 65.8747% | 58.4070% |
| 1 | 29549 | 100 | 100.0000% | 99.2952% |
| 1 | 29549 | 101 | 76.2663% | 67.2223% |
| 1 | 29549 | 110 | 100.0000% | 99.2617% |
| 1 | 29549 | 111 | 65.4830% | 58.0266% |
| 3 | 27748 | 000 | 100.0000% | 99.9553% |
| 3 | 27748 | 001 | 1.6449% | 1.8235% |
| 3 | 27748 | 010 | 100.0000% | 99.9553% |
| 3 | 27748 | 011 | 1.6449% | 1.8235% |
| 3 | 27748 | 100 | 100.0000% | 99.9776% |
| 3 | 27748 | 101 | 1.6449% | 1.8235% |
| 3 | 27748 | 110 | 100.0000% | 99.9776% |
| 3 | 27748 | 111 | 1.6449% | 1.8235% |

| Seed | Stock logit-gradient error | Accurate logit-gradient error | Accurate complete-update derivative |
|---|---:|---:|---:|
| 1 | 2.0541e-06 | 6.3697e-08 | -2.374411 |
| 3 | 0.00092361 | 5.4391e-08 | -0.001469651 |

| Seed | Following features / prior head | Mean term alone | Centered term alone | Complete head displacement |
|---|---:|---:|---:|---:|
| 1 | 99.2617% | 57.9819% | 99.1945% | 58.0266% |
| 3 | 99.9776% | 1.8235% | 99.9776% | 1.8235% |

The feature mean is computed using training examples only and then applied unchanged to held-out examples. This logit decomposition is evaluation-only; it does not specify a training intervention. Parameter hybrid effects identify the captured step and do not determine why later recovery leaves a generalization gap. Full gradient comparisons, group derivatives, hashes, source-independence checks, and predicted-class concentration are retained in the JSON.

Protocol: `../diagnostics_stock_recurrences_protocol.md`.
