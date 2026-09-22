# Accurate-cross-entropy step-30000 endpoints

All five fixed final checkpoints exist, record step 30000, and have SHA256 hashes retained in the full JSON. They continue from the corresponding original step-6000 model, optimizer, and RNG state. This panel describes the final states; the separate dense trajectories determine whether intermediate failures occurred.

| Seed | Train accuracy | Held-out accuracy | Feature mean norm | Head mean norm | Mean cosine | Global mean feature power |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 100.0000% | 100.0000% | 293.18 | 0.010463 | -0.4559 | 21.45% |
| 1 | 100.0000% | 100.0000% | 265.37 | 0.009947 | -0.4372 | 19.79% |
| 2 | 100.0000% | 100.0000% | 389.76 | 0.008030 | -0.3782 | 25.00% |
| 3 | 100.0000% | 100.0000% | 150.24 | 0.007308 | -0.3380 | 10.25% |
| 4 | 100.0000% | 100.0000% | 185.59 | 0.006364 | -0.3742 | 12.75% |

At these final states, the accurate logit derivative used by the training objective matches the same-logit analytic reference with relative L2 error 5.67e-08–2.48e-07. Its downstream readout-gradient error is 1.09e-07–2e-07. Applying stock CE diagnostically to the same states would yield 88.59–90.15% logit-gradient error and zero 100.00–100.00% of target derivatives.

The stock-loss measurements are counterfactual diagnostics of each frozen final model. The actual continuation used accurate cross-entropy. Feature/classifier mean measurements describe the intervention’s endpoint and do not establish unique mediation of its full history.

Raw data: `matched_accurate_final.json`. Compact table: `matched_accurate_final.csv`. The five input hashes are recorded in both.
