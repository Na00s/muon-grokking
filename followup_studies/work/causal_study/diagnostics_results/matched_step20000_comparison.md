# Matched step-20000 intervention endpoints

All twenty fixed checkpoints verify step 20000 and retain input hashes. Original-stock endpoints have recovered high accuracy after their captured failures. Each corrected endpoint has 100% held-out accuracy. The dense trajectory results determine whether an intervention prevents failures during the interval.

| Training backward rule | Held-out accuracy | Feature mean norm | Head mean norm | Mean cosine | Global mean feature power |
|---|---:|---:|---:|---:|---:|
| stock32 | 99.91–100.00% | 620.61–923.12 | 0.01746–0.04307 | -0.850–-0.372 | 54.92–67.07% |
| accurate32 | 100.00–100.00% | 415.77–725.60 | 0.00548–0.00811 | -0.233–-0.097 | 16.75–41.90% |
| target_repair | 100.00–100.00% | 419.51–722.78 | 0.00549–0.00811 | -0.225–-0.099 | 16.93–41.65% |
| row_projection | 100.00–100.00% | 125.37–386.30 | 0.00527–0.00788 | -0.122–0.132 | 3.06–20.09% |

At these matched endpoints, the actual accurate-CE and targeted-repair derivatives match the same-logit reference within 3.2e-7 relative L2 error. The actual projected derivative retains 52.94–86.07% relative error, while its per-example zero-sum residual is only 1.4e-8–2.1e-8 relative to the reference gradient norm. This separates restoration of the zero-sum invariant from complete derivative accuracy. Hypothetical stock-CE errors at those same corrected states are reported in separate columns in the detailed panel and were not the gradients used during corrected training.

After recovery, the stock endpoint derivative errors are smaller than their pre-event values; its feature means still carry 54.92–67.07% of feature power, compared with 16.75–41.90% for accurate/targeted continuations and 3.06–20.09% for projection. These are downstream state differences and do not identify a unique mediator of all earlier updates.

Detailed results: `matched_specificity_final.md`, `matched_stock20000.md`, and the corresponding JSON/CSV files. Five earlier-intervention step-30000 endpoints are reported in `matched_accurate_final.md`; all five have 100% train and held-out accuracy at that endpoint.
