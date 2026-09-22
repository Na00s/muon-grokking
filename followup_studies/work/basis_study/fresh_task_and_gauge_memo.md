# Fresh-seed task and network gauge diagnostics

These analyses separate a long reference interval from the final one-update drop. All maps use original training inputs only; the three class-exclusion groups and random controls are fixed. An adjacent checkpoint can already have reduced accuracy, which is shown explicitly below. These checkpoints share their training trajectories and are dependent observations.

## Task-subspace and output-class transfer

| Comparison | Steps | Reference | Later model | Global backward transport | Task-derived invertible transport | Class-exclusion effect versus random control |
|---|---|---:|---:|---:|---:|---:|
| seed1_adjacent_event | 17720 to 17721 | 94.14% | 9.09% | 94.33% | 94.00% | -0.16 to +0.00 pp |
| seed1_first_event | 17300 to 17721 | 99.19% | 9.09% | 93.04% | 84.20% | -11.52 to -10.34 pp |
| seed2_adjacent_event | 18068 to 18069 | 99.99% | 0.93% | 99.99% | 99.99% | +0.00 to +0.00 pp |
| seed2_first_event | 18000 to 18069 | 100.00% | 0.93% | 100.00% | 100.00% | +0.00 to +0.00 pp |
| seed3_adjacent_event | 16307 to 16308 | 89.54% | 53.60% | 90.49% | 89.81% | -0.22 to +0.00 pp |
| seed3_first_event | 15800 to 16308 | 99.07% | 53.60% | 92.58% | 79.29% | -17.71 to -14.64 pp |
| seed3_peak_event | 15800 to 16309 | 99.07% | 32.12% | 94.68% | 84.29% | -11.66 to -11.11 pp |

The last column is the range of signed, paired accuracy differences across three predefined excluded output-class groups, in percentage points. A negative value means fitting without those classes transfers worse than a random training subset of identical size, evaluated on exactly the same classes. This is a fixed split sensitivity analysis, not a confidence interval.

The task-derived map aligns the full label-conditioned Fourier coefficient matrices. Their full row rank guarantees the existence of an invertible completion. Exact matching of this component is therefore an identifiability limitation rather than independent proof of a historical basis transformation. The table reports deployment on raw residuals, which is a valid held-out prediction test. Label-informed task projection metrics remain separately identified in the JSON files.

## Model-wide residual gauge

| Comparison | Fit | Centered final residual error | Same-matrix network floor | Compensated readout accuracy |
|---|---|---:|---:|---:|
| seed1_adjacent_event | Final-token OLS | 0.0465 | 3.64e-06 | 94.34% |
| seed1_adjacent_event | Shared-site OLS | 0.0647 | 3.37e-06 | 94.37% |
| seed1_first_event | Final-token OLS | 0.4470 | 7.19e-05 | 95.13% |
| seed1_first_event | Shared-site OLS | 0.7437 | 2.52e-06 | 93.71% |
| seed2_adjacent_event | Final-token OLS | 0.0110 | 5.69e-06 | 99.99% |
| seed2_adjacent_event | Shared-site OLS | 0.0164 | 5.62e-06 | 99.99% |
| seed2_first_event | Final-token OLS | 0.0884 | 1.23e-05 | 100.00% |
| seed2_first_event | Shared-site OLS | 0.1515 | 5.50e-06 | 100.00% |
| seed3_adjacent_event | Final-token OLS | 0.1077 | 4.53e-06 | 90.66% |
| seed3_adjacent_event | Shared-site OLS | 0.1413 | 3.15e-06 | 90.58% |
| seed3_first_event | Final-token OLS | 0.5143 | 7.29e-05 | 94.37% |
| seed3_first_event | Shared-site OLS | 0.8482 | 2.16e-06 | 92.50% |

The shared-site map must describe the input residual, post-attention residual, and post-MLP residual at every token position. Each site is normalized by its reference training RMS before fitting. The final-token map uses only the equals-token residual. Both are forward maps; this table uses their inverse for readout compensation, whereas the first table uses the separately fit backward map.

For every learned matrix, its exact residual gauge is planted into the reference network in float32. This matches matrix orientation, conditioning, architecture, and source checkpoint when measuring the numerical floor. Those compensated positive controls preserve source accuracy. Actual representation errors exceed their corresponding floors substantially.

The weaker final-interface account allows internal computations to change. It therefore does not require the stronger common-map constraints. Successful acute transport and poor long-interval transfer can coexist. Ordinary healthy training also departs from an exact gauge, as the earlier temporal controls demonstrate.

Full per-class, per-site, margin, and matrix-spectrum results are in task_subspace_results/ and model_gauge_fresh/actual/. All task coefficients and maps are fit using original training examples only. The task projection explicitly uses labels as a diagnostic; global geometric map fitting uses no labels.
