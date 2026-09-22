# Independent RMS event direction panel

This analysis uses the actual gain-free pre-sublayer and final RMS architecture through `generality_runner.make_model_optimizers(..., normalization="rms")`. Both events are the first every-update joint failures in the completed trajectories. Only training examples are evaluated.

| Arithmetic | Step | Hidden derivative | Embedding derivative | Readout derivative | Complete derivative |
|---|---:|---:|---:|---:|---:|
| stock | 18514 → 18515 | -0.001434554 | -0.01292776 | -1.558806e-06 | -0.01436388 |
| accurate | 28494 → 28495 | -0.2783271 | -11.32015 | -0.003936084 | -11.60242 |

| Arithmetic | Loss at 0 | Loss at 0.001 | Loss at 0.01 | Loss at 0.1 | Loss at 1 |
|---|---:|---:|---:|---:|---:|
| stock | 0.0008129897 | 0.0007987865 | 0.0006843861 | 0.000222263 | 0.9511559 |
| accurate | 0.3420677 | 0.3304998 | 0.2357755 | 0.895634 | 6.810991 |

The primary derivative evaluates the accurate float64 CE derivative on the same float32 logits, then casts that derivative to float32 for the RMS network backward. Accurate-float32 and stock-float32 derivative variants are retained in the JSON. Finite-grid losses evaluate accurate float64 CE on each interpolated float32 network’s logits.

All four interpolation endpoints exactly match the saved preceding/following parameter tensors. All four checkpoint hashes are unchanged. This is an independent local-direction and finite-step test, with no optimizer updates. It does not assume the normalized and unnormalized events have the same causal mechanism.

Protocol: `../diagnostics_rms_joint_protocol.md`. Raw measurements and hashes: `rms_joint_direction.json`.
