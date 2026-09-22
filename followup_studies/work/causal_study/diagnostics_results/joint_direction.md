# Full-network direction of the captured updates

The same-logit accurate derivative is backpropagated through the original float32 network. Its dot product with the actual complete parameter displacement gives the local direction. Loss values use accurate float64 CE on each interpolated float32 network’s logits, evaluated on training examples only.

| Seed | Hidden derivative | Embedding derivative | Readout derivative | Total derivative |
|---|---:|---:|---:|---:|
| 0 | -1.34196e-05 | -1.23303e-05 | -0.000190748 | -0.000216498 |
| 1 | -0.000118516 | -0.00029407 | -0.0021413 | -0.00255389 |
| 2 | -3.03908e-06 | -6.4578e-07 | -8.99047e-05 | -9.35895e-05 |
| 3 | -0.0292708 | -0.101093 | -0.0580972 | -0.188461 |
| 4 | -0.0574825 | -0.185626 | -1.65011 | -1.89322 |

| Seed | Loss at 0 | Loss at 0.001 | Loss at 0.01 | Loss at 0.1 | Loss at 1 |
|---|---:|---:|---:|---:|---:|
| 0 | 3.70252e-06 | 3.49317e-06 | 2.10869e-06 | 3.0689e-07 | 2.85825 |
| 1 | 2.76135e-05 | 2.51797e-05 | 1.11305e-05 | 3.60324e-07 | 13.0745 |
| 2 | 1.6111e-06 | 1.52086e-06 | 9.50022e-07 | 2.84378e-07 | 17.98 |
| 3 | 0.380246 | 0.380061 | 0.378693 | 0.383563 | 7.83768 |
| 4 | 0.145032 | 0.143159 | 0.127814 | 0.0541494 | 11.152 |

All five interpolation endpoints exactly match the saved preceding and following model parameters. The infinitesimal derivative, finite-step loss increase, and group contributions are reported separately. These are the actual captured full-network directions; conclusions about them should not be generalized to every Muon update or every training configuration.

Protocol: `../diagnostics_joint_direction_protocol.md`. Raw derivative variants, training accuracies, hashes, and reconstruction checks: `joint_direction.json`.
