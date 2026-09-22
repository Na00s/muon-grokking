# Readout interventions and optimizer memory

The initial two-seed study completed 115 instantaneous evaluations across six checkpoint pairs and 18 continuation branches of 500 updates each. Every head fit used only original training examples, without labels. Primary accuracy values come from the actual float32 model after changing its head.

## Immediate compensation

| Seed and reference interval | Reference accuracy | Collapsed accuracy | Old reference head | Orthogonal transport | Inverse GL transport |
|---|---:|---:|---:|---:|---:|
| Seed 4, 16050 to 16056 | 100.000% | 66.327% | 99.810% | 99.989% | 99.989% |
| Seed 0, 17200 to 17494 | 99.116% | 31.872% | 14.051% | 95.514% | 96.230% |
| Seed 4, 16050 to 16060 | 100.000% | 18.134% | 99.620% | 99.944% | 99.978% |
| Seed 0, 17490 to 17494 | 96.487% | 31.872% | 96.364% | 96.487% | 96.454% |
| Seed 4, 16055 to 16056 | 91.341% | 66.327% | 98.658% | 89.686% | 89.216% |
| Seed 0, 17493 to 17494 | 95.492% | 31.872% | 96.387% | 95.604% | 95.536% |

The immediately preceding checkpoint can already have degraded generalization. Its measured accuracy is retained in the table. Reusing a nearby old head provides a strong rescue in both seeds. For seed 4, reusing the 16055 head improves accuracy above the preceding checkpoint itself. These interventions support a representation-readout mismatch and show that an explicit fitted basis transformation is not required for an immediate rescue.

The primary inverse-GL maps are invertible in the recorded float64 fit, with condition numbers 665.8 and 3627.2. Their transported heads change by 6.51 and 23.13 times the current head norm. Those large interventions motivate the separately recorded smaller random controls. On the first failures, ten random corrections matched to the orthogonal intervention norm yield 10.35% to 38.01% for seed 4 and 0.91% to 11.51% for seed 0. The actual orthogonal correction yields 99.989% and 95.514%. Opposite-direction corrections yield 46.40% and 10.00%. The rescue is direction-specific.

Backward OLS transport and healthy-logit OLS distillation agree numerically, as expected from linearity. Their apparent agreement is an algebraic consistency check and provides a single line of evidence.

## Continued training and optimizer controls

All continuations use original float32 cross-entropy and Muon arithmetic. Training accuracy is inspected before every update. Heldout accuracy is measured every 10 updates and whenever training accuracy is below 90%. The train-failure counts below use all 501 observed states, including the initial state.

| Head and head-optimizer action | Seed 4 final test | Seed 4 train states below 90% | Seed 0 final test | Seed 0 train states below 90% |
|---|---:|---:|---:|---:|
| Original head, preserve state | 87.336% | 16 | 92.751% | 9 |
| Old head, preserve state | 86.128% | 15 | 95.436% | 11 |
| Orthogonal repair, preserve state | 87.135% | 16 | 95.469% | 9 |
| Inverse GL repair, preserve state | 89.596% | 17 | 53.865% | 10 |
| Random GL-size correction, preserve state | 78.174% | 58 | 32.028% | 149 |
| Random orthogonal-size correction, preserve state | 86.654% | 16 | 93.679% | 12 |
| Original head, clear head state | 95.536% | 6 | 98.702% | 2 |
| Orthogonal repair, clear head state | 97.919% | 5 | 99.586% | 2 |
| Orthogonal repair, zero first moment | 91.599% | 16 | 94.351% | 10 |

Every successful repair with preserved optimizer states falls below 90% training accuracy on the next update. Clearing the head Adam state improves the eventual outcome and reduces the number of failing states, but orthogonal repair plus cleared head state still fails the next update in both seeds. Zeroing only the first moment also permits recollapse, at local update 2 for seed 4 and update 1 for seed 0. Thus head memory affects the subsequent trajectory, while a one-time head correction and head-memory removal are insufficient to maintain stability throughout this window.

Clearing state resets both head moments and its step counter. Zero-first-moment controls retain the second moment and step counter. Every hidden and embedding optimizer state remains exact. Adam state is not transformed covariantly by these controls, so continuation failure should be interpreted as robustness evidence rather than a mathematical test of gauge-equivalent optimizer dynamics.

## Verification and limits

The runner reproduced the six updates from seed 4 step 16050 to 16056 bitwise for every model tensor, optimizer tensor, and RNG state. Monitoring matches the source update exactly. Planted orthogonal and condition-10 invertible transforms recover heldout logits to about 1e-15 relative error. The final audit checked all 18 branch starts and final counters, verified all 9,000 requested updates, and independently reproduced final saved-checkpoint accuracies.

These are controlled experiments in two selected collapse runs of one architecture and task. They establish strong linear readout rescue and directional specificity. Exact global equivalence requires additional geometry evidence, and these head interventions do not establish a unique causal role for representational rotation. Nearby old-head controls are necessary when assessing that interpretation.

Raw tables: intervention_instant_table.csv and intervention_continuation_table.csv. Audit: intervention_final_audit.json. Protocols, full trajectories, fitted maps, readouts, and start/final checkpoints are stored in their corresponding experiment directories. source_versions/ preserves every version of the shared runner used by the completed continuations, with matching hashes.

## Fresh-seed extension

The same fixed instantaneous recipe was then applied to seven pairs from three fresh runs, adding 147 evaluations. The full intervention study therefore contains 262 instantaneous evaluations across five seeds, plus 9,000 continuation updates in the original two seeds. See fresh_interventions/report.md, fresh_interventions/instant_table.csv, and fresh_interventions/summary.json for the complete additional results and verification.

The previous-step head rescues each fresh seed: seed 1 improves from 9.095% to 94.183%, seed 2 from 0.9285% to 100%, and seed 3 from 53.596% to 91.006%. Seed 2 requires a head change of only 1.0876% of the current head norm. These replicas strengthen the evidence that the immediate failure has a large readout contribution.
