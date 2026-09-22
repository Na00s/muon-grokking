# Fresh-seed readout interventions

The fixed intervention recipe was applied to seven pairs from three fresh runs, producing 147 instantaneous evaluations. Every fit used original training rows only, without labels. All outcomes below use actual float32 model forward passes after installing the fitted or control head. These experiments added no continuation training.

| Pair | Reference to evaluated step | Reference test | Collapsed test | Old head | Orthogonal | Inverse GL | Random orthogonal-size range |
|---|---|---:|---:|---:|---:|---:|---:|
| seed1_first | 17300 to 17721 | 99.195% | 9.095% | 22.665% | 90.032% | 95.134% | 0.94% to 7.36% |
| seed1_adjacent | 17720 to 17721 | 94.138% | 9.095% | 94.183% | 94.407% | 94.339% | 8.71% to 17.28% |
| seed2_first | 18000 to 18069 | 100.000% | 0.929% | 78.074% | 100.000% | 100.000% | 0.88% to 27.75% |
| seed2_adjacent | 18068 to 18069 | 99.989% | 0.929% | 100.000% | 99.989% | 99.989% | 0.88% to 1.01% |
| seed3_first | 15800 to 16308 | 99.071% | 53.597% | 35.284% | 88.332% | 94.373% | 1.05% to 6.56% |
| seed3_peak | 15800 to 16309 | 99.071% | 32.118% | 43.830% | 93.031% | 96.163% | 0.89% to 6.41% |
| seed3_adjacent | 16307 to 16308 | 89.540% | 53.597% | 91.006% | 91.252% | 90.659% | 34.01% to 47.26% |

Nearby old heads restore most or all of the preceding reference accuracy in every fresh seed. The preceding reference itself may already have degraded generalization; that value is shown explicitly. This repeats the original two-seed reference-time finding and supports a substantial readout contribution to the sudden failures.

For the farther healthy references, inverse-GL compensation provides a strong rescue, but its accuracy can differ from orthogonal transport and from a nearby old-head swap. A fitted compensation that improves classification establishes linear decodability relative to that reference. Combined with old-head controls, these results do not isolate a unique representational-rotation cause.

The random-direction controls match the magnitude of the orthogonal correction. Some small random perturbations improve a severely collapsed baseline, especially when chance accuracy is the starting point. The directed repairs remain substantially stronger; the complete random range is included rather than selecting a single comparison.

Verification confirmed every source feature and checkpoint hash, every actual-model baseline accuracy, and the requested correction norms. Numerical transport orientation and exact monitoring were validated in the initial intervention suite. The reusable runner is run_instant_interventions.py. Detailed rows are in instant_table.csv, compact pair rows in pair_summary.csv, and provenance plus checks in summary.json.
