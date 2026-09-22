# Exact feature/readout decomposition

`bilinear_diagnostic.py` analyzes all eleven currently available feature pairs without modifying their earlier geometry reports. Results are in `bilinear/<pair>/bilinear.json`; `bilinear/summary.csv` and `summary.json` collect the held-out results. Additional pairs can be passed together with `--features PATH1 PATH2 ...`. The import API is `analyze_bilinear(H0,H1,W0,W1,y,train,heldout)`.

The exact identity is

ΔZ = H0 ΔW + ΔH W0 + ΔH ΔW.

The three terms are the readout change, feature change, and their interaction. Every norm below removes the per-example common class offset. Norms are divided by the norm of total logit change. They are not percentages of explained variation; their squared values require all three pairwise cross terms to sum to one.

| Pair | Readout-change norm / total | Feature-change norm / total | Interaction norm / total |
|---|---:|---:|---:|
| Seed 0 adjacent 17493 to 17494 | 1.006 | 0.0695 | 0.00541 |
| Seed 4 adjacent 16055 to 16056 | 1.035 | 0.0628 | 0.0353 |
| Seed 4 long-reference peak | 1.324 | 0.0693 | 0.335 |
| Seed 0 long-reference event | 1.788 | 1.955 | 0.636 |
| Seed 0 matched healthy interval | 2.429 | 5.805 | 3.464 |

The adjacent comparisons reinforce the acute importance of the observed readout update. The longer intervals have much stronger interactions and cancellations. The matched healthy interval is an especially clear example: each component can greatly exceed the total change while the model retains high accuracy.

## What fitting a representation map adds

Both predictors use the actual later readout. The identity predictor uses H0 W1, and the fitted predictor uses H0 A W1 with A estimated only from original training examples. Errors below are relative to the class-centered actual later logits H1 W1, not relative to the change ΔZ.

| Adjacent pair | Identity prediction error | Fitted prediction error | Identity class agreement | Fitted class agreement |
|---|---:|---:|---:|---:|
| Seed 0 | 5.980% | 1.708% | 95.14% | 98.14% |
| Seed 4 | 7.452% | 3.968% | 80.71% | 86.44% |

The fitted map improves prediction of the later function. The identity predictor already captures most of the adjacent collapse, so the improvement establishes additional representation-prediction fidelity rather than identifying basis drift as the cause.

Full reports include both original train and untouched held-out splits, all 2x2 counterfactual accuracies and margins, logit reconstruction errors, all norm cross terms, exact-identity errors, source hashes, and the fitted map. Five mathematical tests pass, covering the exact identity, readout-only and feature-only changes, cancellation under an exact joint basis transformation, and invariance of the fit to changes in test data and labels. Numerically negligible total changes are explicitly marked and receive no unstable norm ratios.
