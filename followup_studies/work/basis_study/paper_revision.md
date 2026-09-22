# Suggested manuscript revision after the basis study

The new experiments support a more specific acute mechanism and delimit the role of basis freedom. The original PDF has been preserved. The text below is proposed wording for a revision.

## Main result paragraph

We tested whether post-grokking collapse can be explained by a pure invertible transformation of the residual representation. Across five training seeds, train-only linear maps provide useful alignment and readout compensation, but the measured representations depart from exact global basis equivalence by far more than matched full-network numerical controls. Immediately adjacent checkpoint interventions further localize the acute loss: updated representations paired with the preceding readout retain or improve the preceding accuracy in every seed, while updated readouts paired with preceding representations reproduce most of the collapse. In the clearest case, replacing the preceding readout restores test accuracy from 0.93% to 100%. Fresh linear decoders reach 98.20–100% on the analyzed collapsed representations. These results establish substantial retained linear task information and a readout-dominated acute failure, while leaving the upstream optimizer dynamics and a quantitatively approximate basis account open.

## Basis symmetry paragraph

The residual parameterization has an exact joint change-of-basis symmetry: for row activations, the transformation H to H A and W to inverse(A) W preserves all logits. This symmetry supplies a useful hypothesis for representation–readout coadaptation. Our trajectory measurements show that actual training also changes the representation outside a single global invertible map. Successful post-hoc compensation therefore demonstrates predictive alignment without establishing that the observed trajectory follows an exact gauge orbit. Reference-time controls are essential because heads separated by hundreds of successful updates also measure ordinary coadaptation; immediately preceding heads isolate the acute transition.

## Readout intervention paragraph

A one-time head intervention repairs the failed function but does not maintain stability under the original continuing optimizer dynamics in the two tested events. Every successful repair with preserved optimizer states relapses below 90% training accuracy on the next update. Clearing the head's Adam state improves subsequent recovery and still permits immediate relapse. These controls establish the intervention's temporal limits. Since coordinatewise Adam moments do not generally transform covariantly under a general residual basis map, the continuation results assess robustness under explicitly specified optimizer states.

## Specific changes to the existing draft

- **Introduction, lines 39–41:** keep the mathematical invariance statement. Replace the assertion that the observed failure moves along that freedom with the measured distinction between useful approximate alignment and failure of exact global equivalence.
- **Introduction, lines 78–87, and discussion of split routing:** present residual symmetry as a structural motivation. Add the five-seed adjacent readout swaps as direct evidence about the acute update. Distinguish this update attribution from the earlier coupled dynamics that generate it.
- **Discussion of the proposed healthy-to-collapsed transformation test:** replace the untested prediction with the completed train-only geometry, transfer, and compensation results. Include the same-matrix network controls and the nearby-reference sensitivity.
- **Task-subspace and Fourier discussion:** state which analyses use labels. Explain that a full-rank 113-by-128 task coefficient matrix always admits an invertible map to another such matrix, making exact task-only alignment weak evidence for the historical trajectory. Include raw-feature and excluded-class transfer as separate predictive tests.
- **Introduction, lines 96–101:** revise the assertion of no numerical pathology in light of the earlier accurate-loss and precision branches. The present basis study does not supersede those controls.
- **Limitations:** the new basis study concerns five seeds of one modular-addition configuration. Continued-training repair tests concern two selected events. The multiple reference times and interventions from one seed are dependent comparisons.

Suggested section title: **Retained task information and readout-dominated acute collapse**.

The numerical tables, protocol, tests, and reproducible artifacts are in the accompanying report and bundle.
