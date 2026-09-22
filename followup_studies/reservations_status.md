# Status of the paper's remaining reservations

**Historical assessment, written before the causal study.** See the [completed claim decisions](causal_study/report.md#claim-decisions) and [manuscript revisions](causal_study/paper_revision.md) for the current conclusion. The original assessment is retained below.

Assessment dated September 22, 2026, after the five-seed basis study. Printed line numbers refer to the supplied PDF, *Post-Grokking Collapse at the Representation–Readout Interface in Muon-Trained Transformers*. This assessment uses the completed experiments and the original manuscript. It adds no experiments.

**The new evidence resolves major empirical gaps and supports a more precise contribution. Important causal questions remain open, and several original statements require revision.** The strongest supported account is readout-dominated acute collapse with substantial retained linear task information, useful approximate alignment, and demonstrated numerical sensitivity in one studied trajectory.

## Resolved: substantial task information survives the analyzed failures

The draft proposes a fresh-decoder test at printed lines 483–484. That test is now complete across five seeds. Training-only linear decoders recover 98.20–100% held-out accuracy from selected collapsed representations. Their healthy-reference counterparts reach 99.40–100%. The original held-out examples are excluded from fitting and regularization selection, and the reported fits converge.

This establishes substantial preserved, generalizing linear information at those checkpoints. Small residual gaps in three seeds leave room for modest information loss or probe limitations. Wording such as “all information survives every collapse” would exceed the evidence.

## Resolved for the examined acute transitions: readout-update sufficiency

The interface claim appears at printed lines 78–87 and 107–120. Immediately adjacent checkpoint interventions now show that updated features paired with the preceding readout preserve or improve the preceding accuracy in all five seeds. Applying the updated readout to the preceding features reproduces most of each acute failure. In seed 2, replacing only the readout restores accuracy from 0.93% to 100%.

These comparisons directly localize the acute functional loss. Exact replay, actual float32 head replacements, correction-size controls, and saved optimizer states strengthen the result. Earlier representation and optimizer dynamics can still generate the harmful readout update. The five examples establish event-level component sufficiency under the specified interventions.

## Answered with evidence against the literal hypothesis: exact global basis equivalence

The draft links observed failure to residual basis freedom at printed lines 39–41 and 477–481. Exact architecture symmetry is valid. Actual checkpoint trajectories additionally change representations outside a single global invertible map.

Train-only fits, full-domain oracle diagnostics, matched healthy intervals, and planted full-network gauges show reconstruction discrepancies far above the measured numerical benchmarks. Approximate transport remains useful and often repairs predictions. Its success does not identify an exact gauge trajectory. For adjacent events, simply retaining the preceding head already provides a strong repair.

The paper should distinguish the mathematical symmetry, useful approximate prediction, and the optimizer's historical trajectory. An exactly compensated joint basis transformation preserves logits by construction; a collapse account also requires a failure of compensation.

## Open, highest priority: numerical sensitivity and the upstream cause

Printed lines 96–101 assert that numerical pathology is absent and distinguish the mechanism from softmax collapse. The prior follow-up requires revising this passage. Starting from the same seed-4 checkpoint, the original float32 branch falls to 18.13% test accuracy, while accurate cross-entropy and both double-precision branches remain at 100% over the tested continuation window. Fixed-logit checks also measure substantial stock-cross-entropy gradient error before collapse.

These controls demonstrate numerical sensitivity in that trajectory. Its prevalence across seeds, relation to other reported failures, and interaction with normalized Muon updates remain open. Correcting arithmetic only at the final failing update does not prevent that event, consistent with an accumulated effect of the earlier trajectory and optimizer state.

The next focused study should compare original and accurately evaluated cross-entropy across the same five seeds, branch before substantial gradient contamination, use a fixed common horizon, and monitor failures densely. Gradient error, readout moments, and head updates should be recorded around emerging instability. This would directly address the largest unresolved causal question.

## Partly resolved: repair and continued stability

All successful one-time repairs with preserved optimizer states relapse below 90% training accuracy on the next update in the two tested events. Clearing head state improves eventual recovery and still permits immediate relapse. Native-head controls also benefit from state clearing.

These results establish the temporal limits of the tested repairs. They leave the broader cause of instability open. Adam's coordinatewise moments are generally not covariant under arbitrary basis changes, so continuation outcomes require their explicit optimizer-state qualifications. The original anchoring results at printed lines 248–261 remain separate interventions.

## Required mathematical correction: support versus dominant frequencies

Printed lines 421–423 say that the set of **dominant** frequencies is exactly invariant under any invertible basis change. Exact zero/nonzero Fourier support has this property. Power rankings and threshold-defined dominant sets can change under a general invertible map.

For example, take two Fourier coefficient rows `(2, 0)` and `(0, 1)`. Their norms are 2 and 1. Multiplying both by the invertible matrix `diag(0.1, 10)` changes their norms to 0.2 and 10, reversing their ranking while preserving both nonzero frequencies. Orthogonal transformations preserve their powers. The manuscript should name the invariant corresponding to its actual metric.

## Remaining scope and terminology qualifications

The new detailed mechanism study concerns five seeds of one task and configuration. The 18 continuations concern two selected events. The original monitoring definition uses evaluations every 100 steps, as stated at printed lines 630–636. Stability statements should retain this sampling and horizon qualification; brief failures can occur between evaluations.

Finally, the “circuit failure” taxonomy at printed lines 376–398 evaluates a filtered representation through its current readout. A failure of that pathway can coexist with substantial information accessible to a fresh readout. Naming that operational distinction will make the revised contribution clearer and better aligned with the new evidence.
