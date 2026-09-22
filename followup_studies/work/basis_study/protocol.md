# Change-of-basis study protocol

This follow-up tests the basis hypothesis in both directions. The outcome is determined by the experiments, with no requirement that a particular explanation succeed.

## Distinct hypotheses

Use row representations H and readout W, with logits HW.

1. **Exact global representation equivalence:** one invertible, example-independent A satisfies H1 = H0 A across the original task distribution. Its necessary conditions include equal column spaces and zero least-squares residual up to measured arithmetic error. This claim is stronger than retained decodability.
2. **Approximate functional sufficiency:** H0 A W1 reproduces the observed failed model's class-relative logits and errors, with the residual (H1-H0 A)W1 contributing little. This may hold when exact feature equivalence fails.
3. **Task-subspace equivalence:** task-structured projected features can be linearly aligned. The identifiability of this condition must be examined because low-dimensional embeddings may always admit an invertible map.
4. **Model-wide residual gauge:** a common residual basis transforms embeddings, attention and MLP input/output projections consistently. This is a stronger claim than a final-interface relation.
5. **Causal and temporal usefulness:** a train-fitted compensating readout repairs predictions on held-out examples and remains useful during continued training. Every compared continuation preserves the saved optimizer states; this tests an intervention's robustness and does not assume Adam is covariant to general basis changes.

## Geometry and transfer

Run train-only unregularized least squares, orthogonal maps, affine alternatives, inverse and backward transports, principal-angle and rank-sensitivity tests. Centered and uncentered errors are both required. Full-task oracle least squares supplies a separately labeled transductive lower bound for exact representational equivalence, and supplies no prediction claim.

Measure the native arithmetic floor using planted invertible transformations at several condition numbers, repeated identical representations and exact synthetic checks. Add actual network-level planted gauges with compensated and uncompensated readouts in float32 and float64.

Test three fixed output-class exclusions, matched-size random fits, and an operand-stratum exclusion. Excluded examples cannot be used to fit the map or choose hyperparameters. Decompose features using a full Fourier basis of the known modular-addition target only as a labeled task-structure diagnostic; report why perfect alignment in that projection can be non-identifying.

For a fitted forward A, compute the exact decomposition

    H1 W1 - H0 W0 = H0 (A W1 - W0) + (H1 - H0 A) W1.

Report class-centered component norms, their cross term, classification and error-set agreement. Component norms are not additive causal percentages.

## Events and new replications

Existing seeds 0 and 4 provide first joint collapses, seed 4's observed peak, a later generalization decline, adjacent transitions, and healthy temporal controls. Reference-time sensitivity is required because a six-update comparison and a 294-update comparison can have different interpretations.

Three new original-setting seeds, 1, 2 and 3, are specified together before running. Each has a 30,000-step cap. Six consecutive 100-step evaluations at or above 95% test accuracy establish grokking. Training accuracy is subsequently inspected every update; held-out accuracy is evaluated at every training-below-90% trigger. Capture the first simultaneous train/test below 90% event, retain the preceding states and latest confirmed jointly >=99% reference, then continue 200 updates. If no event occurs, report the fixed-horizon outcome.

Additional seeds do not change p=113, the 30% split, the one-block no-normalization architecture, any optimizer hyperparameter, or original float32 arithmetic. The updated monitoring passed an exact model/optimizer/RNG replay check against the existing training API.

## Readout interventions

For existing seed-0 and seed-4 events, compare the native head, old head, orthogonal and general inverse compensation, backward transport, healthy-logit distillation and norm controls, with fits restricted to training rows. Run 500-step continuations for a specified subset while retaining every saved optimizer state. Inspect training accuracy every step and test accuracy every ten steps or on training-failure triggers.

After observing that inverse-GL corrections can have much larger norm than the current readout, add a separately labeled orthogonal-size control: ten predetermined random directions, the opposite correction, and random seed zero as a supplementary continuation. This is a declared adaptive control, not an independently prespecified hypothesis test.

## Reporting constraints

Report every tested seed and arm, including failed repairs, temporary repairs, absent events, and controls. Multiple snapshots of one event are dependent observations. Successful compensation alone does not identify a unique cause. Nonzero geometric drift during healthy training means that geometric deviation alone does not establish collapse causality. Preserve the original repository and earlier deliverables, and place this study's runnable code, results and final report in a separate output bundle.
