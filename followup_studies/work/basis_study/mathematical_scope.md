# What the basis experiments can establish

Write the final residual matrix as H, with one row per input, and the readout as W. Logits are Z = H W. The two saved checkpoints have (H0, W0) and (H1, W1).

## Exact representation equivalence

The claim H1 = H0 A for one invertible A is testable on the complete finite modular-addition domain. When both residual matrices have full column rank, such an A exists if and only if their column spaces coincide. Necessity follows because multiplying on the right by an invertible matrix preserves column space. For sufficiency, equal column spaces give H1 = H0 A; full column rank of H1 forces A to have full rank.

Consequently, a nonzero minimum least-squares residual is an obstruction to exact equivalence of the recorded matrices. Finite-precision execution introduces small departures even for a deliberately planted exact network gauge. The study measures numerical benchmarks with the actual transformer, including the same condition number and orientation as each fitted matrix. These empirical controls characterize the tested transformations; they do not supply a universal bound on floating-point rounding. Observed errors must be compared with these controls. Fitting on all task inputs is explicitly an oracle diagnostic of existence; it is separate from the train-only transfer experiments.

Centering deserves particular care. Large input-independent residual components can dominate the raw Frobenius norm. The report supplies both raw and centered errors. Centered error uses training means and measures the variation among examples. A small raw relative error alone can conceal substantial changes to this variation.

## Exact joint symmetry and failure

An exact joint gauge satisfies H1 = H0 A and W1 = inverse(A) W0. It preserves every logit: H1 W1 = H0 W0. An explanation of collapse through basis motion therefore also needs a failure of the readout to compensate appropriately.

The architecture admits this residual symmetry: transform embeddings by A, residual-facing input projections by inverse(A), and residual-producing output projections by A. Internal attention and MLP activations retain their values. This structural symmetry is a property of the parameterization. Whether an optimizer follows such a transformation is an empirical question.

## Linear prediction and causal attribution

For any fitted A, let E = H1 - H0 A. Then the following decomposition is exact:

    H1 W1 - H0 W0 = H0 (A W1 - W0) + E W1.

If H0 A W1 closely predicts the failed model, the representation update has a useful linear approximation in the directions read by W1. The expression already contains the updated readout. It cannot, by itself, assign the collapse to representation motion.

A second exact identity separates the observed changes directly. With dH = H1 - H0 and dW = W1 - W0:

    dZ = H0 dW + dH W0 + dH dW.

The four observable combinations H0 W0, H1 W0, H0 W1, and H1 W1 assess component sufficiency for the chosen transition. Norm ratios require all cross terms because the components may cancel. They are descriptive quantities, rather than percentages of causal responsibility.

If H1 W0 retains accuracy while H0 W1 reproduces the collapse, the observed readout update is sufficient for that acute loss. This does not explain how earlier coupled optimization produced that readout update. The immediately preceding checkpoint is essential: a head from hundreds of updates earlier also measures ordinary coadaptation during the intervening successful training.

## Why task-only alignment can be automatic

For modular addition modulo 113, a complete real Fourier basis of the output class has 113 columns: a constant and 56 sine/cosine pairs. Fitting H approximately as T(y) C uses the labels and re-expresses the 113 training class means. C is a 113-by-128 matrix.

Whenever C0 and C1 have full row rank, an invertible B with C1 B = C0 always exists. Complete each coefficient matrix with a basis of its nullspace to obtain invertible square matrices E0 and E1. Then B = inverse(E1) E0 has the required action. Its action on the remaining dimensions is arbitrary.

This existence result explains why exact alignment of the full task projection offers limited identification of a historical basis transformation. It leaves within-class variation unconstrained. Projecting a held-out example with T(y) also uses its label, so accuracy on that component is a descriptive task diagnostic. Raw-feature transfer and output classes excluded from map fitting provide separate predictive evidence.

## Continued training

A successful head replacement establishes an instantaneous repair. Continued training tests whether the repair persists under a specified optimizer state. Adam's coordinatewise second moments do not generally transform covariantly under an arbitrary invertible feature map. Preserved-state and reset-state branches therefore test robustness under explicit interventions, without assuming gauge-equivalent optimization dynamics.

The full study concerns this task, architecture, optimizer configuration, and the recorded finite training horizons. Multiple snapshots and interventions from one seed remain dependent observations. The experiments can establish counterexamples to an exact explanation in those runs and evaluate an approximate explanation quantitatively. General claims about every future run require additional evidence.
