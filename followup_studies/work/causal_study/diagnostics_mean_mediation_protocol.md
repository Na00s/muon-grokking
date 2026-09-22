# Training-mean mediation of the acute readout displacement

Declared before the five fixed-event measurements. At each event use the captured following residual matrix H1, prior readout W0, actual readout displacement dW, and the mean mu of H1 over training examples only. Decompose H1 dW = (H1-mu) dW + mu dW exactly in float64 from the stored float32 values.

Evaluate H1 W0 plus the centered-feature contribution alone, the mean contribution alone, and their sum; H1 W0 is the reference. Apply the same training-derived mean to held-out examples. Measure both accuracy and the class-centered logit-component norms, since common shifts across all output classes do not affect predictions. Verify component reconstruction numerically.

This is an evaluation-only mediation of the captured logit change. Subtracting one component from logits is a counterfactual calculation, not a trained architecture or a continuation experiment. It isolates whether the existing feature mean amplifies the actual discriminating readout displacement.
