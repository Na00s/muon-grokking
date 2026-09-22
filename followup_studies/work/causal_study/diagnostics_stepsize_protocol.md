# Supplemental head step-size diagnostic

Declared after the first seed-0 next-update analysis and before evaluating these curves. The seed-0 terminal intervention found a much larger current-gradient contribution than historical first-moment contribution under a fixed Adam denominator.

For each of the same five fixed events, evaluate the actual head displacement at scales -1, -0.1, 0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.75, 1, 1.5, and 2. Hold the features fixed at either the preceding state or the following state. This is a one-dimensional evaluation-only step-size curve. Evaluate stored float32 features and heads in float64, so endpoint accuracy can differ very slightly from the native network. Record endpoint agreement.

Measure the training-loss directional derivative from the analytic accurate gradient and the curvature of this convex fixed-feature readout problem at scale zero. Report the quadratic scale estimate as a local diagnostic. Report the best training-loss grid scale and evaluate its held-out accuracy without selecting on held-out values. This tests whether the acute readout update overshoots along its own direction. It cannot establish why the model entered a sensitive regime.
