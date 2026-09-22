# Experiment protocol, 22 September 2026

Source repository: `Na00s/muon-grokking`, commit `6d64a981af75f1300d9060109e81552d48a81360`.

## Scope and selection

These are fresh deterministic CPU replications of the paper's depth-one, modular-addition condition: p=113, 30% training split, width 128, four heads, MLP width 512, no normalization, full batch. Hidden matrices use Muon (learning rate .03, momentum .95, weight decay .1, five Newton-Schulz iterations); embeddings use AdamW (.001, decay 1); readout uses AdamW (.00025, decay 1).

Seed 4 was selected because the archived trajectory records an early simultaneous training/test collapse at step 15,500. Seed 0 was added as an independent initialization. This selection is suitable for mechanism experiments and does not estimate population collapse frequency. Both fresh runs have a 30,000-step maximum. A qualifying event requires six consecutive 100-step evaluations with test accuracy at least 95%, followed by an evaluation with both training and test accuracy below 90%. Capture the preceding 2,000 steps and following 500 steps at 100-step resolution, then stop that baseline run. If no event occurs, report that outcome at the fixed horizon.

The original serialized initialization and runtime provenance are absent from Git. A separate audit verifies initial-state and five-update equivalence to the source training loop. Exact historical trajectories are not assumed.

## Alignment and decoding

Primary H0 is the latest saved pre-event state with both training and test accuracy at least 99%; H1 is the first qualifying event checkpoint. Report any departure from this selection if that healthy threshold has no available checkpoint. Use raw final residuals before unembedding. The original train/test split is retained; labels and residuals from the test split do not enter fitting or hyperparameter selection.

Fit orthogonal Procrustes and forward/backward regularized linear maps. Select ridge penalty on a deterministic 20% inner training validation split using residual reconstruction. Refit on all original training pairs. Compare native logits, cross-checkpoint readout swaps, orthogonal readout transport, and backward linear transport. Report inverse-forward transport only with adequate numerical rank and conditioning. Include centered representation error and class-centered logit error to expose domination by common offsets. Use planted-rotation and shuffled-correspondence controls, plus a healthy temporal pair when the saved window supports one.

Fit fresh bias-free supervised logistic readouts separately on healthy and collapsed features, with regularization selected on original training data alone. Report optimizer convergence and held-out accuracy. Rescue by a general linear map supports recoverable linear mismatch; successful decoding alone establishes retained linearly decodable information. Neither observation proves a global invertible basis change or prospective prevention.

## Numerical controls

Before a new collapse was observed, specify a prospective stable-CE branch from seed 4 step 6,000 through step 20,000, evaluated every 100 steps. This changes the algebraic evaluation of cross-entropy while retaining the mathematical objective, float32 model, original Muon and all saved optimizer states. The expression computes log1p of the non-target exponential mass when the correct class has the maximal logit, avoiding cancellation in tiny correct-class gradients.

After the first event, compare matched continuations from a saved pre-event checkpoint. Arms are original float32, stable cross-entropy float32, float64 model with original float32 Muon orthogonalization, and a float64 model with dtype-preserving orthogonalization and stable cross-entropy. Report exact starting checkpoint, horizon and sampling grid. These local branches test the observed event; a delayed or absent event within a finite horizon is not evidence of permanent prevention.

Frozen-logit diagnostics compare stock/stable losses and gradients against float64 evaluation of exactly the same logits. Parameter-gradient comparisons use identical float32 weights and deterministic forward passes. Stable-CE synthetic tests and checkpoint restoration tests run before training branches.

## Reporting

Keep original Git source unchanged. Preserve scripts, metadata, trajectories, checkpoint provenance and machine-readable results. Distinguish new experiments from archived-log summaries. Report all completed specified arms, including failed rescues and runs with no qualifying event, and do not treat multiple nearby evaluations as independent experimental replicates.
