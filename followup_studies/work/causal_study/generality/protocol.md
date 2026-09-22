# Generality controls selected before execution

Two configurations are selected together, each at seed 0 and each with a matched pair of original FP32 cross-entropy and accurate FP32 cross-entropy. Each arm runs exactly 30,000 full-batch updates from initialization, even if a collapse is observed. These are separate from the five original-configuration checkpoint continuations.

1. Modular subtraction modulo 113, original bias-free and normalization-free transformer.
2. Modular addition modulo 113, gain-free RMS normalization at each attention input and MLP input, plus the final residual before the readout. RMS epsilon is 1e-6. This is a new normalized architecture; the source initialization and all parameter tensors are shared with the original architecture and no parameters are added.

All arms use dimension 128, four attention heads, MLP dimension 512, one transformer block, training fraction 0.3 (3,830 training and 8,939 test examples), and the original optimizer settings. Hidden matrix weights use Muon with learning rate 0.03, momentum 0.95, weight decay 0.1, five Newton-Schulz iterations. Embeddings use AdamW with learning rate 0.001 and weight decay 1. The readout uses AdamW with learning rate 0.00025 and weight decay 1. Both AdamW groups use betas (0.9, 0.999). No hyperparameters will be tuned based on these runs.

Training accuracy is measured at every state using the full batch already needed for optimization. Held-out accuracy is measured every 100 updates and at every training-accuracy-below-90% state after grokking. Six consecutive held-out grid evaluations at or above 95% confirm grokking. A joint failure is a subsequent state where both train and held-out accuracy are below 90%. Save the first joint failure, immediately preceding state, most recent jointly at-least-99% grid state, lowest observed post-event held-out state, each 1,000-update checkpoint, and the final state. The complete fixed horizon continues after every event.

The accurate loss computes log1p of the summed wrong-class exponentials when the true class has a maximal logit. Other examples retain ordinary cross-entropy. This evaluates the same mathematical objective and retains FP32 parameter storage and model arithmetic. Initial tensor digests, parameter routing, model and optimizer state, software version, source hash, trajectories and completion summaries are saved.

No event within the horizon supports only the stated finite-horizon observation. Failed grokking is recorded and cannot serve as evidence about post-grokking collapse. A single seed per new configuration can establish existence or a bounded counterexample and cannot estimate population prevalence.

## Supplementary diagnostics specified during training

The frozen-logit diagnostics and adjacent-event intervention analyzer were added before any extension collapse was observed. They inspect derivative error against an analytic float64 reference, native and swapped parameter groups, optimizer-state controls, and train-only feature alignment. After the original-five-seed diagnostic audit identified inflated feature means, we added readout-input mean norms and the cosine between feature and classifier means to the saved-checkpoint analysis. This is a mechanistic follow-up motivated by that separate finding, and it changes no training arm, configuration, horizon, or event definition.

After the subtraction event at update 15,281 replicated acute readout sensitivity, the already established original-five-seed mean-mediation diagnostic was applied to this new event. It uses the following state's features, the actual preceding-to-following head displacement, and a training-only feature mean to partition the logit displacement into mean and centered-feature components. This is an evaluation-only follow-up and leaves every training run unchanged.

## Completed RMS failure follow-up

The registered every-update triggers captured brief failures in both RMS arms. The 100-update progress snapshots used for interim status updates missed these episodes; the completed event files and summaries provide the authoritative endpoints. Both worst sampled checkpoints were replayed from update 28,000 and checked for exact model, optimizer, and RNG equality against the recorded state. The immediately preceding states and closest 10-update sampled jointly healthy states were retained. Because the immediately preceding worst-state references were already failing, an additional eight-combination healthy-to-worst parameter factorial was evaluated for each RMS arm across the explicit 10-update and 13-update intervals. These are evaluation-only interventions. The 1,043 replay updates reproduce existing trajectories and are recorded separately from the 120,000 registered training updates.
