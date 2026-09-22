# Independent complete-update direction at the first two RMS events

Declared before measurement. Analyze the first stock-CE and first accurate-CE events in the completed RMS-normalized generality runs, using `previous.pt` and `collapse.pt`. The two events are defined by the runner's every-update joint-failure trigger. They are architecture-generalization cases and are separate from the original unnormalized five-seed panel.

Load each state exclusively through `generality_runner.make_model_optimizers(seed, normalization='rms', checkpoint=state)`. Assert the RMS model class, normalization metadata, and consecutive event steps. Use the checkpoint's operation to generate the original training split. Do not load these tensors into the unnormalized architecture.

Compute stock-float32, accurate-float32, and same-logit analytic-float64-reference gradients. The reference derivative is cast to float32 before propagation through the shared RMS network. Dot each parameter gradient with the actual stored following-minus-preceding parameter displacement, by optimizer group and in total.

Evaluate the full RMS network at interpolation scales [0, 0.001, 0.01, 0.1, 1], constructing interpolation in float64 and storing parameters in float32. Measure accurate CE on training examples only, in float32 and float64 loss arithmetic from the same logits. Require exact saved parameter tensors at both endpoints. Record all checkpoint SHA256 hashes and verify they remain unchanged. No optimizer steps, held-out selection, or new training are performed.

Negative local derivatives with higher endpoint loss support finite-step overshoot for that event; positive local derivatives support ascent for that event. The results do not establish an identical mechanism across normalized and unnormalized architectures.
