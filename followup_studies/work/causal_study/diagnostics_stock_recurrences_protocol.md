# Two late original-stock recurrences

Declared before the new frozen intervention measurements. Analyze the newly captured first joint failures within the original-stock extensions for seed 1 (step 29549) and seed 3 (step 27748), using each extension's exact `previous.pt` and `collapse.pt` states. These are separate recurrence cases and do not change the counts in the original five-seed diagnostic panel.

1. Evaluate stock, accurate-float32, and same-logit accurate-float64-reference derivatives at the preceding state. Use the original unnormalized factory, consistent with checkpoint provenance.
2. Replay exactly one original stock update and require bitwise identity of all model tensors, optimizer buffers, and RNG state with the captured following checkpoint.
3. Evaluate all eight preceding/following parameter-group hybrids for hidden weights, embeddings, and readout, with no subsequent optimization. Compute accurate-gradient dot the actual displacement by group using the same already-computed gradients.
4. Reuse the training-mean-only decomposition of following features times the actual head displacement. Evaluate prior-head logits plus the mean term, centered-feature term, and complete displacement. Held-out labels are used only for evaluation. Verify the component sum numerically.

Record source hashes and keep all results under the independent `stock_recurrences` filenames. The panel localizes each captured recurrent step; it does not attribute the entire later recovery trajectory or final generalization gap. No training is performed beyond the single exact replay for each event.
