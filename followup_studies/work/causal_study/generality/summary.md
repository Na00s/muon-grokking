# Generality control results

Each pair starts from identical seed-0 initialization and runs for 30,000 updates under its prespecified architecture and operation. The accurate-loss arm changes the loss arithmetic throughout training. Every low-training-accuracy state after sustained grokking is checked on held-out data. The event threshold is simultaneous train and held-out accuracy below 90%.

| Condition | Arithmetic | Grokking confirmed | First joint failure | Minimum post-grok train | Minimum sampled post-grok test | Final test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| subtraction | stock | 9300 | 15281 | 27.023% | 5.851% | 100.000% |
| subtraction | accurate | 10500 | None through 30,000 | 100.000% | 96.879% | 99.966% |
| rms | stock | 15700 | 18515 | 0.601% | 0.481% | 100.000% |
| rms | accurate | 19900 | 28495 | 0.992% | 0.526% | 100.000% |

The accuracy panel includes every recorded held-out evaluation; dotted vertical lines indicate the first joint failure. The gradient panel compares the derivative actually used in each arm against an analytic float64 derivative at that same saved logit matrix. This isolates derivative arithmetic from differences in the learned weights. The conclusions cover one seed per configuration and the observed 30,000-update horizon.

The endpoint table uses the every-update failure triggers and completed run summaries. The 100-update progress snapshots missed brief RMS failures that the every-update trigger captured.

## subtraction stock: first registered event at 15281

Exact replay reproduced the captured model, optimizer state, and RNG. The preceding model had 98.646% held-out accuracy. Updating only the readout gives 6.276%; updating only embeddings gives 98.624%; updating hidden matrices and the readout while retaining the preceding embeddings gives 5.929%. Updating hidden matrices and embeddings while retaining the preceding readout gives 98.814%. The complete update gives 5.851%.

At the preceding state, the derivative used in training has relative error 0.0308272 against the analytic float64 logit derivative. The fraction of vanished correct-class derivatives is 7.493%. The preceding feature-mean norm is 12,342.269.

The head-displacement decomposition starts from the preceding readout on the updated features, with 98.814% held-out accuracy. Adding the training-mean contribution gives 5.840%; adding the centered-feature contribution gives 98.758%; adding both gives 5.851%. The exact decomposition identity passed.

In this transition the mean contribution alone reproduces the failure from a healthy baseline, establishing its local sufficiency.

## rms stock: first registered event at 18515

Exact replay reproduced the captured model, optimizer state, and RNG. The preceding model had 100.000% held-out accuracy. Updating only the readout gives 100.000%; updating only embeddings gives 74.315%; updating hidden matrices and the readout while retaining the preceding embeddings gives 100.000%. Updating hidden matrices and embeddings while retaining the preceding readout gives 85.636%. The complete update gives 85.658%.

At the preceding state, the derivative used in training has relative error 0.000159079 against the analytic float64 logit derivative. The fraction of vanished correct-class derivatives is 0.000%. The preceding feature-mean norm is 1.707.

The head-displacement decomposition starts from the preceding readout on the updated features, with 85.636% held-out accuracy. Adding the training-mean contribution gives 85.647%; adding the centered-feature contribution gives 85.658%; adding both gives 85.658%. The exact decomposition identity passed.

The updated-feature baseline has already failed before the readout displacement is applied. This decomposition measures the subsequent readout increment; the parameter-group controls locate the earlier loss.

## rms stock: worst sampled checkpoint at 28540

Exact replay reproduced the captured model, optimizer state, and RNG. The preceding model had 0.694% held-out accuracy. Updating only the readout gives 0.705%; updating only embeddings gives 0.626%; updating hidden matrices and the readout while retaining the preceding embeddings gives 0.526%. Updating hidden matrices and embeddings while retaining the preceding readout gives 0.537%. The complete update gives 0.481%.

At the preceding state, the derivative used in training has relative error 3.88165e-08 against the analytic float64 logit derivative. The fraction of vanished correct-class derivatives is 0.000%. The preceding feature-mean norm is 4.877.

The head-displacement decomposition starts from the preceding readout on the updated features, with 0.537% held-out accuracy. Adding the training-mean contribution gives 0.526%; adding the centered-feature contribution gives 0.503%; adding both gives 0.481%. The exact decomposition identity passed.

The preceding state and the updated-feature baseline are already in failure. This adjacent comparison quantifies changes within that episode. The first-event comparisons identify the acute locus.

The separate healthy-to-worst factorial begins at update 28530 with 100.000% held-out accuracy and ends at 28540. Replacing only the readout with its worst-state value gives 100.000%; replacing only embeddings gives 0.940%; replacing hidden matrices plus readout while retaining the healthy embeddings gives 99.843%; replacing all groups gives 0.481%. This evaluates parameter combinations across the stated multi-update interval.

## rms accurate: first registered event at 28495

Exact replay reproduced the captured model, optimizer state, and RNG. The preceding model had 96.208% held-out accuracy. Updating only the readout gives 96.208%; updating only embeddings gives 0.940%; updating hidden matrices and the readout while retaining the preceding embeddings gives 96.409%. Updating hidden matrices and embeddings while retaining the preceding readout gives 0.895%. The complete update gives 0.884%.

At the preceding state, the derivative used in training has relative error 9.47645e-08 against the analytic float64 logit derivative. The fraction of vanished correct-class derivatives is 0.000%. The preceding feature-mean norm is 0.958.

The head-displacement decomposition starts from the preceding readout on the updated features, with 0.895% held-out accuracy. Adding the training-mean contribution gives 0.884%; adding the centered-feature contribution gives 0.895%; adding both gives 0.884%. The exact decomposition identity passed.

The updated-feature baseline has already failed before the readout displacement is applied. This decomposition measures the subsequent readout increment; the parameter-group controls locate the earlier loss.

## rms accurate: worst sampled checkpoint at 28503

Exact replay reproduced the captured model, optimizer state, and RNG. The preceding model had 0.660% held-out accuracy. Updating only the readout gives 0.671%; updating only embeddings gives 0.626%; updating hidden matrices and the readout while retaining the preceding embeddings gives 0.492%. Updating hidden matrices and embeddings while retaining the preceding readout gives 0.526%. The complete update gives 0.526%.

At the preceding state, the derivative used in training has relative error 3.97717e-08 against the analytic float64 logit derivative. The fraction of vanished correct-class derivatives is 0.000%. The preceding feature-mean norm is 10.269.

The head-displacement decomposition starts from the preceding readout on the updated features, with 0.526% held-out accuracy. Adding the training-mean contribution gives 0.503%; adding the centered-feature contribution gives 0.537%; adding both gives 0.526%. The exact decomposition identity passed.

The preceding state and the updated-feature baseline are already in failure. This adjacent comparison quantifies changes within that episode. The first-event comparisons identify the acute locus.

The separate healthy-to-worst factorial begins at update 28490 with 100.000% held-out accuracy and ends at 28503. Replacing only the readout with its worst-state value gives 100.000%; replacing only embeddings gives 0.682%; replacing hidden matrices plus readout while retaining the healthy embeddings gives 94.530%; replacing all groups gives 0.526%. This evaluates parameter combinations across the stated multi-update interval.

## Scope and verification

The normalized architecture adds gain-free RMS normalization at attention and MLP inputs and before the readout, with epsilon 1e-6. Its parameters and optimizer routing match the source initialization. Both loss arms begin at update zero, so their complete learning paths and grokking times can differ. Across the two extension configurations, both operation and normalization differ. The original seed-0 addition baseline supplies a matched operation for interpreting the RMS result. Any such architecture comparison combines forward-pass and ensuing learning-path changes.

Six runner tests verify source initialization, optimizer routing, exact no-norm source updates, subtraction targets and split, accurate-loss derivatives, normalization arithmetic, and independent checkpoint loading. completion_verification.json audits every saved checkpoint, paired initial model/optimizer/RNG states, source hashes, full 30,000-update completion, event qualification, diagnostic provenance, and consistency of final snapshots. checkpoint_manifest.json lists checkpoint SHA-256 hashes.
