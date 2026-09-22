# Interpretation audit of the functional decomposition

The narrowest supported claim is that the acute failures examined so far involve a harmful readout update acting on representations that retain useful task information. A train-fitted linear model of representation drift helps reproduce the resulting logits. These observations do not establish representation basis drift as the cause of collapse.

## Keep three hypotheses separate

1. **An exact joint basis transformation:** H1 = H0 A and W1 = A^-1 W0. This preserves every logit identically, so an exact joint transformation cannot itself cause an accuracy loss.
2. **An uncompensated representation basis transformation:** H1 is approximately H0 A, while the readout does not maintain the corresponding function. This is a proposed mismatch mechanism. Establishing a fitted A and mismatch does not identify which update caused the mismatch.
3. **A linear approximation to checkpoint changes:** H1 = H0 A + E. This is an always-available regression decomposition. Its empirical usefulness depends on reconstruction quality and held-out functional predictions. It does not require a genuine symmetry or an optimizer trajectory along a basis orbit.

The current “basis-predicted” term H0 A W1 contains the observed later readout. It can therefore predict collapse simply because W1 is harmful, even with A replaced by the identity. Call it **linear-map counterfactual** or **aligned readout-mismatch term** in the report. Treat it as descriptive functional evidence.

## The strongest existing control is the full two-by-two comparison

| Adjacent pair | H0 W0 | H1 W0, feature change alone | H0 W1, readout change alone | H1 W1 |
|---|---:|---:|---:|---:|
| Seed 4, 16055 to 16056 | 91.34% | 98.66% | 66.75% | 66.33% |
| Seed 0, 17493 to 17494 | 95.49% | 96.39% | 31.86% | 31.87% |

For these immediate transitions, retaining the preceding readout prevents the observed accuracy loss under the actual later features. Applying the new readout to the preceding features reproduces almost the whole loss. Thus the actual feature update alone is insufficient to explain the acute failure in these two examples. The readout update is sufficient within these fixed-component, one-step counterfactuals. This conclusion concerns the immediate update and the observed checkpoints; it does not identify the earlier optimization process that produced that update.

The fitted map still improves exact class prediction. In the seed 0 adjacent pair, the later model agrees with H0 W1 on 95.14% of test predictions, and with H0 A W1 on 98.14%. In seed 4 the corresponding agreements are 80.71% and 86.44%. Across the longer seed 0 interval, identity substitution has only 4.83% agreement, whereas the fitted map reaches 74.43%. A's predictive value over longer intervals largely captures successful prior coadaptation as well as the failure. Similar accuracies alone conceal these distinctions.

## Mathematical and inferential checks to retain

- Present H0 W1 as the identity-map baseline beside H0 A W1. Report exact class agreement, class-centered logit error, and margins as well as accuracy. An improvement from fitting A supports representation prediction, not a claim that representation motion caused the event.
- Use the exact bilinear identity ΔZ = (H1-H0)W0 + H0(W1-W0) + (H1-H0)(W1-W0) when separating raw feature and readout changes. Include the interaction term. Neither norm ratios nor accuracy differences are additive causal shares.
- Report immediate preceding references alongside longer healthy references. In seed 0, long-reference geometry error stays around 43.5% while the final update drops accuracy from 95.49% to 31.87%. The old long-reference head was already ineffective before the failure, during successful coadaptation.
- Keep exact-global and approximate-task-subspace claims separate. Nonzero oracle residuals exclude an exact shared map for the observed activation matrices at the stated precision. They do not exclude a useful map on task-relevant directions. Successful linear repair does not identify a unique invertible symmetry.
- Approximate inverse transports need rank, conditioning, update magnitude, and numerical-floor controls. Their success is also consistent with constructing a new effective readout from retained features. Label-free fitting and untouched test inputs make this useful evidence, while leaving the mechanistic interpretation open.
- Successful prospective repair shows that an intervention can control the failure. Distinguishing basis correction from ordinary stabilization requires comparison with the preceding-head reset, head freezing, and appropriately scaled controls under matched optimizer-state handling.

Recommended wording: **“Across these observed adjacent transitions, the readout update is sufficient to reproduce the acute accuracy loss, while the updated representation with the preceding readout retains high accuracy. Train-only linear alignment explains additional details of the changed function, but these experiments do not isolate a basis-drift cause.”**
