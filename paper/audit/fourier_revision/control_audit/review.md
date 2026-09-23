# Independent audit of the new Fourier controls

Audit scope: the new controls under `work/fourier_control`, their mathematical interpretation, and their use in the revised manuscript. Original training observations and original Fourier measurements are preserved. This review does not re-audit those original experiments.

## Verified primary results

`independent_audit.py` imports neither control implementation. It independently reconstructs the five training splits with seeded `torch.randperm`, uses orthonormal torch FFTs, evaluates explicit answer-orbit loops, and compares against the recorded results. All 1,877 checks passed. Saved evidence is in `independent_verification.json`, `primary_recomputed.csv`, and `checkpoint_donors_recomputed.csv`.

Each addition split has 3,830 training and 8,939 held-out examples. Raw held-out correct counts are 79, 68, 79, 81, and 77. The complete diagonal family plus DC gives 8,939 correct in every seed. All 113 answer orbits are covered in each split, with 21 to 45 training donors across the five splits. The corresponding subtraction results also pass. Both operations retain perfect accuracy without DC in these particular controls.

The fixed-size sampling probability that one specified orbit receives no training example is 2.5529551e-18. The union bound across 113 orbits is 2.8848393e-16. The exact fixed-size probability uses the hypergeometric distribution. Treating the training mask as independent Bernoulli draws would give a different calculation.

The raw held-out logits are identically zero. Deterministic smallest-index argmax yields the empirical frequency of class zero; uniform random tie breaking has expected accuracy 1/113. This distinction is stated correctly in the control report and manuscript.

## Mathematical conditions

For every positive modulus p, retaining all diagonal Fourier modes including DC is exactly the uniform average over translations (a+t,b-t). Subtraction uses (a+t,b+t) and the antidiagonal. The identity requires the complete family and a full operand-grid function; it applies channel by channel and commutes with a fixed affine head when DC is retained. Without DC, subtract the global mean. With a bias-free head, either version commutes directly. Independent checks also cover p=6, so primality is clearly unnecessary for this identity.

For a table storing one-hot labels at training inputs and zero elsewhere, the projected output on answer orbit s is n_s/p times the one-hot vector for s. Complete coverage therefore makes the projected predictor perfect. The projection itself supplies the answer equivalence relation, although its FFT implementation does not read labels.

For the training-count-normalized table, the stored nonzero value is p/n_s. This requires n_s>0 for every answer orbit, as verified in all five splits. Its projected class template is exactly the identity matrix. A single conjugate pair k and -k plus DC gives

`z_k(s,c) = [1 + 2 cos(2 pi k(s-c)/p)] / p`.

The maximum is unique at c=s whenever gcd(k,p)=1. At prime p=113 every nonzero k satisfies this condition, so all 56 distinct conjugate pairs classify all inputs correctly. All 280 single-pair cases and their positive margins were checked independently using the closed form. The normalized construction uses training-label counts only. It remains an explicit logical counterexample, with no assertion that the trained transformer implements that lookup table.

The ordinary lookup's conjugate-pair powers are mathematically equal. Sorting their floating-point powers gives an arbitrary numerical ordering. The scripts and report correctly publish a defined ascending-frequency order alongside numerical descending-power order. In the five-pair ordinary lookup, these give respectively 23.81698 to 28.57143 percent and 63.68721 to 87.45945 percent held-out accuracy. The numerical ordering cannot support a preference for particular frequencies.

## Ablation denominators

The intervention `z - Pz + mean(z)` retains DC. For both lookup variants its accuracy is exactly 100 percent on training inputs, zero on held-out inputs, and 3,830/12,769 = 29.9945179732 percent on the full grid. These were recomputed directly from the two-dimensional lookup tables.

The controls establish that full-family sufficiency, sparse sufficiency, low held-out accuracy after ablation, and sensitivity to phase or relocation can each arise for a lookup-only predictor. This particular lookup construction does not reproduce the original paper's near-chance full-grid ablation. That original result remains a distinct numerical observation. The controls alone do not identify the actual network's internal algorithm or invalidate independent raw-feature decoder and adjacent-swap evidence.

## Checkpoint donor domains

All ten adjacent-pair endpoints were independently loaded from their archived residual/head arrays. Their exact split ordering, labels, masks, and train/held-out/full-grid correct counts were verified. For each evaluated held-out row, the held-out-only leave-one-out condition excludes that row and divides by the number of remaining held-out donors in its answer orbit. Training recipients exclude nothing because they are outside the held-out donor set. All donor counts permit this operation.

Before the selected failing updates, held-out-donor leave-one-out accuracy is 100, 100, 100, 99.11623, and 98.17653 percent for seeds 0 through 4. This excludes dependence on training donors for the high orbit-average score at those endpoints. Orbit membership still uses the known task relation, and the procedure uses other held-out inputs' forward outputs. It remains a task-structured diagnostic. Calling it ordinary independent inference or claiming it dates learning of the algorithm would overstate the evidence.

The historical AdamW step-3,000 checkpoint is absent from the inspected local saved features. The donor controls make no claim about its raw donor composition. Its original reported 95.14 percent projection accuracy is preserved.

Report rounding correction: seed 1 raw post-event accuracy is 813/8,939 = 9.0949770668 percent, which rounds to 9.09 percent with two decimal places. Root identified and corrected this manuscript entry.

## Mean-dependent update cross-check

The five `mean_mediation.json` files give absolute differences between mean-only and full-update held-out accuracy of 0.2349256069, 0.3244210762, 0, 0.2572994742, and 0.2572994742 percentage points. The maximum is 0.3244210761830182 percentage points. Thus the manuscript's bound of 0.33 percentage points is valid. The JSON evidence includes each source path and SHA256.

This is an evaluation-only local sufficiency result: add the actual head displacement applied to the training feature mean to the rescued baseline. It does not establish unique upstream causation of feature growth or optimizer-history effects. The current main-text caveat preserves that distinction.

## Secondary implementation check and discovered tie issue

`secondary_direct_synthesis.py` reconstructs all prefix and phase/relocation controls from analytic Fourier coefficients and explicit complex exponential sums, without FFT reconstruction or imports from the control scripts. It verifies every reported prefix accuracy and every isolated-family perturbation accuracy. Its initial independent run exposed an exact-zero tie issue in the auxiliary full-lookup frequency-relocation context. The answer-zero row of the orbit template is invariant under coefficient permutation. Consequently that row's held-out residual logits are mathematically all zero. Floating-point subtraction can leave tiny residual values that change smallest-index argmax. Ten initial one-hot rows differed by the 79 held-out class-zero examples under independent synthesis; maximum minimum-margin disagreement was below 1e-14.

The Fourier-control agent was asked to restore the exact algebraic identity for that row before classification and rerun the secondary outputs. This affects an auxiliary full-lookup context; the primary results and all isolated-family perturbation means were reproduced independently. Final verification status is recorded in `secondary_direct_verification.json` and the companion resolution note.

## Manuscript scope review

The main text's orbit-average caveat, task-informed rescaling interpretation, and separation of decoder/swap evidence are supported. The normalized single-pair counterexample should accompany any statement that sparse sufficiency by itself identifies an arithmetic algorithm. The current revision explicitly recognizes this limitation. The memory controls preserve all original measured spectral outcomes.

For numerical controls, “with corrected CE” is more precise than “under accurate arithmetic,” because the tested intervention corrects cross-entropy evaluation/derivatives while the rest of the model remains finite precision. This wording recommendation was sent to root.
