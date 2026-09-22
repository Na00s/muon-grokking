# Task-subspace alignment and identifiability

The task-conditioned component of two representations can align exactly even when their full residual streams do not. In the existing six checkpoint pairs, a constructed invertible map matches the estimated task component to relative error between 9e-16 and 2.4e-14. Its accuracy on raw held-out representations ranges from 4.62% to 100%. Exact alignment of a projected task component therefore cannot establish that the observed full representation underwent a pure change of basis.

This study covers two independent seeds. Six primary comparisons include four failure or deterioration comparisons and two healthy temporal controls. An additional nearby seed 0 comparison isolates its four-step failure. Comparisons sharing a seed and trajectory are dependent and should not be counted as independent replications.

## Why the task-only map is weak evidence

Let T(y) contain a constant plus sine and cosine features for all 56 nonzero real Fourier frequencies of y modulo 113. For each checkpoint, fit C by least squares on original training examples, so that H is approximated by T C. The complete Fourier basis spans every function of y. Thus this fit is exactly a Fourier re-expression of the 113 training-set class-conditional feature means. It explicitly uses task labels.

Each C is 113 by 128. When C0 and C1 both have full row rank, an invertible matrix B satisfying C1 B = C0 always exists. Complete the rows of each C with a basis of its 15-dimensional nullspace to create invertible square matrices E0 and E1. Then B = inverse(E1) E0 is invertible and has the required action. The completion on the remaining dimensions is arbitrary.

More generally, an invertible right action exists exactly when the two coefficient matrices have identical linear relations among their rows: ker(C0 transpose) equals ker(C1 transpose). Equal rank alone is insufficient in the rank-deficient case. Full row rank makes both kernels zero, so the condition is automatic.

All coefficient matrices in the six existing pairs have full numerical row rank. Their condition numbers range from 165 to 69,429. This is a mathematical existence result for the measured task coefficients. It does not identify the transformation followed during training, constrain its action on within-class variation, or show that optimizer dynamics caused a basis drift.

The empirical coefficients also have sampling uncertainty. Their nonconstant components disagree by 31% to 66% between two fixed, class-stratified halves of the original training set. Consequently, full numerical rank includes weak or unstable directions. The report includes full singular spectra, rank sensitivity, and split-half frequency diagnostics.

## Measurements

All numbers below are percentages of correct predictions on the original held-out examples. The task component uses the held-out label to select its predicted class-conditional representation and is consequently label-informed. Raw-representation accuracies use only the input's captured residual and the fitted linear map at evaluation.

| Checkpoint comparison | Original later model | Global transport | Invertible task transport on raw residual | Same task transport on label-informed task component | Nuisance amplification |
|---|---:|---:|---:|---:|---:|
| Seed 4, acute onset | 66.33 | 99.99 | 99.98 | 100.00 | 1.04x |
| Seed 4, acute peak | 18.13 | 99.98 | 99.94 | 100.00 | 0.73x |
| Seed 0, steps 17200 to 17494 | 31.87 | 95.40 | 85.88 | 100.00 | 1.41x |
| Seed 0, nearby steps 17490 to 17494 | 31.87 | 96.44 | 96.45 | 100.00 | 1.004x |
| Seed 4, steps 16000 to 17000 | 84.37 | 96.63 | 4.62 | 100.00 | 4.12x |
| Seed 4, healthy six-step interval | 100.00 | 100.00 | 100.00 | 100.00 | 1.01x |
| Seed 4, healthy steps 6000 to 7000 | 100.00 | 99.99 | 100.00 | 100.00 | 0.76x |

Global transport is the backward linear map selected using original training examples in the preceding experiment. The task completion is a newly constructed map. “Nuisance” here means the residual after subtracting the training-estimated function of output class, so it includes within-class structure and class-mean estimation error.

The long seed 4 interval provides a useful counterexample to treating task-only alignment as sufficient. Its invertible task map has condition number 795.68 and matches task coefficients to 2.3e-14 relative error. Yet its transformed raw residuals have only 4.62% accuracy and mean correct-class margin -40.97. The original globally fit map has 96.63% accuracy and mean margin 4.87. Different choices of a map that agree on the task coefficients can act very differently on the remaining feature variation.

This counterexample establishes insufficiency of the task-only criterion. It does not establish nonexistence of some other successful global map.

## Transfer to output classes excluded from map fitting

The model's training data remain unchanged. For each of three fixed class groups, y modulo 5 equal to 0, 1, or 2, I excluded those classes only when fitting the post-hoc map from later to earlier residuals. Evaluation used the original held-out examples in the excluded classes. A deterministic random subset with the same number of original training rows provides a sample-size control and is evaluated on exactly the same rows.

No hyperparameter is selected using these results. The primary map is ordinary least squares. The report also contains a fixed relative ridge penalty of 1e-6.

| Comparison | Accuracy on excluded classes, map fit without those classes | Matched random fit on same evaluation classes |
|---|---:|---:|
| Seed 4, acute onset | 100.00 to 100.00 | 100.00 to 100.00 |
| Seed 4, acute peak | 99.78 to 99.94 | 100.00 to 100.00 |
| Seed 0, steps 17200 to 17494 | 88.47 to 89.06 | 95.18 to 95.43 |
| Seed 0, nearby steps 17490 to 17494 | 96.42 to 96.74 | 96.40 to 96.74 |
| Seed 4, steps 16000 to 17000 | 52.37 to 56.02 | 95.72 to 96.98 |
| Seed 4, healthy six-step interval | 100.00 to 100.00 | 100.00 to 100.00 |
| Seed 4, healthy steps 6000 to 7000 | 99.95 to 100.00 | 99.95 to 100.00 |

Excluding one fixed first-operand stratum gave smaller differences. Seed 0 achieved 95.84% versus 95.51% for the random control; the long seed 4 interval achieved 94.84% versus 95.68%. The class-exclusion result is therefore specific to the task directions that the map saw during fitting, beyond a generic reduction in training examples.

The acute seed 4 event supports a strong linear, task-general alignment account: transfer remains near-perfect even to classes excluded from fitting. The nearby seed 0 pair also transfers well: 96.42-96.74% on excluded classes, matching random controls and close to the reference model's 96.49% accuracy. Its task-derived invertible map gives 96.45% raw accuracy with only 1.004x nuisance amplification. Thus the seed 0 transfer deficit in the earlier table belongs to the 294-step comparison, which includes substantial ordinary training drift. It should not be attributed to the four-step acute failure. Longer intervals show that task-general transfer is not universal. Classification recovery can coexist with substantial changes to the residual stream. A pure global basis explanation requires more than fitting task class means or showing high readout accuracy on a random split.

## Reproducibility and validation

`task_subspace.py` runs the decomposition, invertible completion, coefficient uncertainty diagnostics, output-class exclusions, and operand-stratum exclusion. All coefficients and maps use original training examples only. Original inputs are regenerated from each checkpoint's recorded seed and checked against the saved dataset hash before operand analysis.

`task_subspace_results/<pair>/report.json` contains full metrics and the source metadata; `maps.npz` stores the coefficient matrices and transformations. `task_subspace_run.log` records the run. The report contains representation errors with training-mean centering, classification loss and margins, map spectra, task and nuisance errors, and all fixed-ridge sensitivity results.

Eight synthetic tests pass: Fourier orthonormality, equivalence to class means with unequal class counts, existence of invertible maps between independent full-row-rank task encodings, failure on independent nuisance variation, independence from held-out features, independence from held-out labels, recovery of a planted global map on excluded classes, and rejection of a rank-deficient completion.

Run with:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/homebrew/Caskroom/miniforge/base/envs/research/bin/python -m unittest discover -s work/basis_study -p test_task_subspace.py -v
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/homebrew/Caskroom/miniforge/base/envs/research/bin/python -u work/basis_study/task_subspace.py
```

This is a diagnostic of the existing regenerated checkpoints. It supplies evidence for retained linear task structure and limits a claim that this alone demonstrates a historical invertible basis transformation.
