# What the Fourier controls establish

A complete addition-family Fourier filter supplies the equivalence relation induced by the correct answer. Perfect filtered accuracy, sparse filtered sufficiency, or sensitivity to phase perturbations can all occur for a predictor that stores training answers and outputs zero on every unseen input. The new controls establish this limitation directly while preserving the paper's reported measurements.

## The averaging identity

Use the Fourier convention

\[
h(a,b)=\sum_{k,\ell}\widehat h(k,\ell)\exp(2\pi i(ka+\ell b)/p).
\]

Averaging the translated coefficient over \((a+t,b-t)\) multiplies it by

\[
\frac1p\sum_{t=0}^{p-1}\exp(2\pi i(k-\ell)t/p)=\mathbf 1[k=\ell].
\]

Consequently, retaining the entire diagonal family and DC gives

\[
(P_{\rm add}h)(a,b)=\frac1p\sum_{t=0}^{p-1}h(a+t,b-t).
\]

For subtraction, retaining \((k,-k)\) and DC gives

\[
(P_{\rm sub}h)(a,b)=\frac1p\sum_{t=0}^{p-1}h(a+t,b+t).
\]

Each orbit contains exactly the inputs with the same answer. These are orthogonal projectors onto functions of the answer coordinate. Removing DC subtracts the global mean from the corresponding orbit average. An affine head commutes with either projector because the weights are shared across inputs and the projector preserves constants:

\[
P(hW+b)=(Ph)W+b.
\]

The actual model's head is linear and bias free. The identity therefore applies directly to its filtered final residuals and logits. The implementation cross-checks independent FFT, translated-grid, and orbit-reduction versions at moduli 7 and 113, along with idempotence, DC removal, and readout commutation.

## The ordinary lookup control

For the repository's actual training split, store a one-hot class vector at each training input and zero at every held-out input. Write \(n_c\) for the number of training examples whose answer is \(c\). The filtered logits at any input with answer \(c\) are exactly

\[
Pz(a,b)=\frac{n_c}{p}e_c.
\]

The filtering procedure classifies that entire orbit correctly whenever it has one training donor. The predictor itself never evaluates an arithmetic expression at inference. It performs a table lookup. The external filter is given the task's answer-orbit structure.

At \(p=113\) and nominal training fraction 0.30, each split contains 3,830 training examples and 8,939 held-out examples. The five actual addition splits give:

| Seed | Raw held-out correct / 8,939 | Raw held-out accuracy | Filtered held-out correct / 8,939 | Training donors per orbit |
|---|---:|---:|---:|---:|
| 0 | 79 | 0.883768% | 8,939 | 21 to 43 |
| 1 | 68 | 0.760711% | 8,939 | 21 to 45 |
| 2 | 79 | 0.883768% | 8,939 | 21 to 43 |
| 3 | 81 | 0.906142% | 8,939 | 23 to 44 |
| 4 | 77 | 0.861394% | 8,939 | 21 to 45 |

All 113 orbits are covered in all five splits, and filtered training, held-out, and full-grid accuracy are all 100%. The raw full-grid result is 30.5271 to 30.6289%, because training inputs are memorized. Raw held-out outputs are exactly zero. Their deterministic accuracy is the frequency of class zero, since argmax selects the smallest index. Uniform random tie-breaking has expected accuracy \(1/113=0.884956\%\) on every split.

These values use the five repository splits and their measured class-zero frequencies. The subtraction task gives the same 100% filtered result for all five actual splits; raw held-out accuracy ranges from 0.783085 to 0.917329%.

Under uniformly sampled fixed-size splits, the probability that a particular orbit has zero training examples is

\[
q=\frac{\binom{p^2-p}{3830}}{\binom{p^2}{3830}}
 =2.5529551\times10^{-18}.
\]

The union bound over all 113 orbits is \(2.8848393\times10^{-16}\). Full coverage in these runs follows the sampling regime naturally.

Training-donor-only averaging also gives 100% held-out accuracy. Held-out-donor-only averages are identically zero, giving the same tie-dependent accuracy as the raw held-out lookup. Arbitrarily permuting the class labels assigned to complete answer orbits still gives perfect filtered predictions against those permuted labels.

## Sparse sufficiency admits a stronger lookup counterexample

All nonzero diagonal conjugate pairs have exactly the same mathematical power for the ordinary one-hot lookup. Each class channel contributes the same magnitude at every diagonal frequency, with its amplitude determined by \(n_c\). Floating-point ordering therefore supplies arbitrary tie breaks. We publish both ascending-frequency tie breaking and numerical descending-power ranking without treating the latter as a meaningful frequency preference.

With ascending-frequency tie breaking, the ordinary lookup achieves 23.82 to 28.57% held-out accuracy with five conjugate pairs plus DC. Numerical descending-power ranking gives 63.69 to 87.46% on the same five splits. This sensitivity follows the degeneracy of the pair powers. Sparse sufficiency on a trained model therefore contains more information than full-family sufficiency for this specific unnormalized lookup.

A second lookup makes the logical limitation explicit. Store \(p/n_c\) times the one-hot vector for a training example with class \(c\), with zeros on held-out inputs. The normalizer uses training-label counts only. The complete-family projection now equals \(e_c\) on each answer orbit. Retaining any one nonzero conjugate pair \((k,k),(-k,-k)\) and DC gives class logits

\[
z_k(s,c)=\frac{1+2\cos(2\pi k(s-c)/p)}{p},\qquad s=a+b\pmod p.
\]

For prime \(p=113\) and nonzero \(k\), the unique maximum is at \(c=s\). Every one of the 56 distinct conjugate pairs achieves 100% training, held-out, and full-grid accuracy on every seed. The implementation verifies all 280 single-pair cases against the closed form. Each retained-prefix size 1, 2, 3, 5, 10, 20, and 56 also gives 100%. This predictor can be represented as a 113-dimensional lookup residual with an identity head, or padded to the model's 128-dimensional residual width.

The construction addresses a sufficiency claim. It makes no claim that the trained transformer uses this table or that its learned Fourier spectrum has the same power distribution.

## Ablation and phase controls

The paper's family-ablation convention retains DC:

\[
z_{\rm ablated}=z-Pz+\overline z.
\]

For both lookup variants, this intervention preserves all training predictions and produces zero held-out accuracy. Full-grid accuracy is therefore exactly \(3830/12769=29.994518\%\). Full-family sufficiency together with low held-out ablation accuracy can occur under pure memorization. The distinction between denominators is essential: this lookup does not reproduce the paper's near-chance full-grid ablation results.

Phase scrambling and frequency relocation also reduce the lookup's filtered accuracy. We apply 20 fixed-seed replicates per split and predictor for pair-global phases, channelwise phases, and a derangement of pair coefficients. Each intervention preserves DC, conjugate symmetry, and total nonconstant family power. In the isolated family, the ordinary lookup's mean held-out accuracies across the 100 replicates are 1.0755%, 0.9426%, and 3.0680%, respectively. The normalized lookup gives 0%, 0.8850%, and 3.1555%. The full per-replicate distributions, including large occasional outcomes, are retained in `results/phase_metrics.csv`.

Frequency relocation preserves the answer-zero orbit exactly because every Fourier character equals one there. For the full lookup intervention, that orbit therefore has exactly zero held-out logits. An independent audit found that FFT cancellation roundoff could break this mathematical tie. We restore the exact invariant row before evaluation, so smallest-index argmax selects class zero as specified. This correction leaves the isolated-family results above unchanged.

These controls show that phase and frequency sensitivity are properties a filtered lookup can also exhibit. In trained models they characterize the task-aligned representation and its readout. They provide limited grounds for identifying an internal arithmetic implementation by themselves.

## Donor controls on existing adjacent failure pairs

We used both endpoints of the five previously selected adjacent failure pairs. The saved raw residuals and heads came from the original experiment archive. Input order, labels, and masks were reconstructed from the exact repository split and checked. Logits were recomputed as float64 products of archived float32-origin residuals and heads.

| Seed | Steps | Raw held-out before | Held-out donor average, leave one out, before | Raw held-out after | Held-out donor average, leave one out, after |
|---|---|---:|---:|---:|---:|
| 0 | 17,493 to 17,494 | 95.49% | 100% | 31.87% | 30.46% |
| 1 | 17,720 to 17,721 | 94.14% | 100% | 9.09% | 0.93% |
| 2 | 18,068 to 18,069 | 99.99% | 100% | 0.93% | 0.88% |
| 3 | 16,307 to 16,308 | 89.54% | 99.12% | 53.60% | 65.29% |
| 4 | 16,055 to 16,056 | 91.34% | 98.18% | 66.33% | 71.25% |

Each evaluated held-out row is excluded from its donor average. Before these failing updates, the high score survives removal of all training donors, so the signal at these checkpoints extends beyond the ordinary lookup counterexample. The procedure still uses the task-defined answer orbits. It supplies evidence about the logits on unseen inputs while retaining the scope limitation of a task-structured diagnostic. The corresponding full-grid, training-donor, and complete-orbit metrics are also saved.

The historical AdamW step-3,000 checkpoint behind the 95.14% figure was not available among the local checkpoint artifacts inspected. Its reported result remains intact. The donor controls above make no claim about that early checkpoint or whether it was memorizing.

## Implications for the manuscript

The averaging identity and lookup control should accompany the first introduction of full-family Fourier sufficiency. The early 95.14% observation describes an accurate task-aligned projection. Its value is a measurement of task-structured output content; it alone does not date the acquisition of an internal arithmetic algorithm. Terms such as “perfect isolated circuit” should be stated as “100% accuracy after task-family projection” when that is the measured intervention.

The results still support direct comparisons of measured Fourier content and its functional effect under the fixed head. The full-grid ablations, frequency controls, amplitude interventions, and depth experiments retain their measured outcomes. Their joint interpretation should remain about task-aligned components and readout behavior, with the external task structure made explicit.

Readout swaps and decoders fitted using training examples alone provide independent evidence of surviving information. Their primary interpretation does not depend on the Fourier filter. The numerical update analyses and mean-amplification mechanism also remain independent of this counterexample.

The two verification files contain 100 primary checks and 1,452 secondary checks, all passing. The control figure and compact table are generated from saved machine-readable results. The table is sufficient for the main revision; the figure is optional appendix support.
