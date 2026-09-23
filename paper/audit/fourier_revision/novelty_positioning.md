# Contribution relative to the closest work

The revision centers on individual captured post-grokking failures and the information surviving them. Faster grokking, spectral dispersion, and numerical feature inflation provide the experimental context. The original results remain in the manuscript.

| Prior work | Established evidence credited in this manuscript | Additional evidence in this study |
|---|---|---|
| [Liu et al., Grokking or Glitching?](https://arxiv.org/html/2605.06152v3) | Finite-precision class-sum errors, coupled feature/classifier means, amplified Adam updates, projection corrections, and architectural dependence including normalization. | Interventions on the actual captured readout displacement: its product with the post-update training mean reproduces five failing test accuracies within 0.324421 percentage points. Adjacent-state swaps localize the failing parameter update. Corrected-loss continuations test the original Muon experiment's horizon. An RMS case localizes an accurate-loss failure to embeddings. |
| [Chou et al., Two Speeds of Learning](https://arxiv.org/html/2605.27078v1) | Representation/readout decomposition, same-split linear probes, and readout miscalibration during grokking. | The diagnostic is applied at selected post-grokking joint failures and combined with adjacent-state swaps, training-mean interventions, and explicit global basis-equivalence tests. |
| [Wang, The Active Ingredient in Muon's Grokking](https://arxiv.org/html/2607.20512v1) | Faster Muon grokking, speed/stability tradeoffs, accuracy dips, and Fourier dispersion. | Matched interventions trace individual failures, identify the sufficient mean-dependent part of a captured readout update, and test surviving information and corrected-loss persistence. |

The RMS experiment limits the account for the measured unnormalized architecture. It does not dispute Liu et al.'s theorem under its assumptions. The mean-dependent intervention establishes acute local sufficiency. It leaves the preceding contributions of feature growth and optimizer history coupled, and accuracy agreement alone does not establish prediction-by-prediction equivalence.

The Fourier controls address interpretation directly. Full-family filtering supplies the answer-orbit relation, and the count-normalized lookup shows that single-pair filtered sufficiency can also arise from memorization. The original full-grid ablations retain additional information: the simple lookup's ablation accuracy is 29.9945% on that domain. Raw-feature probes and adjacent swaps supply independent evidence of surviving information.

Sources were checked against the linked primary papers on 2026-09-22. The manuscript credits the prior methods without claiming priority for them.
