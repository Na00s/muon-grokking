# Feature and classifier means

Declared before extracting these measurements. Liu Hanqing et al., *Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes* (2026), https://arxiv.org/html/2605.06152v2, predicts growing antiparallel classifier and feature means under its stated assumptions. Those assumptions include approximate neural-collapse geometry and a gradient-descent analysis. Our optimizer splits Muon and AdamW and uses weight decay, so this panel tests empirical compatibility with that mechanism.

At all checkpoints in the preceding frozen-gradient time course, extract the final residual on all 12769 examples. Measure the norm of the classifier class mean, classifier centered norm, training-example feature mean, balanced mean of training class feature means, and exact full-grid feature mean. Measure cosines between each feature mean and the classifier mean. Also report the fraction of total feature squared norm attributable to the global mean and the ratio of within-class to between-class centered feature energy. This explicitly checks some conditions needed before interpreting a mean-based account.

No fitted growth law, threshold selection, or acceptance criterion will be inferred from these descriptive trajectories. The arithmetic interventions supply the causal tests separately.
