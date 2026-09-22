# Matched step-20000 specificity states

Declared before measurement. Diagnose all five original-stock states at step 20000 and all fifteen step-20000 endpoints of continuations from the same saved step-15000 states: accurate float32 CE, targeted correct-class derivative repair, and per-example zero-sum projection. Process available specificity endpoints independently of any still-unavailable stock endpoint.

Use the existing fixed-logit reference and feature/classifier mean measurements. For each specificity endpoint, additionally evaluate its actual training backward rule on exactly the same float32 logits. For targeted repair and row projection, compare the resulting logit and parameter gradients with the analytic float64 derivative cast to float32 through the original network. Keep these actual-rule errors separate from hypothetical stock-CE errors at the same frozen state. Require step 20000 in every checkpoint and retain its SHA256 hash.

Compare full-grid feature-mean norm, mean power, mean cosine, classifier-mean norm, and sampled accuracy. These are downstream state observations. The independently collected dense trajectories supply intervention success or failure through the full interval; endpoint properties do not identify a unique mediator. Additional long runs are outside this diagnostic panel.
