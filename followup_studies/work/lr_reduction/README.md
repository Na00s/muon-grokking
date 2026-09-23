# Post-grokking hidden learning-rate reduction

Five matched continuations restore the original stock-cross-entropy model, optimizer, and RNG states at step 6,000. Immediately after restoration, Muon's hidden learning rate changes from 0.03 to 0.003. Its momentum buffers are retained. Embeddings and readout remain trainable at learning rates 0.001 and 0.00025. All weight-decay coefficients and the stock cross-entropy implementation remain unchanged. Each continuation ends at step 100,000, for 470,000 production updates across five seeds.

`protocol.json` records the settings and source checkpoint hashes before production launch. `source_audit.json` checks checkpoint identity and the historical comparators. `runner_verification.json` checks exact restoration, twelve monitored versus manual updates for every seed, serialized continuation, diagnostic accuracy, and a synthetic train-trigger control. Short verification updates are separate from the five production continuations.

Training accuracy is recorded at every state from 6,000 through 100,000. Held-out accuracy is evaluated at both endpoints, on the 100-step grid, and whenever training accuracy is below 90%. The primary outcome is the number of seeds with a joint train/test accuracy below 90%. Secondary outcomes are the minimum sampled test accuracy, the number of evaluations below 95%, and final test accuracy. Grid-only counts are retained alongside counts over all measured states.

Every 1,000 steps, diagnostics record training-set and full-grid feature-mean norms, plus stock-CE logit-derivative error against accurate float64 derivatives at the same float32 logits. These are observational measurements. The runner's updates match unmonitored source updates bitwise in the short verification runs.

`run_suite.py` launches five CPU workers with one Torch thread each and runs `summarize_and_verify.py` after training. The suite refuses duplicate launches. Its records are `launch.json`, `progress.json`, `training_completion.json`, and `completion.json`; per-seed records are under `runs/seed0` through `runs/seed4`. Retain all existing logs and checkpoints when investigating an interruption.

For a fresh output directory, a single branch can be reproduced with:

```sh
python run_lr_reduction.py \
  --checkpoint ../experiments/seed0_original/step_006000.pt \
  --out /path/to/fresh/seed0 \
  --end-step 100000
```

The binary incidence comparator is 5/5 original failures versus 0/5 corrected-loss failures through the registered horizons. Historical original monitoring differs, so event counts and durations must not be compared with the original runs. Ordinary learning-rate reduction also scales the decoupled weight-decay step. This experiment evaluates stabilization under the stated intervention and horizon.

The paper and submission packages are frozen. This directory is separate from the manuscript. Completed results will be reported for discussion before any paper revision.
