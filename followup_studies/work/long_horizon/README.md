# Accurate-cross-entropy controls through 100,000 steps

This study extends the five corrected addition branches and the corrected subtraction seed-0 branch from their saved step-30,000 states through step 100,000. The horizon matches the original depth-1 experiments. Each extension applies 70,000 updates, for 420,000 new training updates across six branches.

Addition inherits accurate cross-entropy from the matched step-6,000 intervention. Subtraction inherits accurate cross-entropy from initialization. The unchanged model, all three optimizer states, and the Torch random state are restored from each source checkpoint. All training arithmetic and parameter storage remain float32 under the original CPU implementation.

`protocol.json` was written before the extensions began. It records the six jobs, source hashes, evidence for the original horizons, monitoring schedule, stopping rules, and scope. `runner_verification.json` records bitwise equality between uninterrupted training, a saved-state continuation, and the monitored runner for all six sources.

## Monitoring and retained evidence

Training accuracy is checked and logged at every state. Test accuracy is evaluated every 100 global steps, at the source and final states, and at every state where training accuracy is below 90%. This detects every joint train/test event below 90%. Brief test-only excursions between scheduled evaluations can remain undetected.

Each branch retains the start and final states, a 10,000-step checkpoint grid, the first joint event and its preceding state if one occurs, the latest jointly healthy state before that event, and the minimum measured test state. Nonfinite states raise an explicit failure and save a checkpoint. The runner never truncates a requested horizon silently.

`failure_verification.json` records a deliberate nonfinite-state test of the runner. Its files are under `verification_runs/`; the six scientific branches are under `runs/`.

`runs/` contains every training row, progress records, endpoint summaries, arithmetic diagnostics on the checkpoint grid, and raw logs. The summary combines the existing prefix with the extension and counts the shared step-30,000 boundary once. Every conclusion concerns the selected seeds through a finite horizon.

## Reproduction

Use the research environment and unchanged sibling `experiments`, `causal_study`, and `muon-grokking` code supplied with the experiment release. Restore the source checkpoints before running. From the directory that contains `work/`, one extension can be reproduced into a fresh directory with:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python work/long_horizon/run_long_horizon.py \
  --checkpoint work/causal_study/main_runs/seed0_accurate6000/final.pt \
  --out work/reproduction/addition_seed0_100000 \
  --operation addition --end-step 100000
```

Use seeds 1 through 4 in the checkpoint path for the other addition branches. The subtraction source is `work/causal_study/generality/subtraction_accurate/final.pt`, with `--operation subtraction`.

`run_suite.py` is the archived six-worker launch. It refuses to overwrite an existing protocol or any branch directory and expects the original CSV horizon logs beside the original source repository. The single-branch command above is the portable way to launch new training from the published layout. `verify_runner.py` verifies the initial restoration and short continuations in fresh verification directories. `summarize_and_verify.py` audits the complete trajectories, replays final metrics from saved checkpoints, checks source hashes and restored states, and writes the combined summary and checkpoint manifest. When the original runtime paths in `protocol.json` are unavailable, the auditor resolves the source states under the sibling `causal_study` directory and verifies their original hashes. These verification scripts preserve the experimental artifacts.

## Scope

The matched stock addition trajectories already contain joint failures before step 30,000. Their binary event status persists at the longer horizon. Detailed event counts, durations, and minima are not compared across their mixed historical monitoring and the dense corrected branches. The normalized model already provides a distinct accurate-loss failure and is reported separately.
