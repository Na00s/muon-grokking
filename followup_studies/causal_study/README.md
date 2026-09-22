# Causal study: reproduction and artifacts

Start with [report.md](report.md), [paper_revision.md](paper_revision.md), and [verification.json](verification.json). The study completed 364,302 training updates under written finite endpoints. Main and specificity tables are in [results.json](results.json) and [branches.csv](branches.csv).

## Restore all required checkpoints

From the repository root:

```sh
python followup_studies/download_artifacts.py
python followup_studies/download_artifacts.py --manifest CAUSAL_ARTIFACTS.json
cd followup_studies
```

The first command restores the earlier source trajectories and basis artifacts. The second restores all 521 retained causal-study binaries, including every saved model, optimizer, and RNG state. The causal archive is 1.068 GB compressed. Both archive and per-file hashes are checked, and changed local files are preserved. From `followup_studies`, check the two collections with `python download_artifacts.py --verify-only` and `python download_artifacts.py --manifest CAUSAL_ARTIFACTS.json --verify-only`. A locally downloaded archive can be supplied with `--archive /path/to/archive.tar.gz`.

## Runtime and code

CPU, single-threaded PyTorch 2.13.0, Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, and Matplotlib 3.11.1 were used. Changing the arithmetic backend can change trajectories. Original model and optimizer source is frozen under `work/muon-grokking/` at commit `6d64a981af75f1300d9060109e81552d48a81360`.

```sh
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=work/plot_cache
python -m unittest discover -s work/causal_study -p 'test_*.py' -v
python work/causal_study/diagnostics_test.py
```

`work/causal_study/` contains all written protocols, runner source, machine-readable results, plots, test logs, exact replay records, and restored checkpoint files. The main fixed job list and source hashes are in `main_suite_plan.json`. Additional projection and generality protocols are `projection_protocol.json` and `generality/protocol.md`. `claim_adjudication.md` records the claim-specific decision rules.

Checkpoint and Python source bytes are preserved. Text publication normalizes workspace paths and CSV newlines. The source-qualification manifest retains both original and normalized CSV hashes. `runner_versions/` preserves the exact early runner source used before targeted derivative arms were added; full training uses the version hashed in the fixed job plan.

## Repeat a matched continuation

Choose a fresh output directory. The following starts from the exact saved early post-grokking state:

```sh
python work/causal_study/run_dense_arithmetic.py --checkpoint work/experiments/seed0_original/step_006000.pt --out work/reproduction/seed0_accurate --arm stable32 --end-step 30000
python work/causal_study/run_dense_arithmetic.py --checkpoint work/experiments/seed0_original/step_015000.pt --out work/reproduction/seed0_target --arm target_repair --end-step 20000
python work/causal_study/run_dense_arithmetic.py --checkpoint work/experiments/seed0_original/step_015000.pt --out work/reproduction/seed0_projection --arm row_projection --end-step 20000
```

Seeds 1–3 source checkpoints live under `work/basis_study/seedN_baseline/`; seeds 0 and 4 use `work/experiments/seedN_original/`. The job plan lists every original extension and paired source. `original32` selects the stock loss, `stable32` accurately evaluates the same CE objective, `target_repair` changes only the correct-class derivative, and `row_projection` restores the per-example zero class sum. Every arm retains the original float32 model and Muon arithmetic.

## Repeat the operation and architecture controls

```sh
python work/causal_study/generality_runner.py --operation subtraction --normalization none --arithmetic stock --steps 30000 --out work/reproduction/subtraction_stock
python work/causal_study/generality_runner.py --operation subtraction --normalization none --arithmetic accurate --steps 30000 --out work/reproduction/subtraction_accurate
python work/causal_study/generality_runner.py --operation addition --normalization rms --arithmetic stock --steps 30000 --out work/reproduction/rms_stock
python work/causal_study/generality_runner.py --operation addition --normalization rms --arithmetic accurate --steps 30000 --out work/reproduction/rms_accurate
```

RMS checkpoints share parameter keys with the original model but require the RMS forward pass. Load them through `generality_runner.make_model_optimizers(seed, 'rms', checkpoint)`. Subtraction checkpoints require subtraction labels. `generality_analyze.py` handles this metadata correctly. Loading these checkpoints through the original-addition-only helper would analyze a different model or task.

## Recompute analysis

```sh
python work/causal_study/aggregate.py
python work/causal_study/verify_completed.py
python work/causal_study/make_figures.py
python work/causal_study/write_report.py
python work/causal_study/diagnostics_branch_compare.py --checkpoint seed0_accurate=work/causal_study/main_runs/seed0_accurate6000/final.pt --out work/reanalysis/seed0_accurate.json
```

Aggregation validates every recorded main/specificity state, final horizon, and possible joint-failure trigger. The completion audit reloads every final model using the correct task and architecture, checks its saved accuracies, and verifies exact branch-start model/optimizer/RNG states against their sources. The audit and report commands regenerate their outputs under `followup_studies/outputs/causal_study/`, including `verification.json`, `report.md`, `paper_revision.md`, tables, and figures. The published reports remain under `followup_studies/causal_study/`. Generality traces retain every tenth state and every joint failure, while the runner checks training every update. Exact model/optimizer/RNG replay records and source-state hashes accompany the event diagnostics. The runner's generic metadata endpoint string says 30000 even for the specificity arms; their protocols, actual execution end_step, and analysis use 20000. This wording issue is recorded in the final verification.

The original PDF and implementation remain preserved. Earlier reports are retained as historical assessments; the completed causal report gives the current claim decisions.
