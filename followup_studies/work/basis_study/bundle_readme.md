# Basis-change study: evidence and reproduction

Start with `report.md`. `paper_revision.md` contains suggested manuscript wording. `mathematical_scope.md` states the exact hypotheses and algebraic identification limits. Figures are available as PNG and PDF under `figures/`.

The study adds three fresh seeds under unchanged settings and analyzes five seeds in total. It includes 52,698 fresh baseline updates, 18 continuations totaling 9,000 updates, 262 instantaneous interventions, 20 geometry comparisons, and 21 bilinear comparisons. Checkpoint pairs within the same seed are dependent observations.

## Layout

- `work/basis_study/`: all new scripts, protocols, numerical results, raw feature arrays, maps, readouts, trajectories, selected checkpoints, test logs, and audit records.
- `work/experiments/`: previous study helpers and selected source states/results used by this study.
- `work/muon-grokking/`: unchanged source code from commit `6d64a981af75f1300d9060109e81552d48a81360`.
- `manifest.json`: SHA-256 and byte size for every packaged file, excluding the manifest itself.

The original repository is https://github.com/Na00s/muon-grokking. The package is an independent research supplement. It does not edit or publish the original paper or repository.

## Runtime

The experiments ran on CPU with Python 3.14.6, PyTorch 2.13.0, NumPy 2.5.2, SciPy 1.18.0, and Matplotlib 3.11.1. Use a Python environment providing these packages. Float32, optimizer groups, and saved optimizer states are part of the experimental specification. A different arithmetic backend can alter the training trajectory.

Run commands from this extracted directory. Preserve the sibling `basis_study`, `experiments`, and `muon-grokking` directories under `work/`. JSON workspace paths have been normalized relative to this directory. Checkpoints are byte-for-byte copies of the original saved tensors and may retain original provenance paths inside their metadata.

```sh
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=work/plot_cache
```

## Verify the mathematical and architecture checks

```sh
python -m unittest discover -s work/basis_study -p 'test_*.py' -v
```

The 28 tests cover exact planted maps, train/held-out separation, column-space obstructions, rank sensitivity, functional and bilinear identities, task-only nonidentification, and full-network gauge transformations. `runner_verification.json`, `intervention_final_audit.json`, and the fresh intervention summary contain separate saved-state replay and audit results.

## Recompute analyses without training

Use a fresh output directory for any new result. The geometry and bilinear scripts accept the recorded feature arrays directly:

```sh
python work/basis_study/geometry.py --features work/basis_study/seed2_adjacent_event/features.npz --out work/reanalysis/seed2_geometry
python work/basis_study/bilinear_diagnostic.py --features work/basis_study/seed2_adjacent_event/features.npz --out work/reanalysis/bilinear
python work/basis_study/task_subspace.py --pair work/basis_study/seed2_adjacent_event --out work/reanalysis/task_subspace
python work/basis_study/model_gauge.py --skip-planted --pair work/basis_study/seed2_adjacent_event --out work/reanalysis/model_gauge
python work/basis_study/run_instant_interventions.py --pair-dir work/basis_study/seed2_adjacent_event --out work/reanalysis/instant --name seed2_adjacent
```

The complete probe reanalysis is:

```sh
python work/experiments/run_whitened_probe.py --features work/basis_study/seed2_first_event/features.npz --out work/reanalysis/decoder_whitened.json --max-iter 500 --refit-max-iter 1000
```

`summarize_study.py` and `make_figures.py` regenerate the report tables and figures from the packaged results. These scripts write their outputs inside `work/basis_study/`.

## Reproduce fresh baseline training

The three seeds were specified before running. Each has a fixed 30,000-update cap, dense training-failure monitoring after confirmed grokking, and a 200-update follow-up after first joint train/test failure.

```sh
python work/basis_study/train_capture.py --seed 1 --out work/reproduction/seed1 --max-steps 30000 --follow-steps 200
python work/basis_study/train_capture.py --seed 2 --out work/reproduction/seed2 --max-steps 30000 --follow-steps 200
python work/basis_study/train_capture.py --seed 3 --out work/reproduction/seed3 --max-steps 30000 --follow-steps 200
```

The saved initial, healthy, preceding, collapse, peak, and final states include every optimizer group and RNG state. Periodic baseline checkpoints are included when referenced by an analysis; the complete trajectories and rerunnable training code are provided for the remaining intervals.

## Key evidence files

- `work/basis_study/results_overview.json` and `.csv`: five-seed summary and counts.
- `work/basis_study/geometry_summary.csv`: all 20 geometry pairs.
- `work/basis_study/bilinear/summary.csv`: all 21 exact decompositions.
- `work/basis_study/intervention_instant_table.csv`: all 262 instantaneous evaluations.
- `work/basis_study/intervention_continuation_table.csv`: all 18 continuing-training branches.
- `work/basis_study/fresh_task_and_gauge_summary.json`: class transfer and full-network controls for fresh seeds.
- `work/basis_study/protocol.md`: original design, declared adaptations, and interpretation requirements.

Plots and tables report held-out examples from a finite modular-addition domain. Training seeds, rather than individual examples or checkpoint snapshots, are the appropriate units for broad replication claims.
