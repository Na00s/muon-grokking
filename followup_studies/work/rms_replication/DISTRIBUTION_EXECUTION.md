# RMS replication in the reviewer distribution

The supplied records contain the completed historical CPU pilot analysis and four prospective CPU runs, all stopping at their first captured post-confirmation joint failure. All 90 checkpoint files, the frozen source snapshots, raw trajectories, derivative arrays, eight-way swaps, decoder features and coefficients, convergence diagnostics, replay evidence and final audits are included. The historical source and protocol hashes remain as recorded before anonymization. The package-level manifest maps them to the supplied files; identifying provenance strings have been removed while numerical arrays and state values are preserved.

`launch.json` records the original commands and runtime settings with anonymized paths. The original interpreter location describes that execution environment. For a portable, non-overwriting reproduction, use the following commands from the full reviewer repository root. Select an environment matching `protocol.json`: CPU, Python 3.14.6, PyTorch 2.13.0, NumPy 2.5.2, SciPy 1.18.0, with one numerical thread. Hardware or library differences may fail the exact replay checks.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=python3
RMS_SOURCE=followup_studies/work/rms_replication
RMS_REPLAY=followup_studies/work/rms_replication_reproduction
"$PYTHON" "$RMS_SOURCE/prepare_reproduction.py" --out "$RMS_REPLAY"
"$PYTHON" "$RMS_REPLAY/verify_runner.py"
"$PYTHON" "$RMS_REPLAY/freeze_protocol.py"
"$PYTHON" "$RMS_REPLAY/launch_suite.py"
"$PYTHON" "$RMS_REPLAY/analyze_event.py" --path "$RMS_REPLAY/historical_reference" --out "$RMS_REPLAY/analyses/historical_seed0" --cohort historical_cpu_pilot
"$PYTHON" "$RMS_REPLAY/watch_events.py"
"$PYTHON" "$RMS_REPLAY/verify_and_summarize.py"
"$PYTHON" "$RMS_REPLAY/write_report.py"
```

The preparation wrapper copies the supplied source snapshots without changing their implementation, creates a fresh source-hash manifest, and records the analysis scripts before any fitting. It performs no training or decoder fit. The subsequent source-verification and experiment commands perform their declared updates and analyses. The fresh directory's placement as a sibling under `followup_studies/work` lets the original source-verification script reach `causal_study/generality/rms_accurate`; those historical shared inputs are included in the full reviewer repository.

`launch_suite.py` launches the four registered seeds and waits for them. The historical event is analyzed with its separate cohort label. `watch_events.py` analyzes each prospective captured event once, including all eight swaps and the same decoder procedure. The fresh registration reflects the hashes of the actual supplied copies. Existing historical hash-bound audit scripts can reject sanitized metadata when run directly against delivered records; the distribution integrity verifier and this fresh registration workflow keep those two provenance layers explicit.

The added wrapper is a distribution-only utility. It is listed separately in `packaging_manifest.json`. Original experimental scripts retain their original-to-supplied correspondence, including any identifying-string sanitation. The wrapper does not change optimizer routing, arithmetic, initialization, seeds, monitoring, stopping rules, decoder selection or solver budgets.

`independent_final_audit.py` is retained as the historical second-audit implementation. It also checks the original dual-tree canonical layout and original checkpoint hashes, so direct standalone invocation on this anonymized overlay requires a separate adapted audit. It is intentionally absent from the fresh reproduction command list above. Its completed results remain in `independent_final_verification.json`; package-level tensor/state preservation is independently recorded in `packaging_verification.json`.
