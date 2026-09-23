# Completed follow-up studies

This package contains the completed initialization/memorization decoder controls, control-provenance, speed, event and feature-mean audits, and accurate-CE RMS replication and localization. It is an overlay for the full reviewer repository: merge its `followup_studies` directory into the existing reviewer snapshot. The original architecture, data, optimizer implementations, historical results and shared checkpoints remain in that snapshot.

`packaging_manifest.json` maps every original study artifact to its supplied SHA-256 and records the six portability edits. `packaging_verification.json` records numerical preservation and the final metadata scan. Model, optimizer and available RNG tensor values were compared recursively before and after checkpoint metadata sanitization. Numerical arrays are byte-identical. Every numeric JSON field and numeric CSV cell is preserved.

The original experiment registration and audit hashes remain in the historical protocols and manifests. Anonymization changes identifying provenance strings and checkpoint ZIP metadata, so those historical byte hashes refer to the original execution artifacts. Use the distribution manifest to verify the supplied bytes. `python3 verify_distribution.py` verifies this package's contents without modifying files. Original script versions for the documented portability edits are in each study's `source_history` directory; these are provenance references. Use the runnable copies in the study root.

## Decoder controls

The 20-row table in `decoder_controls/summary.csv` compares initialization, sustained memorization, the existing healthy reference and the existing failed state for the same five trajectories. Ten new selected fits and their unregularized sensitivity fits are complete. The original ten reference/failed results are retained. `verification.json` records five exact early replays and 40 independent saved-coefficient prediction checks. `protocol.json` contains the fixed selection, validation indices, solver budgets and source hashes.

To run a fresh reproduction, use a new sibling directory under `followup_studies/work`. This preserves the delivered records and registers the actual hashes of the reviewer copies. Run these commands from the full reviewer repository root, in an environment matching the versions and CPU numerical settings recorded in the protocol. Exact replay checks can expose backend or numerical-library differences.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=python3
DECODER_SOURCE=followup_studies/work/decoder_controls
DECODER_REPLAY=followup_studies/work/decoder_controls_reproduction
mkdir "$DECODER_REPLAY"
cp "$DECODER_SOURCE/run_controls.py" "$DECODER_REPLAY/"
cp "$DECODER_SOURCE/verify_controls.py" "$DECODER_REPLAY/"
cp "$DECODER_SOURCE/build_tables.py" "$DECODER_REPLAY/"
"$PYTHON" "$DECODER_REPLAY/run_controls.py" register
for seed in 0 1 2 3 4; do
  "$PYTHON" "$DECODER_REPLAY/run_controls.py" prepare --seed "$seed"
  "$PYTHON" "$DECODER_REPLAY/run_controls.py" fit --seed "$seed"
done
"$PYTHON" "$DECODER_REPLAY/run_controls.py" summarize
"$PYTHON" "$DECODER_REPLAY/verify_controls.py"
"$PYTHON" "$DECODER_REPLAY/build_tables.py"
```

The executable can be selected through `PYTHON`. The recorded environment uses Python 3.14.6, PyTorch 2.13.0, NumPy 2.5.2, SciPy 1.18.0 and `threadpoolctl`. The early-control fits, source checkpoints, features, fitted coefficients and full convergence reports are all included.

## Control provenance and feature-mean audit

The report identifies the two Table 2 controls and four observations around steps 44,700 to 44,710 with their distinct trajectories and schedules. Their recorded statistics are retained. The controlled three-program replay establishes implementation parity on an explicitly adapted CPU checkpoint; the historical step-44,000 checkpoint and runtime manifests remain unavailable. The raw replay CSVs, source checkpoint, branch checkpoints, final RNG states and comparisons are included.

The feature-mean figure uses 88 archived checkpoint measurements from the selected Muon and AdamW configurations. Its CSV and manifest identify the full-grid mean definition, source rows, sampling density, historical events and backend limitations. The plot PDF and PNG are unchanged from the completed audit.

The runnable scripts accept `MUON_SOURCE_REPO`; their default resolves the full reviewer repository root from this package layout. A fresh sibling directory allows regeneration without changing delivered records:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=python3
CONTROL_SOURCE=followup_studies/work/control_provenance
CONTROL_REPLAY=followup_studies/work/control_provenance_reproduction
mkdir "$CONTROL_REPLAY"
cp "$CONTROL_SOURCE"/*.py "$CONTROL_REPLAY/"
"$PYTHON" "$CONTROL_REPLAY/audit_records.py"
"$PYTHON" "$CONTROL_REPLAY/replay_control_implementations.py"
MPLCONFIGDIR="$CONTROL_REPLAY/mplcache" "$PYTHON" "$CONTROL_REPLAY/build_feature_mean_timeline.py"
"$PYTHON" "$CONTROL_REPLAY/verify_feature_mean_timeline.py"
```

The parity replay performs three 2,000-update CPU branches and refuses to replace existing checkpoint folders. The original saved step-16,000 checkpoint under `followup_studies/work/experiments/seed0_original` is its input. The timeline verifier uses the corresponding saved step-6,000 checkpoint. Shared implementations and archived source CSVs resolve from the reviewer repository root. All these inputs are part of the full supporting repository.

## Accurate-CE RMS replication and localization

The RMS study includes the separately identified historical CPU pilot and four prospective CPU seeds. All four prospective runs reached a captured joint failure after grokking confirmation, and the same event analysis is complete for all five events. The package includes 90 checkpoints, 20 numerical-array archives, exact-state replay checks, derivative references, all eight parameter-group swaps per event, selected and unregularized decoder reports, frozen source snapshots and per-seed monitoring records. `analysis_environment.json` records the verified analysis libraries.

The study-specific [execution guide](followup_studies/work/rms_replication/DISTRIBUTION_EXECUTION.md) supplies portable commands that create a fresh sibling reproduction directory. The distribution-only `prepare_reproduction.py` wrapper copies sources and records their actual supplied hashes before any new fitting. It leaves the delivered scientific records intact. The supplementary final audits retain their original source hashes and historical runtime context. Their completed results are provided alongside the distribution preservation checks; rerunning hash-bound historical audits on sanitized copies requires adapting the original-byte hash checks to this package's correspondence.

The independent RMS packaging audit compared all 90 supplied checkpoints, all 20 numerical-array archives and six summary tables against the canonical records. Its full result is embedded in `packaging_verification.json`; its original-layout source is preserved in `packaging_audits/rms_preservation_check_original_layout.py`. That source is a provenance record for the completed comparison and requires both original and supplied trees if repeated.
