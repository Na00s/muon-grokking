# Accurate-CE RMS replication

The frozen study extends the historical CPU seed 0 RMS trajectory with seeds 1–4. The prospective runs stop at the first joint train/test accuracy below 90% after confirmed grokking, or 100,000 updates. The historical pilot is analyzed separately at its first captured failure.

The immutable training protocol is `protocol.json`. All source dependencies are snapshotted under `source_snapshot/`, with hashes in `source_manifest.json`. Model and Newton–Schulz arithmetic remain float32. The CPU backend, original optimizer implementations, seeded data split and gain-free RMS configuration are preserved.

Execution commands and worker PIDs are in `launch.json`. Worker logs are `runs/seedN.log`, per-state measurements are `runs/seedN/trajectory.csv`, and periodic recovery states include model, optimizer and CPU/Python/NumPy RNG states. No GPU RNG participates in CPU execution.

Event analysis uses `analyze_event.py`. It saves the same-logit derivative arrays, all eight embedding/hidden/readout swaps, exact replay evidence, normalized readout-input features, fixed training-only validation identities, and decoder diagnostics. `watch_events.py` starts these analyses as the prospective runs finish. It never restarts training.

After all work finishes:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /opt/homebrew/Caskroom/miniforge/base/envs/research/bin/python work/rms_replication/verify_and_summarize.py
/opt/homebrew/Caskroom/miniforge/base/envs/research/bin/python work/rms_replication/write_report.py
```

The completion audit independently recomputes per-state monitoring, stopping rules, endpoint metrics, stored derivative errors and predictions from the fitted decoders. Preserve interruption records and all existing raw data if recovery is needed.
