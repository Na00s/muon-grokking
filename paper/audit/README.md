# Verification records

These records contain numerical recomputations, saved-state replay comparisons, Fourier controls, and source hashes for the published study artifacts. The original reports and execution protocols are supplied under `followup_studies/`.

`followup/claim_ledger.json` records computed checks and their evidence paths. `followup/checkpoint_replay_audit.json` records matched-state comparisons and repeated update-direction panels. The numerical checks and scientific inputs remain available. Recorded source identifiers and line numbers accompany the numeric checks without requiring manuscript text.

Restore saved arrays and checkpoints using the appropriate manifest before repeating computational analyses:

```sh
python followup_studies/download_artifacts.py --manifest ARTIFACTS.json --verify-only
python followup_studies/download_artifacts.py --manifest CAUSAL_ARTIFACTS.json --verify-only
python followup_studies/download_artifacts.py --manifest FOURIER_HORIZON_ARTIFACTS.json --verify-only
```

The study-specific guides record the runtime and exact-state requirements.
