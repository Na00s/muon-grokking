# New-experiment audit

Scope: independently verify the added experiments, their results, and their claims. Original experimental results and observations are retained. Their interpretation and positioning are reviewed in the manuscript.

`followup/claim_ledger.json` contains 598 computed checks and six explicitly labeled manual checks. `followup/checkpoint_replay_audit.json` records 28 matched-state comparisons and seven repeated update-direction panels. `input_source/` is the immutable pre-audit text snapshot used for ledger locations.

From the repository root, restore the previously released binary evidence if needed:

```sh
python followup_studies/download_artifacts.py
python followup_studies/download_artifacts.py --manifest CAUSAL_ARTIFACTS.json
```

Then use the research environment specified in `followup_studies/README.md`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/audit/followup/checkpoint_replays.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/audit/followup/audit_followup.py
```

The scripts reuse saved checkpoints and arrays. They do not repeat training. `MUON_AUDIT_REPO` and `MUON_AUDIT_EVIDENCE_ROOT` support a separate checkout or data location. Scripts write the audit outputs beside themselves. `literature_review.md` records the separate claim-attribution review against 19 primary sources.
