# Resolution review of follow-up evidence

All requested follow-up corrections are present in the revised manuscript. The changes preserve the verified numerical results and state the experimental scope more precisely. No unresolved follow-up claim or numerical issue remains from this audit.

| Issue | Revised location | Resolution |
| --- | --- | --- |
| Original five-check criterion versus new six-check confirmation | `followup_appendix.tex:9` | Both criteria and reporting points are now explicit. The incorrect universal 500-update offset is removed. |
| Coordinate system for decoder regularization | `followup_appendix.tex:34` | The penalty is explicitly on the preconditioned coefficients. |
| Cohort behind 4.62% task-only-map example | `followup_appendix.tex:60` | The example identifies seed 4, steps 16,000 and 17,000, and distinguishes this pair from the adjacent failures. |
| Backward-repair reduction precision | `followup_appendix.tex:89` | Both class sums use float64, followed by a cast to float32. This matches `diagnostics_repairs.py`. |
| SGD comparison precision | `followup_appendix.tex:91` | The new range `(2.8–4.1) x 10^-16` agrees with the recorded extrema 2.814533392071649e-16 and 4.059831247850319e-16. |
| RMS generality scope | `followup_appendix.tex:108,136`, `main.tex:163` | The prose states one seed per operation/architecture condition and attributes the observed acute damage to the embedding update in those runs. |
| Mean-intervention interpretation | `followup_appendix.tex:101`, Discussion | Local sufficiency is retained. The text preserves the unresolved contribution of accumulated feature growth and optimizer history. |
| Figure-script default paths | `figures/build_followup_figures.py:29–38` | The default data root is the repository's `followup_studies`, the default output is paper source, and a standalone archive requires an explicit data root. |

I also read the revised abstract, introduction, the mechanism paragraph preceding Figure 4, Section 4.4, Figure 4's caption, and Discussion. The five selected trajectories, first versus later probe checkpoints, finite horizons, test-only monitoring limitation, and normalized counterexample remain clear. The prose uses concrete experimental statements. The revised Discussion avoids claiming either a unique mediator or a complete weight-level algorithm.

## Portable audit scripts

`audit_followup.py` and `checkpoint_replays.py` now support:

- `MUON_AUDIT_REPO`: a repository root containing `runs/` and `followup_studies/`.
- `MUON_AUDIT_EVIDENCE_ROOT`: the directory containing released `work/` evidence. Default: `<repository>/followup_studies`.
- Repository autodetection from script ancestors. The current workspace layout is also recognized through `<ancestor>/work/muon-grokking`.
- Immutable baseline locations under `paper/audit/input_source`, when that snapshot exists. The fallback is `paper/source`.
- Output files beside each script, so copying them to `paper/audit/followup/` needs no output-path edits.
- Relative metadata paths such as `work/basis_study/...`. Absolute workstation paths retained in original provenance are relocated by their `work/...` suffix beneath the selected evidence root. Missing files produce an explicit missing-artifact error rather than silently falling back to a different data tree.

The scalar audit was rerun against the published repository's `followup_studies` tree, with both binary bundles already restored. All 604 ledger entries are verified: 598 computed comparisons and six explicitly marked manual checks comprising four source-code monitoring inspections and two linear-algebra statements. There are no failures. The separate portable checkpoint audit also passes all 28 saved-state comparisons and reproduces every scalar in the seven direction/interpolation panels exactly.

The figure script was separately regenerated in a simulated repository using its defaults. Both new figures were produced, all 15 input hashes matched, and all six original figure PDFs were preserved. The source README's standalone command remains correct:

```sh
python figures/build_followup_figures.py --workspace /path/to/followup_studies --output .
```

## Lightweight packaging

Package these files under `paper/audit/followup/`:

- `audit_followup.py`
- `checkpoint_replays.py`
- `claim_ledger.json` and `claim_ledger.csv`
- `recomputed_datasets.json`
- `checkpoint_replay_audit.json`
- `evidence_manifest.json`
- `report.md`, `resolution_review.md`, and `audit_summary.json`

The two fragment scripts and `plot_smoke/` are construction/test intermediates and can be omitted. Feature arrays, decoder matrices, and checkpoints are already in the released binary bundles and should remain there. The ledger, diagnostics, and hashes are small text artifacts.

From a fresh repository checkout, restore the external evidence once:

```sh
python followup_studies/download_artifacts.py
python followup_studies/download_artifacts.py --manifest CAUSAL_ARTIFACTS.json
```

Then, using the recorded research runtime and single-threaded numerical libraries:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/audit/followup/checkpoint_replays.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/audit/followup/audit_followup.py
```

If data is stored outside the repository, set `MUON_AUDIT_EVIDENCE_ROOT` to the directory containing its `work/` tree. The audit uses saved model data and evaluations; it does not rerun the training horizons.
