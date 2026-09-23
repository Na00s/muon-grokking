# Review of final integration and publication preparation

The four preparation scripts were reviewed against the actual `run_long_horizon.py`, `run_suite.py`, `summarize_and_verify.py`, `verify_runner.py`, artifact downloader, figure builder, and current evidence schemas. Training, canonical manuscript files, and publication outputs were untouched. Only the four requested preparation scripts and this audit's fixture/evidence files were edited.

## Changes

- `incorporate_completed_horizons.py`: validates all audit checks, the exact five addition seeds plus subtraction seed 0, 70,000 updates per extension, and the inherited correction start. Every text anchor must match exactly once. All three manuscript/figure-script changes are staged before the first write, so a late missing anchor cannot leave partial integration. Repeated integration stops before mutation. Describes the verified random state as PyTorch's and the six-plus-six comparison as a resumed continuation; the verifier restores its middle snapshot in memory rather than saving that midpoint to disk.
- `write_completed_docs.py`: validates completed status, all audit checks, exact branch identities and correction starts, and the stated perfect addition endpoints. Makes the readme notices repeatable. Replaces the downward-rounded claim “within 0.324421” with an approximately stated observed maximum. Describes the random-state check precisely. Creates the release-note parent directory when needed.
- `verify_revision.py`: checks all individual check records, independently computes the coverage probability using both exact binomial coefficients and an exact rational product, compares the published bound, verifies the audited control scripts/results have unchanged hashes, and checks the exact Section 6 wording requested by the user. Checks the expected six branch identities, zero-event interpretation, perfect addition endpoints, current runner/protocol hashes, updated figure label and source-data hashes. The final verification now records hashes for the summary, runner, protocol, and checkpoint manifest so packaging can enforce freshness.
- `package_revision.py`: rejects a stale reviewed PDF, source tree, control audit, long-horizon audit, summary, runner, protocol, or checkpoint manifest before copying. Verifies checkpoint binaries and Fourier publication-manifest inputs/outputs, and checks the published-path reproduction result. Binary inventory is derived from current experiment sources, avoiding stale destination-only binaries. Archives the visual-review record and the four preparation scripts. Creates necessary publication directories.

## Verification

`check_preparation_scripts.py` executes integration and documentation only on temporary copies with explicitly synthetic completed records. Five fixture checks pass:

1. A deliberately missing late figure anchor leaves all three input files unchanged.
2. Synthetic completed records produce the expected new table and escaped 100,000-step plotting label.
3. Repeated integration rejects without changing files.
4. Running the documentation writer twice produces identical files without duplicate notices.
5. Packaging a changed reviewed PDF rejects before modifying publication files.

The 14 currently runnable pre-horizon checks in `verify_revision.py` pass, including the exact probability identity and all independently audited control hashes. The requested Section 6 wording is present. The Fourier publication manifest currently matches every referenced file. All four preparation scripts parse successfully.

The initial concern about the escaped plot-label newline was resolved by AST inspection: the existing label replacement already matched once. The explicit anchor check now guards that replacement too. No newline bug was present.

The complete final verifier, actual integration, and actual packager were deliberately not run before training completion. Their remaining gates require completed real records and a reviewed final PDF. The verifier uses `pypdf`; run it in the existing document/PDF environment, since the research training environment does not include that package.

## Required order and scope

After all runs complete, run the long-horizon summarizer/verifier, incorporate the actual completed outcomes, write the release documentation, rebuild the figures against the current experiment workspace, compile and review the PDF, run final verification, then package. The figure builder must receive a workspace with the completed `work/long_horizon/summary.json`; before packaging copies it into `followup_studies`, use the original workspace explicitly or synchronize the completed results first.

The current templates intentionally require zero joint events and perfect final addition accuracy. If the actual completed records differ, they stop before stating those outcomes. Such an outcome needs a corresponding scientific revision from the saved event/metrics. The finite-horizon and test-only-excursion limitations are retained. The separate packaged-archive verifier remains owned by root and was not edited.
