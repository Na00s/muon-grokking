# Selective revision and audit

## Fourier controls and extended horizons

The latest changes are documented in [revision_addendum.md](revision_addendum.md). They add Appendix K and Figure 9, move the mean-update mechanism into Section 4.4 with Table 3, extend corrected-loss evidence to step 100,000, simplify the abstract/contribution list, and apply the four requested wording changes. The original numerical results are retained. `fourier_revision.patch` compares against commit ed5fa274735994a14d047ec73898b28d5ea1ed9f.

## Prior revision and audit

The paper incorporates the completed follow-up experiments while retaining the original experimental results and observations. The latest audit verifies the new experiments and reviews claim strength, interpretation, and research prose throughout the affected sections.

- Abstract/introduction: distinguish test instability from joint train/test failures; state decoder recovery, measured basis rejection, numerical contribution, and architecture dependence.
- Section 4: preserve matched freezes and observed non-finite ablations; add adjacent swaps, probes, arithmetic controls, and finite-step diagnostics.
- Sections 5-7 and original appendix: clarify Fourier/DC conventions, evaluation domain, margin competitor, dispersion definitions, and the scope of stage interventions. Original figure PDFs and numerical table cells are preserved.
- Related work/discussion: credit existing functional Fourier methods and numerical corrections; state the remaining limits without weakening the reported observations.
- Appendix J: specify training-only regularization coordinates, monitoring criteria, derivative-repair precision, and the exact cohort behind the task-only example.

New Figure 4 summarizes the main causal evidence. New Figure 8 shows the RMS architecture boundary. All plotted values in these figures passed the new-experiment audit.

See `audit_report.md` for scope and results, `audit/followup/` for the 604-check ledger and portable scripts, and `audit/literature_review.md` for the primary-source citation review. `targeted_revision.patch` compares the original submitted LaTeX with the current source; `audit_revision.patch` compares the previously revised version with the audited version.
