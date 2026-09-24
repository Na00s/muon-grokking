# Four targeted manuscript corrections

All four authorized corrections are implemented. The manuscript remains nine main-text pages and 39 pages total. Existing experimental records and figure artwork are unchanged.

1. Regenerated Table 21 from the same 41 checkpoints with native full-grid accuracy at least 95% used in Section 7 and Appendix I. Both accuracy columns now use two decimal places. The overall intermediate maximum remains 17.97%. The selected depth-4 Stable Muon maximum is 17.58%; the former 18.1% entry comes from the 210,000-step checkpoint with 92.41% native accuracy, which lies outside the stated subset. Appendix I retains and explains that observation. The corrected table also resolves the depth-2 AdamW range and final-column rounding discrepancies. The source includes a portable generator and exact source/checkpoint range audit.
2. Figure 4 now refers to embedding and readout parameter subsets. Table 2 explicitly identifies the historical shared AdamW group and points to Appendix A.1.
3. Table 6 states that the paired CPU runs have identical logged pre-freeze metrics. It points to the historical provenance account without attributing the unresolved divergence to backend nondeterminism. The reproducibility statement scopes shared-state checks to the matched depth-4 freeze and fresh arithmetic/LR continuations.
4. Renamed Section 7 to “Depth, stability, and final-readout compatibility.”

Only PDF text on pages 4, 7, 9, 13, 15, 20 and 21 changes. All seven pages were inspected after rebuilding; no overflow, clipping or unresolved references remains. The editable source ZIP is rebuilt independently and its PDF text and decoded page streams are compared with the delivered PDF.

No new training or decoder fitting was required. Prior studies, raw measurements and checkpoints are preserved.
