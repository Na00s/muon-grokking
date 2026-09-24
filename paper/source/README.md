# Revised submission source

Compile `main.tex` with Tectonic or a standard LaTeX/BibTeX installation. Figures, tables, bibliography, and the conference style are included. Experimental downloads are unnecessary for compiling the paper.

```sh
tectonic main.tex
```

The scientific main text occupies nine pages. The title and central abstract framing are preserved. Figure 1 opens with adjacent swaps, training-only decoders, and the arithmetic/LR comparison. The spectral comparison is Figure 3 in the later spectral discussion. Detailed projection analysis is retained in Appendix L; stagewise depth results remain in Appendix I. Appendix J includes arithmetic interventions, feature-mean timelines, early decoder controls, and the accurate-CE RMS replication.

All original measured results are retained. Table 2 now identifies the separate historical control continuations and their provenance limits. The four prospective RMS seeds are distinguished from the historical pilot. The change log and verification record in the full repository describe the scientific changes and checks.

## Figure and table reproduction

Run these commands from this source directory in the complete repository layout:

```sh
python figures/build_followup_figures.py --workspace ../../followup_studies --output .
python figures/build_fourier_control_figure.py --workspace ../../followup_studies --output .
python figures/build_rms_replication_figure.py --study ../../followup_studies/work/rms_replication --output .
python figures/build_rms_tables.py --study ../../followup_studies/work/rms_replication --output tables
python ../../followup_studies/work/decoder_controls/build_tables.py
MUON_SOURCE_REPO=../.. python figures/build_feature_mean_timeline.py
cp figures/feature_mean_timeline.pdf .
```

The decoder command regenerates its table within the study directory; copy `decoder_phase_controls_table.tex` to `tables/decoder-phase-controls.tex`. The mean-timeline commands regenerate the printed figure and its CSVs from the original archived measurements. The full study records preserve exact input hashes, fitted coefficients, raw measurements, protocols, and replay evidence. In a standalone source archive, pass the separately downloaded study paths explicitly.

The original scientific vector artwork is retained in `figures/legibility_originals/`. Typography and axis-label scripts modify labels while preserving plotted geometry. To reproduce the current percentage axes and final-head labels:

```sh
python figures/accuracy_axes_percent.py --font /path/to/LiberationSerif-Regular.ttf --bold-font /path/to/LiberationSerif-Bold.ttf
python figures/clarify_projection_depth_axes.py --font /path/to/LiberationSerif-Regular.ttf
```

These scripts require pypdf, pdfplumber, and reportlab. Their JSON provenance records list each label change and source/output hash. The current figures use “projection failure” and “projection masking.” Historical typography records describe their own earlier editing stage; the current manuscript verification records the final printed sizes and layout.

## Targeted consistency corrections

Table 21 uses the same 41 checkpoints with native full-grid accuracy at least 95% as Section 7. Both accuracy columns use two decimal places. Regenerate the table and its source audit from the repository root with:

```bash
python paper/source/tables/build_depth_readout_table.py --repo .
```

The accompanying `tables/depth_readout_ranges.json` lists the selected checkpoints, exact ranges, input hashes and excluded 210,000-step observation.
