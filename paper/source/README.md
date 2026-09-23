# Revised submission source

Compile `main.tex` with Tectonic or a standard LaTeX/BibTeX installation. Figures, tables, bibliography, and the conference style are included.

```sh
tectonic main.tex
```

The revision preserves the title, anonymous authors, section order, original figure data, and the numerical cells of the 17 original appendix table files. Figure 4 uses the labels "projection failure" and "projection masking"; its original PDF is retained as `figures/figure5_modes_original.pdf`. All six original figures have larger labels, and their pre-enlargement assets are retained in `figures/legibility_originals/`. Section 4.4 contains the mean-dependent update equation and Table 3. The causal follow-up figure reports the swaps, decoders, and original-CE, reduced-hidden-LR, and corrected-loss outcomes through step 100,000. Appendix J documents these experiments; Appendix K gives the Fourier averaging proof and memorizer controls, including Figure 9.

The scientific main text occupies nine pages. The reproducibility and required AI use statements are separate and excluded from the [ICLR 2027 initial-submission limit](https://iclr.cc/Conferences/2027/AuthorGuidelines). The quiet-window and Fourier-grid figures appear in Appendices C and G. Included template files are byte-identical to the official ICLR 2027 counterparts despite their retained legacy filenames.

The figure scripts use saved experimental results and record their input/output hashes. From a released `followup_studies` directory containing `work/`:

```sh
python figures/build_followup_figures.py --workspace /path/to/followup_studies --output .
python figures/build_fourier_control_figure.py --workspace /path/to/followup_studies --output .
```

The scripts resolve the data root automatically in the repository layout. In a standalone source archive, pass it explicitly. This source archive compiles without downloading experimental binaries.

To reproduce the two Figure 4 annotation changes, run `python figures/relabel_projection_figure.py --font /path/to/LiberationSerif-Regular.ttf`. This requires pypdf and reportlab; the script preserves the original vector geometry and records the source, font, and output hashes.

The three `figures/enlarge_*_labels.py` scripts reproduce the typography changes from the retained assets. Use each script's `--help` for font and output arguments. Their provenance records describe the figure-editing stage; `paper/verification.json` in the full repository records the final manuscript layout and printed label sizes. Plot data and numeric labels are preserved.

This public source package links to the public code repository in the reproducibility statement. All scientific content and figures match the current manuscript.
