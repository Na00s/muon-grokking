# Revised submission source

Compile `main.tex` from this directory using Tectonic, or a standard LaTeX/BibTeX installation. All figure PDFs, table sources, bibliography files, and the supplied conference style are included.

```sh
tectonic main.tex
```

The revision preserves the supplied title, anonymous author block, conference template, six original figure PDFs, and 17 original table files. The T1 font-encoding declaration preserves the requested Times face with Tectonic. New material is in Section 4.4, Figure 4, and Appendix J (including Figure 8). Existing causal and spectral interpretations are corrected where affected by these results.

The two new figures were generated from recorded measurements. `figures/followup_figure_provenance.json` contains every plotted value and the hashes of 15 input result files. To regenerate them, install NumPy and Matplotlib and point the plotting script to the released follow-up results tree, which contains the `work/` subdirectory:

```sh
python figures/build_followup_figures.py --workspace /path/to/followup_studies --output .
```

The existing figure PDFs require no regeneration to compile the paper. The plotting script verifies preservation of all six original figure files.
