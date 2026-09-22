# Audited submission source

Compile `main.tex` in this directory with Tectonic or a standard LaTeX/BibTeX installation. Figures, tables, bibliography, and the supplied conference style are included.

```sh
tectonic main.tex
```

The revision preserves the title, anonymous authors, section order, all six original figure PDFs, and the numerical cells in the 17 original appendix table files. Captions, definitions, analysis, and claims are revised where needed. New material appears in Section 4.4, Figure 4, and Appendix J, including Figure 8.

The two new figures use recorded experimental measurements. Their complete values and the hashes of 15 input files are in `figures/followup_figure_provenance.json`. To regenerate from the released follow-up results tree:

```sh
python figures/build_followup_figures.py --workspace /path/to/followup_studies --output .
```

From the repository layout, the script also resolves the data and output paths automatically. In a standalone source archive, pass the data root explicitly. The script checks that the original figures are preserved. The source archive is sufficient to compile the paper without downloading experimental binaries.
