# Revised submission source

Compile `main.tex` with Tectonic or a standard LaTeX/BibTeX installation. Figures, tables, bibliography, and the conference style are included.

```sh
tectonic main.tex
```

The revision preserves the title, anonymous authors, section order, all six original figure PDFs, and the numerical cells of the 17 original appendix table files. Section 4.4 contains the mean-dependent update equation and Table 3. The causal follow-up figure reports the swaps, decoders, and corrected-loss outcomes through step 100,000. Appendix J documents these experiments; Appendix K gives the Fourier averaging proof and memorizer controls, including Figure 9.

The scientific main text occupies nine pages. The reproducibility and required AI use statements are separate and excluded from the [ICLR 2027 initial-submission limit](https://iclr.cc/Conferences/2027/AuthorGuidelines). The quiet-window and Fourier-grid figures appear in Appendices C and G. Included template files are byte-identical to the official ICLR 2027 counterparts despite their retained legacy filenames.

The figure scripts use saved experimental results and record their input/output hashes. From a released `followup_studies` directory containing `work/`:

```sh
python figures/build_followup_figures.py --workspace /path/to/followup_studies --output .
python figures/build_fourier_control_figure.py --workspace /path/to/followup_studies --output .
```

The scripts resolve the data root automatically in the repository layout. In a standalone source archive, pass it explicitly. This source archive compiles without downloading experimental binaries.
