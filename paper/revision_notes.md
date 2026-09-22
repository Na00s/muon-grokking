# Targeted revision of the Muon grokking submission

The revised paper retains the title, section order, anonymous author block, original measurements, six figure PDFs, and all 17 original table files. Of the original archive's 30 files, 27 are unchanged. Only `main.tex`, `appendix.tex`, and `references.bib` are edited; new evidence is added in separate files.

## Changes to affected passages

| Location | Revision |
| --- | --- |
| Abstract and introduction | Replace the unproven basis-drift explanation with measured readout sensitivity, surviving task information, numerical controls, and the RMS boundary. Preserve the original phenomenon and framing. |
| Section 4.1 | Retain quiet-window measurements while separating displacement scales from causal interpretation. |
| Section 4.2 | Retain matched freezes; add adjacent-update swaps and training-only decoder results. Scope necessity and prevention to tested interventions and horizons. |
| Section 4.3 | Preserve every freeze result. Remove the uniqueness claim, since accurate-loss branches also avoid joint collapse over a shorter horizon. |
| New Section 4.4 | Summarize the five early arithmetic branches, fifteen specificity branches, finite-step overshoot, mean-dependent update, subtraction control, and RMS counterexample. |
| Section 6 and captions | Define circuit failure through the current readout. Distinguish exact nonzero Fourier support from dominant-frequency membership; correct the general-invertible-map invariance claim. |
| Related work and discussion | Credit the numerical-arithmetic literature, replace tested predictions with completed findings, and state the remaining scope limits. |
| Existing appendix | Make three small protocol/terminology qualifications. Preserve all existing table files. |
| New Appendix J | Add the decoder, basis, arithmetic, mean-update, and architecture protocols, with four tables and the RMS figure. |

## Added figures

**Figure 4, main text:** three panels show the five adjacent feature/readout swaps, recovery with fresh linear decoders, and 5/5 original versus 0/5 corrected joint-failure incidence. The caption distinguishes selected probe snapshots, reused seeds, and the 6,000-to-30,000 intervention horizon.

**Figure 8, appendix:** one-step parameter interventions at the first detected failures in subtraction and the two RMS controls. The accurate-loss RMS case places the acute failure in the embeddings, bounding the main account. These are one-seed controls. Later minima are reported separately in the adjacent table.

## Claims supported by the additions

- At the five recorded unnormalized transitions, the new readout sharply reduces accuracy while the preceding readout retains accuracy on updated features.
- Training-only fresh decoders recover 98.20-100% at selected collapsed checkpoints. This establishes substantial surviving linear task information.
- Exact global basis equivalence fails for the measured pairs; approximate transport can still help.
- Early accurate cross-entropy gives no joint train/test failures through step 30,000 in five matched branches. The three specificity corrections each give 0/5 through step 20,000.
- The actual captured updates locally descend accurate loss and overshoot at their applied scale. Terminal gradient replacement alone fails to prevent the five events.
- An accurate-loss RMS trajectory still fails through its embeddings. The numerical/readout explanation therefore has an explicit architectural scope.

The revision does not claim permanent prevention, a unique historical mediator, or universal prevalence from these selected trajectories. The new joint-event endpoint differs from the original test-below-95% stability definition; both are stated explicitly.

## Build and preservation checks

The original ZIP is untouched. All six original figure PDFs and 17 table files match their source hashes. The revised PDF has **10 main-text pages, 2 reference pages, and 12 appendix pages**, for **24 pages total**. The original main text also occupied 10 pages.

The paper compiles successfully with resolved citations and references, no overfull boxes, and no em dashes in the edited manuscript sources. Every page was rendered for visual inspection. A T1 font-encoding declaration preserves Times with Tectonic, a small table-size adjustment resolves an existing overfull table, and references and the new appendix start on fresh pages. One stray closing brace in the supplied bibliography was removed and one numerical-arithmetic citation was added.

`targeted_revision.patch` shows the complete changes to original text files. `verification.json` records preservation checks and file hashes. `source/` and the source ZIP contain the complete compilable manuscript; `figures/` contains the two new figures as PDF and PNG.
