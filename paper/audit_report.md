# Audit of the revised submission

The experimental audit covers the new decoder, basis, arithmetic, and architecture studies. The original experiments and observations are retained as established results, following the author's clarification. Review of the original sections concerns their interpretation, analysis, claim strength, and relation to prior work. Missing old artifacts are not used to remove reported results.

## New experiments and results

The audit contains **604 checks: 598 computed comparisons and six manual checks**, with zero failures. This is a count of checks, including repeated claims and protocol checks, rather than a count of unique numbers. It covers all four new tables, both new figures, the new appendix, and the corresponding main-text claims.

- Recomputed all 20 adjacent feature/readout swaps and all 10 reference/collapsed decoder accuracies from saved arrays and coefficients.
- Refit the 15 reconstruction comparisons by training-only least squares and checked held-out, mean-centered errors.
- Checked decoder validation selection, all 50 inner fits and 10 final-fit convergence records, data masks, dimensions, and regularization settings.
- Scanned the 24 causal branch trajectories and the four operation/architecture trajectories for the stated endpoints, monitoring, minima, and horizons.
- Compared 28 saved initial states, including model parameters, optimizer buffers, and random state.
- Re-evaluated all seven whole-update directional/interpolation panels from checkpoints. Every recorded scalar reproduced.
- Reconstructed the five mean/centered readout interventions from saved arrays.
- Regenerated both new figures and checked their plotted values and input hashes. Fixed the plotting script's default paths.

These checks re-evaluate saved experimental evidence. They do not repeat the full training runs.

## Claim and method refinements

The follow-up confirmation requires six consecutive checks and reports the final check. The original sweep definition uses the start of a five-check streak. The paper now states both. Decoder regularization is explicitly defined in preconditioned coordinates; derivative-repair reductions are identified as float64 before conversion to float32. The 4.62% task-only alignment example now names its seed and interval.

The paper retains the central findings: substantial linear task information survives the selected failures; the measured residual pairs fail exact global basis-equivalence tests; early arithmetic corrections prevent joint collapse through the stated horizons; captured updates locally descend accurate loss and overshoot at full scale; the RMS control can fail through embeddings under accurate cross-entropy.

The original results remain in the paper, including the observed non-finite terminations. All six original figure PDFs and the numerical cells of the original 17 appendix table files are preserved. Captions and analysis clarify the diagnostic domain, DC convention, margin competitor, distinct dispersion measures, and perturbation realization. Scope is stated explicitly when a claim concerns sampled checkpoints, one perturbation, a selected seed set, or a finite horizon.

## Research prose and positioning

The revision keeps the title, section order, anonymous authors, and conference style. The prose uses direct experimental statements and gives the relevant setting before the interpretation. It removes unsupported universal statements and rhetorical claims such as a spectral 'certificate' of circuit presence. It credits Nanda et al.'s functional progress measures and causal Fourier ablations, separates weight spectra from operand-frequency dispersion, and distinguishes the Collatz decoder study from the representation/readout speed study.

The literature review checks all 19 references against primary sources. No style detector is used: the standard is clear methods, concrete results, precise attribution, and claims proportional to evidence. The manuscript contains no em dashes.

## Figures retained in the revision

**Figure 4, main text:** adjacent swaps, fresh decoders, and corrected-arithmetic incidence. The caption identifies reused seeds, distinct probe snapshots, and the fixed horizon.

**Figure 8, appendix:** parameter interventions in subtraction and the two RMS controls. The figure exposes the architectural boundary of the readout account. Each condition uses one seed.

## Remaining scientific limits

Exact global equivalence is rejected for the measured pairs. Approximate transformations can still help. Feature growth and optimizer history remain coupled, so the experiments do not identify a unique historical mediator. Prevention is established over the completed horizons. The selected five trajectories do not estimate population prevalence, and the one-seed operation/normalization controls establish examples. Brief test-only excursions can occur between scheduled test evaluations.

The claim ledger and replay records document the new results. `verification.json` records the final build, preservation, and layout checks.
