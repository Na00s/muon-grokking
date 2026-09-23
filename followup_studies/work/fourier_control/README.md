# Fourier averaging and memorizer controls

These controls address what full-family Fourier filtering can establish about a modular-arithmetic predictor. No model was trained or selected for this study. Existing experimental results are unchanged.

## Reproduce

From the research workspace, using a Python environment with NumPy, Torch, and Matplotlib:

```sh
python work/fourier_control/run_control.py --repo work/muon-grokking --work-root work --out work/fourier_control/results
python work/fourier_control/run_frequency_controls.py --repo work/muon-grokking --out work/fourier_control/results
python work/fourier_control/make_report.py
```

The primary synthetic controls require the repository's `data.py` only. Omit `--work-root` to run them without the archived feature pairs. To run the complete optional donor analysis, provide the directory containing `basis_study/adjacent_seed0`, `seed1_adjacent_event`, `seed2_adjacent_event`, `seed3_adjacent_event`, and `adjacent_seed4`, each with `features.npz` and `metadata.json`. Source hashes are in `results/manifest.json`.

`protocol.md` was written before primary results. `protocol_extension.md` records the additional frequency, phase, and ablation controls requested after the primary results and before those additional computations. The result manifests preserve the hash of each applicable protocol.

## Files

- `report.md`: derivation, interpretation, measured results, and limitations.
- `run_control.py`: independent FFT/orbit/translation identities, actual five-seed repository splits, addition/subtraction memorizers, and ten preselected real-checkpoint endpoints.
- `run_frequency_controls.py`: ablation, frequency prefixes, all 56 individual conjugate pairs for the normalized lookup, and phase/frequency controls.
- `results/`: complete machine-readable data, exact numerator/denominator metrics, source hashes, and verification results.
- `fourier_memorizer_control.pdf` and `.png`: optional scientific figure.
- `suggested_table.tex`: compact manuscript table, including full-grid and held-out denominators.

All synthetic classification uses the smallest-index argmax tie convention. Uniform random tie-breaking expectations are reported separately. The scientific identity is exact; numerical identities are checked with float64 tolerances. The Fourier-only 1D implementation in the frequency control is checked against the independently computed 2D FFT.

The secondary implementation explicitly restores the mathematically invariant answer-zero row after coefficient relocation. This ensures that cancellation roundoff does not turn an exact all-zero full-lookup held-out logit vector into an arbitrary argmax. The correction is documented in the secondary result manifest and report.
