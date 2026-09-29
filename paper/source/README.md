# Figure and table assets

This directory contains the plotted measurements, figure assets, vector inputs, and analysis builders used with the modular-arithmetic studies. Training and numerical-analysis records are under `followup_studies/`; the repository root describes the study-specific execution commands and artifact downloads.

The `figures/` builders read the saved study measurements. The `tables/` directory contains scientific tabulations and their source manifests. The inputs retain the execution and source hashes recorded with the corresponding study.

From the repository root, the depth summary and its checkpoint-selection audit can be generated with:

```sh
python paper/source/tables/build_depth_readout_table.py --repo .
```

The selected rows have native full-grid accuracy at least 95%. `tables/depth_readout_ranges.json` identifies the input rows, checkpoint identities, exact ranges, and source hashes.
