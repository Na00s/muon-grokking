# Exact execution commands

Run from the repository root. Python 3.14.6, PyTorch 2.13.0, NumPy 2.5.2, SciPy 1.18.0; CPU with one Torch and one BLAS thread. The registered source manifest identifies the implementation.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/research/bin/python
SCRIPT=followup_studies/work/decoder_controls/run_controls.py
"$PYTHON" "$SCRIPT" register
for seed in 0 1 2 3 4; do
  "$PYTHON" "$SCRIPT" prepare --seed "$seed"
  "$PYTHON" "$SCRIPT" fit --seed "$seed"
done
"$PYTHON" "$SCRIPT" summarize
"$PYTHON" followup_studies/work/decoder_controls/verify_controls.py
"$PYTHON" followup_studies/work/decoder_controls/build_tables.py
```

The original working-directory invocation used `work/muon-grokking/` as a prefix. An initial registration used the first single scheduled train evaluation at 99%; before any probe fitting, `protocol_amendment.json` replaced this with the paper's five-evaluation 99.9% definition. Seed 0 had already been prepared; its selected step remained 200, with the original preparation audit preserved. All fitting uses the amended protocol. Existing reference and failure decoder reports are reused byte for byte.
