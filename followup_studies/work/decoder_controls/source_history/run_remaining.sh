#!/bin/sh
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/research/bin/python
SCRIPT=work/muon-grokking/followup_studies/work/decoder_controls/run_controls.py
for seed in 1 2 3 4; do
  "$PYTHON" "$SCRIPT" prepare --seed "$seed"
  "$PYTHON" "$SCRIPT" fit --seed "$seed"
done
