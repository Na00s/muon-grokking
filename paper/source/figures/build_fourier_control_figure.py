"""Regenerate the Fourier-control figure from released numerical records."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path, default=HERE.parents[2]/'followup_studies',
                    help='Directory containing the released work/fourier_control tree.')
parser.add_argument('--output', type=Path, default=HERE.parent)
args = parser.parse_args()
study = args.workspace.resolve()/'work/fourier_control'
if not (study/'make_report.py').is_file():
    parser.error('Supply --workspace /path/to/followup_studies containing work/fourier_control.')
subprocess.run([sys.executable, str(study/'make_report.py')], check=True)
args.output.mkdir(parents=True, exist_ok=True)
for name in ['fourier_memorizer_control.pdf','fourier_memorizer_control.png']:
    shutil.copy2(study/name,args.output/name)
(args.output/'figures').mkdir(exist_ok=True)
shutil.copy2(study/'results/publication_manifest.json',args.output/'figures/fourier_control_provenance.json')
print('Copied the regenerated figure and its provenance to',args.output.resolve())
