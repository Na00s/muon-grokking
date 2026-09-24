"""Regenerate final-head readout ranges from the declared solved-checkpoint subset."""
from pathlib import Path
import argparse
import csv
import hashlib
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    names = ['runs/depth_fourier_mode_model_summary_seed_0.csv',
             'runs/depth_fourier_mode_layer_summary_seed_0.csv']
    records = [list(csv.DictReader((args.repo/name).open())) for name in names]
    models, layers = records
    selected = {(r['run'], r['step']): r for r in models if float(r['full_accuracy']) >= .95}
    assert len(selected) == 41
    residuals = [r for r in layers if r['stage_kind'] == 'post_block_residual'
                 and (r['run'], r['step']) in selected]
    groups = [(2, 'adamw', 'AdamW'), (2, 'muon', 'Muon'),
              (2, 'stable_muon', 'Stable Muon'), (4, 'muon', 'Muon'),
              (4, 'stable_muon', 'Stable Muon')]
    rows = []
    for depth, regime, label in groups:
        points = [r for r in residuals if int(r['depth']) == depth and r['regime'] == regime]
        assert points, (depth, regime, sorted({r['regime'] for r in residuals}))
        earlier = [100*float(r['direct_readout_full_accuracy']) for r in points if int(r['layer']) < depth]
        final = [100*float(r['direct_readout_full_accuracy']) for r in points if int(r['layer']) == depth]
        checkpoints = sorted({(r['run'], int(r['step'])) for r in points})
        rows.append(dict(depth=depth, regime=regime, label=label, checkpoints=checkpoints,
                         earlier_min=min(earlier), earlier_max=max(earlier),
                         final_min=min(final), final_max=max(final)))
    assert sum(len(r['checkpoints']) for r in rows) == 41
    maximum = max((r for r in residuals if int(r['layer']) < int(r['depth'])),
                  key=lambda r: float(r['direct_readout_full_accuracy']))
    assert round(100*float(maximum['direct_readout_full_accuracy']), 2) == 17.97
    excluded = next(r for r in models if r['run'] == 'depth4_stable_muon_observed' and r['step'] == '210000')
    excluded_layer = next(r for r in layers if r['run'] == excluded['run'] and r['step'] == excluded['step']
                          and r['stage_kind'] == 'post_block_residual' and r['layer'] == '3')
    assert float(excluded['full_accuracy']) < .95
    header = r'''\begin{table}[t]
\centering\small
\caption{Direct-readout accuracy by post-block residual, decoding with the final unembedding and
skipping the remaining blocks. Values are ranges over the same $41$ checkpoints used in
Section~\ref{sec:depth}, each with native full-grid accuracy of at least $95\%$. Both columns
report percentages to two decimal places.}
\label{tab:depthreadout}
\begin{tabular}{llrr}
\toprule
Depth & Regime & Earlier blocks & Final block \\
\midrule
'''
    body = ''.join(f"${r['depth']}$ & {r['label']} & ${r['earlier_min']:.2f}$--${r['earlier_max']:.2f}\\%$ & ${r['final_min']:.2f}$--${r['final_max']:.2f}\\%$ \\\\\n" for r in rows)
    footer = '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'depthreadout.tex').write_text(header+body+footer)
    audit = dict(status='passed', selection='Native full-grid accuracy >=0.95, joined on (run, step); post_block_residual stages; earlier layer < depth; final layer == depth.',
                 checkpoints=41, percentage_decimal_places=2,
                 sources=[dict(path=n, sha256=hashlib.sha256((args.repo/n).read_bytes()).hexdigest()) for n in names],
                 rows=rows, maximum_earlier=dict(run=maximum['run'], step=int(maximum['step']), layer=int(maximum['layer']), percent=100*float(maximum['direct_readout_full_accuracy'])),
                 excluded_previous_maximum=dict(run=excluded['run'], step=210000, layer=3,
                                               native_full_percent=100*float(excluded['full_accuracy']),
                                               direct_readout_percent=100*float(excluded_layer['direct_readout_full_accuracy'])))
    (args.out/'depth_readout_ranges.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'status':'passed','checkpoints':41,'rows':len(rows),'maximum_earlier_percent':audit['maximum_earlier']['percent']}))

if __name__ == '__main__':
    main()
