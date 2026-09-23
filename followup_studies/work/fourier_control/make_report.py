"""Build the manuscript-ready summary and control figure from saved results."""
from __future__ import annotations
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
primary = json.loads((OUT / 'memorizer_results.json').read_text())
addition = [r for r in primary if r['operation'] == 'addition']
checkpoints = json.loads((OUT / 'checkpoint_donor_results.json').read_text())
with (OUT / 'ablation_metrics.csv').open() as f:
    ablation = list(csv.DictReader(f))
with (OUT / 'frequency_prefix_metrics.csv').open() as f:
    prefix = list(csv.DictReader(f))
with (OUT / 'phase_metrics.csv').open() as f:
    phase = list(csv.DictReader(f))

def pct_range(values):
    x = 100 * np.array(list(values), dtype=float)
    return {'min': float(x.min()), 'max': float(x.max()), 'mean': float(x.mean()), 'values': x.tolist()}

summary = {'memorizer_addition': {}, 'checkpoint_donors': {}, 'frequency_controls': {}, 'phase_controls': {}}
for condition in addition[0]['conditions']:
    summary['memorizer_addition'][condition] = {split: pct_range(r['conditions'][condition][split]['accuracy'] for r in addition) for split in ('training', 'heldout', 'full_grid')}
for endpoint in (0,1):
    summary['checkpoint_donors'][str(endpoint)] = {condition: pct_range(r['conditions'][condition]['heldout']['accuracy'] for r in checkpoints if r['endpoint']==endpoint) for condition in checkpoints[0]['conditions']}
for kind in ('one_hot','training_count_normalized'):
    summary['frequency_controls'][kind] = {str(n): pct_range(float(r['accuracy']) for r in prefix if r['lookup']==kind and r['ranking']=='ascending_frequency_mathematical_tie' and int(r['retained_pairs'])==n and r['split']=='heldout') for n in (1,2,3,5,10,20,56)}
    summary['phase_controls'][kind] = {control: pct_range(float(r['accuracy']) for r in phase if r['lookup']==kind and r['control']==control and r['context']=='isolated_family_with_dc' and r['split']=='heldout') for control in ('pair_global_phase','channelwise_phase','frequency_derangement')}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8})
fig, axes = plt.subplots(1,2,figsize=(5.5,3.0),gridspec_kw={'width_ratios':[1,1.05]})
colors = {'heldout':'#254C73','full_grid':'#BD7436'}
labels = {'heldout':'Held-out pairs','full_grid':'All input pairs'}
conditions=('raw_lookup','complete_family_with_dc','family_ablated_dc_retained')
for j, split in enumerate(('heldout','full_grid')):
    vals = [100*np.array([float(r['accuracy']) for r in ablation if r['lookup']=='one_hot' and r['condition']==condition and r['split']==split]) for condition in conditions]
    means=np.array([v.mean() for v in vals])
    lower=means-np.array([v.min() for v in vals]); upper=np.array([v.max() for v in vals])-means
    positions=np.arange(3)+(j-.5)*.30
    axes[0].bar(positions,means,width=.27,color=colors[split],label=labels[split],zorder=3)
    axes[0].errorbar(positions,means,yerr=np.vstack([lower,upper]),fmt='none',color='#17202A',lw=.7,capsize=2,zorder=4)
axes[0].set_xticks(np.arange(3),['Raw\nlookup','Full family\n+ DC','Ablated\nDC retained'])
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('(a) Lookup memorizer',loc='left',pad=36)
axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,
               borderaxespad=0,handlelength=1.5,handletextpad=.5,labelspacing=.3)
counts=np.array([1,2,3,5,10,20,56])
for kind,color,label,marker in (('one_hot','#254C73','One-hot lookup','o'),('training_count_normalized','#8B466E','Count-normalized lookup','s')):
    vals=[summary['frequency_controls'][kind][str(n)] for n in counts]
    means=np.array([v['mean'] for v in vals]); lower=np.array([v['min'] for v in vals]); upper=np.array([v['max'] for v in vals])
    axes[1].plot(counts,means,color=color,marker=marker,ms=3,lw=1.2,label=label,zorder=3)
    axes[1].fill_between(counts,lower,upper,color=color,alpha=.14,lw=0,zorder=2)
axes[1].set_xscale('log')
axes[1].set_xticks([1,2,3,5,10,20,56],[1,2,3,5,10,20,56])
axes[1].set_xlabel('Conjugate pairs retained (+ DC)')
axes[1].set_ylabel('Held-out accuracy (%)')
axes[1].set_title('(b) Frequency prefixes',loc='left',pad=36)
axes[1].legend(loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,
               borderaxespad=0,handlelength=1.5,handletextpad=.5,labelspacing=.3)
for ax in axes:
    ax.set_ylim(-2,108)
    ax.set_yticks([0,25,50,75,100])
    ax.grid(axis='y',color='#DDDDDD',lw=.5,zorder=0)
    ax.tick_params(length=3)
# Preserve the physical 5.5-inch paper width and an eight-point text floor.
fig.subplots_adjust(left=.10,right=.985,bottom=.22,top=.74,wspace=.48)
fig.savefig(HERE/'fourier_memorizer_control.pdf')
fig.savefig(HERE/'fourier_memorizer_control.png',dpi=250)
plt.close(fig)

( HERE / 'suggested_table.tex').write_text(r'''\begin{table}[t]
\centering
\caption{Task-structured Fourier controls on lookup-only predictors. All values are percentages over the five original addition splits ($p=113$, $3{,}830$ training and $8{,}939$ held-out inputs). Ranges reflect splits. The count-normalized lookup stores $p/n_c$ at its memorized class $c$ and zero elsewhere. Every one of the $56$ individual nonzero conjugate diagonal pairs succeeds on all five splits.}
\label{tab:fourier-memorizer}
\small
\begin{tabular}{lrrr}
\toprule
Predictor and intervention & Train & Held out & Full grid \\
\midrule
One-hot lookup, raw & $100$ & $0.76$--$0.91$ & $30.53$--$30.63$ \\
Full addition family $+$ DC & $100$ & $100$ & $100$ \\
Family ablation, DC retained & $100$ & $0$ & $29.99$ \\
Training-donor orbit average & $100$ & $100$ & $100$ \\
Held-out-donor orbit average & $0.84$--$1.17$ & $0.76$--$0.91$ & $0.88$ \\
Count-normalized lookup, one pair $+$ DC & $100$ & $100$ & $100$ \\
\bottomrule
\end{tabular}
\end{table}
''')
# Check exact ranges used in the typeset table, including the all-zero donor control.
assert round(summary['memorizer_addition']['heldout_donors_only']['training']['min'],2)==.84
assert round(summary['memorizer_addition']['heldout_donors_only']['training']['max'],2)==1.17
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

publication_manifest = {
    'figure_size_inches': [5.5, 3.0],
    'minimum_text_size_points': 8,
    'title_size_points': 9,
    'layout_note': 'Native paper text width, legends above axes, fixed physical PDF dimensions.',
    'plotted_statistics': 'Unchanged seed means and minima/maxima; frequency axis remains logarithmic.',
    'source_sha256': {str(path.relative_to(HERE)): digest(path)
                      for path in sorted(HERE.iterdir())
                      if path.suffix in {'.py', '.md', '.tex', '.txt'}},
    'input_sha256': {str(path.relative_to(HERE)): digest(path) for path in
                     [OUT/'memorizer_results.json', OUT/'checkpoint_donor_results.json',
                      OUT/'ablation_metrics.csv', OUT/'frequency_prefix_metrics.csv', OUT/'phase_metrics.csv']},
    'output_sha256': {str(path.relative_to(HERE)): digest(path) for path in
                      [HERE/'fourier_memorizer_control.pdf', HERE/'fourier_memorizer_control.png',
                       HERE/'suggested_table.tex', OUT/'summary.json']},
}
(OUT/'publication_manifest.json').write_text(json.dumps(publication_manifest, indent=2)+'\n')
print('Wrote summary.json, figure PDF/PNG, suggested_table.tex, and publication_manifest.json.')
