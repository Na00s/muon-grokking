"""Create scientific figures from the completed causal-study artifacts."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'

def read(p):return json.loads(p.read_text())
def measured(p):
    with p.open() as f:
        return {int(x['step']):float(x['test_accuracy'])*100 for x in csv.DictReader(f) if x.get('test_accuracy')}
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=190)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)

def main():
    data=read(HERE/'analysis'/'results.json');assert data['complete']
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'stock':'#b63a38','accurate':'#14756e','target':'#4f68ac','projection':'#a56d12'}
    fig,axes=plt.subplots(3,2,figsize=(11.5,9),layout='constrained',sharex=True,sharey=True)
    for seed,ax in enumerate(axes.flat):
        if seed==5:ax.axis('off');continue
        base=HERE.parent/('experiments' if seed in (0,4) else 'basis_study')/f'seed{seed}_{"original" if seed in (0,4) else "baseline"}'
        stock=measured(base/'trajectory.csv')
        extension=HERE/'main_runs'/f'seed{seed}_stock_extension'/'trajectory.csv'
        if extension.exists():stock.update(measured(extension))
        event=data['primary'][seed]
        stock[event['stock_event_step']]=event['stock_event_test_accuracy']*100
        stock={k:v for k,v in sorted(stock.items()) if k>=6000}
        accurate=measured(HERE/'main_runs'/f'seed{seed}_accurate6000'/'trajectory.csv')
        ax.plot(list(stock),list(stock.values()),color=colors['stock'],lw=.9,label='Original CE')
        ax.plot(list(accurate),list(accurate.values()),color=colors['accurate'],lw=1.4,label='Accurate CE from 6,000')
        ax.scatter([event['stock_event_step']],[event['stock_event_test_accuracy']*100],color=colors['stock'],s=28,zorder=5,label='Verified original joint failure')
        ax.axhline(90,color='#999999',lw=.7,ls=':')
        ax.set(title=f'Seed {seed}',ylim=(-3,104),xlim=(6000,30000),ylabel='Held-out accuracy (%)',xlabel='Global update')
        ax.set_xticks([6000,12000,18000,24000,30000],['6k','12k','18k','24k','30k'])
        ax.tick_params(axis='x',labelbottom=True)
        ax.grid(alpha=.16)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles,labels,loc='upper left',bbox_to_anchor=(.04,.86),frameon=False)
    axes.flat[5].text(.05,.38,'Original prefixes have mixed evaluation grids.\nMarkers identify densely verified joint failures.\nEvery low-training state in corrected branches\nwas checked on held-out examples.',transform=axes.flat[5].transAxes,fontsize=9,va='top')
    fig.suptitle('Matched arithmetic intervention through update 30,000',fontsize=14)
    save(fig,'matched_trajectories')

    fig,axs=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
    for seed in range(5):
        scales=read(HERE/'diagnostics_results'/f'seed{seed}'/'stepsize.json')['feature_arms']['pre_features']
        curve=[x for x in scales['curve'] if x['scale']>=0]
        axs[0].plot([x['scale'] for x in curve],[x['train']['loss'] for x in curve],'-o',ms=3,label=f'Seed {seed}')
    axs[0].set(xlabel='Fraction of captured readout displacement',ylabel='Accurate training cross-entropy',yscale='log',title='The full readout step overshoots')
    axs[0].legend(frameon=False,fontsize=9);axs[0].grid(alpha=.16)
    matrix=[]
    for row in data['specificity']:
        matrix.append([float(row[arm]['event']) for arm in ['accurate','target_repair','row_projection']])
    matrix=np.array(matrix)
    axs[1].imshow(matrix.T,vmin=0,vmax=1,cmap=matplotlib.colors.ListedColormap(['#d7eee9','#e9b4b0']),aspect='auto')
    for y in range(3):
        for x in range(5):axs[1].text(x,y,'Failure' if matrix[x,y] else 'No failure',ha='center',va='center',fontsize=9)
    axs[1].set(xticks=range(5),xticklabels=[f'Seed {s}' for s in range(5)],yticks=range(3),yticklabels=['Accurate CE','Target derivative repair','Zero-sum projection'],title='Matched controls: updates 15,000 to 20,000')
    save(fig,'specificity_and_overshoot')

if __name__=='__main__':main()
