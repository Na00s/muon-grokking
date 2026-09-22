"""Figures from saved measurements only; no fitting or outcome selection."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 180, 'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
COLORS = ['#2563a6', '#df7132', '#228b72', '#9460ad', '#ba4f68']


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def fresh_trajectories():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, seed, color in zip(axes, [1, 2, 3], COLORS):
        root = HERE / f'seed{seed}_baseline'
        r = rows(root / 'trajectory.csv')
        t = [x for x in r if x['test_accuracy']]
        ax.plot([int(x['step']) for x in r], [100*float(x['train_accuracy']) for x in r],
                lw=1, color='#a9b1ba', label='Training')
        ax.plot([int(x['step']) for x in t], [100*float(x['test_accuracy']) for x in t],
                lw=1.4, color=color, label='Held out')
        if (root / 'event.json').exists():
            e = read(root / 'event.json')
            ax.scatter(e['step'],100*e['test_accuracy'],color='#ba2632',s=25,zorder=5)
            ax.axvline(e['step'],color='#ba2632',alpha=.4,lw=.7)
        ax.set(title=f'Seed {seed}', xlabel='Training update', ylim=(-2,103))
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(loc='lower right',frameon=False)
    fig.suptitle('Three additional seeds under the original training settings',y=1.04)
    fig.tight_layout()
    save(fig,'fresh_trajectories')


def acute_swaps():
    names = [(f'adjacent_seed0','Seed 0'), (f'seed1_adjacent_event','Seed 1'),
             (f'seed2_adjacent_event','Seed 2'), (f'seed3_adjacent_event','Seed 3'),
             (f'adjacent_seed4','Seed 4')]
    entries=[]
    for name,label in names:
        f=HERE/name/'features.npz'
        if not f.exists():continue
        d=np.load(f);m=d['heldout'];y=d['y'][m]
        vals=[100*np.mean((d[h][m]@d[w]).argmax(1)==y)
              for h,w in [('H0','W0'),('H1','W0'),('H0','W1'),('H1','W1')]]
        entries.append((label,vals))
    fig,ax=plt.subplots(figsize=(10,4.4))
    x=np.arange(len(entries));width=.19
    for j,(label,color) in enumerate(zip(['Preceding model','Updated features + preceding readout',
                                        'Preceding features + updated readout','Updated model'],
                                       ['#777f88','#228b72','#df7132','#ba4f68'])):
        bars=ax.bar(x+(j-1.5)*width,[v[j] for _,v in entries],width,label=label,color=color)
        ax.bar_label(bars,fmt=lambda v:f'{v:.1f}'.rstrip('0').rstrip('.'),fontsize=8,padding=2)
    ax.set_xticks(x,[s for s,_ in entries]);ax.set(ylim=(0,115),ylabel='Held-out accuracy (%)',
             title='The immediately preceding readout isolates the acute update')
    ax.legend(ncol=2,loc='lower center',bbox_to_anchor=(.5,1.04),frameon=False,fontsize=9)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'acute_readout_swaps')


def gauge_floors():
    entries=[]
    for root in [HERE/'model_gauge_results'/'actual', HERE/'model_gauge_fresh'/'actual']:
        for f in sorted(root.glob('*.json')):
            if f.name=='summary.json':continue
            d=read(f);m=d['metadata'];name=f.stem
            if 'peak' in name or 'near' in name or '6000' in name:continue
            a=d['maps']['final_token_ols']
            actual=a['sites']['block0_after_mlp']['tokens']['2']['centered_relative_error']
            floor=a['same_A_network_positive_control']['known_A_sites']['block0_after_mlp']['tokens']['2']['centered_relative_error']
            label=f"Seed {m['seed']}\n{m.get('healthy_step',m.get('first_step'))} to {m.get('collapsed_step',m.get('second_step'))}"
            entries.append((label,actual,floor,'healthy' in name))
    fig,ax=plt.subplots(figsize=(10,4))
    for j,(label,actual,floor,healthy) in enumerate(entries):
        ax.plot([j,j],[floor,actual],color='#b2b8bf',lw=2,zorder=1)
        ax.scatter(j,actual,color='#228b72' if healthy else '#ba4f68',marker='o',s=55,zorder=2)
        ax.scatter(j,floor,color='#2563a6',marker='s',s=35,zorder=2)
    ax.scatter([],[],color='#ba4f68',label='Observed collapse comparison')
    ax.scatter([],[],color='#228b72',label='Observed healthy comparison')
    ax.scatter([],[],color='#2563a6',marker='s',label='Same-map planted network benchmark')
    ax.set(yscale='log',ylabel='Centered held-out residual error')
    ax.set_title('Observed representations depart from exact residual gauges',pad=45)
    ax.set_xticks(range(len(entries)),[e[0] for e in entries],fontsize=8)
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,fontsize=8);ax.grid(axis='y',alpha=.18)
    fig.tight_layout();save(fig,'gauge_floors')


def continuation():
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharey=True)
    arms=[('interventions','original','Original head','#777f88'),
          ('interventions','orthogonal','Orthogonal repair','#2563a6'),
          ('optimizer_memory_results','native_head_clear_state','Original + cleared head state','#df7132'),
          ('optimizer_memory_results','orthogonal_clear_state','Repair + cleared head state','#228b72')]
    for i,seed in enumerate([4,0]):
        for group,arm,label,color in arms:
            rr=rows(HERE/group/f'seed{seed}_first'/'continuation'/arm/'trajectory.csv')
            r=[x for x in rr if x.get('test_accuracy')]
            for col,limit in enumerate([30,500]):
                chosen=[x for x in r if int(x['local_step'])<=limit]
                axes[i,col].plot([int(x['local_step']) for x in chosen],
                                 [100*float(x['test_accuracy']) for x in chosen],
                                 label=label,color=color,lw=1.4,marker='.' if col==0 else None,ms=3)
                axes[i,col].set(xlim=(0,limit),ylim=(-2,103),title=f'Seed {seed}, first {limit} resumed updates',xlabel='Updates after intervention')
                axes[i,col].grid(alpha=.17)
        axes[i,0].set_ylabel('Held-out accuracy (%)')
    axes[0,1].legend(loc='lower right',fontsize=8,frameon=False)
    fig.suptitle('A one-time readout repair permits renewed collapse',y=1.01)
    fig.tight_layout();save(fig,'continuation')


def timecourse():
    fig,axes=plt.subplots(2,2,figsize=(12,6),sharex='col')
    for col,seed in enumerate([4,0]):
        r=rows(HERE/f'timecourse_seed{seed}'/'trajectory.csv')
        lo=16040 if seed==4 else 17450
        r=[x for x in r if int(x['step'])>=lo]
        x=[int(q['step']) for q in r]
        for key,label,color in [('native_heldout_accuracy','Actual model','#ba4f68'),('old_head_accuracy','Fixed earlier readout','#228b72'),('readout_change_only_accuracy','Fixed earlier features + current readout','#df7132')]:
            axes[0,col].plot(x,[100*float(q[key]) for q in r],label=label,color=color,marker='.',ms=4)
        axes[1,col].plot(x,[100*float(q['backward_centered_heldout_error']) for q in r],color='#2563a6',marker='.',ms=4)
        axes[0,col].set(title=f'Seed {seed}, reference update {r[0]["reference_step"]}',ylim=(0,103))
        axes[1,col].set_xlabel('Training update')
        for a in axes[:,col]:a.grid(alpha=.18)
    axes[0,0].set_ylabel('Held-out accuracy (%)');axes[1,0].set_ylabel('Centered reconstruction error (%)')
    axes[0,0].legend(fontsize=8,frameon=False,loc='lower left')
    fig.suptitle('Reference timing separates gradual drift from the acute failure',y=1.02)
    fig.tight_layout();save(fig,'temporal_reference')


if __name__=='__main__':
    fresh_trajectories();acute_swaps();gauge_floors();continuation();timecourse()
    print(json.dumps({'figures':sorted(x.name for x in FIG.iterdir())}))
