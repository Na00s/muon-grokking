"""Render manuscript-ready figures from the additional Muon experiments.

Example:
  python plot_results.py --baseline seed4_original/trajectory.csv \
      --prospective seed4_prospective/trajectory.csv --alignment seed4_event \
      --branch seed4_original32 --branch seed4_true64 --out figures

All accuracies use full 0-100% axes. Input artifacts remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE.parent / 'tmp' / 'matplotlib'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', str(HERE.parent / 'tmp' / 'cache'))
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, MaxNLocator

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
ARM_LABELS = {
    'original32': 'Original FP32',
    'stable32': 'FP32, accurate CE',
    'model64': 'FP64 model, FP32 NS',
    'true64': 'FP64 model and NS, accurate CE',
}
ARM_COLORS = {'original32': '#222222', 'stable32': '#0072B2', 'model64': '#D55E00', 'true64': '#009E73'}
ARM_STYLES = {'original32': '-', 'stable32': '--', 'model64': '-.', 'true64': ':'}


def read_json(path: Path, required=False):
    if not path.exists() and not required:
        return {}
    with path.open() as stream:
        return json.load(stream)


def read_trajectory(path: Path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty trajectory: {path}')
    needed = ['step', 'train_accuracy', 'test_accuracy']
    for key in needed:
        if key not in rows[0]:
            raise ValueError(f'{path} lacks {key}')
    result = {key: [float(row[key]) for row in rows] for key in needed}
    if any(b <= a for a, b in zip(result['step'], result['step'][1:])):
        raise ValueError(f'Trajectory steps must increase: {path}')
    for key in needed[1:]:
        if any(not 0 <= value <= 1 for value in result[key]):
            raise ValueError(f'Accuracy must be a fraction in [0,1]: {path}, {key}')
    return result


def seed_from(metadata, path):
    seed = metadata.get('seed', metadata.get('data_config', {}).get('seed'))
    if seed is None:
        match = re.search(r'seed[_-]?(\d+)', str(path))
        seed = int(match.group(1)) if match else None
    return seed


def seed_prefix(metadata, path):
    seed = seed_from(metadata, path)
    return f'Seed {seed}: ' if seed is not None else ''


def configure_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 9,
        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'lines.linewidth': 1.25,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 220, 'figure.facecolor': 'white',
    })


def style_accuracy_axes(ax, split):
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.set_ylabel(f'{split} accuracy (%)')
    ax.grid(axis='y', color='#D5DADF', linewidth=.55, alpha=.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.margins(x=.015)


def save(fig, out, stem, manifest, source_files, notes=None):
    output_paths=[]
    for suffix in ['png', 'pdf']:
        path = out / f'{stem}.{suffix}'
        fig.savefig(path, bbox_inches='tight', facecolor='white', metadata={'Creator': 'Muon experiment plotting helper'} if suffix=='pdf' else None)
        output_paths.append(str(path.resolve()))
    manifest.append({'figure': stem, 'source_files': [str(Path(p).resolve()) for p in source_files], 'outputs': output_paths, 'notes': notes or []})
    plt.close(fig)


def plot_overview(baselines, prospective, out, manifest):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.1), sharex=True)
    lines=[]; labels=[]
    for index, path in enumerate(baselines):
        rows=read_trajectory(path); metadata=read_json(path.parent/'metadata.json')
        label=seed_prefix(metadata,path)+'original FP32'
        color=COLORS[index % len(COLORS)]
        for ax, split in zip(axes,['train','test']):
            line,=ax.plot(rows['step'],[100*x for x in rows[f'{split}_accuracy']],color=color,label=label)
        lines.append(line);labels.append(label)
    if prospective is not None:
        rows=read_trajectory(prospective); metadata=read_json(prospective.parent/'metadata.json')
        arm=metadata.get('arm')
        if arm in ARM_LABELS:
            description=ARM_LABELS[arm]
        elif 'freeze' in str(metadata.get('condition','')).lower():
            description='frozen embeddings and readout'
        else:
            description=str(metadata.get('condition', 'prospective control'))
        label=seed_prefix(metadata,prospective)+description
        for ax, split in zip(axes,['train','test']):
            line,=ax.plot(rows['step'],[100*x for x in rows[f'{split}_accuracy']],color='#222222',ls='--',label=label,zorder=3)
        lines.append(line);labels.append(label)
    for ax, split in zip(axes,['Train','Test']):style_accuracy_axes(ax,split)
    axes[-1].set_xlabel('Training step')
    fig.legend(lines,labels,loc='lower center',bbox_to_anchor=(.52,.015),ncol=min(2,len(labels)),frameon=False)
    fig.subplots_adjust(left=.12,right=.98,top=.97,bottom=.18,hspace=.12)
    save(fig,out,'training_overview',manifest,[*baselines,*([prospective] if prospective else [])],['Accuracy spans 0-100%. Each curve starts at its first saved evaluation.'])


def nested_accuracy(obj, *path):
    for key in path:
        obj=obj[key]
    value=float(obj)
    if not 0 <= value <= 1:
        raise ValueError(f'Invalid accuracy at {path}: {value}')
    return 100*value


def plot_alignment(directory, out, manifest):
    alignment=read_json(directory/'alignment.json',required=True)
    metadata=read_json(directory/'metadata.json')
    decoder=read_json(directory/'decoder.json')
    whitened=read_json(directory/'decoder_whitened.json')
    heldout=alignment['classification']['heldout']
    native0=metadata.get('native_healthy',{}).get('heldout',{}).get('accuracy')
    native1=metadata.get('native_collapsed',{}).get('heldout',{}).get('accuracy')
    train1=metadata.get('native_collapsed',{}).get('train',{}).get('accuracy')
    is_collapse=native1 is not None and train1 is not None and native1 < .9 and train1 < .9
    later='Collapsed' if is_collapse else 'Later'
    values=[100*native0 if native0 is not None else nested_accuracy(heldout,'healthy_original','accuracy'),100*native1 if native1 is not None else nested_accuracy(heldout,'collapsed_original','accuracy'),nested_accuracy(heldout,'healthy_readout_on_collapsed','accuracy'),nested_accuracy(heldout,'orthogonal_transport','accuracy'),nested_accuracy(heldout,'linear_transport','accuracy')]
    names=['Healthy checkpoint\nnative readout',f'{later} checkpoint\nnative readout',f'{later} features\nhealthy readout',f'{later} features\northogonal transport',f'{later} features\nlinear transport']
    colors=['#8E9BA6','#222222','#CC79A7','#0072B2','#009E73']
    notes=['Transports and decoder are fit using original training examples only; bars report untouched test examples.','Native model metrics are used for the first two bars when available.']
    probe=decoder.get('collapsed',decoder) if decoder else {}
    whitened_probe=whitened.get('cv_selected',{}).get('collapsed',{})
    use_whitened=whitened_probe.get('refit',{}).get('converged') is True
    visible_note=None
    if use_whitened:
        original_probe=probe
        probe=whitened_probe
        transform=probe.get('feature_transform',{})
        dimension=transform.get('dimension')
        floored=transform.get('floored_singular_values')
        penalty=probe.get('selected_regularization')
        notes.append('The plotted fresh readout uses the converged exploratory whitening control selected by inner-validation CE; the initial decoder remains part of the recorded analysis.')
        notes.append(f'Whitening preserves all {dimension} feature dimensions; {floored} singular values were floored. Selected regularization: {penalty}.')
        if dimension is not None and floored==0:
            visible_note=f'Whitening preserves all {dimension} dimensions; no singular-value flooring was applied.'
        if original_probe.get('refit',{}).get('converged') is False:
            old_accuracy=100*float(original_probe['classification']['heldout']['accuracy'])
            notes.append(f'The original RMS-scaled decoder reached {old_accuracy:.4f}% held-out accuracy without convergence; that optimization-limited fit is retained in decoder.json and is not an estimate of information capacity.')
    elif whitened:
        notes.append('A whitening probe was available but its selected refit did not converge, so it was not preferred over the recorded initial probe.')
    if probe:
        if 'classification' in probe:
            values.append(nested_accuracy(probe,'classification','heldout','accuracy'))
            converged=probe.get('refit',{}).get('converged')
            suffix='*' if converged is False else ''
            readout_label='Fresh readout (whitened)' if use_whitened else f'fresh decoder{suffix}'
            names.append(f'{later} features\n{readout_label}')
            colors.append('#D55E00')
            if converged is False:notes.append('* Fresh decoder reached its iteration limit; the fit has not converged.')
    fig,ax=plt.subplots(figsize=(7.0, max(3.4,.61*len(values))))
    ax.barh(range(len(values)),values,color=colors,height=.67)
    ax.set_yticks(range(len(names)),labels=names)
    ax.invert_yaxis();ax.set_xlim(0,100)
    ax.xaxis.set_major_locator(MultipleLocator(25));ax.set_xlabel('Held-out test accuracy (%)')
    ax.grid(axis='x',color='#D5DADF',linewidth=.55,alpha=.8);ax.set_axisbelow(True)
    ax.tick_params(axis='y',length=0)
    for i,value in enumerate(values):
        inside=value>=90
        ax.text(value-1.3 if inside else value+1.2,i,f'{value:.2f}%',ha='right' if inside else 'left',va='center',fontsize=8,color='white' if inside else '#222222')
    healthy_step=metadata.get('healthy_step');later_step=metadata.get('collapsed_step')
    seed=seed_from(metadata,directory)
    context=[]
    if seed is not None:context.append(f'Seed {seed}')
    if healthy_step is not None and later_step is not None:context.append(f'Healthy step {healthy_step:,}; {later.lower()} step {later_step:,}')
    if context:ax.text(0,1.02,' | '.join(context),transform=ax.transAxes,fontsize=8,va='bottom')
    if any(note.startswith('*') for note in notes):
        fig.text(.30,.015,'* Decoder fit reached its iteration limit.',fontsize=7)
    elif visible_note:
        fig.text(.30,.015,visible_note,fontsize=7)
    fig.subplots_adjust(left=.30,right=.97,top=.93,bottom=.15)
    source_files=[directory/'alignment.json',*([directory/'metadata.json'] if metadata else []),*([directory/'decoder.json'] if decoder else [])]
    if whitened:
        source_files.append(directory/'decoder_whitened.json')
        if (directory/'decoder_whitened_protocol.json').exists():
            source_files.append(directory/'decoder_whitened_protocol.json')
    save(fig,out,'heldout_readout_comparison',manifest,source_files,notes)


def plot_branches(directories, out, manifest):
    fig,axes=plt.subplots(2,1,figsize=(7.0,5.4),sharex=True)
    lines=[];labels=[];sources=[];starts=[];hashes=set()
    for index,directory in enumerate(directories):
        path=directory/'trajectory.csv';sources.append(path)
        rows=read_trajectory(path);metadata=read_json(directory/'metadata.json')
        arm=metadata.get('arm',directory.name)
        label=seed_prefix(metadata,directory)+ARM_LABELS.get(arm,str(arm))
        color=ARM_COLORS.get(arm,COLORS[index%len(COLORS)])
        style=ARM_STYLES.get(arm,'-')
        for ax,split in zip(axes,['train','test']):
            line,=ax.plot(rows['step'],[100*x for x in rows[f'{split}_accuracy']],color=color,ls=style,label=label)
        lines.append(line);labels.append(label)
        starts.append(metadata.get('start_step',rows['step'][0]))
        if metadata.get('source_checkpoint_sha256'):hashes.add(metadata['source_checkpoint_sha256'])
    for ax,split in zip(axes,['Train','Test']):style_accuracy_axes(ax,split)
    axes[-1].set_xlabel('Training step')
    note=''
    if len(set(starts))==1:note=f'Branch step {int(starts[0]):,}'
    if len(hashes)==1 and len(directories)>1:note+=' | Shared saved model and optimizer state'
    if note:axes[0].text(0,1.035,note.strip(' |'),transform=axes[0].transAxes,fontsize=8)
    fig.legend(lines,labels,loc='lower center',bbox_to_anchor=(.53,.015),ncol=2,frameon=False)
    legend_rows=(len(lines)+1)//2
    fig.subplots_adjust(left=.12,right=.98,top=.93,bottom=.16+.025*legend_rows,hspace=.12)
    save(fig,out,'precision_branch_trajectories',manifest,sources,['Accuracy spans 0-100%. Different sampling intervals are retained.','NS denotes Newton-Schulz orthogonalization; accurate CE denotes the cancellation-resistant cross-entropy implementation.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline',type=Path,action='append',default=[],help='Original trajectory CSV; repeat for additional seeds.')
    parser.add_argument('--prospective',type=Path,help='Optional prospective-control trajectory CSV.')
    parser.add_argument('--alignment',type=Path,help='Directory containing alignment.json and optional decoder.json.')
    parser.add_argument('--branch',type=Path,action='append',default=[],help='Precision-arm directory; repeat for additional arms.')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not any([args.baseline,args.prospective,args.alignment,args.branch]):parser.error('Provide at least one input artifact.')
    args.out.mkdir(parents=True,exist_ok=True)
    configure_style();manifest=[]
    if args.baseline or args.prospective:plot_overview(args.baseline,args.prospective,args.out,manifest)
    if args.alignment:plot_alignment(args.alignment,args.out,manifest)
    if args.branch:plot_branches(args.branch,args.out,manifest)
    (args.out/'figure_manifest.json').write_text(json.dumps({'figures':manifest,'matplotlib_version':matplotlib.__version__},indent=2)+'\n')
    print(json.dumps({'figures':[item['figure'] for item in manifest],'output_directory':str(args.out.resolve())}))


if __name__=='__main__':main()
