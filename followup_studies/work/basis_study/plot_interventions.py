"""Scientific figures for the completed readout intervention experiments."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
COLORS = {'original': '#C44E52', 'old_head': '#555555', 'orthogonal': '#2C7FB8', 'inverse_gl': '#8172B2',
          'orthogonal_clear_state': '#238B45', 'orthogonal_zero_first_moment': '#E68613'}


def main():
    out = ROOT / 'intervention_figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'figure.dpi': 160})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    pairs = [
        [('interventions', 'seed4_first', '16050 to 16056'), ('interventions', 'seed4_peak', '16050 to 16060'), ('intervention_adjacent', 'seed4_adjacent', '16055 to 16056')],
        [('interventions', 'seed0_first', '17200 to 17494'), ('intervention_near_reference', 'seed0_near', '17490 to 17494'), ('intervention_adjacent', 'seed0_adjacent', '17493 to 17494')],
    ]
    labels = {'original': 'Collapsed head', 'old_head': 'Old reference head', 'orthogonal': 'Orthogonal transport', 'inverse_gl': 'Inverse GL transport'}
    for ax, seed, selected in zip(axes, [4, 0], pairs):
        for index, (group, event, label) in enumerate(selected):
            path = ROOT / group / event
            results = json.loads((path / 'instant.json').read_text())
            meta = json.loads((path / 'metadata.json').read_text())
            for j, name in enumerate(labels):
                ax.bar(index + (j - 1.5) * .18, results[name]['heldout']['accuracy'], width=.17,
                       color=COLORS[name], label=labels[name] if index == 0 else None)
            reference = meta['native_healthy']['heldout']['accuracy']
            ax.plot([index - .38, index + .38], [reference, reference], color='black', linestyle=':', linewidth=1.3,
                    label='Reference checkpoint' if index == 0 else None)
        ax.set_xticks(range(3), [x[2] for x in selected], rotation=15, ha='right')
        ax.set_xlabel('Reference step to evaluated step')
        ax.set_title(f'Seed {seed}')
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Heldout accuracy after changing the head')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Immediate rescue depends on the reference checkpoint', y=.98)
    fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.29, wspace=.12)
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'reference_sensitivity.{ext}', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    arms = [('interventions', 'original', 'Original head, preserve state'),
            ('interventions', 'orthogonal', 'Orthogonal, preserve state'),
            ('optimizer_memory_results', 'orthogonal_clear_state', 'Orthogonal, clear head state'),
            ('optimizer_memory_results', 'orthogonal_zero_first_moment', 'Orthogonal, zero first moment')]
    for ax, event, seed in zip(axes, ['seed4_first', 'seed0_first'], [4, 0]):
        for group, arm, label in arms:
            path = ROOT / group / event / 'continuation' / arm / 'trajectory.csv'
            rows = [r for r in csv.DictReader(path.open()) if r['test_accuracy']]
            ax.plot([int(r['local_step']) for r in rows], [float(r['test_accuracy']) for r in rows],
                    linewidth=1.3, color=COLORS[arm], label=label, alpha=.9)
        ax.axhline(.9, color='#999999', linewidth=.9, linestyle=':')
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(f'Seed {seed}')
        ax.set_xlabel('Optimizer updates after the intervention')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Heldout accuracy')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Head optimizer memory changes recovery after compensation', y=.98)
    fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.25, wspace=.12)
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'optimizer_memory_trajectories.{ext}', bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    for index, seed in enumerate(range(5)):
        if seed in [0, 4]:
            directory = ROOT / 'intervention_adjacent' / f'seed{seed}_adjacent'
        else:
            name = f'seed{seed}_adjacent'
            directory = ROOT / 'fresh_interventions' / name / name
        values = json.loads((directory / 'instant.json').read_text())
        metadata = json.loads((directory / 'metadata.json').read_text())
        for j, name in enumerate(labels):
            ax.bar(index + (j - 1.5) * .18, values[name]['heldout']['accuracy'], width=.17,
                   color=COLORS[name], label=labels[name] if index == 0 else None)
        reference = metadata['native_healthy']['heldout']['accuracy']
        ax.plot([index - .38, index + .38], [reference, reference], color='black', linestyle=':', linewidth=1.3,
                label='Previous-step accuracy' if index == 0 else None)
    ax.set_xticks(range(5), [f'Seed {seed}' for seed in range(5)])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Heldout accuracy at the first joint failure')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.set_title('Previous-step readouts restore accuracy across five seeds')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.11), ncol=3, frameon=False)
    fig.subplots_adjust(left=.09, right=.99, top=.87, bottom=.23)
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'adjacent_five_seed_repair.{ext}', bbox_inches='tight')
    plt.close(fig)
    (out / 'manifest.json').write_text(json.dumps(dict(figures=['reference_sensitivity.png', 'reference_sensitivity.pdf', 'optimizer_memory_trajectories.png', 'optimizer_memory_trajectories.pdf', 'adjacent_five_seed_repair.png', 'adjacent_five_seed_repair.pdf'],
                                                         source='All source JSON and CSV files are read directly by plot_interventions.py'), indent=2) + '\n')


if __name__ == '__main__':
    main()
