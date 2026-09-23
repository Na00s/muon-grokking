"""Build compact manuscript tables and summaries from recorded decoder results."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent


def main():
    rows = json.loads((HERE / 'summary.json').read_text())['rows']
    protocol = json.loads((HERE / 'protocol.json').read_text())
    labels = {'initialization': 'Initialization', 'memorization': 'Memorization',
              'healthy_reference': 'Reference', 'failed': 'Failed'}
    stats = {}
    for role in labels:
        subset = [r for r in rows if r['role'] == role]
        stats[role] = {
            'minimum_decoder_test_accuracy': min(r['decoder_test_accuracy'] for r in subset),
            'maximum_decoder_test_accuracy': max(r['decoder_test_accuracy'] for r in subset),
            'refit_converged_count': sum(r['refit_converged'] for r in subset),
            'unregularized_refit_converged_count': sum(r['unregularized_refit_converged'] for r in subset),
        }
    (HERE / 'stage_summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    lines = [r'\begin{table}[t]', r'\centering\small', r'\setlength{\tabcolsep}{3.2pt}',
        r'\caption{Training-only decoder controls on the five trajectories used for the reference and failed-state probes. All accuracies are percentages. Memorization is the first of five consecutive scheduled training evaluations at $\geq99.9\%$, before grokking and with test accuracy below $95\%$. Refit reports convergence and iterations; Cand. gives converged candidate fits out of five. The ten reference and failure results are unchanged.}',
        r'\label{tab:decoder-phase-controls}', r'\begin{tabular}{clrrrrcc}', r'\toprule',
        r'Seed & State & Step & Native train & Native test & Decoder test & Refit & Cand. \\', r'\midrule']
    for i, r in enumerate(rows):
        if i and r['seed'] != rows[i-1]['seed']:
            lines.append(r'\midrule')
        step = f"{r['step']:,}".replace(',', '{,}')
        status = 'Yes' if r['refit_converged'] else 'No'
        fields = [str(r['seed']), labels[r['role']], step,
            f"{100*r['native_train_accuracy']:.2f}", f"{100*r['native_test_accuracy']:.2f}",
            f"{100*r['decoder_test_accuracy']:.2f}", f"{status} ({r['refit_iterations']})",
            f"{sum(r['candidate_convergence'])}/5"]
        lines.append(' & '.join(fields) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (HERE / 'decoder_phase_controls_table.tex').write_text('\n'.join(lines))
    means = []
    for e in protocol['seeds']:
        new = np.load(HERE / f"seed{e['seed']}" / 'features.npz')
        old = np.load(WORK / e['existing_features']['path'])
        phases = [('initialization', 0, new['initialization']),
            ('memorization', e['memorization_step'], new['memorization']),
            ('healthy_reference', e['healthy_step'], old['H0']), ('failed', e['failed_step'], old['H1'])]
        for role, step, H in phases:
            means.append({'seed': e['seed'], 'role': role, 'step': step, 'backend': 'CPU',
                'configuration': 'unnormalized hidden Muon0.03, embedding AdamW0.001, readout AdamW0.00025, stockCE',
                'native_dtype': 'float32', 'reduction_dtype': 'float64',
                'definition': 'Euclidean norm of arithmetic mean of actual readout-input feature vectors',
                'training_count': int(new['train'].sum()), 'full_grid_count': len(H),
                'training_feature_mean_norm': float(np.linalg.norm(H[new['train']].mean(axis=0))),
                'full_grid_feature_mean_norm': float(np.linalg.norm(H.mean(axis=0)))})
    (HERE / 'feature_mean_controls.json').write_text(json.dumps({'rows': means,
        'sampling': 'Four selected phase checkpoints per seed; no interpolation or additional temporal density implied.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
