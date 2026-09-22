"""Write compact tables from geometry runs without changing prior outputs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for path in sorted(ROOT.glob('*/geometry.json')):
    d = json.loads(path.read_text())
    b = d['train_only_maps']['backward']['linear']['rank_sensitivity']['float64_solver']
    a = d['train_only_maps']['backward']['affine']['rank_sensitivity']['float64_solver']
    o = d['oracle_geometry']['uncentered']['by_rank_tolerance']['float64_solver']
    f = d['functional_decomposition']['splits']['heldout']
    c = f['counterfactuals']
    rows.append({
        'pair': path.parent.name.removeprefix('geometry_'),
        'healthy_accuracy': c['healthy']['classification']['accuracy'],
        'later_accuracy': c['actual_later']['classification']['accuracy'],
        'backward_linear_accuracy': b['transported_classification']['heldout']['accuracy'],
        'backward_relative_error': b['reconstruction']['heldout']['relative_error'],
        'backward_centered_error': b['reconstruction']['heldout']['centered_relative_error'],
        'backward_affine_centered_error': a['reconstruction']['heldout']['centered_relative_error'],
        'backward_map_condition': b['mapping_spectrum']['condition_number'],
        'oracle_backward_relative_error': o['x_outside_y_column_space_relative'],
        'median_principal_angle_degrees': o['angle_median_degrees'],
        'max_principal_angle_degrees': o['angle_max_degrees'],
        'basis_predicted_accuracy': c['basis_predicted']['classification']['accuracy'],
        'residual_only_accuracy': c['residual_only']['classification']['accuracy'],
        'readout_change_only_accuracy': c['readout_change_only']['classification']['accuracy'],
        'basis_predicted_argmax_agreement': c['basis_predicted']['argmax_agreement_with_actual'],
        'basis_predicted_actual_flip_recall': c['basis_predicted']['actual_flip_recall'],
        'basis_norm_over_delta': f['basis_norm_over_delta'],
        'residual_norm_over_delta': f['residual_norm_over_delta'],
        'normalized_cross_term': f['normalized_cross_term'],
        'native_cond100_backward_relative_floor': d.get('controls', {}).get('planted', {}).get('100', {}).get('float32_matmul', {}).get('backward', {}).get('heldout', {}).get('relative_error'),
    })
with (ROOT / 'geometry_summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(ROOT / 'geometry_summary.json').write_text(json.dumps(rows, indent=2))
print(json.dumps({'pairs': len(rows), 'summary': str(ROOT / 'geometry_summary.csv')}))
