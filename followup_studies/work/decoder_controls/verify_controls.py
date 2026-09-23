"""Independently verify saved decoder predictions, selection and provenance."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
WORK = HERE.parent


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def metrics(logits, y):
    correct = int(np.sum(np.argmax(logits, axis=1) == y))
    # Independent CE evaluation at the saved final coefficients.
    losses = logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]
    return {'correct': correct, 'count': len(y), 'accuracy': correct / len(y), 'cross_entropy': float(losses.mean())}


def main():
    protocol = read(HERE / 'protocol.json')
    audits = []
    for entry in protocol['seeds']:
        seed = entry['seed']; out = HERE / f'seed{seed}'
        for kind in ['trajectory', 'pair_metadata', 'existing_decoder', 'existing_features']:
            r = entry[kind]
            assert sha(WORK / r['path']) == r['sha256']
        for r in entry['checkpoints'].values():
            assert sha(WORK / r['path']) == r['sha256']
        prepared = read(out / 'prepared.json')
        assert not any(prepared['exact_state_mismatches'].values())
        assert all(all(row['checks'].values()) for row in prepared['scheduled_evaluations'])
        assert prepared['protocol_sha256'] == sha(HERE / 'protocol.json')
        features = np.load(out / 'features.npz')
        assert sha(out / 'features.npz') == prepared['features']['sha256']
        assert np.flatnonzero(features['inner_validation']).tolist() == entry['inner_validation_indices']
        assert not np.any(features['inner_validation'] & features['heldout'])
        for role in ['initialization', 'memorization']:
            r = read(out / f'{role}_decoder.json')
            weights = np.load(out / f'{role}_decoder.npz')
            assert r['protocol_sha256'] == sha(HERE / 'protocol.json')
            assert sha(WORK / r['checkpoint']['path']) == r['checkpoint']['sha256']
            for key, prefix in [('cv_selected', 'cv'), ('unregularized_control', 'unregularized')]:
                report = r[key]
                assert report['split']['inner_validation_indices'] == entry['inner_validation_indices']
                assert report['initialization'] == 'zero'
                selected = min(report['candidates'], key=lambda c: (c['validation']['cross_entropy'], c['regularization']))
                assert selected['regularization'] == report['selected_regularization']
                assert report['feature_transform']['dimension'] == 128
                assert report['feature_transform']['centered'] is False
                assert report['feature_transform']['relative_singular_floor'] == 1e-6
                W = weights[prefix + '_readout']
                P = weights[prefix + '_feature_matrix']
                Wt = weights[prefix + '_transformed_readout']
                assert W.dtype == np.float64 and P.dtype == np.float64
                assert np.allclose(W, P @ Wt, rtol=1e-12, atol=1e-12)
                for split, mask in [('train', features['train']), ('heldout', features['heldout'])]:
                    measured = metrics(features[role][mask] @ W, features['y'][mask])
                    recorded = report['classification'][split]
                    assert measured['accuracy'] == recorded['accuracy']
                    assert np.isclose(measured['cross_entropy'], recorded['cross_entropy'], rtol=1e-10, atol=1e-10)
                    audits.append({'seed': seed, 'role': role, 'decoder': key, 'split': split,
                        **measured, 'refit_converged': report['refit']['converged'],
                        'all_candidate_statuses_present': all('status' in c['optimization'] for c in report['candidates'])})
    result = {'passed': True, 'protocol_sha256': sha(HERE / 'protocol.json'),
        'all_registered_source_artifacts_unchanged': True,
        'all_five_early_replays_exact': True, 'validation_indices_preserved_and_test_disjoint': True,
        'independent_saved_coefficient_evaluations': audits}
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'independent_prediction_checks': len(audits)}))


if __name__ == '__main__':
    main()
