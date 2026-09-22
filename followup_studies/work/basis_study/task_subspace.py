"""Task-conditioned decomposition and out-of-class alignment diagnostics.

All maps fit original training rows only. The Fourier task projector explicitly
uses labels, including evaluation labels to *measure* task/nuisance decomposition;
it is descriptive and must never be presented as independent prediction evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / 'experiments'
sys.path.insert(0, str(EXPERIMENTS))
from alignment_core import classification_metrics, fit_ridge_map, reconstruction_metrics, spectrum_metrics

PAIRS = ['seed4_dense_event', 'seed4_peak_event', 'seed0_dense_event',
         'seed4_generalization_drop_16000_17000',
         'seed4_healthy_temporal_16044_16050', 'seed4_healthy_temporal_6000_7000']


def fourier_task(labels, modulus):
    """Complete real Fourier basis, with columns orthonormal over uniform y."""
    labels = np.asarray(labels)
    if modulus % 2 != 1 or labels.ndim != 1:
        raise ValueError('An odd modulus and one-dimensional labels are required')
    angles = 2 * np.pi * labels[:, None] * np.arange(1, (modulus + 1) // 2)[None, :] / modulus
    return np.column_stack([np.ones(len(labels)), np.sqrt(2) * np.cos(angles), np.sqrt(2) * np.sin(angles)])


def fit_task_coefficients(H, y, train, modulus):
    if set(np.unique(y[train])) != set(range(modulus)):
        raise ValueError('Full task decomposition requires every output class in the training rows')
    T = fourier_task(y, modulus)
    return np.linalg.lstsq(T[train], H[train], rcond=None)[0]


def complete_invertible_backward(C0, C1):
    """Construct B such that C1 B = C0 when both C matrices have full row rank.

    Ck has q <= d rows. Complete its rows with an orthonormal basis of its
    nullspace, scaled by the median singular value for finite numerical scale.
    E1 B = E0 gives B = solve(E1, E0), which is invertible because E0, E1 are.
    The nullspace matching is arbitrary and has no causal identification claim.
    """
    q, d = C0.shape
    if C1.shape != C0.shape or q > d:
        raise ValueError('Coefficient matrices need equal shape with q <= d')
    _, s0, v0 = np.linalg.svd(C0, full_matrices=True)
    _, s1, v1 = np.linalg.svd(C1, full_matrices=True)
    threshold = max(q, d) * np.finfo(float).eps
    if min(s0[-1] / s0[0], s1[-1] / s1[0]) <= threshold:
        raise ValueError('Both coefficient matrices must have full numerical row rank')
    scale = np.sqrt(np.median(s0) * np.median(s1))
    E0 = np.vstack([C0, scale * v0[q:]])
    E1 = np.vstack([C1, scale * v1[q:]])
    B = np.linalg.solve(E1, E0)
    return B, dict(relative_coefficient_error=float(np.linalg.norm(C1 @ B - C0) / np.linalg.norm(C0)),
                   condition_number=float(np.linalg.cond(B)), smallest_singular_value=float(np.linalg.svd(B,compute_uv=False)[-1]),
                   completion_scale=float(scale), coefficient_rank=q,
                   nullspace_dimension=d-q, E0_condition=float(np.linalg.cond(E0)), E1_condition=float(np.linalg.cond(E1)))


def recon(pred, target, train, evaluate):
    return reconstruction_metrics(pred[evaluate], target[evaluate],
        prediction_train_mean=pred[train].mean(0), target_train_mean=target[train].mean(0))


def energy(matrix, center):
    return float(np.linalg.norm(matrix - center) ** 2)


def map_metrics(H0, H1, W0, y, fit, evaluate, B):
    """Same held-out masks are used for restricted and random matched-size fits."""
    P = H1 @ B
    return dict(n_fit=int(fit.sum()), n_evaluation=int(evaluate.sum()),
        representation=recon(P, H0, fit, evaluate),
        classification=classification_metrics((P @ W0)[evaluate], y[evaluate]),
        map_spectrum=spectrum_metrics(B))


def restricted_map_tests(H0, H1, W0, y, train, heldout, inputs=None):
    fixed = [(f'output_class_mod5_{r}', y % 5 == r) for r in range(3)]
    if inputs is not None:
        fixed += [('first_operand_mod5_0', inputs[:, 0] % 5 == 0)]
    rows = []
    for k, (name, excluded) in enumerate(fixed):
        restricted = train & ~excluded
        rng = np.random.default_rng(41000 + k)
        random_mask = np.zeros(len(y), dtype=bool)
        random_mask[rng.choice(np.flatnonzero(train), restricted.sum(), replace=False)] = True
        for method, penalty in [('ols', 0.0), ('fixed_ridge_1e-6', 1e-6)]:
            for fit_name, fit in [('restricted', restricted), ('random_matched_size', random_mask)]:
                B = fit_ridge_map(H1[fit], H0[fit], penalty)
                # The exclusion criterion is fixed in advance and never used for model training.
                results = {ename:map_metrics(H0,H1,W0,y,fit,emask,B) for ename,emask in [
                    ('heldout_excluded', heldout & excluded), ('heldout_included', heldout & ~excluded)]}
                rows.append(dict(split=name, method=method, fit=fit_name, relative_ridge=penalty,
                    original_model_training_unchanged=True, results=results))
    return rows


def task_coefficient_reliability(H0,H1,y,train,p):
    """Diagnose sampling error using two label-stratified halves of training rows."""
    rng=np.random.default_rng(41113)
    half_a=np.zeros(len(y),bool)
    for label in range(p):
        indices=np.flatnonzero(train&(y==label))
        rng.shuffle(indices)
        half_a[indices[::2]]=True
    half_b=train&~half_a
    results={}
    for name,H in [('healthy',H0),('collapsed',H1)]:
        C=fit_task_coefficients(H,y,train,p)
        A=fit_task_coefficients(H,y,half_a,p);B=fit_task_coefficients(H,y,half_b,p)
        k=(p-1)//2
        frequencies=[]
        for f in range(k):
            rows=[1+f,1+k+f];ca,cb,cc=A[rows],B[rows],C[rows]
            frequencies.append(dict(frequency=f+1,training_power=float(np.sum(cc**2)),
                split_half_cosine=float(np.sum(ca*cb)/(np.linalg.norm(ca)*np.linalg.norm(cb))),
                split_half_cross_power=float(np.sum(ca*cb))))
        powers=np.array([v['training_power'] for v in frequencies])
        order=np.argsort(powers)[::-1]
        count=int(np.searchsorted(np.cumsum(powers[order]),.95*powers.sum())+1)
        results[name]=dict(n_half_a=int(half_a.sum()),n_half_b=int(half_b.sum()),
            nonconstant_split_half_relative_disagreement=float(np.linalg.norm(A[1:]-B[1:])/np.linalg.norm(C[1:])),
            frequencies_for_95percent_training_nonconstant_power=[int(i+1) for i in order[:count]],
            all_frequencies=frequencies,
            note='Split-half disagreement measures finite training-sample uncertainty in class-conditional task means. Full coefficient rank can include weak or unstable task frequencies.')
    return results


def analyze_task(H0, H1, W0, W1, y, train, heldout, *, global_B=None, inputs=None):
    p = W0.shape[1]
    T = fourier_task(y, p)
    C0, C1 = [fit_task_coefficients(H,y,train,p) for H in (H0,H1)]
    P0, P1 = T @ C0, T @ C1
    N0, N1 = H0-P0, H1-P1
    mean0, mean1 = H0[train].mean(0), H1[train].mean(0)
    report = dict(protocol={
        'fitting':'All task coefficients and maps fit original training examples only.',
        'labels':'Full Fourier task basis uses known y. Held-out y is used only to evaluate the descriptive task projection; projection accuracy is label-informed and is not prediction evidence.',
        'map_tests':'Three prespecified excluded output-class groups y modulo 5 equal 0, 1, or 2. An additional first-operand group is excluded if original inputs are available. Fixed OLS and relative ridge 1e-6, with deterministic size-matched random fits. No selection by test performance.',
        'global_map':'Uses the previously train-selected backward map if supplied, otherwise unregularized training-row least squares.',
        'task_map':'Least squares C1 B = C0 and an arbitrary invertible nullspace completion. Both descriptive; neither identifies an actual historical transformation.',
        'identifiability':'For full-row-rank q x d coefficient matrices with q <= d, an invertible B mapping C1 to C0 always exists. Task-subspace transport by itself is consequently weak evidence for an observed global basis drift.'},
        dimensions=dict(n=len(y),d=H0.shape[1],modulus=p,train=int(train.sum()),heldout=int(heldout.sum())),
        coefficients={k:spectrum_metrics(C) for k,C in [('healthy',C0),('collapsed',C1)]},
        coefficient_rank_sensitivity={k:{str(tol):int(np.count_nonzero(np.linalg.svd(C,compute_uv=False)>tol*np.linalg.svd(C,compute_uv=False)[0])) for tol in [1e-7,1e-6,1e-5,1e-4,1e-3]} for k,C in [('healthy',C0),('collapsed',C1)]},
        decomposition={}, mapping={}, coefficient_sampling_reliability=task_coefficient_reliability(H0,H1,y,train,p))
    for tag, H, P, N, W, mean in [('healthy',H0,P0,N0,W0,mean0),('collapsed',H1,P1,N1,W1,mean1)]:
        report['decomposition'][tag] = dict(
            task_prediction_of_representation=recon(P,H,train,heldout),
            task_energy_fraction_of_centered_heldout=energy(P[heldout],mean)/energy(H[heldout],mean),
            residual_energy_fraction_of_centered_heldout=energy(N[heldout],np.zeros(H.shape[1]))/energy(H[heldout],mean),
            note='Held-out task and nuisance are not exactly orthogonal. Fractions need not sum to one.',
            raw_classification=classification_metrics((H@W)[heldout],y[heldout]),
            label_informed_task_classification=classification_metrics((P@W)[heldout],y[heldout]),
            residual_only_classification=classification_metrics((N@W)[heldout],y[heldout]),
        )
    if global_B is None:
        global_B=fit_ridge_map(H1[train],H0[train],0)
    maps={'global_train_selected_backward':global_B,
          'task_minimum_norm_backward':fit_ridge_map(C1,C0,0),
          'task_fixed_ridge_1e-6_backward':fit_ridge_map(C1,C0,1e-6)}
    try:
        maps['task_invertible_completion'], report['invertible_completion'] = complete_invertible_backward(C0,C1)
    except ValueError as exc:
        report['invertible_completion']={'unavailable':str(exc)}
    for name,B in maps.items():
        raw,task,nuis=H1@B,P1@B,N1@B
        report['mapping'][name] = dict(
            raw_representation=recon(raw,H0,train,heldout),
            task_representation=recon(task,P0,train,heldout),
            nuisance_representation=recon(nuis,N0,train,heldout),
            raw_classification=classification_metrics((raw@W0)[heldout],y[heldout]),
            label_informed_task_classification=classification_metrics((task@W0)[heldout],y[heldout]),
            residual_only_classification=classification_metrics((nuis@W0)[heldout],y[heldout]),
            heldout_nuisance_amplification=float(np.linalg.norm(nuis[heldout])/np.linalg.norm(N1[heldout])),
            map_spectrum=spectrum_metrics(B))
    report['restricted_map_tests']=restricted_map_tests(H0,H1,W0,y,train,heldout,inputs)
    return report, {'C0':C0,'C1':C1,**maps}


def load_inputs(meta, y):
    try:
        import torch
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'muon-grokking'))
        from data import generate_modular_addition_data
        torch.set_num_threads(1)
        tx,ty,vx,vy=generate_modular_addition_data(seed=meta['seed'],modulus=meta['config']['modulus'])
        x=torch.cat([tx,vx]).numpy()
        labels=torch.cat([ty,vy]).numpy()
        if not np.array_equal(labels,y): raise ValueError('Regenerated input ordering does not match feature labels')
        digest=hashlib.sha256(x.tobytes()+labels.tobytes()).hexdigest()
        if 'dataset_sha256' in meta and digest != meta['dataset_sha256']: raise ValueError('Dataset hash mismatch')
        return x
    except ImportError:
        return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pair',action='append',default=[])
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'task_subspace_results')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for name in (args.pair or PAIRS):
        start=time.perf_counter()
        source=EXPERIMENTS/name
        if not source.exists(): source=Path(name)
        data=dict(np.load(source/'features.npz'))
        meta=json.loads((source/'metadata.json').read_text())
        inputs=load_inputs(meta,data['y'])
        mapfile=source/'maps_and_readouts.npz'
        maps=np.load(mapfile) if mapfile.exists() else None
        B=maps['linear_backward'] if maps is not None else None
        report, arrays=analyze_task(**data,global_B=B,inputs=inputs)
        report['source']=dict(directory=str(source.resolve()),metadata=meta)
        report['elapsed_seconds']=time.perf_counter()-start
        out=args.out/source.name
        out.mkdir(exist_ok=True)
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        np.savez_compressed(out/'maps.npz',**arrays)
        concise={key:dict(raw_accuracy=val['raw_classification']['accuracy'],task_accuracy=val['label_informed_task_classification']['accuracy'],task_relerr=val['task_representation']['relative_error'],nuisance_relerr=val['nuisance_representation']['relative_error'],nuisance_amplification=val['heldout_nuisance_amplification']) for key,val in report['mapping'].items()}
        print(json.dumps(dict(pair=source.name,elapsed=report['elapsed_seconds'],maps=concise)),flush=True)


if __name__=='__main__': main()
