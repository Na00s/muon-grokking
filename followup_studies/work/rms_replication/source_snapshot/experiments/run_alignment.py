"""Analyze two saved checkpoints using only original training pairs for fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from run_collapse import make_model_optimizers, generate_modular_addition_data, residuals, evaluate
from alignment_core import analyze_alignment, classification_metrics, fit_ridge_map


def load_features(path, inputs):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model, _ = make_model_optimizers(state['seed'], checkpoint=state)
    features = residuals(model, inputs).double().numpy()
    readout = model.unembedding.weight.detach().T.double().numpy()
    return state, model, features, readout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--healthy', type=Path, required=True)
    parser.add_argument('--collapsed', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--decoder', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    healthy = torch.load(args.healthy, map_location='cpu', weights_only=False)
    train_x, train_y, test_x, test_y = generate_modular_addition_data(seed=healthy['seed'])
    inputs = torch.cat([train_x, test_x])
    labels = torch.cat([train_y, test_y]).numpy()
    train = np.arange(len(inputs)) < len(train_x)
    heldout = ~train
    state0, model0, H0, W0 = load_features(args.healthy, inputs)
    state1, model1, H1, W1 = load_features(args.collapsed, inputs)
    if state0['seed'] != state1['seed'] or state0['model_config'] != state1['model_config']:
        raise ValueError('Checkpoint seeds and configurations must agree')
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = dict(healthy_checkpoint=str(args.healthy.resolve()), collapsed_checkpoint=str(args.collapsed.resolve()),
        healthy_step=state0['step'], collapsed_step=state1['step'], seed=state0['seed'],
        config=state0['model_config'], torch_version=torch.__version__, numpy_version=np.__version__,
        native_healthy=dict(train=evaluate(model0,train_x,train_y),heldout=evaluate(model0,test_x,test_y)),
        native_collapsed=dict(train=evaluate(model1,train_x,train_y),heldout=evaluate(model1,test_x,test_y)),
        dataset_sha256=hashlib.sha256(inputs.numpy().tobytes()+labels.tobytes()).hexdigest(),
        protocol='Raw final residuals. Original training examples only for fits and regularization validation. All test rows remain held out. No Fourier filtering.')
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    np.savez_compressed(args.out/'features.npz', H0=H0,H1=H1,W0=W0,W1=W1,y=labels,train=train,heldout=heldout)
    result=analyze_alignment(H0,H1,W0,W1,labels,train,heldout)
    native_epsilon=float(np.finfo(np.float32).eps)
    rank_thresholds={'eps32':native_epsilon,'feature_count_times_eps32':H0.shape[1]*native_epsilon,
                     'max_shape_times_eps32':max(H0[train].shape)*native_epsilon,
                     'relative_1e-6':1e-6,'relative_1e-5':1e-5,'relative_1e-4':1e-4}
    result.report['native_precision_rank_sensitivity']={'representation_origin_dtype':'float32',
        'fitting_dtype':'float64','relative_thresholds':rank_thresholds,
        'note':'Ranks are sensitivity diagnostics. Full numerical rank of task activations alone does not establish a global invertible change of basis.',
        'ranks':{name:{threshold:int(np.count_nonzero(np.asarray(spectrum['singular_values'])>rtol*spectrum['singular_values'][0]))
                        for threshold,rtol in rank_thresholds.items()}
                 for name,spectrum in result.report['training_representation_spectra'].items()}}
    (args.out/'alignment.json').write_text(json.dumps(result.report,indent=2))
    np.savez_compressed(args.out/'maps_and_readouts.npz',**result.transforms,**result.transported_readouts)
    controls={}
    rng=np.random.default_rng(915)
    Q,_=np.linalg.qr(rng.normal(size=(H0.shape[1],H0.shape[1])))
    planted=analyze_alignment(H0,H0@Q,W0,Q.T@W0,labels,train,heldout)
    controls['planted_orthogonal']={key:planted.report[key] for key in ['classification','reconstruction','healthy_function_reconstruction']}
    # Permute training target correspondence only, leaving evaluation inputs untouched.
    # This provides a reconstruction negative control without contaminating test fits.
    wrong_targets=H0[train][rng.permutation(train.sum())]
    shuffled_B=fit_ridge_map(H1[train],wrong_targets,result.report['selection']['backward_relative_ridge'])
    controls['shuffled_correspondence']={name:classification_metrics((H1@shuffled_B@W0)[mask],labels[mask]) for name,mask in [('train',train),('heldout',heldout)]}
    (args.out/'controls.json').write_text(json.dumps(controls,indent=2))
    if args.decoder:
        from decoder_probe import fit_decoder
        decoders={}
        for name,H in [('healthy',H0),('collapsed',H1)]:
            decoder=fit_decoder(H,labels,train,heldout)
            decoders[name]=decoder.report
            np.save(args.out/f'{name}_decoder.npy',decoder.readout)
        (args.out/'decoder.json').write_text(json.dumps(decoders,indent=2))
    (args.out/'completion.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started),indent=2))
    print(json.dumps(dict(metadata=metadata,heldout=result.report['classification']['heldout']),indent=2),flush=True)


if __name__=='__main__':
    main()
