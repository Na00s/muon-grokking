"""Audit raw monitoring, fitted predictions, derivative arrays and final artifacts."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
import runner as r
sys.path.insert(0,str(r.HERE/'source_snapshot/experiments'))
from alignment_core import classification_metrics
from analyze_event import metrics
HERE=Path(__file__).resolve().parent

def audit_run(folder):
    info=json.loads((folder/'completion.json').read_text());seed=info['seed'];rows=list(csv.DictReader((folder/'trajectory.csv').open()))
    terminal=info['terminal_step'];assert [int(x['step']) for x in rows]==list(range(terminal+1))
    streak=0;confirmation=None;initial_streak=None;test_evaluations=0;post_count=0;below=0;minimum=None;events=[]
    for row in rows:
        step=int(row['step']);train=int(row['train_correct'])/int(row['train_total']);assert train==float(row['train_accuracy'])
        scheduled=step%100==0;expected=scheduled or (confirmation is not None and train<.9) or step==100000
        assert bool(row['test_accuracy'])==expected,(seed,step,'test_schedule')
        if expected:
            acc=int(row['test_correct'])/int(row['test_total']);assert acc==float(row['test_accuracy']);test_evaluations+=1
        if scheduled:
            streak=streak+1 if acc>=.95 else 0
            if streak>=6 and confirmation is None:confirmation=step;initial_streak=step-500
        joint=bool(confirmation is not None and step>confirmation and train<.9 and expected and acc<.9)
        assert (row['grok_confirmed']=='True')==(confirmation is not None)
        assert (row['joint_failure']=='True')==joint
        if joint:events.append(step)
        if confirmation is not None and expected:post_count+=1;below+=int(acc<.95);minimum=acc if minimum is None else min(minimum,acc)
    mon=info['monitor'];assert mon['grok_confirmation_step']==confirmation;assert mon['first_qualifying_scheduled_step']==initial_streak
    assert mon['test_evaluations']==test_evaluations;assert mon['post_confirmation_test_evaluations']==post_count;assert mon['post_confirmation_below95_evaluations']==below;assert mon['minimum_post_confirmation_sampled_test_accuracy']==minimum
    if events:assert events==[terminal] and info['status']=='confirmed_grokking_with_failure'
    else:assert terminal==100000 and info['status'] in ['confirmed_grokking_event_free_completion','failure_to_confirm_grokking']
    state=torch.load(folder/'terminal.pt',map_location='cpu',weights_only=False);assert state['step']==terminal
    model,opts=r.build(seed,state);data=r.source.generate_modular_addition_data(seed=seed,operation='addition');observed=metrics(model,data)
    assert observed['train']['correct']==int(rows[-1]['train_correct']);assert observed['test']['correct']==int(rows[-1]['test_correct'])
    # Compare every checkpoint configuration and optimizer settings to the frozen choices.
    init=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
    assert all(v.dtype in [torch.float32,torch.bool] for v in init['model_state_dict'].values())
    fresh,freshopts=r.build(seed)
    assert r.equal(fresh.state_dict(),init['model_state_dict'])
    assert r.equal({k:o.state_dict() for k,o in freshopts.items()},init['optimizer_state_dicts'])
    assert torch.equal(torch.get_rng_state(),init['torch_rng_state'])
    for key,wanted in [('muon',dict(learning_rate=.03,momentum=.95,weight_decay=.1,newton_schulz_steps=5,nesterov=True)),('auxiliary_adamw',dict(lr=.001,weight_decay=1.,eps=1e-8,betas=(.9,.999))),('unembedding_adamw',dict(lr=.00025,weight_decay=1.,eps=1e-8,betas=(.9,.999)))]:
        for group in init['optimizer_state_dicts'][key]['param_groups']:
            for name,value in wanted.items():assert group[name]==value,(name,group[name],value)
    return dict(seed=seed,status=info['status'],terminal_step=terminal,confirmation_step=confirmation,first_of_confirmation_streak=initial_streak,all_states_logged=len(rows),test_evaluations=test_evaluations,post_confirmation_test_evaluations=post_count,post_confirmation_below95_evaluations=below,minimum_post_confirmation_sampled_test_accuracy=minimum,terminal=observed,monitoring_recomputed=True,terminal_metrics_recomputed=True,configuration_verified=True)


def audit_analysis(folder):
    result=json.loads((folder/'summary.json').read_text());replay=json.loads((folder/'replay_verification.json').read_text());assert replay['passed']
    protocol=json.loads((folder/'analysis_protocol.json').read_text());validation=np.array(protocol['validation_indices']);feature=np.load(folder/'decoder_features.npz');weights=np.load(folder/'decoder_weights.npz');reports=json.loads((folder/'decoders.json').read_text());checks=[]
    for label,key in [('previous','H0'),('failed','H1')]:
        # Recompute numerical errors directly from stored float32 logits/derivative.
        a=np.load(folder/f'{label}_derivative_arrays.npz');z=a['logits'].astype(np.float64);y=a['targets'];shift=z-z.max(axis=1,keepdims=True);exp=np.exp(shift);den=exp.sum(1,keepdims=True);ref=exp/den;exp[np.arange(len(y)),y]=0;ref[np.arange(len(y)),y]=-exp.sum(1)/den[:,0];ref/=len(y)
        np.testing.assert_allclose(ref,a['reference64_gradient'],rtol=2e-14,atol=1e-300)
        report=json.loads((folder/f'{label}_derivative.json').read_text());diff=a['accurate32_gradient'].astype(np.float64)-ref
        np.testing.assert_allclose(np.linalg.norm(diff)/np.linalg.norm(ref),report['relative_l2_error'],rtol=1e-8,atol=1e-14)
        assert np.array_equal(np.flatnonzero(feature['validation']),validation)
        for kind in ['cv_selected','unregularized_control']:
            rep=reports[kind][label];w=weights[f'{label}_{kind}_readout'];h=feature[key].astype(np.float64)
            assert rep['split']['inner_validation_indices']==validation.tolist()
            scores=h@w
            for split,maskname in [('train','train'),('heldout','heldout')]:
                got=classification_metrics(scores[feature[maskname]],feature['y'][feature[maskname]])
                assert got['correct_count']==rep['classification'][split]['correct_count'];np.testing.assert_allclose(got['cross_entropy'],rep['classification'][split]['cross_entropy'],rtol=1e-12)
            best=min(rep['candidates'],key=lambda x:(x['validation']['cross_entropy'],x['regularization']))
            assert best['regularization']==rep['selected_regularization']
            assert rep['refit']['iteration_limit']==1000
            assert all(x['optimization']['iteration_limit']==500 for x in rep['candidates'])
            checks.append(dict(checkpoint=label,method=kind,heldout_correct=rep['classification']['heldout']['correct_count'],heldout_accuracy=rep['classification']['heldout']['accuracy'],refit_converged=rep['refit']['converged'],candidate_convergence=[x['optimization']['converged'] for x in rep['candidates']]))
    assert [x['mask'] for x in result['swaps']]==[format(k,'03b') for k in range(8)]
    assert result['swaps'][0]['metrics']==result['native_previous'];assert result['swaps'][-1]['metrics']==result['native_failed']
    for entry in json.loads((folder/'artifact_manifest.json').read_text()):assert r.sha(folder/entry['path'])==entry['sha256'],entry['path']
    return dict(seed=result['seed'],cohort=result['cohort'],passed=True,replay_passed=True,derivative_arrays_independently_recomputed=True,decoder_predictions_independently_recomputed=True,validation_indices_verified=True,source_artifact_hashes_verified=True,decoder_checks=checks)


def main(partial=False):
    r.runtime();protocol=json.loads((HERE/'protocol.json').read_text())
    for name,digest in protocol['immutable_hashes'].items():assert r.sha(HERE/name)==digest,name
    for item in json.loads((HERE/'source_manifest.json').read_text()):assert r.sha(HERE/'source_snapshot'/item['source'])==item['sha256']
    runs=[];analyses=[];missing=[];source_summaries=[]
    historical=HERE/'analyses/historical_seed0'
    if (historical/'completion.json').exists():analyses.append(audit_analysis(historical));source_summaries.append(json.loads((historical/'summary.json').read_text()))
    else:missing.append('historical_seed0_event_analysis')
    for seed in [1,2,3,4]:
        folder=HERE/f'runs/seed{seed}'
        if (folder/'completion.json').exists():
            run=audit_run(folder);runs.append(run)
            if run['status']=='confirmed_grokking_with_failure':
                analysis=HERE/f'analyses/prospective_seed{seed}'
                if (analysis/'completion.json').exists():analyses.append(audit_analysis(analysis));source_summaries.append(json.loads((analysis/'summary.json').read_text()))
                else:missing.append(f'seed{seed}_event_analysis')
        else:missing.append(f'seed{seed}_training')
    report=dict(complete=not missing,protocol_sha256=r.sha(HERE/'protocol.json'),immutable_source_hashes_verified=True,runs=runs,event_audits=analyses,remaining=missing)
    r.write_json(HERE/('partial_verification.json' if missing else 'verification.json'),report)
    if missing and not partial:raise RuntimeError('Missing work: '+','.join(missing))
    summary=dict(prospective_seed_count=4,prospective_completed=len(runs),prospective_failure_count=sum(x['status']=='confirmed_grokking_with_failure' for x in runs),historical_pilot_separate=True,historical_reference=dict(seed=0,confirmation_step=19900,first_event_step=28495,monitoring='Historical train inspected everyupdate, CSV every10updates plus events; test every100updates and train triggers; full historical horizon30000'),runs=runs,event_summaries=source_summaries,remaining=missing)
    r.write_json(HERE/('partial_summary.json' if missing else 'summary.json'),summary)
    print(json.dumps(dict(complete=not missing,prospective_completed=len(runs),event_audits=len(analyses),remaining=missing)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
