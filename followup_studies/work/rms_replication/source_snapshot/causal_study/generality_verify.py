"""Independent completion, paired-state, checkpoint and source audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch
import generality_runner as g
from test_generality_runner import equal_tree


def file_hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(base):
    expected={
        'subtraction_stock':('subtraction','none','stock'),
        'subtraction_accurate':('subtraction','none','accurate'),
        'rms_stock':('addition','rms','stock'),
        'rms_accurate':('addition','rms','accurate')}
    initial_states={};checkpoints=[];runs=[]
    source_hash=file_hash(Path(g.__file__))
    for name,condition in expected.items():
        folder=base/name
        metadata=json.loads((folder/'metadata.json').read_text())
        completion=json.loads((folder/'completion.json').read_text())
        assert metadata['script_sha256']==source_hash
        assert tuple(metadata[key] for key in ['operation','normalization','arithmetic'])==condition
        assert completion['steps']==30000
        assert (folder/'analysis_completion.json').is_file()
        states={}
        for path in sorted(folder.rglob('*.pt')):
            state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['seed']==0
            assert tuple(state[key] for key in ['operation','normalization','arithmetic'])==condition
            assert all(bool(torch.isfinite(value).all()) for value in state['model_state_dict'].values())
            checkpoints.append(dict(path=str(path.relative_to(base)),step=int(state['step']),
                                    size_bytes=path.stat().st_size,sha256=file_hash(path)))
            if path.parent==folder and path.name in ['initial.pt','final.pt','step_030000.pt','previous.pt','collapse.pt']:
                states[path.name]=state
        assert len(list(folder.glob('step_*.pt')))==31
        for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state']:
            assert equal_tree(states['final.pt'][key],states['step_030000.pt'][key])
        assert states['final.pt']['step']==30000 and states['initial.pt']['step']==0
        initial_states[name]=states['initial.pt']
        rows=list(csv.DictReader((folder/'trajectory.csv').open()))
        assert int(rows[0]['step'])==0 and int(rows[-1]['step'])==30000
        assert all(int(left['step'])<int(right['step']) for left,right in zip(rows,rows[1:]))
        qualified=[row for row in rows if row['grok_confirmed']=='True']
        assert (len(qualified)>0)==(completion['grok_step'] is not None)
        if qualified: assert int(qualified[0]['step'])==completion['grok_step']
        event_rows=[row for row in rows if row['joint_failure']=='True']
        assert (len(event_rows)>0)==(completion['event_step'] is not None)
        if event_rows:
            assert int(event_rows[0]['step'])==completion['event_step']
            assert states['previous.pt']['step']+1==states['collapse.pt']['step']==completion['event_step']
            event=json.loads((folder/'event_analysis.json').read_text())
            assert event['exact_next_update_replay'] is True
            for kind,filename in [('previous','previous.pt'),('collapse','collapse.pt')]:
                assert event['checkpoint_sha256'][kind]==file_hash(folder/filename)
        analysis=json.loads((folder/'analysis_completion.json').read_text())
        assert analysis['runner_script_sha256']==source_hash
        assert analysis['analysis_script_sha256']==file_hash(Path(__file__).with_name('generality_analyze.py'))
        timecourse=json.loads((folder/'gradient_timecourse.json').read_text())
        assert len(timecourse)==8
        for entry in timecourse:
            assert entry['checkpoint_sha256']==file_hash(folder/f"step_{entry['step']:06d}.pt")
        peak_folder=folder/'peak_event'
        if peak_folder.exists():
            replay=json.loads((peak_folder/'completion.json').read_text())
            assert replay['exact_saved_peak_replay'] is True
            assert replay['target_step']==completion['peak_step']
            assert replay['recorded_peak_sha256']==file_hash(folder/'peak.pt')
            original_peak=torch.load(folder/'peak.pt',map_location='cpu',weights_only=False)
            replay_peak=torch.load(peak_folder/'collapse.pt',map_location='cpu',weights_only=False)
            for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state']:
                assert equal_tree(original_peak[key],replay_peak[key])
            peak_analysis=json.loads((peak_folder/'event_analysis.json').read_text())
            assert peak_analysis['exact_next_update_replay'] is True
            for kind,filename in [('previous','previous.pt'),('collapse','collapse.pt')]:
                assert peak_analysis['checkpoint_sha256'][kind]==file_hash(peak_folder/filename)
            panel=json.loads((peak_folder/'healthy_hybrids.json').read_text())
            assert panel['healthy_checkpoint_sha256']==file_hash(peak_folder/'healthy.pt')
            assert panel['peak_checkpoint_sha256']==file_hash(peak_folder/'collapse.pt')
            assert len(panel['parameter_hybrids'])==8
        runs.append(dict(name=name,steps=completion['steps'],grok_step=completion['grok_step'],event_step=completion['event_step'],
                         checkpoint_count=len(list(folder.rglob('*.pt'))),saved_rows=len(rows),gradient_checkpoints=len(timecourse),
                         peak_replay_updates=replay['replayed_updates'] if peak_folder.exists() else 0))
    for first,second in [('subtraction_stock','subtraction_accurate'),('rms_stock','rms_accurate')]:
        for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state']:
            assert equal_tree(initial_states[first][key],initial_states[second][key])
    assert equal_tree(initial_states['subtraction_stock']['model_state_dict'],initial_states['rms_stock']['model_state_dict'])
    result=dict(all_checks_passed=True,completed_updates=sum(row['steps'] for row in runs),
                peak_replay_updates=sum(row['peak_replay_updates'] for row in runs),
                paired_initial_model_optimizer_rng_exact=True,cross_architecture_initial_tensor_exact=True,
                runner_script_sha256=source_hash,runs=runs,checkpoint_count=len(checkpoints))
    (base/'checkpoint_manifest.json').write_text(json.dumps(checkpoints,indent=2))
    (base/'completion_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,default=Path(__file__).parent/'generality')
    g.configure_runtime();run(parser.parse_args().base)
