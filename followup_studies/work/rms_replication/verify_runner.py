"""Independent source-update and monitored-update checks before cohort launch."""
import copy,json,time
from pathlib import Path
import torch
import runner as r
HERE=Path(__file__).resolve().parent
HIST=HERE.parent/'causal_study/generality/rms_accurate'

def check_tree(a,b,label):
    assert r.equal(a,b),label

def main():
    env=r.runtime();records=[]
    seed=0;data=r.source.generate_modular_addition_data(seed=seed,operation='addition')
    hist=torch.load(HIST/'initial.pt',map_location='cpu',weights_only=False)
    model,opts=r.build(0)
    check_tree(model.state_dict(),hist['model_state_dict'],'pilot_initial_model')
    check_tree({k:o.state_dict() for k,o in opts.items()},hist['optimizer_state_dicts'],'pilot_initial_optimizers')
    check_tree(torch.get_rng_state(),hist['torch_rng_state'],'pilot_initial_torch_rng')
    for origin in [hist,torch.load(HIST/'previous.pt',map_location='cpu',weights_only=False)]:
        manual,mopts=r.build(0,origin)
        before=r.snapshot(manual,mopts,origin['step'],0)
        r.source.original_step(manual,mopts,*data[:2],loss_fn=r.source.accurate_cross_entropy)
        expected=r.snapshot(manual,mopts,origin['step']+1,0)
        monitored,opts=r.build(0,before)
        z,loss=r.train_forward(monitored,opts,*data[:2]);train_correct=int((z.detach().argmax(-1)==data[1]).sum());test=r.evaluate(monitored,*data[2:])
        # Same immediately preceding snapshot work as the live monitor.
        checkpoint=r.snapshot(monitored,opts,origin['step'],0)
        r.apply(loss,opts);actual=r.snapshot(monitored,opts,origin['step']+1,0)
        for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state','rng_states']:check_tree(actual[key],expected[key],key)
        if origin['step']>0:
            captured=torch.load(HIST/'collapse.pt',map_location='cpu',weights_only=False)
            for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state']:check_tree(actual[key],captured[key],'historical_replay_'+key)
        restored,ropts=r.build(0,checkpoint);rz,rloss=r.train_forward(restored,ropts,*data[:2]);r.apply(rloss,ropts);restored_state=r.snapshot(restored,ropts,origin['step']+1,0)
        for key in ['model_state_dict','optimizer_state_dicts','rng_states']:check_tree(restored_state[key],expected[key],'checkpoint_restore_'+key)
        records.append(dict(origin_step=int(origin['step']),manual_monitored_model_optimizer_rng_exact=True,restored_replay_exact=True,train_correct=train_correct,test=test,historical_event_exact=origin['step']>0))
    model,opts=r.build(1);data=r.source.generate_modular_addition_data(seed=1,operation='addition');start=time.perf_counter()
    for step in range(30):
        z,loss=r.train_forward(model,opts,*data[:2]);n=int((z.detach().argmax(-1)==data[1]).sum());cp=r.snapshot(model,opts,step,1);r.apply(loss,opts)
    result=dict(passed=True,pilot_initial_exact=True,environment=env,checks=records,benchmark=dict(updates=30,seconds_per_update=(time.perf_counter()-start)/30),source_manifest_sha256=r.sha(HERE/'source_manifest.json'))
    r.write_json(HERE/'runner_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
