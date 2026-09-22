"""Reproduce a saved worst sampled state and retain its exact preceding state."""
import argparse
import csv
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import generality_runner as g
import generality_analyze as a


def run(folder):
    target=a.load(folder/'peak.pt');target_step=int(target['step'])
    start_step=(target_step//1000)*1000
    if start_step==target_step:start_step-=1000
    source=folder/f'step_{start_step:06d}.pt';start=a.load(source)
    out=folder/'peak_event';out.mkdir(exist_ok=False)
    model,opts=a.build(start)
    data=g.generate_modular_addition_data(seed=start['seed'],operation=start['operation'])
    x,y,vx,vy=data
    fn=F.cross_entropy if start['arithmetic']=='stock' else g.accurate_cross_entropy
    def snapshot(step):return g.snapshot(model,opts,step,start['seed'],start['operation'],start['normalization'],start['arithmetic'])
    healthy=None
    with (out/'replay.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['step','train_accuracy','test_accuracy']);writer.writeheader()
        for step in range(start_step,target_step+1):
            model.train()
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            logits=model(x);loss=fn(logits,y)
            train=float((logits.detach().argmax(-1)==y).double().mean())
            te=g.evaluate_preserving_runtime(model,vx,vy) if step%10==0 or train<.9 or step==target_step else None
            if te:writer.writerow(dict(step=step,train_accuracy=train,test_accuracy=te['accuracy']));file.flush()
            if te and train>=.99 and te['accuracy']>=.99:healthy=snapshot(step)
            if step==target_step-1:torch.save(snapshot(step),out/'previous.pt')
            if step==target_step:
                actual=snapshot(step)
                for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state']:
                    assert a.equal_tree(actual[key],target[key]),key
                torch.save(actual,out/'collapse.pt')
                if healthy is not None:torch.save(healthy,out/'healthy.pt')
                break
            loss.backward()
            for opt in opts.values():opt.step()
    metadata=dict(selection='Worst sampled held-out checkpoint in the complete registered horizon',
        start_step=start_step,target_step=target_step,healthy_step=healthy['step'] if healthy else None,
        replayed_updates=target_step-start_step,exact_saved_peak_replay=True,
        source_checkpoint_sha256=a.sha256(source),recorded_peak_sha256=a.sha256(folder/'peak.pt'),
        replay_script_sha256=a.sha256(Path(__file__)))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    a.event_report(out,data)
    (out/'completion.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--path',type=Path,required=True)
    args=parser.parse_args();g.configure_runtime();run(args.path)
