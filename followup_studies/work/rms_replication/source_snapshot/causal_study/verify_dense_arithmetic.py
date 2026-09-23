"""Check monitoring and diagnostics preserve exact updates in both loss arms."""
import json
from pathlib import Path
import torch
from run_dense_arithmetic import run,make_arm,original,configure_runtime

HERE=Path(__file__).resolve().parent


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    configure_runtime(1)
    checkpoint=HERE.parent/'experiments'/'seed4_original'/'step_006000.pt'
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    report={}
    for arm in ['original32','stable32']:
        model,opts,loss=make_arm(source,arm)
        tx,ty,_,_=original.generate_modular_addition_data(seed=4)
        for _ in range(3):original.train_step(model,opts,tx,ty,loss_fn=loss)
        expected=original.snapshot(model,opts,6003,4)
        out=HERE/'runner_verification'/arm
        run(checkpoint,out,arm,6003)
        actual=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
        checks={k:same(actual[k],expected[k]) for k in ['model_state_dict','optimizer_state_dicts','torch_rng_state']}
        assert all(checks.values()),checks
        report[arm]=checks
    (HERE/'runner_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
