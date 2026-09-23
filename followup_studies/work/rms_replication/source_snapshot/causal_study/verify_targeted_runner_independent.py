"""Independent three-update audit of targeted dense arithmetic branches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from diagnostics_repairs import ce_target_repair, ce_row_projection
from run_dense_arithmetic import run
from run_precision_branches import make_arm, configure_runtime
import run_collapse as original

HERE=Path(__file__).resolve().parent


def compare(a,b,path="root"):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
            raise AssertionError(f"Tensor mismatch at {path}")
        return 1
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():raise AssertionError(f"Dictionary mismatch at {path}")
        return sum(compare(a[k],b[k],f"{path}.{k}") for k in a)
    if isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):raise AssertionError(f"Sequence mismatch at {path}")
        return sum(compare(x,y,f"{path}[{i}]") for i,(x,y) in enumerate(zip(a,b)))
    if a!=b:raise AssertionError(f"Scalar mismatch at {path}: {a!r} != {b!r}")
    return 0


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    configure_runtime(1)
    checkpoint=HERE.parent/"experiments/seed4_original/step_015000.pt"
    source=torch.load(checkpoint,map_location="cpu",weights_only=False)
    report={"source_checkpoint":str(checkpoint),"source_sha256":digest(checkpoint),
            "runner_sha256":digest(HERE/"run_dense_arithmetic.py"),
            "helper_sha256":digest(HERE/"diagnostics_repairs.py"),
            "method":"Three direct training updates without dense monitoring versus the dense runner. Exact tensor dtype, shape, value, scalar state, and RNG comparisons.",
            "arms":{}}
    for arm,loss_fn in (("target_repair",ce_target_repair),("row_projection",ce_row_projection)):
        model,opts,_=make_arm(source,"original32")
        tx,ty,_,_=original.generate_modular_addition_data(seed=4)
        for _ in range(3):
            model.train()
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(tx),ty)
            loss.backward()
            for opt in opts.values():opt.step()
        expected=original.snapshot(model,opts,15003,4)
        out=HERE/"targeted_runner_independent_verification"/arm
        summary=run(checkpoint,out,arm,15003)
        actual=torch.load(out/"final.pt",map_location="cpu",weights_only=False)
        checks={}
        for key in ("model_state_dict","optimizer_state_dicts","torch_rng_state"):
            tensors=compare(expected[key],actual[key],key)
            checks[key]={"bitwise_equal":True,"tensor_count":tensors}
        assert summary["end_step"]==15003 and summary["updates"]==3
        assert actual["step"]==15003
        report["arms"][arm]={"checks":checks,"updates":3,"passed":True,"output":str(out)}
    dest=HERE/"targeted_runner_independent_verification.json"
    dest.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
