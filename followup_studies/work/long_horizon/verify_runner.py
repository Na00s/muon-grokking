"""Compare the monitored runner with uninterrupted source updates and a resume."""
import contextlib
import copy
import io
import json
from pathlib import Path
import time

import torch
import run_long_horizon as runner

HERE = Path(__file__).resolve().parent


def equal(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a,b), path
        return 1
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        return sum(equal(a[k], b[k], path+'.'+str(k)) for k in a)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        return sum(equal(x,y,path+f'[{i}]') for i,(x,y) in enumerate(zip(a,b)))
    assert a == b, (path,a,b)
    return 0


def compare(a,b):
    return sum(equal(a[key],b[key],key) for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state'])


def cases():
    result = [(f'addition_seed{seed}', HERE.parent/'causal_study'/'main_runs'/f'seed{seed}_accurate6000'/'final.pt','addition') for seed in range(5)]
    result.append(('subtraction_seed0', HERE.parent/'causal_study'/'generality'/'subtraction_accurate'/'final.pt','subtraction'))
    return result


def main():
    runner.configure_runtime(1)
    results = []
    for name,path,operation in cases():
        source = torch.load(path,map_location='cpu',weights_only=False)
        start,seed = source['step'],source['seed']
        model,opts,loss_fn = runner.make_arm(source,'stable32')
        tx,ty,vx,vy = runner.original.generate_modular_addition_data(seed=seed,operation=operation)
        initial_count = compare(source, runner.original.snapshot(model,opts,start,seed))
        begin = time.perf_counter()
        middle = None
        for i in range(12):
            runner.original.train_step(model,opts,tx,ty,loss_fn)
            if i == 5:
                middle = runner.original.snapshot(model,opts,start+6,seed)
        direct = runner.original.snapshot(model,opts,start+12,seed)
        direct_seconds = time.perf_counter()-begin
        resumed_model,resumed_opts,resumed_loss = runner.make_arm(middle,'stable32')
        for i in range(6):
            runner.original.train_step(resumed_model,resumed_opts,tx,ty,resumed_loss)
        resumed = runner.original.snapshot(resumed_model,resumed_opts,start+12,seed)
        resume_count = compare(direct,resumed)
        out = HERE/'verification_runs'/name
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = runner.run(path,out,start+12,operation=operation)
        monitored = torch.load(out/'final.pt',map_location='cpu',weights_only=False)
        monitored_count = compare(direct,monitored)
        result = dict(name=name,source_sha256=runner.file_sha256(path),source_step=start,
            initial_tensor_checks=initial_count,resume_tensor_checks=resume_count,monitored_tensor_checks=monitored_count,
            uninterrupted_equals_split_resume=True,uninterrupted_equals_monitored=True,
            uninterrupted_seconds_per_update=direct_seconds/12,monitored_seconds_per_update=metrics['elapsed_seconds']/12)
        results.append(result)
        print(json.dumps(result),flush=True)
    report = dict(status='passed',checks=results,runner_sha256=runner.file_sha256(runner.__file__))
    runner.write_json(HERE/'runner_verification.json',report)


if __name__ == '__main__': main()
