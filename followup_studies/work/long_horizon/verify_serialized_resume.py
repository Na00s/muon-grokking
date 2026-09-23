"""Verify a disk checkpoint round trip in a late corrected-loss state."""
import json
from pathlib import Path

import torch

import run_long_horizon as runner
from verify_runner import compare

HERE = Path(__file__).resolve().parent


def main():
    runner.configure_runtime(1)
    plan = json.loads((HERE/'protocol.json').read_text())
    records = []
    for job in plan['jobs']:
        source_path = HERE/'runs'/job['name']/'step_090000.pt'
        source = torch.load(source_path,map_location='cpu',weights_only=False)
        model,opts,loss_fn = runner.make_arm(source,'stable32')
        tx,ty,_,_ = runner.original.generate_modular_addition_data(seed=source['seed'],operation=job['operation'])
        out = HERE/'verification_runs'/'serialized_resume'/job['name']
        out.mkdir(parents=True,exist_ok=False)
        for i in range(12):
            runner.original.train_step(model,opts,tx,ty,loss_fn)
            if i == 5:
                middle = runner.original.snapshot(model,opts,90006,source['seed'])
                torch.save(middle,out/'midpoint.pt')
        direct = runner.original.snapshot(model,opts,90012,source['seed'])
        restored_middle = torch.load(out/'midpoint.pt',map_location='cpu',weights_only=False)
        roundtrip_tensors = compare(middle,restored_middle)
        resumed_model,resumed_opts,resumed_loss = runner.make_arm(restored_middle,'stable32')
        for i in range(6):
            runner.original.train_step(resumed_model,resumed_opts,tx,ty,resumed_loss)
        resumed = runner.original.snapshot(resumed_model,resumed_opts,90012,source['seed'])
        replay_tensors = compare(direct,resumed)
        records.append(dict(name=job['name'],source_step=90000,end_step=90012,
            source_sha256=runner.file_sha256(source_path),disk_roundtrip_tensor_checks=roundtrip_tensors,
            resumed_tensor_checks=replay_tensors,uninterrupted_equals_disk_resumed=True))
    runner.write_json(HERE/'serialized_resume_verification.json',dict(status='passed',checks=records))
    print(json.dumps(dict(status='passed',checks=records),indent=2))


if __name__ == '__main__': main()
