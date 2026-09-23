"""Check that a deliberately nonfinite state is retained and fails explicitly."""
import json
from pathlib import Path

import torch

import run_long_horizon as runner

HERE = Path(__file__).resolve().parent


def main():
    source_path = HERE.parent/'causal_study'/'main_runs'/'seed0_accurate6000'/'final.pt'
    source_hash = runner.file_sha256(source_path)
    source = torch.load(source_path,map_location='cpu',weights_only=False)
    parameter_name = next(iter(source['model_state_dict']))
    source['model_state_dict'][parameter_name].flatten()[0] = float('nan')
    root = HERE/'verification_runs'/'nonfinite_injection'
    root.mkdir(parents=True,exist_ok=False)
    injected = root/'injected_source.pt'
    torch.save(source,injected)
    error = None
    try:
        runner.run(injected,root/'run',int(source['step'])+1)
    except FloatingPointError as failure:
        error = str(failure)
    assert error is not None
    assert (root/'run'/'nonfinite.pt').is_file()
    assert (root/'run'/'failure.json').is_file()
    assert not (root/'run'/'summary.json').exists()
    assert not (root/'run'/'final.pt').exists()
    assert runner.file_sha256(source_path) == source_hash
    retained = torch.load(root/'run'/'nonfinite.pt',map_location='cpu',weights_only=False)
    assert torch.isnan(retained['model_state_dict'][parameter_name]).any()
    result = dict(status='passed',injected_parameter=parameter_name,error=error,
                  source_unchanged=True,nonfinite_state_retained=True,failure_explicit=True,
                  completed_summary_absent=True)
    runner.write_json(HERE/'failure_verification.json',result)
    print(json.dumps(result))


if __name__ == '__main__': main()
