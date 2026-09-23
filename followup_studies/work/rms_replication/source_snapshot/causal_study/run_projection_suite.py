"""Prospectively recorded secondary zero-sum-gradient projection controls."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
from run_main_suite import base,sha,now

HERE=Path(__file__).resolve().parent

def main():
    protocol=dict(created_utc=now(),seeds=list(range(5)),start_step=15000,end_step=20000,workers=2,
        arm='row_projection',endpoint='Any joint train/test accuracy below90% within the resumed15000-to20000 window.',
        motivation='Separate common-class gradient error from remaining class-relative error, motivated by prior fixed-logit diagnostics and Hanqing et al. arXiv2605.06152v2 zero-sum projection experiments.',
        limitation='The projection retains most fixed-logit L2 derivative error. Failure would not refute target-derivative cancellation; rescue would support a consequential class-common component.',
        runner_sha256=sha(HERE/'run_dense_arithmetic.py'),
        metadata_erratum='The runner primary_endpoint string says30000 for every arm. This protocol and actual end_step=20000 govern the specificity and projection windows.')
    path=HERE/'projection_protocol.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def worker(seed):
        root=HERE/'projection_runs';root.mkdir(exist_ok=True)
        out=root/f'seed{seed}';checkpoint=base(seed)/'step_015000.pt'
        command=[sys.executable,str(HERE/'run_dense_arithmetic.py'),'--checkpoint',str(checkpoint),'--out',str(out),'--arm','row_projection','--end-step','20000']
        print(json.dumps(dict(seed=seed,status='started',time=now())),flush=True)
        with (root/f'seed{seed}.log').open('w') as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(seed=seed,exit_code=r.returncode,time=now())),flush=True)
        return r.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:codes=list(pool.map(worker,range(5)))
    if any(codes):raise RuntimeError(codes)
    print('All projection controls complete.',flush=True)

if __name__=='__main__':main()
