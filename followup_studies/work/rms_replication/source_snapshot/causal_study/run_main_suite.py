"""Run the fixed primary and targeted arithmetic comparisons with bounded workers."""
import concurrent.futures
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

HERE=Path(__file__).resolve().parent
LOCK=threading.Lock()

def now():return datetime.now(timezone.utc).isoformat()
def base(seed):return HERE.parent/('experiments' if seed in (0,4) else 'basis_study')/f'seed{seed}_{"original" if seed in (0,4) else "baseline"}'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    jobs=[]
    for seed in range(5):
        jobs.append(dict(name=f'seed{seed}_accurate6000',seed=seed,arm='stable32',checkpoint=str(base(seed)/'step_006000.pt'),end=30000))
    for seed in range(5):
        for arm in ('target_repair','stable32'):
            jobs.append(dict(name=f'seed{seed}_{arm}15000',seed=seed,arm=arm,checkpoint=str(base(seed)/'step_015000.pt'),end=20000))
    for seed in range(4):
        checkpoint=base(seed)/('step_018000.pt' if seed==0 else 'final.pt')
        jobs.append(dict(name=f'seed{seed}_stock_extension',seed=seed,arm='original32',checkpoint=str(checkpoint),end=30000))
    for job in jobs:
        assert Path(job['checkpoint']).is_file(),job
        job['checkpoint_sha256']=sha(Path(job['checkpoint']))
    plan=dict(created_utc=now(),workers=6,jobs=jobs,
              primary='Five matched arithmetic contrasts from saved step6000 through30000; cached stock prefixes contain verified positive joint-collapse events.',
              specificity='Five paired accurate-CE and target-derivative-only repair continuations from step15000 through20000.',
              stock_extensions='Four originals resumed through30000. Seed4 already reached30000.',
              interpretation='Fixed finite horizons. Mixed cached-stock monitoring supports binary incidence only. Detailed rules are in claim_adjudication.md.',
              runner_sha256=sha(HERE/'run_dense_arithmetic.py'),target_helper_sha256=sha(HERE/'diagnostics_repairs.py'))
    plan_path=HERE/'main_suite_plan.json'
    if plan_path.exists():raise FileExistsError('Plan exists; inspect it and progress before resuming.')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    status={job['name']:dict(status='queued') for job in jobs}
    def update(name,**items):
        with LOCK:
            status[name].update(items)
            temporary=HERE/'main_suite_progress.tmp.json'
            temporary.write_text(json.dumps(dict(updated_utc=now(),jobs=status),indent=2)+'\n')
            temporary.replace(HERE/'main_suite_progress.json')
            print(json.dumps(dict(job=name,**items)),flush=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def worker(job):
        name=job['name'];out=HERE/'main_runs'/name;out.parent.mkdir(exist_ok=True)
        update(name,status='running',started_utc=now())
        command=[sys.executable,str(HERE/'run_dense_arithmetic.py'),'--checkpoint',job['checkpoint'],'--out',str(out),'--arm',job['arm'],'--end-step',str(job['end'])]
        with (HERE/'main_runs'/f'{name}.log').open('w') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        update(name,status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,ended_utc=now())
        return result.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        codes=list(pool.map(worker,jobs))
    if any(codes):raise RuntimeError(f'Failed jobs: {sum(code!=0 for code in codes)}')
    print('All fixed main-suite jobs completed.',flush=True)

if __name__=='__main__':main()
