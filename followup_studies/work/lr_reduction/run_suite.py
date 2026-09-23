"""Launch five fixed-horizon learning-rate continuations and audit completion."""
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from run_lr_reduction import file_sha256, write_json
HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()
def now(): return datetime.now(timezone.utc).isoformat()

def main():
    plan = json.loads((HERE/'protocol.json').read_text())
    verified = json.loads((HERE/'runner_verification.json').read_text())
    assert verified['status'] == 'passed'
    assert plan['runner_sha256'] == file_sha256(HERE/'run_lr_reduction.py')
    assert verified['runner_sha256'] == plan['runner_sha256']
    assert len(plan['jobs']) == 5 and sum(j['end_step']-j['start_step'] for j in plan['jobs']) == 470000
    assert not (HERE/'launch.json').exists(), 'A launch already exists; inspect before resuming.'
    for job in plan['jobs']:
        assert file_sha256(HERE.parent/job['checkpoint']) == job['source_checkpoint_sha256']
        assert not (HERE/'runs'/job['name']).exists()
    write_json(HERE/'launch.json', dict(started_utc=now(), pid=os.getpid(), workers=5, protocol_sha256=file_sha256(HERE/'protocol.json')))
    status = {j['name']:dict(status='queued') for j in plan['jobs']}
    (HERE/'runs').mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    def update(name, **kwargs):
        with LOCK:
            status[name].update(kwargs)
            write_json(HERE/'progress.json',dict(updated_utc=now(),jobs=status))
            print(json.dumps(dict(job=name,**kwargs)),flush=True)
    def worker(job):
        name=job['name'];update(name,status='running',started_utc=now())
        cmd=[sys.executable,str(HERE/'run_lr_reduction.py'),'--checkpoint',str(HERE.parent/job['checkpoint']),
             '--out',str(HERE/'runs'/name),'--end-step',str(job['end_step'])]
        with (HERE/'runs'/f'{name}.log').open('w') as log:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            update(name,pid=p.pid)
            rc=p.wait()
        update(name,status='completed' if rc==0 else 'failed',exit_code=rc,ended_utc=now())
        return rc
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        codes=list(pool.map(worker,plan['jobs']))
    if any(codes):
        write_json(HERE/'failure.json',dict(status='failed',ended_utc=now(),jobs=status))
        raise RuntimeError('One or more continuations failed; retain all records and inspect.')
    write_json(HERE/'training_completion.json',dict(status='completed',ended_utc=now(),jobs=status,completed_updates=470000))
    auditor=HERE/'summarize_and_verify.py'
    if not auditor.exists():
        raise FileNotFoundError('Training completed; completion auditor is missing.')
    with (HERE/'completion_audit.log').open('w') as log:
        result=subprocess.run([sys.executable,str(auditor)],env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(HERE/'audit_failure.json',dict(status='verification_failed',exit_code=result.returncode))
        raise RuntimeError('Training completed; final verification requires inspection.')
    write_json(HERE/'completion.json',dict(status='completed_and_verified',ended_utc=now(),completed_updates=470000,
         summary_sha256=file_sha256(HERE/'summary.json'),verification_sha256=file_sha256(HERE/'verification.json')))
    print('All five continuations completed and verified.',flush=True)

if __name__=='__main__': main()
