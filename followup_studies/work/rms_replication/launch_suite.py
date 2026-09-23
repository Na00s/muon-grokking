"""Launch the four frozen prospective CPU runs, then event analyses."""
from pathlib import Path
import subprocess,os,sys,json,time,hashlib
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
assert not (HERE/'launch.json').exists(),'Refusing duplicate launch'
protocol=json.loads((HERE/'protocol.json').read_text())
for name,digest in protocol['immutable_hashes'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
(HERE/'runs').mkdir(exist_ok=True)
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
workers=[];records=[]
for seed in protocol['prospective_seeds']:
    cmd=[sys.executable,str(HERE/'runner.py'),'--seed',str(seed),'--out',str(HERE/f'runs/seed{seed}'),'--steps','100000']
    stream=(HERE/f'runs/seed{seed}.log').open('w')
    worker=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env);workers.append((seed,worker,stream))
    records.append(dict(seed=seed,pid=worker.pid,command=cmd))
(HERE/'launch.json').write_text(json.dumps(dict(launched_at_utc=datetime.now(timezone.utc).isoformat(),supervisor_pid=os.getpid(),environment_overrides={k:env[k] for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONUNBUFFERED']},workers=records),indent=2)+'\n')
print(json.dumps(records),flush=True)
for seed,worker,stream in workers:
    code=worker.wait();stream.close()
    record=dict(seed=seed,returncode=code,finished_at_utc=datetime.now(timezone.utc).isoformat())
    (HERE/f'runs/seed{seed}_process_exit.json').write_text(json.dumps(record,indent=2)+'\n')
(HERE/'training_processes_complete.json').write_text(json.dumps(dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),seeds=protocol['prospective_seeds']),indent=2)+'\n')
