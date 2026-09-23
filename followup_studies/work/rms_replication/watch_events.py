"""Analyze each completed prospective event once, preserving terminal evidence."""
from pathlib import Path
import os,subprocess,sys,time,json
HERE=Path(__file__).resolve().parent
lock=HERE/'analysis_supervisor.json'
with lock.open('x') as f:json.dump(dict(pid=os.getpid(),started=time.time(),command=sys.argv),f,indent=2)
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
remaining=set([1,2,3,4]);records=[]
while remaining:
    for seed in sorted(remaining):
        folder=HERE/f'runs/seed{seed}';completion=folder/'completion.json';interruption=folder/'interruption.json'
        if completion.exists():
            info=json.loads(completion.read_text());record=dict(seed=seed,status=info['status'])
            if info['status']=='confirmed_grokking_with_failure':
                cmd=[sys.executable,str(HERE/'analyze_event.py'),'--path',str(folder),'--out',str(HERE/f'analyses/prospective_seed{seed}'),'--cohort','prospective_cpu']
                with (HERE/f'analyses_seed{seed}.log').open('w') as log:
                    proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
                record.update(analysis_returncode=proc.returncode,command=cmd)
            else:record['event_analysis']='No captured event to analyze'
            records.append(record);remaining.remove(seed);print(json.dumps(record),flush=True)
        elif interruption.exists():
            records.append(dict(seed=seed,status='technical_interruption',record=str(interruption)));remaining.remove(seed)
    (HERE/'analysis_progress.json').write_text(json.dumps(dict(completed=records,awaiting_training=sorted(remaining)),indent=2)+'\n')
    if remaining:time.sleep(5)
(HERE/'analysis_supervisor_completion.json').write_text(json.dumps(records,indent=2)+'\n')
