"""Run final independent audits and preserve a canonical copy after completion."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
HERE=Path(__file__).resolve().parent
TARGET=HERE.parent/'muon-grokking/followup_studies/work/rms_replication'
with (HERE/'completion_supervisor.json').open('x') as stream:json.dump(dict(pid=os.getpid(),started=time.time()),stream,indent=2)
while not ((HERE/'analysis_supervisor_completion.json').exists() and (HERE/'training_processes_complete.json').exists() and (HERE/'analyses/historical_seed0/completion.json').exists()):time.sleep(5)
commands=[[sys.executable,str(HERE/'verify_and_summarize.py')],[sys.executable,str(HERE/'write_report.py')]]
for index,cmd in enumerate(commands):
    with (HERE/f'completion_audit_{index}.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        (HERE/'completion_failure.json').write_text(json.dumps(dict(command=cmd,returncode=result.returncode,log=f'completion_audit_{index}.log'),indent=2)+'\n');raise SystemExit(result.returncode)
manifest=[]
for p in sorted(HERE.rglob('*.pt')):
    manifest.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(HERE/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'completion_status.json').write_text(json.dumps(dict(verified=True,completed_at_unix=time.time(),checkpoint_files=len(manifest),checkpoint_bytes=sum(x['bytes'] for x in manifest),execution_commands=commands,canonical_copy=str(TARGET)),indent=2)+'\n')
if TARGET.exists():raise FileExistsError('Canonical RMS destination exists; refusing overwrite')
shutil.copytree(HERE,TARGET,ignore=shutil.ignore_patterns('__pycache__','*.tmp'))
print(json.dumps(dict(verified=True,canonical_copy=str(TARGET),checkpoint_files=len(manifest))),flush=True)
