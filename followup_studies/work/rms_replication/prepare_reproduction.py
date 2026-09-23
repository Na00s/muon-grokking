"""Prepare a fresh sibling RMS reproduction from supplied source copies.

This administrative wrapper copies files and records their hashes. It performs
no training, analysis fit, optimizer update, or change to the delivered results.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();out=args.out.resolve()
 if out.parent!=HERE.parent:raise ValueError('Choose a new sibling under followup_studies/work so shared historical inputs resolve correctly.')
 out.mkdir(exist_ok=False)
 copied=[]
 for src in sorted(HERE.glob('*.py')):
  if src.name=='prepare_reproduction.py':continue
  dst=out/src.name;shutil.copyfile(src,dst);copied.append({'path':src.name,'source_sha256':sha(src),'copy_sha256':sha(dst)})
 shutil.copytree(HERE/'source_snapshot',out/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copytree(HERE/'historical_reference',out/'historical_reference',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 manifest=[]
 for src in sorted((out/'source_snapshot').rglob('*.py')):
  manifest.append({'source':str(src.relative_to(out/'source_snapshot')),'sha256':sha(src),'bytes':src.stat().st_size})
 dump(out/'source_manifest.json',manifest)
 dump(out/'analysis_source_registration.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Fresh reproduction registration of the supplied anonymous analysis implementations before fitting. Delivered historical registration remains unchanged.','files':{n:sha(out/n) for n in ['analyze_event.py','verify_and_summarize.py','write_report.py','watch_events.py']}})
 dump(out/'reproduction_preparation.json',{'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'origin':'Delivered RMS anonymous study','delivered_protocol_sha256':sha(HERE/'protocol.json'),'delivered_source_manifest_sha256':sha(HERE/'source_manifest.json'),'copied_scripts':copied,'training_updates':0,'probe_fits':0,'next_steps':['verify_runner.py','freeze_protocol.py','launch_suite.py','analyze_event.py for the historical_reference','watch_events.py','verify_and_summarize.py','write_report.py']})
 print(json.dumps({'prepared':out.name,'source_files':len(manifest),'training_updates':0,'probe_fits':0}))
if __name__=='__main__':main()
