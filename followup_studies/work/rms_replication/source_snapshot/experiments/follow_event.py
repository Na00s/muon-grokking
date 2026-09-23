"""Run predefined checkpoint probes and precision arms after event capture."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).resolve().parent
    environment={**os.environ,'OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    print('Waiting for a qualifying event or the fixed baseline horizon.',flush=True)
    while not (args.baseline/'event.json').exists():
        if (args.baseline/'completion.json').exists():
            (args.out/'completion.json').write_text(json.dumps(dict(status='no_qualifying_event',baseline=str(args.baseline)),indent=2))
            print('No qualifying baseline event within the predefined horizon.',flush=True)
            return
        time.sleep(5)
    event=json.loads((args.baseline/'event.json').read_text())
    step=event['step']
    with (args.baseline/'trajectory.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    healthy=[row for row in rows if step-2000<=int(row['step'])<step and float(row['train_accuracy'])>=.99 and float(row['test_accuracy'])>=.99]
    selection_note='Latest >=99% train and test checkpoint within the preceding 2,000 steps.'
    if not healthy:
        healthy=[row for row in rows if int(row['step'])<step and int(row['step'])%1000==0 and float(row['train_accuracy'])>=.99 and float(row['test_accuracy'])>=.99]
        selection_note='No >=99% checkpoint in the recent window; used latest qualifying archived 1,000-step snapshot. The longer interval is a protocol departure.'
    if not healthy:
        raise RuntimeError('No saved healthy checkpoint meets the >=99% threshold anywhere in this run')
    healthy_step=int(healthy[-1]['step'])
    healthy_path=args.baseline/f'event_window_{healthy_step:06d}.pt'
    if not healthy_path.exists():healthy_path=args.baseline/f'step_{healthy_step:06d}.pt'
    collapsed_path=args.baseline/f'event_window_{step:06d}.pt'
    branch_path=args.baseline/f'event_window_{step-1000:06d}.pt'
    specification=dict(event_step=step,healthy_step=healthy_step,branch_start_step=step-1000,branch_end_step=step+1000,
        eval_every=10,save_every=100,arms=['original32','stable32','model64','true64'],
        primary_selection='First qualifying event at 100-step baseline sampling.',healthy_selection_note=selection_note)
    (args.out/'specification.json').write_text(json.dumps(specification,indent=2))
    print(json.dumps(specification),flush=True)
    children=[]
    for arm in specification['arms']:
        log=(args.out/f'{arm}.log').open('w')
        command=[sys.executable,str(scripts/'run_precision_branches.py'),'--checkpoint',str(branch_path),'--out',str(args.out/arm),
                 '--arm',arm,'--steps','2000','--threads','1','--eval-every','10','--save-every','100']
        child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        children.append((arm,child,log))
    analysis_log=(args.out/'alignment.log').open('w')
    command=[sys.executable,str(scripts/'run_alignment.py'),'--healthy',str(healthy_path),'--collapsed',str(collapsed_path),
             '--out',str(args.out/'alignment'),'--decoder']
    analysis_code=subprocess.run(command,env=environment,stdout=analysis_log,stderr=subprocess.STDOUT).returncode
    analysis_log.close()
    print(f'Alignment analysis exit code: {analysis_code}',flush=True)
    gradient_log=(args.out/'gradient.log').open('w')
    command=[sys.executable,str(scripts/'gradient_diagnostics.py'),'--checkpoint',str(healthy_path),'--out',str(args.out/'pre_event_gradient.json')]
    gradient_code=subprocess.run(command,env=environment,stdout=gradient_log,stderr=subprocess.STDOUT).returncode
    gradient_log.close()
    codes={}
    for arm,child,log in children:
        codes[arm]=child.wait();log.close()
        print(f'{arm} exit code: {codes[arm]}',flush=True)
    success=analysis_code==0 and gradient_code==0 and all(code==0 for code in codes.values())
    (args.out/'completion.json').write_text(json.dumps(dict(status='completed' if success else 'failed',branches=codes,
        alignment_exit_code=analysis_code,gradient_exit_code=gradient_code),indent=2))
    if not success:raise RuntimeError('A child experiment failed; inspect its saved log')


if __name__=='__main__':main()
