"""Launch the fixed six-branch 100,000-step extension on the local CPU."""
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from verify_runner import cases
from run_long_horizon import file_sha256, write_json

HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()


def now(): return datetime.now(timezone.utc).isoformat()


def main():
    verification = json.loads((HERE/'runner_verification.json').read_text())
    assert verification['status'] == 'passed'
    assert verification['runner_sha256'] == file_sha256(HERE/'run_long_horizon.py')
    jobs = [dict(name=name, checkpoint=str(path.resolve()), operation=operation,
                 source_checkpoint_sha256=file_sha256(path), start_step=30000, end_step=100000)
            for name,path,operation in cases()]
    plan = dict(created_utc=now(), status='prespecified_before_extension', workers=6, jobs=jobs,
        purpose='Extend the existing accurate-cross-entropy controls to the original depth-1 100,000-step budget.',
        original_horizon_evidence=[dict(path=str(p),sha256=file_sha256(p)) for p in
            [HERE.parent/'muon-grokking'/'runs'/f'seedstudy_muon_seed_{seed}.csv' for seed in range(5)] +
            [HERE.parent/'muon-grokking'/'runs'/'subtraction_depth_1_muon_seed_0.csv']],
        primary='Whether any joint train/test accuracy below 90% occurs in each of five corrected addition branches between steps 30,000 and 100,000.',
        secondary='The same endpoint for the corrected subtraction seed-0 branch.',
        inherited_interventions='Addition accurate CE was introduced at matched stock step 6,000. Subtraction used accurate CE from initialization.',
        training_monitor='Every pre-update state and final state, with every state recorded in trajectory.csv.',
        test_monitor='Every 100 global steps, every training state below 90%, start and final states.',
        finite_horizon_scope='Completion at 100,000 tests persistence through the original budget. No claim of indefinite prevention follows.',
        test_only_scope='Brief test-only excursions between 100-step evaluations can remain undetected.',
        prior_stock_scope='Matched stock branches have verified joint failures by 30,000. Their binary event status persists at longer horizons without rerunning them.',
        stopping='Complete every requested horizon. Nonfinite states fail explicitly and retain a failure checkpoint; never truncate silently.',
        checkpoint_policy='Retain start/final, 10,000-step grid, first joint event/preceding state/last jointly >=99% state, and minimum measured test state.',
        runtime='CPU float32, original model and three optimizer objects, accurate CE, deterministic algorithms, one Torch thread per worker.',
        validation='All six restored model/optimizer/RNG states match source exactly. Twelve uninterrupted source updates match monitored updates and a six-plus-six save/restore continuation bitwise.',
        runner_sha256=file_sha256(HERE/'run_long_horizon.py'),
        validation_sha256=file_sha256(HERE/'runner_verification.json'))
    if (HERE/'protocol.json').exists():
        raise FileExistsError('Protocol already exists. Inspect suite state before starting anything again.')
    write_json(HERE/'protocol.json',plan)
    status = {job['name']:dict(status='queued') for job in jobs}
    (HERE/'runs').mkdir(exist_ok=True)
    def update(name,**items):
        with LOCK:
            status[name].update(items)
            write_json(HERE/'progress.json',dict(updated_utc=now(),jobs=status))
            print(json.dumps(dict(job=name,**items)),flush=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def worker(job):
        name = job['name']
        update(name,status='running',started_utc=now())
        command = [sys.executable,str(HERE/'run_long_horizon.py'),'--checkpoint',job['checkpoint'],
            '--out',str(HERE/'runs'/name),'--end-step',str(job['end_step']),'--operation',job['operation']]
        with (HERE/'runs'/f'{name}.log').open('w') as log:
            result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        update(name,status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,ended_utc=now())
        return result.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(worker,jobs))
    if any(codes):
        raise RuntimeError(f'{sum(code != 0 for code in codes)} branches failed. Raw failure records retained.')
    write_json(HERE/'completion.json',dict(status='completed',ended_utc=now(),jobs=status))
    print('All six prespecified horizons completed.',flush=True)


if __name__ == '__main__': main()
