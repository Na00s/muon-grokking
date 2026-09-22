"""Create a portable, hashed bundle of the completed Muon experiments.

The work root can be the experiments directory or its parent work directory.
No source files are modified. The destination must be new or empty.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

RESULT_DIRECTORIES = (
    'seed4_original', 'seed0_original', 'seed4_stable32_prospective',
    'seed4_healthy_temporal_6000_7000', 'seed4_event', 'seed0_event',
    'seed4_generalization_drop_16000_17000', 'seed4_dense_capture', 'seed4_dense_event',
    'seed4_precision_15000_18000',
    'seed4_matched_healthy_replay', 'seed4_healthy_temporal_16044_16050',
    'seed4_peak_replay', 'seed4_peak_event', 'seed4_one_step_attribution',
    'seed0_dense_capture', 'seed0_dense_event',
)
RESULT_EXTENSIONS = {'.csv', '.json', '.npz', '.npy', '.log'}


def safe_relative(path):
    """Exclude hidden, temporary and cache paths before traversing artifacts."""
    return all(not part.startswith('.') and part != '__pycache__'
               and not part.endswith('.tmp') and '.tmp.' not in part
               for part in path.parts)


def files_under(directory):
    if not directory.is_dir():
        return
    for path in sorted(directory.rglob('*')):
        if path.is_file() and not path.is_symlink() and safe_relative(path.relative_to(directory)):
            yield path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def find_experiments(work_root):
    for candidate in [work_root, work_root/'experiments']:
        if (candidate/'run_collapse.py').is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Cannot find run_collapse.py in {work_root} or its experiments subdirectory')


def plan_files(experiments, repository):
    plan={}
    missing=[]

    def add(source, destination, required=False):
        if source.is_file() and not source.is_symlink():
            if destination in plan and plan[destination] != source:
                raise ValueError(f'Conflicting package destination: {destination}')
            plan[destination]=source
        elif required:
            missing.append(str(source))

    def add_experiment(source, required=False):
        add(source, Path('experiments')/source.relative_to(experiments), required)

    # Executable experiment definitions, protocol and top-level diagnostics.
    for source in sorted(experiments.iterdir()):
        if not source.is_file() or not safe_relative(Path(source.name)):
            continue
        if source.suffix in {'.py', '.json', '.log'} or source.name=='protocol.md':
            add_experiment(source)

    for dirname in RESULT_DIRECTORIES:
        root=experiments/dirname
        for source in files_under(root):
            if source.suffix.lower() in RESULT_EXTENSIONS:
                add_experiment(source)
            if source.name in {'start.pt','final.pt','pre_step.pt','post_step.pt','stable_ce_one_step.pt'}:
                add_experiment(source)
            if source.name=='metadata.json':
                metadata=json.loads(source.read_text())
                for key in ['healthy_checkpoint','collapsed_checkpoint','source_checkpoint']:
                    reference=metadata.get(key)
                    if isinstance(reference,str) and reference.endswith('.pt'):
                        checkpoint=Path(reference)
                        if not checkpoint.is_absolute():
                            checkpoint=(source.parent/checkpoint).resolve()
                        else:
                            checkpoint=checkpoint.resolve()
                        if checkpoint.is_relative_to(experiments):
                            add_experiment(checkpoint,required=True)

    # Diagnostics and nested result provenance can name additional essential
    # checkpoint inputs, including gradient probes and one-step attribution.
    def checkpoint_references(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if (key=='checkpoint' or key.endswith('_checkpoint')
                        or key.endswith('_checkpoint_path') or key=='checkpoint_path'):
                    if isinstance(item,str) and item.endswith('.pt'):
                        yield item
                yield from checkpoint_references(item)
        elif isinstance(value,list):
            for item in value:
                yield from checkpoint_references(item)

    for relative,source in list(plan.items()):
        if source.suffix!='.json':
            continue
        for reference in checkpoint_references(json.loads(source.read_text())):
            checkpoint=Path(reference)
            candidates=([checkpoint.resolve()] if checkpoint.is_absolute() else
                        [(source.parent/checkpoint).resolve(),(experiments/checkpoint).resolve()])
            candidates=[p for p in candidates if p.is_relative_to(experiments)]
            if candidates:
                selected=next((p for p in candidates if p.exists()),candidates[0])
                add_experiment(selected,required=True)

    # Lightweight verification evidence. Checkpoint binaries are deliberately excluded.
    for directory in sorted(experiments.iterdir()):
        if directory.is_dir() and (
            directory.name.startswith('precision_runner_verif')
            or directory.name=='dense_capture_verified'
            or directory.name=='pipeline_identity_smoke'
            or directory.name.startswith('test_')
        ):
            for source in files_under(directory):
                if source.suffix=='.json':
                    add_experiment(source)

    # Initial/final state of each baseline, plus the saved temporal controls.
    for dirname in ['seed4_original', 'seed0_original']:
        directory=experiments/dirname
        if directory.exists():
            for name in ['initial.pt', 'latest.pt']:
                add_experiment(directory/name, required=True)
            for step in [6000,7000,15000,16000,17000]:
                add_experiment(directory/f'step_{step:06d}.pt')

    prospective=experiments/'seed4_stable32_prospective'
    if prospective.exists():
        for name in ['start.pt','final.pt']:
            add_experiment(prospective/name,required=True)

    # Event specifications select the primary pair and a pre-event branch state.
    for seed in [4,0]:
        event=experiments/f'seed{seed}_event'
        specification=event/'specification.json'
        if not specification.exists():
            continue
        spec=json.loads(specification.read_text())
        baseline=experiments/f'seed{seed}_original'
        for field in ['healthy_step','event_step','branch_start_step']:
            if field not in spec:
                raise ValueError(f'{specification} lacks {field}')
            step=int(spec[field])
            window=baseline/f'event_window_{step:06d}.pt'
            standard=baseline/f'step_{step:06d}.pt'
            selected=window if window.exists() else standard
            add_experiment(selected,required=True)
        for arm in spec.get('arms',['original32','stable32','model64','true64']):
            for name in ['start.pt','final.pt']:
                add_experiment(event/arm/name,required=True)

    # Minimal source snapshot preserves imports such as experiments.depth.*.
    for source in files_under(repository):
        relative=source.relative_to(repository)
        if source.suffix=='.py':
            add(source,Path('muon-grokking')/relative)
    for source in sorted(repository.glob('LICENSE*')):
        if source.is_file():add(source,Path('muon-grokking')/source.name)

    if missing:
        raise FileNotFoundError('Required key checkpoints are missing; finish the runs before packaging:\n'+'\n'.join(missing))
    return plan


def build_readme(experiments, commit):
    event_specs=[]
    for seed in [4,0]:
        event=experiments/f'seed{seed}_event'
        if (event/'specification.json').exists():
            event_specs.append((seed,json.loads((event/'specification.json').read_text())))
    blocks=[f'''# Muon grokking experiment bundle

This bundle contains experiment scripts, selected checkpoints, numerical arrays, trajectories,
and verification records. The source repository snapshot is commit `{commit}`.
The separate research report and figures summarize completed results.

## Layout

- `experiments/`: added scripts, protocol, retained results and checkpoint files.
- `muon-grokking/`: Python source snapshot from the unchanged paper repository.
- `manifest.json`: SHA-256 and byte count for every file packaged here.

The two source directories must remain siblings because the experiment scripts import
`../muon-grokking`. Saved JSON metadata retains the original absolute paths as provenance;
the commands below use paths within this bundle. Serialized checkpoints are PyTorch files.

## Environment

Use Python with PyTorch, NumPy, SciPy and Matplotlib. Recorded run metadata specifies the
versions used for these results. New runs can follow different trajectories when libraries
or hardware change. The recorded runs used CPU execution with one thread.

From this bundle directory:

```sh
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m unittest discover -s experiments -p 'test_*.py'
python experiments/test_numerical_controls.py
```

Run the checkpoint integration verification separately:

```sh
python experiments/test_precision_branches.py \\
  --checkpoint experiments/seed4_original/step_006000.pt \\
  --out rerun/checkpoint_verification
```

## Reproduce the temporal control

```sh
python experiments/run_alignment.py \\
  --healthy experiments/seed4_original/step_006000.pt \\
  --collapsed experiments/seed4_original/step_007000.pt \\
  --decoder --out rerun/healthy_temporal_6000_7000
```

The second checkpoint in this command is the later healthy temporal control. The script
uses the same `--collapsed` argument for both healthy controls and collapse pairs.

## Start a fresh baseline

```sh
python experiments/run_collapse.py --seed 4 --threads 1 --steps 30000 \\
  --out rerun/seed4_original
```

Result directories must be new. The packaged results are read-only inputs to these commands.
''']
    for seed,spec in event_specs:
        baseline=experiments/f'seed{seed}_original'
        paths={}
        for field in ['healthy_step','event_step','branch_start_step']:
            step=int(spec[field]);window=baseline/f'event_window_{step:06d}.pt'
            selected=window if window.exists() else baseline/f'step_{step:06d}.pt'
            paths[field]=str(Path('experiments')/selected.relative_to(experiments))
        nsteps=int(spec['branch_end_step'])-int(spec['branch_start_step'])
        blocks.append(f'''## Reproduce seed {seed}'s selected event

The event and pair selection rules are recorded in
`experiments/seed{seed}_event/specification.json`.

```sh
python experiments/run_alignment.py \\
  --healthy {paths['healthy_step']} \\
  --collapsed {paths['event_step']} \\
  --decoder --out rerun/seed{seed}_event_alignment

python experiments/run_precision_branches.py \\
  --checkpoint {paths['branch_start_step']} \\
  --arm true64 --steps {nsteps} --threads 1 --eval-every {int(spec.get('eval_every',10))} \\
  --save-every {int(spec.get('save_every',100))} --out rerun/seed{seed}_true64
```

Repeat the branch command with `original32`, `stable32`, and `model64` in separate output
directories for the other arms. `stable32` is the accurate cross-entropy control.
''')
    blocks.append('''## Figures

```sh
python experiments/plot_results.py \\
  --baseline experiments/seed4_original/trajectory.csv \\
  --baseline experiments/seed0_original/trajectory.csv \\
  --prospective experiments/seed4_stable32_prospective/trajectory.csv \\
  --out rerun/figures
```

Add `--alignment experiments/seed4_dense_event` for the first joint-collapse probes. Add
repeated `--branch experiments/seed4_precision_15000_18000/ARM` arguments for the four
matched numerical controls.

## Reproduce the densely captured collapse and readout probe

```sh
python experiments/run_dense_capture.py \\
  --checkpoint experiments/seed4_original/step_015000.pt \\
  --steps 5000 --threads 1 --out rerun/seed4_dense_capture

python experiments/run_alignment.py \\
  --healthy experiments/seed4_dense_capture/pre_event_train_healthy.pt \\
  --collapsed experiments/seed4_dense_capture/collapse.pt \\
  --out rerun/seed4_dense_alignment

python experiments/run_whitened_probe.py \\
  --features rerun/seed4_dense_alignment/features.npz \\
  --out rerun/seed4_dense_alignment/decoder_whitened.json \\
  --max-iter 500 --refit-max-iter 1000

python experiments/run_precision_branches.py \\
  --checkpoint experiments/seed4_original/step_015000.pt \\
  --arm stable32 --steps 3000 --threads 1 --eval-every 10 \\
  --save-every 100 --out rerun/seed4_accurate_ce
```

Repeat the last command with `original32`, `model64`, and `true64`, using distinct output
directories. `stable32` changes the arithmetic used for the same cross-entropy objective;
the paper's frozen-auxiliary Stable Muon intervention is a separate experiment.

## Retention and provenance

The bundle retains initial and latest baseline states, temporal-control checkpoints,
the selected healthy/event/branch checkpoints, and each precision arm's start/final state.
Intermediate training checkpoints and verification checkpoint binaries are omitted.
Verification JSON and saved trajectories remain available.

The archive event summary reanalyzes the paper's preexisting CSVs. Its input filenames are
listed in the summary; those upstream CSVs are outside this minimal source snapshot. A full
checkout of the recorded repository commit supplies them for independent reanalysis.

`manifest.json` hashes files created by the packaging step. Reports and figures added later
are outside that manifest unless a later final manifest is supplied.
''')
    return '\n'.join(blocks)


def package(work_root,repository,out):
    experiments=find_experiments(Path(work_root).resolve())
    repository=Path(repository).resolve();out=Path(out).resolve()
    if (out.is_relative_to(experiments) or out in experiments.parents
            or out.is_relative_to(repository) or out in repository.parents):
        raise ValueError('Output must be outside the source directories and their ancestors')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory must be new or empty: {out}')
    commit=subprocess.run(['git','-C',str(repository),'rev-parse','HEAD'],check=True,text=True,capture_output=True).stdout.strip()
    dirty=subprocess.run(['git','-C',str(repository),'status','--porcelain'],check=True,text=True,capture_output=True).stdout.strip()
    if dirty:
        raise ValueError('Source repository has uncommitted changes; preserve an unambiguous source snapshot before packaging')
    plan=plan_files(experiments,repository)
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for relative,source in sorted(plan.items(),key=lambda item:str(item[0])):
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if source.is_relative_to(experiments):origin='work/'+str(source.relative_to(experiments))
        else:origin='repository/'+str(source.relative_to(repository))
        records.append(dict(path=str(relative),source=origin,sha256=digest(target),bytes=target.stat().st_size))
    readme=out/'README.md';readme.write_text(build_readme(experiments,commit))
    records.append(dict(path='README.md',source='generated by package_results.py',sha256=digest(readme),bytes=readme.stat().st_size))
    manifest=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),repository_commit=commit,repository_dirty=False,files=records,file_count=len(records),total_bytes=sum(r['bytes'] for r in records),scope='Files copied/generated by packaging; later report and figure additions require a separate final manifest.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();manifest=package(a.work_root,a.repository,a.out)
    print(json.dumps({'output_directory':str(a.out.resolve()),'file_count':manifest['file_count'],'total_bytes':manifest['total_bytes'],'repository_commit':manifest['repository_commit']}))


if __name__=='__main__':main()
