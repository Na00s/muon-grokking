"""Build a separate, portable evidence bundle without touching earlier outputs."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs'/'basis_change_study'
ZIP=OUT.with_suffix('.zip')


def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)


def strings(obj):
    if isinstance(obj,str):yield obj
    elif isinstance(obj,dict):
        for v in obj.values():yield from strings(v)
    elif isinstance(obj,list):
        for v in obj:yield from strings(v)


def localize(obj):
    if isinstance(obj,str):
        prefix=str(ROOT)+'/'
        return obj[len(prefix):] if obj.startswith(prefix) else obj
    if isinstance(obj,dict):return {k:localize(v) for k,v in obj.items()}
    if isinstance(obj,list):return [localize(v) for v in obj]
    return obj


def main():
    if OUT.exists():raise FileExistsError(f'Existing deliverable is preserved: {OUT}')
    OUT.mkdir(parents=True)
    # Retain the earlier reproducibility dependencies and selected source states.
    prior=ROOT/'outputs'/'muon_followup'
    for name in ['experiments','muon-grokking']:
        shutil.copytree(prior/name,OUT/'work'/name)
    copy(prior/'report.md',OUT/'prior_followup_report.md')
    # New numerical results, complete code, and all non-periodic saved states.
    for src in HERE.rglob('*'):
        if not src.is_file() or '__pycache__' in src.parts or 'mpl_cache' in src.parts:continue
        if src.name=='package.log':continue
        if src.suffix=='.pt' and src.name.startswith('step_'):continue
        copy(src,OUT/'work'/'basis_study'/src.relative_to(HERE))
    # Include checkpoints directly referenced by any saved study metadata.
    required=set()
    for src in HERE.rglob('*.json'):
        for value in strings(json.loads(src.read_text())):
            if value.endswith('.pt'):
                p=Path(value)
                if p.is_absolute() and p.is_relative_to(ROOT) and p.exists():required.add(p)
    # Exact previous-step reference paths and long-reference control inputs.
    for value in [
        'work/experiments/seed0_original/event_window_016900.pt',
        'work/experiments/seed0_dense_capture/pre_event_latest.pt',
        'work/experiments/seed4_one_step_attribution/pre_step.pt',
        'work/experiments/seed4_one_step_attribution/post_step.pt',
    ]:
        p=ROOT/value
        if p.exists():required.add(p)
    for src in required:copy(src,OUT/src.relative_to(ROOT))
    # Expose the main deliverables without requiring navigation through raw files.
    for name in ['report.md','paper_revision.md','mathematical_scope.md']:
        copy(HERE/name,OUT/name)
    shutil.copytree(HERE/'figures',OUT/'figures')
    copy(HERE/'bundle_readme.md',OUT/'README.md')
    # JSON path strings become relative to the bundle root. Checkpoints are copied
    # byte-for-byte, preserving every original model, optimizer and RNG tensor.
    for p in OUT.rglob('*.json'):
        d=json.loads(p.read_text())
        p.write_text(json.dumps(localize(d),indent=2,allow_nan=False)+'\n')
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file():files.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,
                                         sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (OUT/'manifest.json').write_text(json.dumps(dict(files=files,
        file_count=len(files),total_bytes=sum(f['bytes'] for f in files),
        checkpoint_handling='Byte-for-byte copies. Periodic source states are included when explicitly referenced; baseline training can be rerun from the recorded initial states and fixed configuration.',
        path_handling='Absolute workspace-root paths in JSON are normalized to bundle-relative paths. Run commands from the extracted bundle root.'),indent=2)+'\n')
    with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.relative_to(OUT)))
    print(json.dumps(dict(directory=str(OUT),archive=str(ZIP),archive_bytes=ZIP.stat().st_size,
                         files=len(files)+1,checkpoints=sum(f['path'].endswith('.pt') for f in files))))


if __name__=='__main__':main()
