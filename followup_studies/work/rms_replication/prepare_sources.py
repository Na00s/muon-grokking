from pathlib import Path
import shutil,hashlib,json
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
files=[]
for folder in ['experiments','causal_study']:
    files.extend((WORK/folder).glob('*.py'))
for file in (WORK/'muon-grokking').glob('*.py'): files.append(file)
for folder in ['experiments/depth','optimizers']:
    files.extend((WORK/'muon-grokking'/folder).glob('*.py'))
manifest=[]
for src in sorted(files):
    rel=src.relative_to(WORK); target=HERE/'source_snapshot'/rel
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    manifest.append(dict(source=str(rel),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),bytes=src.stat().st_size))
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(len(manifest))
