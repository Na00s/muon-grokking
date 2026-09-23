"""Read-only comparison of staged anonymous checkpoints and numerical arrays."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'muon-grokking/followup_studies/work/rms_replication'
SUPPLIED=HERE.parent/'revision_controls_20260923/anonymous_new_artifacts/followup_studies/work/rms_replication'

def compare(a,b,route,counts):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),route;counts['tensors']+=1;return
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),route;counts['numpy_arrays']+=1;return
    if isinstance(a,dict):
        assert a.keys()==b.keys(),route
        for key in a:compare(a[key],b[key],route+'/'+str(key),counts)
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),route
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,route+f'/{i}',counts)
    elif isinstance(a,str):assert isinstance(b,str),route
    elif isinstance(a,(float,int,bool,np.number)):
        assert type(a)==type(b) and (a==b or (isinstance(a,float) and math.isnan(a) and math.isnan(b))),route;counts['numeric_scalars']+=1
    else:assert a==b,route

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    counts=dict(tensors=0,numpy_arrays=0,numeric_scalars=0);records=[]
    pts=sorted(ORIGINAL.rglob('*.pt'));npzs=sorted(ORIGINAL.rglob('*.npz'));assert len(pts)==90 and len(npzs)==20
    assert {x.relative_to(ORIGINAL) for x in pts}=={x.relative_to(SUPPLIED) for x in SUPPLIED.rglob('*.pt')}
    assert {x.relative_to(ORIGINAL) for x in npzs}=={x.relative_to(SUPPLIED) for x in SUPPLIED.rglob('*.npz')}
    for p in pts:
        rel=p.relative_to(ORIGINAL);q=SUPPLIED/rel;a=torch.load(p,map_location='cpu',weights_only=False);b=torch.load(q,map_location='cpu',weights_only=False);compare(a,b,str(rel),counts)
        records.append(dict(path=str(rel),format='pt',all_tensors_optimizer_rng_and_numeric_fields_equal=True,original_sha256=sha(p),supplied_sha256=sha(q)))
    for p in npzs:
        rel=p.relative_to(ORIGINAL);q=SUPPLIED/rel
        with np.load(p,allow_pickle=False) as a,np.load(q,allow_pickle=False) as b:
            assert a.files==b.files,str(rel)
            for name in a.files:compare(a[name],b[name],str(rel)+'/'+name,counts)
        records.append(dict(path=str(rel),format='npz',all_arrays_identical=True,original_sha256=sha(p),supplied_sha256=sha(q)))
    tables=['run_summary.csv','event_summary.csv','parameter_swaps.csv','derivative_summary.csv','decoder_summary.csv','decoder_optimization.csv']
    for name in tables:
        with (ORIGINAL/name).open(newline='') as f,(SUPPLIED/name).open(newline='') as g:assert list(csv.reader(f))==list(csv.reader(g)),name
    result=dict(passed=True,comparison='Canonical completed RMS study versus staged anonymous RMS study; read-only check. Text values may be anonymized; every checkpoint tensor, optimizer/RNG numeric field and NPZ array is exact.',checkpoints_checked=len(pts),npz_files_checked=len(npzs),summary_csvs_checked=len(tables),**counts,files=records)
    (HERE/'anonymous_preservation_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}))
if __name__=='__main__':main()
