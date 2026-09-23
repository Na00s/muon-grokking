"""Independent verification of the new Fourier controls only.
Uses torch FFTs, independently reconstructed torch.randperm splits, explicit orbit loops,
and analytic single-frequency logits. Does not import either control implementation.
"""
from pathlib import Path
import json, hashlib, math
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[3];WORK=ROOT/'work';OUT=Path(__file__).parent;RESULT=WORK/'fourier_control/results'
torch.set_num_threads(1)
p=113;N=p*p;m=int(.30*N);a,b=np.indices((p,p));checks=[];rows=[];primary=json.loads((RESULT/'memorizer_results.json').read_text())
def check(name,got,expected,tol=0):
 if isinstance(got,(np.ndarray,list)) or isinstance(expected,(np.ndarray,list)):
  ga,ea=np.asarray(got),np.asarray(expected);error=float(np.max(np.abs(ga-ea))) if ga.size else 0.;passed=bool(np.allclose(ga,ea,rtol=0,atol=tol));actual_summary=dict(shape=list(ga.shape),max_absolute_error=error)
 else:passed=bool(abs(got-expected)<=tol);actual_summary=got
 checks.append(dict(name=name,passed=passed,actual=actual_summary,expected=expected if not isinstance(expected,(list,np.ndarray)) else 'array compared elementwise',absolute_tolerance=tol))
 if not passed:raise AssertionError((name,got,expected))
def metrics(z,y,mask):
 zz=z[mask];yy=y[mask];pred=zz.argmax(1);maxes=zz.max(1);ties=(zz==maxes[:,None]).sum(1);istrue=zz[np.arange(len(yy)),yy]==maxes
 wrong=zz.copy();wrong[np.arange(len(yy)),yy]=-np.inf;margin=zz[np.arange(len(yy)),yy]-wrong.max(1)
 return dict(n=len(yy),correct=int((pred==yy).sum()),accuracy=float((pred==yy).mean()),uniform_tie_expected_accuracy=float((istrue/ties).mean()),tied_maximum_rows=int((ties>1).sum()),positive_margin_rows=int((margin>0).sum()))
for op in ['addition','subtraction']:
 y=((a+b) if op=='addition' else (a-b))%p;y=y.ravel()
 for seed in range(5):
  perm=torch.randperm(N,generator=torch.Generator().manual_seed(seed)).numpy();train=np.zeros(N,bool);train[perm[:m]]=True
  counts=np.array([sum(train[y==c]) for c in range(p)]);given=next(x for x in primary if x['seed']==seed and x['operation']==op)
  check(f'{op}_{seed}_traincount',train.sum(),3830);check(f'{op}_{seed}_heldoutcount',(~train).sum(),8939);check(f'{op}_{seed}_orbitcounts',counts,given['training_orbit_counts']);check(f'{op}_{seed}_allcovered',np.count_nonzero(counts),113)
  z=np.zeros((N,p));z[np.flatnonzero(train),y[train]]=1
  spectrum=torch.fft.fft2(torch.from_numpy(z.reshape(p,p,p)),dim=(0,1),norm='ortho');kk,ll=torch.meshgrid(torch.arange(p),torch.arange(p),indexing='ij');mask=kk==ll if op=='addition' else ((kk+ll)%p==0)
  fft=torch.fft.ifft2(spectrum*mask[:,:,None],dim=(0,1),norm='ortho').real.numpy().reshape(N,p)
  oracle=np.zeros_like(z);oracle[np.arange(N),y]=counts[y]/p
  check(f'{op}_{seed}_torchFFT_vs_countformula',fft,oracle,tol=1e-14)
  native=metrics(z,y,~train);filtered=metrics(fft,y,~train)
  for field in native:check(f'{op}_{seed}_raw_{field}',native[field],given['conditions']['raw_lookup']['heldout'][field],tol=1e-14)
  for field in ['n','correct','accuracy','tied_maximum_rows','positive_margin_rows']:check(f'{op}_{seed}_filtered_{field}',filtered[field],given['conditions']['complete_family_with_dc']['heldout'][field])
  nodc=fft-z.mean(0);check(f'{op}_{seed}_withoutDCcorrect',metrics(nodc,y,~train)['correct'],8939)
  # Direct combinatorial product independently computed in logarithms.
  probability=math.exp(sum(math.log(N-m-i)-math.log(N-i) for i in range(p)))
  check(f'{op}_{seed}_coverage_probability',probability,given['probability_one_orbit_uncovered_fixed_size'],tol=1e-29)
  rows.append(dict(operation=op,seed=seed,raw_correct=native['correct'],heldout_n=native['n'],raw_accuracy=native['accuracy'],filtered_correct=filtered['correct'],minimum_donors=int(counts.min()),maximum_donors=int(counts.max()),probability_one_uncovered=probability))
# Independent synthetic identity for odd and even moduli: orbit identity itself needs no primality.
rng=np.random.default_rng(221917)
for q in [6,7,113]:
 aa,bb=np.indices((q,q));h=rng.normal(size=(q,q,3));H=torch.tensor(h,dtype=torch.float64);s=torch.fft.fft2(H,dim=(0,1));kk,ll=np.indices((q,q))
 for op in ['addition','subtraction']:
  projected=torch.fft.ifft2(s*torch.from_numpy((kk==ll) if op=='addition' else ((kk+ll)%q==0))[:,:,None],dim=(0,1)).real.numpy();orbits=np.empty_like(h)
  for i in range(q):
   for j in range(q):orbits[i,j]=np.mean([h[(i+t)%q,(j-t if op=='addition' else j+t)%q] for t in range(q)],axis=0)
  check(f'identity_p{q}_{op}_explicit_translations',projected,orbits,tol=2e-14)
# Check actual checkpoint donor domains with independent loops in original train/test ordering.
checkpoint=json.loads((RESULT/'checkpoint_donor_results.json').read_text());donor_rows=[]
for pair in sorted(set(r['pair'] for r in checkpoint)):
 seed=next(r['seed'] for r in checkpoint if r['pair']==pair);data=np.load(WORK/'basis_study'/pair/'features.npz');perm=torch.randperm(N,generator=torch.Generator().manual_seed(seed)).numpy();y=((a+b)%p).ravel()[perm];train=np.arange(N)<m
 check(f'{pair}_labels',data['y'],y);check(f'{pair}_splitmask',data['train'].astype(int),train.astype(int))
 for endpoint in [0,1]:
  z=data[f'H{endpoint}'].astype(np.float64)@data[f'W{endpoint}'].astype(np.float64);arms={k:np.empty_like(z) for k in ['complete_family_with_dc','training_donors_only','heldout_donors_only','heldout_donors_leave_one_out']};arms['raw']=z
  for c in range(p):
   idx=np.flatnonzero(y==c);tr=idx[idx<m];te=idx[idx>=m];arms['complete_family_with_dc'][idx]=z[idx].mean(0);arms['training_donors_only'][idx]=z[tr].mean(0);arms['heldout_donors_only'][idx]=z[te].mean(0)
   # Explicit LOO excludes only the recipient when it belongs to held-out donors.
   total=z[te].sum(0);arms['heldout_donors_leave_one_out'][tr]=total/len(te);arms['heldout_donors_leave_one_out'][te]=(total-z[te])/(len(te)-1)
  ref=next(r for r in checkpoint if r['pair']==pair and r['endpoint']==endpoint)
  for name,zz in arms.items():
   for split,mm in [('training',train),('heldout',~train),('full_grid',np.ones(N,bool))]:
    calculated=metrics(zz,y,mm)
    for field in ['n','correct','accuracy','tied_maximum_rows','positive_margin_rows']:check(f'{pair}_{endpoint}_{name}_{split}_{field}',calculated[field],ref['conditions'][name][split][field],tol=1e-14)
    if split=='heldout':donor_rows.append(dict(pair=pair,seed=seed,endpoint=endpoint,step=ref['step'],condition=name,**calculated))
# Secondary single-pair construction: cos kernel, prime p => unique maximizer at target.
single=pd.read_csv(RESULT/'single_pair_normalized_metrics.csv');single_errors=[]
for seed in range(5):
 for k in range(1,57):
  analytic=(1+2*np.cos(2*np.pi*k*(np.arange(p)[:,None]-np.arange(p)[None,:])/p))/p
  check(f'normalized_singlepair_seed{seed}_k{k}_unique_prediction',analytic.argmax(1),np.arange(p));row=single[(single.seed==seed)&(single.frequency==k)].iloc[0];check(f'normalized_singlepair_seed{seed}_k{k}_recordedcorrect',int(row.heldout_correct),8939)
  wrong=analytic.copy();np.fill_diagonal(wrong,-np.inf);minmargin=float(np.min(np.diag(analytic)-wrong.max(1)));check(f'normalized_singlepair_seed{seed}_k{k}_margin',minmargin,float(row.minimum_margin),tol=5e-14);single_errors.append(abs(minmargin-row.minimum_margin))
# Direct2D ablation verifies full-grid denominator and source residual outsidefamily.
abl=pd.read_csv(RESULT/'ablation_metrics.csv')
for seed in range(5):
 perm=torch.randperm(N,generator=torch.Generator().manual_seed(seed)).numpy();train=np.zeros(N,bool);train[perm[:m]]=True;y=((a+b)%p).ravel();counts=np.array([sum(train[y==c]) for c in range(p)])
 for kind in ['one_hot','training_count_normalized']:
  amp=np.ones(p) if kind=='one_hot' else p/counts;z=np.zeros((N,p));z[np.flatnonzero(train),y[train]]=amp[y[train]];orbit=np.zeros_like(z);orbit[np.arange(N),y]=counts[y]*amp[y]/p;ablation=z-orbit+z.mean(0)
  for split,mm in [('training',train),('heldout',~train),('full_grid',np.ones(N,bool))]:
   got=metrics(ablation,y,mm);ref=abl[(abl.seed==seed)&(abl.lookup==kind)&(abl.condition=='family_ablated_dc_retained')&(abl.split==split)].iloc[0]
   for field in ['n','correct','accuracy']:check(f'ablated_{seed}_{kind}_{split}_{field}',got[field],float(ref[field]),tol=1e-14)
# Main mean/full gap, recomputed directly from original five raw intervention dictionaries.
mean_rows=[]
for seed in range(5):
 path=WORK/f'muon-grokking/followup_studies/work/causal_study/diagnostics_results/seed{seed}/mean_mediation.json';d=json.loads(path.read_text());arms=d['arms'];mean=arms['mean_only']['heldout']['accuracy']*100;full=arms['full']['heldout']['accuracy']*100;mean_rows.append(dict(seed=seed,mean_only_test_percent=mean,full_test_percent=full,absolute_gap_percentage_points=abs(mean-full),source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
maxgap=max(r['absolute_gap_percentage_points'] for r in mean_rows);check('maximum_mean_full_gap_percentage_points',maxgap,.3244210761830182,tol=1e-13)
summary=dict(check_count=len(checks),all_passed=all(c['passed'] for c in checks),primary=rows,checkpoint_heldout=donor_rows,mean_update=mean_rows,max_mean_full_gap_percentage_points=maxgap,single_pair_max_margin_error=float(max(single_errors)),checks=checks,scope='Independent torch FFT, public torch.randperm splits, explicit orbit loops, explicit checkpoint donor exclusion and analytic singlepair kernel. All600phase/relocation draws independently synthesized in companion secondary_direct_synthesis.py; see secondary_direct_verification.json.')
(OUT/'independent_verification.json').write_text(json.dumps(summary,indent=2,default=lambda value: value.item() if isinstance(value,np.generic) else str(value))+'\n');pd.DataFrame(rows).to_csv(OUT/'primary_recomputed.csv',index=False);pd.DataFrame(donor_rows).to_csv(OUT/'checkpoint_donors_recomputed.csv',index=False)
print('checks',len(checks),'all_passed',summary['all_passed']);print(pd.DataFrame(rows).to_string(index=False));print('max mean accuracy gap',maxgap)
