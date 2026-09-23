"""Independent synthesis audit using analytic Fourier coefficients and complex exponentials.
Does not import original control scripts or use FFT to reconstruct modified logits.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent;RESULT=ROOT/'work/fourier_control/results'
p=113;N=p*p;m=3830;labels=np.indices((p,p)).sum(0).ravel()%p;ks=np.arange(1,57);classes=np.arange(p)
basis=np.exp(2j*np.pi*classes[:,None]*ks/p);coef_unit=np.exp(-2j*np.pi*ks[:,None]*classes/p)
prefix=pd.read_csv(RESULT/'frequency_prefix_metrics.csv');phase=pd.read_csv(RESULT/'phase_metrics.csv');checks=[];near_ties=[]
def validate(name,z,weights,ref):
 pred=z.argmax(1);correct=int(weights[pred==classes].sum());expected=int(ref['correct']);wrong=z.copy();wrong[classes,classes]=-np.inf;minimum_margin=float((z.diagonal()-wrong.max(1)).min());err=abs(minimum_margin-float(ref['minimum_margin']))
 checks.append(dict(name=name,correct=correct,expected=expected,passed=correct==expected,minimum_margin_error=err))
 if correct!=expected:near_ties.append(dict(name=name,correct=correct,expected=expected,minimum_margin=minimum_margin))
for seed in range(5):
 perm=torch.randperm(N,generator=torch.Generator().manual_seed(seed)).numpy();counts=np.bincount(labels[perm[:m]],minlength=p)
 for kind in ['one_hot','training_count_normalized']:
  amplitude=np.ones(p) if kind=='one_hot' else p/counts;q=counts*amplitude/p;coefs=coef_unit*q;dc=np.broadcast_to(q/p,(p,p));raw=np.diag(amplitude);template=np.diag(q)
  selected_rows=prefix[(prefix.seed==seed)&(prefix.lookup==kind)]
  for (ranking,count),group in selected_rows.groupby(['ranking','retained_pairs']):
   selected=np.array([int(k) for k in group.iloc[0].selected_frequencies.split('|')]);z=dc+2*np.real(basis[:,selected-1]@coefs[selected-1])/p
   for _,r in group.iterrows():validate(f'prefix_{seed}_{kind}_{ranking}_{count}_{r.split}',z,counts if r.split=='training' else p-counts if r.split=='heldout' else np.full(p,p),r)
  selected_rows=phase[(phase.seed==seed)&(phase.lookup==kind)]
  for (control,replicate),group in selected_rows.groupby(['control','replicate']):
   rng=np.random.default_rng(int(group.iloc[0].rng_seed));changed=coefs.copy()
   if control=='frequency_derangement':
    while True:
     dest=rng.permutation(56)
     if all(dest!=np.arange(56)):break
    changed[dest]=coefs
   else:
    factors=np.exp(1j*rng.uniform(0,2*np.pi,size=(56,1 if control=='pair_global_phase' else p)));changed*=factors
   modified=dc+2*np.real(basis@changed)/p
   # Permuting all nonzero frequencies preserves evaluation at s=0 exactly.
   # Restore this analytic identity so residual cancellation gives exact zero ties.
   if control=='frequency_derangement':modified[0]=template[0]
   for context,g in group.groupby('context'):
    train_z=modified if context=='isolated_family_with_dc' else raw-template+modified;held_z=modified if context=='isolated_family_with_dc' else -template+modified
    for _,r in g.iterrows():
     if r.split=='full_grid':
      correct=int(counts[train_z.argmax(1)==classes].sum()+(p-counts)[held_z.argmax(1)==classes].sum());checks.append(dict(name=f'phase_{seed}_{kind}_{control}_{replicate}_{context}_full_grid',correct=correct,expected=int(r.correct),passed=correct==int(r.correct)))
     else:validate(f'phase_{seed}_{kind}_{control}_{replicate}_{context}_{r.split}',train_z if r.split=='training' else held_z,counts if r.split=='training' else p-counts,r)
summary=dict(check_count=len(checks),all_passed=all(c['passed'] for c in checks),differing_integer_counts=near_ties,maximum_margin_error=max(c.get('minimum_margin_error',0) for c in checks),checks=checks)
(OUT/'secondary_direct_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))
