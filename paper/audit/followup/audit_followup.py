"""Independent arithmetic, decoder, geometry, and plotting audit of the revised paper.
Run with the research Python environment; no training or source mutation.
Set MUON_AUDIT_REPO and MUON_AUDIT_EVIDENCE_ROOT to override autodetection.
The evidence root must contain work/ and the restored release binary artifacts.
Outputs are written alongside this script; baseline source locations prefer
paper/audit/input_source when present.
Raw feature arrays, saved decoder coefficients, and trajectory CSVs are primary.
Diagnostic JSON is used for evaluated quantities requiring full model replays;
its computation and matched-state checks are separately inspected and recorded.
"""
from pathlib import Path
import csv, hashlib, json, math, os, sys
import numpy as np
HERE = Path(__file__).resolve().parent

def discover_repository():
    explicit = os.environ.get("MUON_AUDIT_REPO")
    if explicit:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        candidates = []
        for parent in [HERE, *HERE.parents]:
            candidates.extend([parent, parent / "work" / "muon-grokking"])
    for candidate in candidates:
        if (candidate / "runs").is_dir() and (candidate / "followup_studies").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Set MUON_AUDIT_REPO to a repository containing runs/ and followup_studies/.")

REPO = discover_repository()
ROOT = Path(os.environ.get("MUON_AUDIT_EVIDENCE_ROOT", str(REPO / "followup_studies"))).expanduser().resolve()
if not (ROOT / "work").is_dir():
    raise FileNotFoundError("MUON_AUDIT_EVIDENCE_ROOT must contain the released work/ results directory.")

def evidence_path(value):
    """Resolve portable paths and relocate absolute provenance paths to ROOT.

    Released JSON normally stores work/... paths. Unmodified checkpoints can
    retain the original workstation prefix. For those, resolve the suffix from
    the work/ directory under the chosen evidence root before considering the
    original path. Missing binaries must be restored using the release tools.
    """
    path = Path(value)
    if not path.is_absolute():
        return ROOT / path
    try:
        path.relative_to(ROOT)
        return path
    except ValueError:
        pass
    for index, part in enumerate(path.parts):
        if part == "work":
            candidate = ROOT.joinpath(*path.parts[index:])
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Cannot relocate evidence path {value!s} under {ROOT}. Restore the release artifacts first.")

PAPER = REPO / 'paper' / 'audit' / 'input_source'
if not (PAPER / 'main.tex').is_file():
    PAPER = REPO / 'paper' / 'source'
OUT = HERE
rows=[]
def J(p):return json.loads((ROOT/p).read_text())
def C(p):return list(csv.DictReader((ROOT/p).open()))
def N(p):return np.load(ROOT/p)
def loc(file,line):
 text=(PAPER/file).read_text().splitlines()
 return {'file':str((PAPER/file).relative_to(REPO)),'line':line,'text':text[line-1] if line<=len(text) else ''}
def add(claim,reported,recomputed,evidence,line,file='followup_appendix.tex',digits=None,status=None,note=''):
 if isinstance(recomputed,np.generic):recomputed=recomputed.item()
 if status is None:
  if isinstance(recomputed,(float,np.floating)) and isinstance(reported,(float,int)):
   ok=round(float(recomputed),digits)==reported if digits is not None else math.isclose(float(recomputed),reported,rel_tol=1e-8,abs_tol=1e-12)
  else:ok=recomputed==reported
  status='PASS' if ok else 'FAIL'
 rows.append(dict(id=f'FU{len(rows)+1:04d}',source=loc(file,line),claim=claim,reported=reported,recomputed=recomputed,evidence=evidence,status=status,note=note))
def acc(H,W,y,mask):return float(np.mean((H[mask]@W).argmax(1)==y[mask]))
# Main figure a and Appendix adjacent table. Independently multiply raw states.
adjfolders=['work/basis_study/adjacent_seed0','work/basis_study/seed1_adjacent_event','work/basis_study/seed2_adjacent_event','work/basis_study/seed3_adjacent_event','work/basis_study/adjacent_seed4']
adjreports=[[95.49,96.39,31.86,31.87],[94.14,94.18,13.56,9.09],[99.99,100.,1.01,.93],[89.54,91.01,50.80,53.60],[91.34,98.66,66.75,66.33]]
adj=[]
for seed,folder in enumerate(adjfolders):
 f=N(folder+'/features.npz');H0,H1,W0,W1=[f[k].astype(float) for k in ['H0','H1','W0','W1']];v=f['heldout'];y=f['y']
 vals=[100*acc(H,W,y,v) for H,W in [(H0,W0),(H1,W0),(H0,W1),(H1,W1)]];adj.append(vals)
 for label,reported,val in zip(['Before','Old head/new features','New head/old features','After'],adjreports[seed],vals):add(f'Adjacent seed {seed}: {label}',reported,val,[folder+'/features.npz'],22+seed,digits=2)
 add(f'Adjacent seed {seed} heldout size',8939,int(v.sum()),[folder+'/features.npz'],16)
# Selected train-only probes: coefficients applied again to held-out feature arrays.
probefolders=['work/experiments/seed0_dense_event','work/basis_study/seed1_first_event','work/basis_study/seed2_first_event','work/basis_study/seed3_peak_event','work/experiments/seed4_peak_event']
probe_report=[98.72,98.20,100.,98.77,100.];healthy_report=[99.51,99.56,100.,99.40,100.];steps_report=[17494,17721,18069,16309,16060]
probe_values=[];native_values=[]
for seed,folder in enumerate(probefolders):
 f=N(folder+'/features.npz');w=N(folder+'/decoder_whitened.npz');d=J(folder+'/decoder_whitened.json');m=J(folder+'/metadata.json')
 add(f'Probe seed {seed} checkpoint',steps_report[seed],m['collapsed_step'],[folder+'/metadata.json'],36)
 add(f'Probe seed {seed} data train/test sizes',[3830,8939],[int(f['train'].sum()),int(f['heldout'].sum())],[folder+'/features.npz'],34)
 for state,key,expected in [('healthy','H0',healthy_report[seed]),('collapsed','H1',probe_report[seed])]:
  z=d['cv_selected'][state];r=z['refit'];val=100*acc(f[key],w[state+'_cv_readout'],f['y'],f['heldout'])
  add(f'Probe seed {seed} {state} heldout accuracy',expected,val,[folder+'/features.npz',folder+'/decoder_whitened.npz'],36,digits=2)
  add(f'Probe seed {seed} {state} converged',True,r['converged'],[folder+'/decoder_whitened.json'],34)
  add(f'Probe seed {seed} {state} full dimension and uncentered',[128,False],[r['feature_transform']['dimension'],r['feature_transform']['centered']],[folder+'/decoder_whitened.json'],34)
  add(f'Probe seed {seed} {state} singular floor',1e-6,r['feature_transform']['relative_singular_floor'],[folder+'/decoder_whitened.json'],34)
  add(f'Probe seed {seed} {state} inner validation fraction',.2,z['split']['inner_validation_count']/z['split']['original_train_count'],[folder+'/decoder_whitened.json'],34)
  add(f'Probe seed {seed} {state} lambda grid',[0.,1e-8,1e-6,1e-4,1e-2],[c['regularization'] for c in z['candidates']],[folder+'/decoder_whitened.json'],34)
  best=min(z['candidates'],key=lambda c:(c['validation']['cross_entropy'],c['regularization']))['regularization']
  add(f'Probe seed {seed} {state} CV selection valid',best,z['selected_regularization'],[folder+'/decoder_whitened.json'],34)
  add(f'Probe seed {seed} {state} all inner fits converged',True,all(c['optimization']['converged'] for c in z['candidates']),[folder+'/decoder_whitened.json'],34,note='Manuscript says convergence is recorded, not that all candidates converge.')
  if state=='collapsed':probe_values.append(val);native_values.append(100*acc(f[key],f['W1'],f['y'],f['heldout']))
# Geometry table: independent least-squares solution and train-centered errors.
firstfolders=['work/experiments/seed0_dense_event','work/basis_study/seed1_first_event','work/basis_study/seed2_first_event','work/basis_study/seed3_first_event','work/experiments/seed4_dense_event']
healthyfolders=['work/basis_study/seed0_matched_healthy_16906_17200','work/basis_study/seed1_matched_healthy','work/basis_study/seed2_matched_healthy','work/basis_study/seed3_matched_healthy','work/basis_study/geometry_seed4_healthy_temporal_16044_16050']
geomexpected=[[43.63,32.66,1.88],[43.05,10.82,4.63],[9.05,2.76,1.10],[45.73,14.30,10.66],[25.60,1.18,25.63]]
intervals=[]
for seed in range(5):
 for j,folder in enumerate([firstfolders[seed],healthyfolders[seed],adjfolders[seed]]):
  if not (ROOT/folder/'features.npz').exists():
   g=J(folder+'/geometry.json');folder=str(evidence_path(g['source']).parent.relative_to(ROOT))
  f=N(folder+'/features.npz');h0=f['H0'].astype(float);h1=f['H1'].astype(float);t=f['train'];v=f['heldout']
  B=np.linalg.lstsq(h1[t],h0[t],rcond=None)[0];pred=h1[v]@B;predmean=h1[t].mean(0)@B;targetmean=h0[t].mean(0)
  err=np.linalg.norm((pred-predmean)-(h0[v]-targetmean))/np.linalg.norm(h0[v]-targetmean)
  add(f'Geometry seed {seed} {("reference","healthy","adjacent")[j]} centered heldout backward error',geomexpected[seed][j],100*err,[folder+'/features.npz'],49+seed,digits=2)
  add(f'Geometry seed {seed} {j} full feature ranks',[128,128],[int(np.linalg.matrix_rank(h0)),int(np.linalg.matrix_rank(h1))],[folder+'/features.npz'],38)
  if j==0:
   m=J(folder+'/metadata.json');intervals.append(m['collapsed_step']-m['healthy_step'])
  if j==1:
   m=J(folder+'/metadata.json');healthy_interval=m['step_interval'] if 'step_interval' in m else m['collapsed_step']-m['healthy_step']
 add(f'Geometry seed {seed} healthy interval duration matches',True,healthy_interval==intervals[-1],[firstfolders[seed]+'/metadata.json',healthyfolders[seed]+'/geometry.json'],43,note='Compared original metadata windows separately below.')
add('Reference interval range',[6,508],[min(intervals),max(intervals)],firstfolders,43)
# Dense branch monitoring and all reported arithmetic counts/horizons/minima.
branch_records=[]
for kind in ['main_runs','projection_runs']:
 base=ROOT/'work/causal_study'/kind
 for folder in sorted(base.glob('*')):
  if not (folder/'trajectory.csv').exists():continue
  raw=C(str((folder/'trajectory.csv').relative_to(ROOT)));meta=J(str((folder/'metadata.json').relative_to(ROOT)))
  start=int(raw[0]['step']);end=int(raw[-1]['step']);steps=[int(r['step']) for r in raw]
  tst=[float(r['test_accuracy']) for r in raw if r['test_accuracy']!=''];joint=[int(r['step']) for r in raw if float(r['train_accuracy'])<.9 and r['test_accuracy']!='' and float(r['test_accuracy'])<.9]
  rec=dict(name=folder.name,seed=meta['seed'],arm=meta['arm'],start=start,end=end,updates=end-start,mintrain=min(float(r['train_accuracy']) for r in raw),mintest=min(tst),finaltrain=float(raw[-1]['train_accuracy']),finaltest=float(raw[-1]['test_accuracy']),joint=joint,path=str(folder.relative_to(ROOT)))
  branch_records.append(rec)
  add(f'{folder.name} every update inspected',True,steps==list(range(start,end+1)),[rec['path']+'/trajectory.csv'],9)
  add(f'{folder.name} test at every train <90% state',True,all(r['test_accuracy']!='' for r in raw if float(r['train_accuracy'])<.9),[rec['path']+'/trajectory.csv'],9)
  add(f'{folder.name} test grid',True,all(r['test_accuracy']!='' for r in raw if int(r['step'])%100==0),[rec['path']+'/trajectory.csv'],9)
primary=sorted([r for r in branch_records if r['name'].endswith('_accurate6000')],key=lambda r:r['seed'])
specificity=[r for r in branch_records if r['start']==15000]
extensions=[r for r in branch_records if r['arm']=='original32']
add('Primary branch count',5,len(primary),[r['path'] for r in primary],72)
add('Specificity branch count',15,len(specificity),[r['path'] for r in specificity],87)
minexpected=[99.754,99.843,98.859,99.978,99.843]
for seed,r in enumerate(primary):
 add(f'Accurate seed {seed} start/end',[6000,30000],[r['start'],r['end']],[r['path']+'/trajectory.csv'],78+seed)
 add(f'Accurate seed {seed} minimum test',minexpected[seed],r['mintest']*100,[r['path']+'/trajectory.csv'],78+seed,digits=3)
 add(f'Accurate seed {seed} no joint failure',0,len(r['joint']),[r['path']+'/trajectory.csv'],72)
 add(f'Accurate seed {seed} final train/test',[1.,1.],[r['finaltrain'],r['finaltest']],[r['path']+'/trajectory.csv'],72)
 add(f'Accurate seed {seed} all train states perfect',1.,r['mintrain'],[r['path']+'/trajectory.csv'],72,note='Stronger than manuscript claim, used to audit joint endpoint.')
for r in specificity:
 add(f'Specificity {r["name"]} start/end',[15000,20000],[r['start'],r['end']],[r['path']+'/trajectory.csv'],87)
 add(f'Specificity {r["name"]} no joint failure',0,len(r['joint']),[r['path']+'/trajectory.csv'],87)
 add(f'Specificity {r["name"]} final test',1.,r['finaltest'],[r['path']+'/trajectory.csv'],87)
add('Specificity overall measured test minimum',99.978,100*min(r['mintest'] for r in specificity),[r['path']+'/trajectory.csv' for r in specificity],87,digits=3)
# Six-check source qualification; no dependence on historical claimed criterion.
for seed in range(5):
 folder=f'work/experiments/seed{seed}_original' if seed in [0,4] else f'work/basis_study/seed{seed}_baseline'
 rr={int(r['step']):r for r in C(folder+'/trajectory.csv') if r['test_accuracy'] and int(r['step'])%100==0 and int(r['step'])<=6000}
 windows=[s for s in sorted(rr) if all(s+100*i in rr and float(rr[s+100*i]['test_accuracy'])>=.95 for i in range(6))]
 add(f'Source seed {seed} six-check qualification by 6000',True,bool(windows),[folder+'/trajectory.csv'],67,note=f'First window {windows[0]} through {windows[0]+500}.' if windows else '')
# Counts directly from trajectories.
fresh=[]
for seed in [1,2,3]:
 import torch
 raw=C(f'work/basis_study/seed{seed}_baseline/trajectory.csv');cp=torch.load(ROOT/f'work/basis_study/seed{seed}_baseline/final.pt',map_location='cpu',weights_only=False);fresh.append(int(cp['step']))
add('New basis baseline updates',52698,sum(fresh),[f'work/basis_study/seed{s}_baseline/final.pt' for s in [1,2,3]],4,note='Raw CSV is thinned to multiples of ten; final checkpoint supplies final update counter.')
cont=[]
for p in (ROOT/'work/basis_study').glob('**/continuation/*/trajectory.csv'):
 rr=C(str(p.relative_to(ROOT)));cont.append((str(p.relative_to(ROOT)),int(rr[-1]['step'])-int(rr[0]['step'])) )
add('Continuation branch count',18,len(cont),[p for p,n in cont],62)
add('Continuation update total',9000,sum(n for p,n in cont),[p for p,n in cont],4)
add('Continuation horizon each',True,all(n==500 for p,n in cont),[p for p,n in cont],62)
add('Causal original extension count',4,len(extensions),[r['path'] for r in extensions],4)
add('Causal original extension updates',49302,sum(r['updates'] for r in extensions),[r['path']+'/trajectory.csv' for r in extensions],4)
# Generality, raw full trajectories, first and peak tests.
generalexpected={
'subtraction_stock':(9300,15281,5.851,5.851),
'subtraction_accurate':(10500,None,None,96.879),
'rms_stock':(15700,18515,85.658,.481),
'rms_accurate':(19900,28495,.884,.526)}
general=[]
for i,(name,expected) in enumerate(generalexpected.items()):
 folder='work/causal_study/generality/'+name;rr=C(folder+'/trajectory.csv');meta=J(folder+'/metadata.json');steps=[int(r['step']) for r in rr]
 streak=0;confirm=None
 for r in rr:
  if int(r['step'])%100==0:
   streak=streak+1 if float(r['test_accuracy'])>=.95 else 0
   if streak>=6 and confirm is None:confirm=int(r['step'])
 post=[r for r in rr if int(r['step'])>=confirm];events=[r for r in post if float(r['train_accuracy'])<.9 and r['test_accuracy'] and float(r['test_accuracy'])<.9]
 measured=[r for r in post if r['test_accuracy']];worst=min(measured,key=lambda r:float(r['test_accuracy']));first=events[0] if events else None
 general.append(dict(name=name,updates=steps[-1]-steps[0],confirm=confirm,event_step=int(first['step']) if first else None,min=float(worst['test_accuracy']),peak=int(worst['step'])))
 for key,reported,value in [('confirmation',expected[0],confirm),('event step',expected[1],int(first['step']) if first else None),('event test',expected[2],100*float(first['test_accuracy']) if first else None),('minimum test',expected[3],100*float(worst['test_accuracy']))]:
  add(f'Generality {name} {key}',reported,value,[folder+'/trajectory.csv'],119+i,digits=3 if isinstance(value,float) else None)
 add(f'Generality {name} seed and horizon',[0,30000],[meta['seed'],steps[-1]-steps[0]],[folder+'/metadata.json',folder+'/trajectory.csv'],108)
 add(f'Generality {name} dense train inspection',True,True,['work/causal_study/generality_runner.py:107-118'],9,status='MANUAL_VERIFIED',note='Source inspects every update; CSV retains every tenth state plus every joint failure. Raw CSV alone cannot demonstrate all train-only triggers.')
 add(f'Generality {name} all post-grok train triggers evaluated',True,all(r['test_accuracy'] for r in post if float(r['train_accuracy'])<.9),[folder+'/trajectory.csv'],9)
 if name.startswith('rms'):add(f'Generality {name} final accuracy',100.,100*float(rr[-1]['test_accuracy']),[folder+'/trajectory.csv'],113)
 if name=='subtraction_accurate':add('Subtraction accurate final test',99.966,100*float(rr[-1]['test_accuracy']),[folder+'/trajectory.csv'],127,digits=3)
add('Causal study update total',364302,sum(r['updates'] for r in branch_records)+sum(r['updates'] for r in general),[r['path']+'/trajectory.csv' for r in branch_records]+[f'work/causal_study/generality/{r["name"]}/trajectory.csv' for r in general],4)
# Mean decomposition and gradient interventions, every reported accuracy independently from feature arrays.
meanexpected=[32.11,9.42,.93,53.34,66.07];centerexpected=[95.26,92.48,100.,87.20,98.46];meannorms=[]
for seed,folder in enumerate(adjfolders):
 f=N(folder+'/features.npz');h0=f['H0'].astype(float);h=f['H1'].astype(float);w0=f['W0'].astype(float);dw=f['W1'].astype(float)-w0;t=f['train'];v=f['heldout'];y=f['y'];mu=h[t].mean(0);base=h@w0
 terms=[np.broadcast_to(mu@dw,base.shape),(h-mu)@dw];full=h@dw
 for term,reported,label in zip(terms,[meanexpected[seed],centerexpected[seed]],['mean only','centered only']):add(f'Mean intervention seed {seed} {label}',reported,100*float(((base[v]+term[v]).argmax(1)==y[v]).mean()),[folder+'/features.npz'],101,digits=2)
 err=np.linalg.norm(terms[0]+terms[1]-full)/np.linalg.norm(full)
 add(f'Mean decomposition seed {seed} relative error bound',True,err<3e-16,[folder+'/features.npz'],101,note=f'Independent float64 recomputation {err:.17g}; numerical reduction order affects last bits.')
 meannorms.append(float(np.linalg.norm(h0.mean(0))))
 event=J(f'work/causal_study/diagnostics_results/seed{seed}/event.json')
 add(f'Actual update seed {seed} exact replay',True,event['exact_replay'],[f'work/causal_study/diagnostics_results/seed{seed}/event.json'],11)
add('Pre-event full-grid feature mean range',[10374,12203],[round(min(meannorms)),round(max(meannorms))],[f+'/features.npz' for f in adjfolders],96)
# Directional interpolation all groups, scales and endpoints.
for kind,line in [('joint_direction',103),('rms_joint_direction',136)]:
 p=f'work/causal_study/diagnostics_results/{kind}.json';d=J(p)
 for event in d['events']:
  who=str(event.get('seed',event.get('condition')))
  deriv=event['directional_derivatives']['reference64_cast32'];interp={r['scale']:r for r in event['interpolation']}
  add(f'{kind} {who} all directional contributions negative',True,deriv['total']<0 and all(v<0 for v in deriv['by_group'].values()),[p],line)
  losskey='training_accurate_loss64' if kind=='joint_direction' else 'train_accurate_loss64'
  base=interp[0.][losskey]
  for a in [.001,.01,1.]:add(f'{kind} {who} interpolation at {a}',True,interp[a][losskey]<base if a<1 else interp[a][losskey]>base,[p],line,note=f'Loss {base:.15g} to {interp[a][losskey]:.15g}.')
  add(f'{kind} {who} exact interpolation endpoints',True,all(event['endpoint_parameters_exact'].values()),[p],line)
# Diagnostic ranges, single-checkpoint repair, and complete-update intervention counts.
p='work/causal_study/derivative_identity_check.json';d=next(r for r in J(p)['checkpoints'] if r['step']==16000)
for method,expected in [('stock',32.02),('row_mean_projected',31.88)]:add(f'Seed4 step16000 {method} derivative error',expected,100*d['methods'][method]['relative_l2_error'],[p],89,digits=2)
add('Seed4 step16000 target repair derivative error mantissa',4.19,d['methods']['target_repaired']['relative_l2_error']/1e-7,[p],89,digits=2)
add('Target repair leaves wrong-class derivatives bitwise unchanged',True,d['wrong_class_derivatives_bitwise_preserved'],[p,'work/causal_study/diagnostics_repairs.py'],89)
p='work/causal_study/diagnostics_results/matched_specificity_final.json';proj=[r for r in J(p)['comparisons'] if 'row_projection' in r['label']]
# Locate the actual backward-rule measurements (distinct from hypothetical stock diagnostics).
def find_key(tree,key):
 if isinstance(tree,dict):
  if key in tree:yield tree[key]
  for v in tree.values():yield from find_key(v,key)
 elif isinstance(tree,list):
  for v in tree:yield from find_key(v,key)
# Kept explicit in the source records.
for r in proj:
 assert r['step']==20000
actuals=[r['actual_training_gradient'] for r in proj]
errs=[r['logit_gradient']['relative_l2_error'] for r in actuals]
zs=[r['logit_gradient']['row_sum_relative_l2'] for r in actuals]
add('Projected step20000 derivative error minimum',52.94,100*min(errs),[p],89,digits=2)
add('Projected step20000 derivative error maximum',86.07,100*max(errs),[p],89,digits=2)
add('Projected step20000 class-sum residual bound',True,max(zs)<2.1e-8,[p],89,note=f'Maximum {max(zs)}')
timecourse=[J(f'work/causal_study/diagnostics_results/seed{s}/timecourse.json') for s in range(5)]
g15000=[next(r for r in rr if r['step']==15000) for rr in timecourse]
for field,expected in [('logit',[10.56,21.86]),('head',[18.95,43.42])]:
 vals=[r['logit_gradients']['stock32']['relative_l2_error'] if field=='logit' else r['parameter_gradients']['stock32']['readout']['relative_l2_error'] for r in g15000]
 for side,val,ex in zip(['min','max'],[min(vals),max(vals)],expected):add(f'Stock step15000 {field} gradient error {side}',ex,100*val,[f'work/causal_study/diagnostics_results/seed{s}/timecourse.json' for s in range(5)],91,digits=2)
common=[next(r for r in J(f'work/causal_study/diagnostics_results/seed{s}/adam_common.json')['checkpoints'] if r['step']==15000) for s in range(5)]
cf=[r['variants']['reference_plus_common']['actual_update_difference_from_reference']['centered_fraction'] for r in common]
for side,val,ex in zip(['min','max'],[min(cf),max(cf)],[45.85,73.37]):add('Adam common error to centered update '+side,ex,100*val,[f'work/causal_study/diagnostics_results/seed{s}/adam_common.json' for s in range(5)],91,digits=2)
sgd=[r['variants']['reference_plus_common']['sgd_update_difference_from_reference']['centered_fraction'] for r in common]
add('SGD centered fraction approximately 3e-16',True,all(2e-16<v<5e-16 for v in sgd),[f'work/causal_study/diagnostics_results/seed{s}/adam_common.json' for s in range(5)],91,note=str(sgd))
p='work/causal_study/diagnostics_results/matched_early_arithmetic.json';early=J(p)['comparisons'];rat=[]
for seed in range(5):
 a=next(r for r in early if r['label']==f'seed{seed}_stock15000');b=next(r for r in early if r['label']==f'seed{seed}_accurate15000');rat.append(a['full_grid_features']['mean_norm']/b['full_grid_features']['mean_norm'])
for label,value,expected in zip(['min','max'],[min(rat),max(rat)],[2.22,3.83]):add('Stock/accurate feature mean ratio '+label,expected,value,[p],91,digits=2)
for seed in range(5):
 p=f'work/causal_study/diagnostics_results/seed{seed}/event.json';e=J(p);r=next(r for r in e['gradient_swap_arms'] if r['mask']=='111')['metrics']
 add(f'Seed{seed} accurate terminal complete gradient still jointly fails',True,r['train_accuracy']<.9 and r['test_accuracy']<.9,[p],103)
# Full-network planted gauge controls and fresh adjacent measured error floors.
fixedfiles=list((ROOT/'work/basis_study/model_gauge_results/planted').glob('float*_cond*.json'))
add('Fixed-condition planted controls',8,len(fixedfiles),[str(p.relative_to(ROOT)) for p in fixedfiles],58)
for pth in fixedfiles:
 p=str(pth.relative_to(ROOT));d=J(p)
 add(pth.stem+' exact compensation accuracy',1.,d['native_compensated']['accuracy'],[p],58)
 for fit,r in d['fits'].items():add(pth.stem+' '+fit+' compensation accuracy',1.,r['compensated_readout_classification']['accuracy'],[p],58)
actualfiles=[]
for dirname in ['model_gauge_results','model_gauge_fresh']:
 actualfiles.extend(p for p in (ROOT/'work/basis_study'/dirname/'actual').glob('*.json') if p.name!='summary.json')
add('Observed-matrix planted control count',24,sum(len(J(str(p.relative_to(ROOT)))['maps']) for p in actualfiles),[str(p.relative_to(ROOT)) for p in actualfiles],58)
for seed,expected in zip([1,2,3],[4.65,1.10,10.77]):
 p=f'work/basis_study/model_gauge_fresh/actual/seed{seed}_adjacent_event.json';d=J(p)['maps']['final_token_ols'];observed=d['sites']['block0_after_mlp']['tokens']['2']['centered_relative_error'];floor=d['same_A_network_positive_control']['known_A_sites']['block0_after_mlp']['tokens']['2']['centered_relative_error']
 add(f'Fresh adjacent seed{seed} forward-map error',expected,100*observed,[p],58,digits=2)
 add(f'Fresh adjacent seed{seed} planted floor rounds to stated range',True,.0004<=round(100*floor,4)<=.0006,[p],58,note=f'Raw percent {100*floor}')
for seed,expected in zip([1,2,3],[95.13,100.,94.37]):
 p=f'work/basis_study/model_gauge_fresh/actual/seed{seed}_first_event.json';d=J(p)['maps']['final_token_ols']
 add(f'First event seed{seed} inverse-linear compensation',expected,100*d['compensated_readout_classification']['accuracy'],[p],60,digits=2)
f=N('work/experiments/seed0_dense_event/features.npz')
add('Seed0 distant head test at collapse',14.05,100*acc(f['H1'],f['W0'],f['y'],f['heldout']),['work/experiments/seed0_dense_event/features.npz'],60,digits=2)
add('Seed0 distant head step',17200,J('work/experiments/seed0_dense_event/metadata.json')['healthy_step'],['work/experiments/seed0_dense_event/metadata.json'],60)
add('Seed0 adjacent head step',17493,J('work/basis_study/adjacent_seed0/metadata.json')['first_step'],['work/basis_study/adjacent_seed0/metadata.json'],60)
p='work/basis_study/task_subspace_results/seed4_generalization_drop_16000_17000/report.json';d=J(p)
add('Task-only coefficient shape',[113,128],[d['dimensions']['modulus'],d['dimensions']['d']],[p],60)
add('Task-only full row rank',113,d['invertible_completion']['coefficient_rank'],[p],60)
add('Task-only invertible raw-feature accuracy',4.62,100*d['mapping']['task_invertible_completion']['raw_classification']['accuracy'],[p],60,digits=2)
# Every immediate repair continuation; raw logs and preservation flags.
for r in C('work/basis_study/intervention_continuation_table.csv'):
 if float(r['initial_test_accuracy'])>=.9 and r['every_optimizer_state_preserved']=='True':
  rr=C(str(evidence_path(r['source']).parent.relative_to(ROOT))+'/trajectory.csv');post=next(z for z in rr if int(z['step'])>int(rr[0]['step']))
  add('Successful state-preserving repair '+r['event']+'/'+r['arm']+' fails next update',True,float(post['train_accuracy'])<.9,[str(evidence_path(r['source']).parent.relative_to(ROOT))+'/trajectory.csv'],62)
# Generality adjacent parameter groups and all first-event Figure8 values.
provenance=json.loads((REPO/'paper/source/figures/followup_figure_provenance.json').read_text())
for i,name in enumerate(['subtraction_stock','rms_stock','rms_accurate']):
 p=f'work/causal_study/generality/{name}/event_analysis.json';d=J(p);hy={r['mask']:r for r in d['parameter_hybrids']}
 for j,mask in enumerate(['000','001','010','111']):
  add(f'Figure8 {name} {mask}',provenance['appendix_first_events'][i]['test_accuracy'][j],100*hy[mask]['metrics']['heldout']['accuracy'],[p],132)
 add(f'Figure8 {name} adjacent step',d['pre_step']+1,d['post_step'],[p],132)
 add(f'Figure8 {name} exact next update',True,d['exact_next_update_replay'],[p],136)
 if name=='rms_accurate':
  for mask,reported,label in [('000',96.208,'before'),('111',.884,'after'),('001',96.208,'head only'),('010',.940,'embedding only'),('110',.895,'freeze head')]:add('Accurate RMS '+label,reported,100*hy[mask]['metrics']['heldout']['accuracy'],[p],136,digits=3)
  grad=d['pre_gradient']['logit_gradients']['accurate']
  add('Accurate RMS derivative error mantissa',9.48,grad['relative_l2_error']/1e-8,[p],136,digits=2)
  add('Accurate RMS vanished targets',0,d['pre_gradient']['logit_gradients']['accurate']['zero_target_fraction'],[p],136)
p='work/causal_study/generality/subtraction_stock/mean_mediation.json';d=J(p)
for arm,expected in [('reference',98.81),('mean_only',5.84),('centered_only',98.76)]:add('Subtraction '+arm,expected,100*d['arms'][arm]['heldout']['accuracy'],[p],127,digits=2)
for r in general:
 if r['name'].startswith('rms'):add(r['name']+' minimum step',28540 if r['name']=='rms_stock' else 28503,r['peak'],[f'work/causal_study/generality/{r["name"]}/trajectory.csv'],136)
# Figure4 manifest validated against independently recomputed data.
for i in range(5):
 for j in range(4):add(f'Figure4a seed{i} column{j}',provenance['main_panel_a']['test_accuracy_percent'][i][j],adj[i][j],[adjfolders[i]+'/features.npz'],152,'main.tex')
 add(f'Figure4b seed{i} decoder',provenance['main_panel_b']['fresh_decoder_test_accuracy_percent'][i],probe_values[i],[probefolders[i]+'/features.npz',probefolders[i]+'/decoder_whitened.npz'],152,'main.tex')
 add(f'Figure4b seed{i} native',provenance['main_panel_b']['native_test_accuracy_percent'][i],native_values[i],[probefolders[i]+'/features.npz'],152,'main.tex')
for rel,digest in provenance['source_sha256'].items():
 canonical=REPO/'followup_studies'/rel
 add('Published figure input hash '+rel,digest,hashlib.sha256(canonical.read_bytes()).hexdigest(),[str(canonical)],152,'main.tex')

# Remaining replicated statement and group-count checks.
for seed,folder in enumerate(adjfolders):
 f=N(folder+'/features.npz');m=J(folder+'/metadata.json');step=m['collapsed_step'] if 'collapsed_step' in m else m['second_step'];test=100*acc(f['H1'],f['W1'],f['y'],f['heldout']);train=acc(f['H1'],f['W1'],f['y'],f['train'])
 add(f'Original event seed{seed} recorded step',[17494,17721,18069,16308,16056][seed],step,[folder+'/metadata.json'],78+seed)
 add(f'Original event seed{seed} test',[31.872,9.095,.929,53.597,66.327][seed],test,[folder+'/features.npz'],78+seed,digits=3)
 add(f'Original event seed{seed} joint below90',True,train<.9 and test<90,[folder+'/features.npz'],87)
 g=next(r for r in timecourse[seed] if r['step']==6000)['logit_gradients']['stock32']['relative_l2_error']
 add(f'All five source6000 already measurable derivative error: seed{seed}',True,g>1e-3,[f'work/causal_study/diagnostics_results/seed{seed}/timecourse.json'],67,note=f'Relative error {g}')
for arm in ['stable32','target_repair','row_projection']:
 armrows=[r for r in specificity if r['arm']==arm]
 add('Specificity arm '+arm+' replicate count',5,len(armrows),[r['path'] for r in armrows],87)
 add('Specificity arm '+arm+' event incidence',0,sum(bool(r['joint']) for r in armrows),[r['path'] for r in armrows],87)
contmeta=[J(str(Path(p).parent/'summary.json')) for p,n in cont]
counts={}
for r in contmeta:counts[r['event']]=counts.get(r['event'],0)+1
add('Continuation events and arms per event',[9,9],sorted(counts.values()),[p for p,n in cont],62)
add('Fixed planted dtype/condition factorial grid',[(dt,c) for dt in ['float32','float64'] for c in [1,3,10,100]],sorted((J(str(p.relative_to(ROOT)))['dtype'],J(str(p.relative_to(ROOT)))['planted_condition']) for p in fixedfiles),[str(p.relative_to(ROOT)) for p in fixedfiles],58)
for name in generalexpected:
 p=f'work/causal_study/generality/{name}/metadata.json';m=J(p)
 if name.startswith('rms'):add(name+' RMS epsilon',1e-6,m['rms_epsilon'],[p],108)
# Repeated main text numbers use the same independently reconstructed datasets.
add('Main minimum fresh decoder',98.20,min(probe_values),[f+'/decoder_whitened.npz' for f in probefolders],24,'main.tex',digits=2)
add('Main maximum fresh decoder',100.,max(probe_values),[f+'/decoder_whitened.npz' for f in probefolders],24,'main.tex')
add('Main corrected minimum measured test',98.86,100*min(r['mintest'] for r in primary),[r['path']+'/trajectory.csv' for r in primary],159,'main.tex',digits=2)
add('Figure4 original event incidence',5,len(adjfolders),[f+'/features.npz' for f in adjfolders],152,'main.tex')
add('Figure4 accurate event incidence',0,sum(bool(r['joint']) for r in primary),[r['path']+'/trajectory.csv' for r in primary],152,'main.tex')
replay=json.loads((OUT/'checkpoint_replay_audit.json').read_text())
for r in replay['matched_states']:
 add('Fresh independent matched-state comparison: '+r.get('branch',r.get('condition')),True,r.get('source_state_bitwise_equal',r.get('initial_parameters_optimizers_rng_bitwise_equal')),[str(OUT/'checkpoint_replay_audit.json')],11)
for label,r in replay['panels'].items():add('Fresh independent all-scalar direction replay '+label,True,r['fresh_checkpoint_recomputation_identical'],[str(OUT/'checkpoint_replay_audit.json')],103 if label=='unnormalized' else 136)
# Mathematical assertions reviewed separately: full-rank feature column spaces,
# full-row-rank task coefficient extension, exact Fourier support and power symmetry.
add('Full-column-rank global GL criterion','Column spaces equal iff invertible map exists','Column spaces equal iff invertible map exists',['work/basis_study/geometry.py',str(OUT/'report.md')],38,status='MANUAL_VERIFIED',note='Linear-algebra proof: with rank d, coordinate matrix between two bases of one d-dimensional column space is invertible.')
add('Full-row-rank task coefficient GL extension','Any two rank113 matrices in R^(113x128) admit invertible alignment','Any two rank113 matrices in R^(113x128) admit invertible alignment',['work/basis_study/task_subspace.py',str(OUT/'report.md')],60,status='MANUAL_VERIFIED',note='Extend the 113 independent coefficient rows to bases of R^128, then map the completed bases.')

# Persist the complete ledger.
(OUT/'claim_ledger.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'recomputed_datasets.json').write_text(json.dumps(dict(adjacent=adj,probe=probe_values,native_probe=native_values,branches=branch_records,generality=general,continuations=cont),indent=2)+'\n')
with (OUT/'claim_ledger.csv').open('w', newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=['id','file','line','claim','reported','recomputed','status','evidence','note'])
 writer.writeheader()
 for row in rows:
  writer.writerow(dict(id=row['id'],file=row['source']['file'],line=row['source']['line'],claim=row['claim'],reported=json.dumps(row['reported']),recomputed=json.dumps(row['recomputed']),status=row['status'],evidence='; '.join(row['evidence']),note=row['note']))
manifest={}
for row in rows:
 for evidence in row['evidence']:
  path=Path(evidence)
  if not path.is_absolute():path=ROOT/path
  if path.is_file():manifest[evidence]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
(OUT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'checks':len(rows),'failures':[r for r in rows if r['status']=='FAIL']},indent=2))
