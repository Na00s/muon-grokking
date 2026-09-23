import os,csv,hashlib,json,math,re,statistics
from pathlib import Path
B=Path(__file__).resolve().parent;R=Path(os.environ.get('MUON_SOURCE_REPO',str(B.parent/'muon-grokking')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(path):return list(csv.DictReader(path.open()))
def dump(name,obj):(B/name).write_text(json.dumps(obj,indent=2)+'\n')
def fst(a,k,n=5,t=.95):return next((int(a[i]['step']) for i in range(len(a)-n+1) if all(float(x[k])>=t for x in a[i:i+n])),None)
runids=['branch_control_from_44000','aux_control_from_44000','branch_control_from_44000_instrumented','collapse_spectral_replay_from_44000']
controls=[]
for name in runids:
 p=R/'runs'/f'{name}.csv';a=rows(p);fail=[x for x in a if float(x['train_accuracy'])<.9];mi=min(a,key=lambda x:float(x['test_accuracy']))
 controls.append(dict(run_id=name,csv=str(p.relative_to(R)),sha256=sha(p),evaluations=len(a),start_step=int(a[0]['step']),end_step=int(a[-1]['step']),evaluation_intervals=sorted(set(int(x['step'])-int(y['step']) for x,y in zip(a[1:],a[:-1]))),first_sampled_train_below90=int(fail[0]['step']),first_failure_train_accuracy=float(fail[0]['train_accuracy']),first_failure_test_accuracy=float(fail[0]['test_accuracy']),minimum_sampled_test_accuracy=float(mi['test_accuracy']),minimum_sampled_test_step=int(mi['step']),final_test_accuracy=float(a[-1]['test_accuracy']),maximum_sampled_auxiliary_gradient_norm=max(float(x['auxiliary_gradient_norm']) for x in a[1:]),train_below90_evaluations=len(fail),initial_metrics={k:float(a[0][k]) for k in ['train_loss','train_accuracy','test_loss','test_accuracy']},source_checkpoint_status='Historical path named by original scripts/manuscript; exact checkpoint binary unavailable in inspected artifacts',historical_backend_status='Not independently recorded; source only supports CUDA or MPS and auto prefers CUDA',rng_source_semantics='Python and torch reseeded to stored seed; CUDA seeded when available; saved RNG not restored. Full-batch model has no stochastic layer.',saved_branch_state_semantics='Model plus Muon and auxiliary AdamW state dicts, seed and source path; original branch checkpoint writer does not include RNG or backend.'))
first=rows(R/'runs'/f'{runids[0]}.csv');second=rows(R/'runs'/f'{runids[1]}.csv')
shared=['train_loss','train_accuracy','test_loss','test_accuracy','hidden_gradient_norm','auxiliary_gradient_norm','hidden_delta_norm','muon_applied_update_norm']
firstdiff=next(int(a['step']) for a,b in zip(first,second) if any(a[k]!=b[k] for k in shared))
dump('historical_controls.json',{'controls':controls,'native_log_match':{'file':'runs/muon_lr_0p01_seed_0_full.csv','step':44000,'matching_metrics':['train_loss','train_accuracy','test_loss','test_accuracy'],'scope':'Scalar match only; exact model/optimizer provenance unavailable'},'control_initial_scalars_equal':controls[0]['initial_metrics']==controls[1]['initial_metrics'],'first_distinct_logged_step':firstdiff,'bitwise_historical_start_equivalence':'Unverified: matching four metrics does not certify equal model/optimizer tensors. Original claimed shared source is retained with the limitation stated.'})
speed={}
for regime in ['adamw','muon']:
 p=R/'runs'/f'{regime}_sweep_summary.csv';a=rows(p);yes=[x for x in a if x['sustained_95_test_step']];vals=[int(float(x['sustained_95_test_step'])) for x in yes]
 speed[regime]={'source':str(p.relative_to(R)),'sha256':sha(p),'configurations':len(a),'successes':len(yes),'unsuccessful':[x['run_name'] for x in a if not x['sustained_95_test_step']],'successful_steps':vals,'success_conditional_mean':statistics.mean(vals),'fastest_successful_step':min(vals),'stable_configurations':[x['run_name'] for x in yes if x['stable_after_grokking']=='True'],'grid':sorted({(float(x['learning_rate']),float(x['weight_decay'])) if regime=='adamw' else (float(x['muon_learning_rate']),float(x['muon_weight_decay'])) for x in a})}
 med=[]
 for s in range(5):
  p=R/'runs'/f'seedstudy_{"adamw_stable" if regime=="adamw" else "muon"}_seed_{s}.csv';a=rows(p);g=fst(a,'test_accuracy');med.append(g)
 speed[regime]['selected_five_seed_grokking_steps']=med;speed[regime]['selected_five_seed_median']=statistics.median(med)
speed['ratios']={'selected_configuration_medians':speed['adamw']['selected_five_seed_median']/speed['muon']['selected_five_seed_median'],'fastest_successful_single_seed_sweep_entries':speed['adamw']['fastest_successful_step']/speed['muon']['fastest_successful_step'],'success_conditional_sweep_means':speed['adamw']['success_conditional_mean']/speed['muon']['success_conditional_mean']}
speed['selection']='AdamW baseline: fastest strictly stable sweep entry, lr.001 wd3; Muon: fastest entry, lr.03 wd.1, among nine configurations all showing at least one post-grokking sampled test accuracy<95%. Grids differ; sweep means exclude four AdamW failures at100000.'
dump('speed_audit.json',speed)
depth=[]
for arm in ['control','freeze']:
 p=R/'runs'/f'depth4_matched_freeze_seed_0_from_126200_{arm}.csv';a=rows(p);post=[x for x in a if int(x['step'])>126200]
 depth.append(dict(arm=arm,source=str(p.relative_to(R)),sha256=sha(p),all_rows=len(a),post_branch_evaluations=len(post),start_step=int(a[0]['step']),end_step=int(a[-1]['step']),step_interval=sorted(set(int(x['step'])-int(y['step']) for x,y in zip(a[1:],a[:-1]))),below90=sum(float(x['test_accuracy'])<.9 for x in post),below95=sum(float(x['test_accuracy'])<.95 for x in post),mean_test_accuracy=statistics.mean(float(x['test_accuracy']) for x in post),final_test_accuracy=float(a[-1]['test_accuracy'])))
dump('depth4_audit.json',depth)
chance=[]
for p in (R/'paper/source').rglob('*.tex'):
 for num,line in enumerate(p.read_text().splitlines(),1):
  if re.search(r'100\s*/\s*113|1\s*/\s*113|0\.885',line):chance.append({'file':str(p.relative_to(R)),'line':num,'text':line,'incorrect_100_over113':bool(re.search(r'100\s*/\s*113',line))})
dump('chance_notation_audit.json',{'chance_accuracy':1/113,'chance_percent':100/113,'incorrect_occurrences':sum(x['incorrect_100_over113'] for x in chance),'occurrences':chance})
print('audits written',speed['ratios'])
