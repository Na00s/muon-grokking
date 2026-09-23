"""Exercise preparation scripts in isolated temporary roots only.
No canonical manuscript, runner, or publication directory is modified.
"""
import ast, hashlib, json, shutil, tempfile
from pathlib import Path
REAL=Path(__file__).resolve().parents[3];PREP=REAL/'work/fourier_revision';checks=[]
def load(name,root):
 tree=ast.parse((PREP/name).read_text())
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('ROOT','HERE') for t in node.targets):
   key=node.targets[0].id;value=root if key=='ROOT' else root/'work/fourier_revision'
   node.value=ast.parse(f'Path({str(value)!r})',mode='eval').body
 ast.fix_missing_locations(tree);return compile(tree,str(PREP/name),'exec')
def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))
def hashes(path):return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}
with tempfile.TemporaryDirectory(prefix='muon-preparation-audit-') as tmp:
 root=Path(tmp);src=root/'work/muon-grokking/paper/source';src.mkdir(parents=True)
 for name in ['main.tex','followup_appendix.tex','figures/build_followup_figures.py']:
  p=src/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REAL/'work/muon-grokking/paper/source'/name,p)
 recs=[dict(operation=op,seed=seed,end_step=100000,extension_updates=70000,arithmetic_start_step=6000 if op=='addition' else 0,monitoring_start_step=6000 if op=='addition' else 10500,minimum_measured_test_accuracy=.9886 if op=='addition' else .9687884550844613,final_train_accuracy=1.,final_test_accuracy=1.,first_joint_failure_step=None) for op,seed in [('addition',i) for i in range(5)]+[('subtraction',0)]]
 write(root/'work/long_horizon/summary.json',{'status':'completed','records':recs});write(root/'work/long_horizon/verification.json',{'status':'passed','passed':1,'total':1,'checks':[{'passed':True}]})
 # A deliberately absent late figure anchor must leave every manuscript input unchanged.
 fig=src/'figures/build_followup_figures.py';original_fig=fig.read_text();fig.write_text(original_fig.replace('assert incidence == [5, 0]','assert incidence == [999, 999]'))
 before=hashes(src)
 try:exec(load('incorporate_completed_horizons.py',root),{'__file__':str(PREP/'incorporate_completed_horizons.py'),'__name__':'fixture'})
 except AssertionError:pass
 else:raise AssertionError('Missing-anchor fixture should have failed')
 assert hashes(src)==before;checks.append('all integration anchors checked before any manuscript mutation')
 fig.write_text(original_fig)
 exec(load('incorporate_completed_horizons.py',root),{'__file__':str(PREP/'incorporate_completed_horizons.py'),'__name__':'fixture'})
 assert r'"6,000 to 100,000\nupdates"' in fig.read_text();assert r'\label{tab:long-horizon}' in (src/'followup_appendix.tex').read_text();checks.append('completed-data integration, 100000 plot label, and table succeed in fixture')
 before=hashes(src)
 try:exec(load('incorporate_completed_horizons.py',root),{'__file__':str(PREP/'incorporate_completed_horizons.py'),'__name__':'fixture'})
 except AssertionError:pass
 else:raise AssertionError('Repeated integration should stop')
 assert hashes(src)==before;checks.append('duplicate integration rejected without mutation')
 for name in ['paper/audit_report.md','paper/revision_notes.md','README.md','followup_studies/README.md']:
  p=root/'work/muon-grokking'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REAL/'work/muon-grokking'/name,p)
 (root/'work/fourier_revision').mkdir(parents=True)
 for _ in range(2):
  exec(load('write_completed_docs.py',root),{'__file__':str(PREP/'write_completed_docs.py'),'__name__':'fixture'})
  current=hashes(root/'work/muon-grokking')
  if _==0:first=current
  else:assert current==first
 checks.append('completed documentation is repeatable without duplicate notices')
 # Package preflight must reject changed reviewed PDF before writing copied outputs.
 work=root/'work/fourier_revision';write(root/'work/long_horizon/completion.json',{'status':'completed'})
 for name in ['verification.json','frequency_verification.json']:write(root/'work/fourier_control/results'/name,{'all_passed':True})
 write(work/'control_audit/independent_verification.json',{'all_passed':True})
 (work/'build').mkdir();(work/'build/main.pdf').write_bytes(b'fixture')
 write(work/'final_verification.json',{'all_passed':True,'visual_review_complete':True,'checks':[{'passed':True}],'pdf_sha256':'deliberately-stale'})
 scope={'__file__':str(PREP/'package_revision.py'),'__name__':'fixture'};exec(load('package_revision.py',root),scope)
 before=hashes(root/'work/muon-grokking')
 try:scope['main']()
 except AssertionError as e:assert 'Reviewed PDF changed' in str(e)
 else:raise AssertionError('Stale PDF should be rejected')
 assert hashes(root/'work/muon-grokking')==before;checks.append('packager rejects stale reviewed PDF before publication mutation')
result={'all_passed':True,'check_count':len(checks),'checks':checks,'scope':'All integration/documentation/package executions used isolated temporary fixtures. No canonical source or training output was mutated.'}
(PREP/'control_audit/preparation_script_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
