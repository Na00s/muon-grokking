"""Re-evaluate saved-state equivalence and all seven finite-direction panels.
No persistent experiment or manuscript files are mutated.
MUON_AUDIT_REPO and MUON_AUDIT_EVIDENCE_ROOT override repository and evidence
autodetection. Restore both released binary bundles before running. Absolute
provenance checkpoint paths are relocated to the chosen evidence root.
"""
from pathlib import Path
import json,os,sys
import torch
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

sys.path.insert(0,str(ROOT/'work/causal_study'))
import diagnostics_joint_direction as jd
import diagnostics_rms_joint_direction as rd
from diagnostics import configure_runtime,assert_identical_tree
configure_runtime(1)
records=[]
for top in ['main_runs','projection_runs']:
 for p in sorted((ROOT/'work/causal_study'/top).glob('*/metadata.json')):
  m=json.loads(p.read_text());source=torch.load(evidence_path(m['source_checkpoint']),weights_only=False,map_location='cpu');start=torch.load(p.parent/'start.pt',weights_only=False,map_location='cpu')
  for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state']:
   assert_identical_tree(source[key],start[key],key)
  records.append({'branch':str(p.parent.relative_to(ROOT)),'start_step':start['step'],'source_state_bitwise_equal':True})
for name in ['subtraction_stock','subtraction_accurate','rms_stock','rms_accurate']:
 p=ROOT/f'work/causal_study/generality/{name}/initial.pt';cp=torch.load(p,weights_only=False,map_location='cpu')
 if name=='subtraction_stock':reference=cp
 for key in ['model_state_dict','optimizer_state_dicts','torch_rng_state']:
  assert_identical_tree(reference[key],cp[key],key)
 records.append({'condition':name,'initial_parameters_optimizers_rng_bitwise_equal':True})
panels={}
for label,oldpath,calls in [('unnormalized','work/causal_study/diagnostics_results/joint_direction.json',[lambda s=s:jd.run(s) for s in range(5)]),('rms','work/causal_study/diagnostics_results/rms_joint_direction.json',[lambda a=a:rd.run(a) for a in ['stock','accurate']])]:
 old=json.loads((ROOT/oldpath).read_text())['events'];new=[f() for f in calls]
 for a,b in zip(old,new):
  for key in ['directional_derivatives','interpolation','endpoint_parameters_exact']:
   assert a[key]==b[key],(label,key)
 panels[label]={'fresh_checkpoint_recomputation_identical':True,'events':new}
( Path(__file__).parent/'checkpoint_replay_audit.json').write_text(json.dumps({'matched_states':records,'panels':panels},indent=2)+'\n')
print(json.dumps({'matched_branches':len(records),'direction_panels_recomputed':sum(len(v['events']) for v in panels.values()),'all_bitwise_or_scalar_exact':True},indent=2))
