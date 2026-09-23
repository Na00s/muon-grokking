import os,csv,hashlib,json,math,sys
from pathlib import Path
import torch
B=Path(__file__).resolve().parent;R=Path(os.environ.get('MUON_SOURCE_REPO',str(B.parents[2])));sys.path.insert(0,str(R));torch.set_num_threads(1)
from experiments.depth.train_depth_variant import DepthModularAdditionTransformer as Model
from data import generate_modular_addition_data
cp=B.parent/'experiments/seed0_original/step_006000.pt';c=torch.load(cp,map_location='cpu',weights_only=False)
m=Model(**c['model_config']);m.load_state_dict(c['model_state_dict']);m.eval();x,y,tx,ty=generate_modular_addition_data(seed=0)
values=[]
def h(module,args):values.append(args[0].detach())
hook=m.unembedding.register_forward_pre_hook(h)
with torch.no_grad():
 for a in torch.cat([x,tx]).split(1024):m(a)
hook.remove();features=torch.cat(values).double();allx=torch.cat([x,tx]);grid=torch.empty((113,113,features.shape[-1]),dtype=torch.float64);grid[allx[:,0],allx[:,1]]=features
ft=torch.fft.fft2(grid,dim=(0,1),norm='ortho');direct=float(features.mean(0).norm());fft=float(ft[0,0].norm()/113)
a=list(csv.DictReader((B/'feature_mean_timeline.csv').open()));raw=list(csv.DictReader((R/'runs/seedstudy_fourier_mode_layer_summary.csv').open()));checks=[]
for r in a:
 s=raw[int(r['source_row'])-2];p=float(s['total_spectral_power'])*float(s['dc_power_fraction_total']);norm=math.sqrt(p/12769)
 assert abs(norm-float(r['feature_mean_norm']))<1e-9
 assert r['run']==s['run'] and r['step']==s['step']
 # Independently using total-minus-nonDC has single-precision cancellation;
 # retain its deviation as a diagnostic rather than force exact agreement.
 alternative=math.sqrt(max(0,float(s['total_spectral_power'])-float(s['non_dc_spectral_power']))/12769)
 checks.append(abs(alternative-norm)/norm)
report={'status':'passed','input_checkpoint':str(cp.relative_to(B.parent)),'input_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'source_step':6000,'direct_full_grid_mean_norm':direct,'orthonormal_fft_recovered_mean_norm':fft,'relative_error':abs(direct-fft)/direct,'recomputed_source_rows':len(a),'source_row_identity_all_passed':True,'maximum_relative_discrepancy_using_float32_total_minus_nondc':max(checks),'note':'Identity checked independently on a saved fresh-cohort checkpoint, separately from archived checkpoints unavailable as tensors. All88plotted row calculations and source identities independently rechecked.'}
assert report['relative_error']<1e-12
(B/'feature_mean_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
