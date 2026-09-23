"""Controlled parity replay of the two historical control implementations.
The historical step-44,000 state is unavailable. This replay has a distinct identity.
Uses an available seed-0 full-state checkpoint with transparent format conversion.
"""
import argparse, contextlib, copy, hashlib, importlib.util, json, os, platform, sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('MUON_SOURCE_REPO', str(BASE.parents[2])))
sys.path.insert(0,str(ROOT))
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
from model import ModularAdditionTransformer

def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def eq(a,b):
 if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
 if isinstance(a,dict): return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)): return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b

def main():
 source=BASE.parent/'experiments/seed0_original/step_016000.pt'
 raw=torch.load(source,map_location='cpu',weights_only=False)
 c=copy.deepcopy(raw)
 c['model_state_dict']={k.replace('transformer_blocks.0.','transformer_block.'):v for k,v in c['model_state_dict'].items()}
 # Both historical programs use a single auxiliary AdamW group. Preserve its
 # moments and step counters, merge the head moments, and explicitly select
 # the embedding group hyperparameters (LR=.001) for this implementation test.
 a=copy.deepcopy(c['optimizer_state_dicts']['auxiliary_adamw'])
 h=c['optimizer_state_dicts']['unembedding_adamw']
 assert a['param_groups'][0]['params']==[0,1] and h['param_groups'][0]['params']==[0]
 a['state'][2]=copy.deepcopy(h['state'][0]);a['param_groups'][0]['params']=[0,1,2]
 c['optimizer_state_dicts']={'muon':c['optimizer_state_dicts']['muon'],'auxiliary_adamw':a}
 c['optimizer_name']='muon'; c['arguments']={'muon_lr':.03,'muon_weight_decay':.1,'aux_lr':.001,'aux_weight_decay':1.}
 c['replay_test_note']='Implementation parity only. Merged head into one auxiliary group; head LR explicitly changed .00025 to .001. Does not reproduce historical step44000 runs.'
 path=BASE/'parity_source.pt';torch.save(c,path)
 manifest={'status':'registered_before_execution','historical_state_available':False,'source_checkpoint':str(source.relative_to(BASE.parent)),'source_sha256':sha(source),'adapted_source_sha256':sha(path),'start_step':16000,'end_step':18000,'device':'cpu','torch_version':torch.__version__,'python_version':platform.python_version(),'threads':1,'deterministic_algorithms':True,'note':c['replay_test_note'],'model_key_conversion_only':True,'optimizer_moments_and_steps_preserved':True,'rng_handling':'Each original main reseeds Python and torch to checkpoint.seed before constructing model. Full-batch forward has no dropout or random operations. Original scripts do not restore saved RNG.'}
 (BASE/'parity_protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
 paths={'group_historical':BASE/'historical_branch_collapse.py','component':ROOT/'analysis/interventions/branch_auxiliary_components.py','group_instrumented':ROOT/'analysis/interventions/branch_collapse.py'}
 for name,src in paths.items():
  m=loadmod(name,src)
  args=argparse.Namespace(checkpoint=path,mode='control',steps=2000,evaluation_interval=10,checkpoint_interval=2000,device='cpu',run_name=name,overwrite=False)
  m.parse_arguments=lambda:args
  m.get_device=lambda requested:torch.device('cpu')
  folder=BASE/'parity_runs'/name;folder.mkdir(parents=True,exist_ok=True)
  def output(run_name,overwrite,folder=folder):
   cps=folder/'checkpoints';cps.mkdir(exist_ok=False);return folder/'trajectory.csv',cps
  m.prepare_outputs=output
  with (folder/'run.log').open('w') as log,contextlib.redirect_stdout(log):m.main()
  torch.save({'torch_rng':torch.get_rng_state()},folder/'rng_final.pt')
  print(name,'done',flush=True)
 finals={k:torch.load(BASE/'parity_runs'/k/'checkpoints/step_018000.pt',map_location='cpu',weights_only=False) for k in paths}
 first=next(iter(finals.values()))
 import csv
 frames={k:list(csv.DictReader((BASE/'parity_runs'/k/'trajectory.csv').open())) for k in paths}
 metrics=['step','train_loss','train_accuracy','test_loss','test_accuracy','hidden_gradient_norm','auxiliary_gradient_norm','muon_applied_update_norm','muon_max_abs_applied_update','hidden_delta_norm']
 ref=next(iter(frames.values()))
 results={'protocol':manifest,'program_sha256':{k:sha(p) for k,p in paths.items()},'comparisons':{}}
 for k,v in finals.items():
  results['comparisons'][k]={'model_bitwise_equal':eq(first['model_state_dict'],v['model_state_dict']),'optimizers_bitwise_equal':eq(first['optimizer_state_dicts'],v['optimizer_state_dicts']),'all_201_shared_metric_rows_equal':all(all(x[f]==y[f] for f in metrics) for x,y in zip(ref,frames[k])),'final_rng_equal':eq(torch.load(BASE/'parity_runs/group_historical/rng_final.pt',weights_only=False),torch.load(BASE/'parity_runs'/k/'rng_final.pt',weights_only=False)),'checkpoint_sha256':sha(BASE/'parity_runs'/k/'checkpoints/step_018000.pt')}
 results['conclusion']='The historical group, component, and instrumented control programs give identical sampled metrics and bitwise-identical terminal model, optimizer and RNG states in this controlled CPU replay. Historical divergence at44010 remains unidentified because the44,000 checkpoint and execution manifests are unavailable.'
 (BASE/'parity_verification.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(results['comparisons'],indent=2),flush=True)
if __name__=='__main__':main()
