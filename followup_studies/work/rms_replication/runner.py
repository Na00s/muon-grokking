"""Prospective accurate-CE RMS first-joint-failure replication on CPU."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,os,platform,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source_snapshot'/'causal_study'))
import generality_runner as source


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def save(path,value):
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);os.replace(tmp,path)
def runtime():
    source.configure_runtime()
    return dict(backend='cpu',platform=platform.platform(),python=sys.version,torch=str(torch.__version__),numpy=np.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),float32_matmul_precision=torch.get_float32_matmul_precision(),default_dtype=str(torch.get_default_dtype()),torch_build=torch.__config__.show(),mps_available=torch.backends.mps.is_available(),cuda_available=torch.cuda.is_available(),active_gpu=False)
def rng():return dict(torch_cpu=torch.get_rng_state().clone(),python=random.getstate(),numpy=np.random.get_state(),cuda=None,mps=None,gpu_note='CPU execution; no GPU RNG is used.')
def restore_rng(r):
    torch.set_rng_state(r['torch_cpu']);random.setstate(r['python']);np.random.set_state(r['numpy'])
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def snapshot(model,opts,step,seed,monitor=None):
    s=source.snapshot(model,opts,step,seed,'addition','rms','accurate');s['rng_states']=rng();s['backend']='cpu';s['monitor']=copy.deepcopy(monitor);s['protocol_sha256']=sha(HERE/'protocol.json') if (HERE/'protocol.json').exists() else None
    return s

def build(seed,checkpoint=None):
    random.seed(seed);np.random.seed(seed)
    model,opts=source.make_model_optimizers(seed,'rms',checkpoint)
    if checkpoint is not None and 'rng_states' in checkpoint:restore_rng(checkpoint['rng_states'])
    return model,opts

@torch.no_grad()
def evaluate(model,x,y):
    before=rng();training=model.training
    model.eval();z=torch.cat([model(x[i:i+1024]) for i in range(0,len(x),1024)]);model.train(training)
    assert equal(before,rng()),'Evaluation changed RNG'
    correct=int((z.argmax(-1)==y).sum());n=len(y)
    return dict(correct=correct,total=n,accuracy=correct/n,accurate_ce32=float(source.accurate_cross_entropy(z,y)),accurate_ce64=float(source.accurate_cross_entropy(z.double(),y)))

def train_forward(model,opts,x,y):
    model.train()
    for opt in opts.values():opt.zero_grad(set_to_none=True)
    z=model(x);loss=source.accurate_cross_entropy(z,y)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training objective')
    return z,loss

def apply(loss,opts):
    loss.backward()
    for opt in opts.values():opt.step()

FIELDS=['step','train_correct','train_total','train_accuracy','train_accurate_ce32','test_correct','test_total','test_accuracy','test_accurate_ce32','test_accurate_ce64','test_scheduled','grok_confirmed','grok_confirmation_step','joint_failure','elapsed_seconds']

def run(seed,out,steps=100000):
    env=runtime();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    model,opts=build(seed);data=source.generate_modular_addition_data(seed=seed,operation='addition');tx,ty,vx,vy=data
    names={id(p):n for n,p in model.named_parameters()}
    meta=dict(seed=seed,cohort='prospective_cpu',protocol_sha256=sha(HERE/'protocol.json'),runner_sha256=sha(__file__),environment=env,config=source.CONFIG,source_initial_tensor_sha256=source.tensor_digest(model),parameter_groups={k:[dict(**{n:v for n,v in group.items() if n!='params'},parameters=[names[id(p)] for p in group['params']]) for group in opt.param_groups] for k,opt in opts.items()},data={label:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest()) for label,a in zip(['train_inputs','train_targets','test_inputs','test_targets'],data)})
    write_json(out/'metadata.json',meta)
    monitor=dict(streak=0,grok_confirmation_step=None,first_qualifying_scheduled_step=None,minimum_post_confirmation_sampled_test_accuracy=None,test_evaluations=0,post_confirmation_test_evaluations=0,post_confirmation_below95_evaluations=0)
    save(out/'initial.pt',snapshot(model,opts,0,seed,monitor))
    previous=None;start=time.perf_counter()
    try:
        with (out/'trajectory.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
            for step in range(steps+1):
                z,loss=train_forward(model,opts,tx,ty);correct=int((z.detach().argmax(-1)==ty).sum());acc=correct/len(ty)
                scheduled=step%100==0
                test=evaluate(model,vx,vy) if scheduled or (monitor['grok_confirmation_step'] is not None and acc<.9) or step==steps else None
                if test:monitor['test_evaluations']+=1
                confirmation=False
                if scheduled:
                    monitor['streak']=monitor['streak']+1 if test['accuracy']>=.95 else 0
                    if monitor['streak']>=6 and monitor['grok_confirmation_step'] is None:
                        monitor['grok_confirmation_step']=step;monitor['first_qualifying_scheduled_step']=step-500;confirmation=True
                grokked=monitor['grok_confirmation_step'] is not None
                joint=bool(grokked and step>monitor['grok_confirmation_step'] and acc<.9 and test and test['accuracy']<.9)
                if grokked and test:
                    monitor['post_confirmation_test_evaluations']+=1
                    monitor['post_confirmation_below95_evaluations']+=int(test['accuracy']<.95)
                    old=monitor['minimum_post_confirmation_sampled_test_accuracy'];monitor['minimum_post_confirmation_sampled_test_accuracy']=min(old,test['accuracy']) if old is not None else test['accuracy']
                row=dict(step=step,train_correct=correct,train_total=len(ty),train_accuracy=acc,train_accurate_ce32=float(loss.detach()),test_correct=test['correct'] if test else None,test_total=test['total'] if test else None,test_accuracy=test['accuracy'] if test else None,test_accurate_ce32=test['accurate_ce32'] if test else None,test_accurate_ce64=test['accurate_ce64'] if test else None,test_scheduled=scheduled,grok_confirmed=grokked,grok_confirmation_step=monitor['grok_confirmation_step'],joint_failure=joint,elapsed_seconds=time.perf_counter()-start)
                writer.writerow(row)
                if scheduled or joint:f.flush();write_json(out/'progress.json',row)
                current=None
                if step%1000==0 or confirmation or joint or step==steps:
                    current=snapshot(model,opts,step,seed,monitor)
                    if step%1000==0:
                        save(out/f'step_{step:06d}.pt',current);save(out/'recovery.pt',current)
                        print(json.dumps(row),flush=True)
                    if confirmation:save(out/'grok_confirmation.pt',current)
                if joint or step==steps:
                    save(out/'terminal.pt',current)
                    status='confirmed_grokking_with_failure' if joint else ('confirmed_grokking_event_free_completion' if grokked else 'failure_to_confirm_grokking')
                    if joint:
                        assert previous is not None and previous['step']+1==step
                        save(out/'previous.pt',previous);save(out/'collapse.pt',current);write_json(out/'event.json',row)
                    result=dict(seed=seed,status=status,terminal_step=step,steps_executed=step,monitor=monitor,terminal=row,elapsed_seconds=time.perf_counter()-start)
                    write_json(out/'completion.json',result);print('COMPLETE '+json.dumps(result),flush=True);return result
                if grokked:previous=current or snapshot(model,opts,step,seed,monitor)
                apply(loss,opts)
    except BaseException:
        write_json(out/'interruption.json',dict(seed=seed,status='technical_interruption',step=step,traceback=traceback.format_exc(),last_recovery_checkpoint='recovery.pt'));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=100000);a=p.parse_args();run(a.seed,a.out,a.steps)
