"""Second event audit with parameter groups reconstructed from parameter names."""
from pathlib import Path
import copy,csv,json
import torch
import runner as r
HERE=Path(__file__).resolve().parent

def identical(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a==b

@torch.no_grad()
def direct_measure(model,tx,ty,vx,vy):
    result={};model.eval()
    for split,x,y,batch in [('train',tx,ty,tx.shape[0]),('test',vx,vy,1024)]:
        z=torch.cat([model(x[j:j+batch]) for j in range(0,len(x),batch)])
        n=int(len(y));k=int(torch.count_nonzero(z.argmax(dim=1)==y));result[split]=dict(correct=k,total=n,accuracy=k/n,accurate_ce32=float(r.source.accurate_cross_entropy(z,y)),accurate_ce64=float(r.source.accurate_cross_entropy(z.double(),y)))
    return result

def main():
    r.runtime();reports=[]
    for seed in range(5):
        path=HERE/'historical_reference' if seed==0 else HERE/f'runs/seed{seed}'
        analysis=HERE/'analyses'/('historical_seed0' if seed==0 else f'prospective_seed{seed}')
        old=torch.load(path/'previous.pt',map_location='cpu',weights_only=False);new=torch.load(path/'collapse.pt',map_location='cpu',weights_only=False)
        model,opts=r.build(seed,old);data=r.source.generate_modular_addition_data(seed=seed,operation='addition')
        model.train()
        for opt in opts.values():opt.zero_grad(set_to_none=True)
        loss=r.source.accurate_cross_entropy(model(data[0]),data[1]);loss.backward()
        for opt in opts.values():opt.step()
        assert identical(model.state_dict(),new['model_state_dict'])
        assert identical({k:o.state_dict() for k,o in opts.items()},new['optimizer_state_dicts'])
        assert torch.equal(torch.get_rng_state(),new['torch_rng_state'])
        if seed:assert r.equal(r.rng(),new['rng_states'])
        names=list(dict(model.named_parameters()))
        groups={
            'embeddings':[name for name in names if name in ['token_embedding.weight','position_embedding.weight']],
            'hidden':[name for name in names if name.startswith('transformer_blocks.')],
            'readout':[name for name in names if name=='unembedding.weight'],
        }
        assert sorted(sum(groups.values(),[]))==sorted(names)
        stored=json.loads((analysis/'parameter_swaps.json').read_text());records=[]
        for code in range(8):
            bits=[(code>>2)&1,(code>>1)&1,code&1];model,_=r.build(seed,old)
            weights=copy.deepcopy(old['model_state_dict'])
            for group,updated in zip(['embeddings','hidden','readout'],bits):
                if updated:
                    for name in groups[group]:weights[name]=new['model_state_dict'][name]
            model.load_state_dict(weights);observed=direct_measure(model,*data);record=stored['rows'][code]
            assert record['mask']==format(code,'03b')
            assert observed==record['metrics'],(seed,code)
            records.append(dict(mask=record['mask'],train_correct=observed['train']['correct'],test_correct=observed['test']['correct'],metrics_bitwise_identical=True))
        reports.append(dict(seed=seed,cohort='historical_cpu_pilot' if seed==0 else 'prospective_cpu',from_step=old['step'],to_step=new['step'],manual_replay_exact=True,verified_rng=['torch_cpu'] if seed==0 else ['torch_cpu','Python','NumPy'],parameter_groups_independently_reconstructed=True,swaps=records))
    checkpoint_records=json.loads((HERE/'checkpoint_manifest.json').read_text())
    for item in checkpoint_records:assert r.sha(HERE/item['path'])==item['sha256'],item['path']
    canonical=HERE.parent/'muon-grokking/followup_studies/work/rms_replication'
    for item in checkpoint_records:assert r.sha(canonical/item['path'])==item['sha256'],item['path']
    for name in ['summary.json','verification.json','run_summary.csv','event_summary.csv','parameter_swaps.csv','derivative_summary.csv','decoder_summary.csv','decoder_optimization.csv','report.md','protocol.json','source_manifest.json']:
        assert r.sha(HERE/name)==r.sha(canonical/name),name
    result=dict(passed=True,events=len(reports),independent_parameter_swaps=8*len(reports),all_checkpoint_hashes_checked=len(checkpoint_records),canonical_checkpoint_and_report_hashes_verified=True,event_checks=reports)
    r.write_json(HERE/'independent_final_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='event_checks'}))
if __name__=='__main__':main()
