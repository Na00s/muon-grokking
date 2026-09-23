"""Validate completed fixed-horizon branches and summarize their finite endpoints."""
import argparse
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text())

def inspect_branch(directory,expected_end):
    summary=read(directory/'summary.json')
    metadata=read(directory/'metadata.json')
    assert summary['status']=='completed'
    assert summary['end_step']==expected_end==metadata['end_step']
    with (directory/'trajectory.csv').open() as stream:rows=list(csv.DictReader(stream))
    start=summary['start_step']
    assert len(rows)==expected_end-start+1,(directory,len(rows))
    assert [int(x['step']) for x in rows]==list(range(start,expected_end+1))
    failing=[x for x in rows if float(x['train_accuracy'])<.9]
    assert all(x['test_accuracy'] for x in failing)
    joint=[x for x in failing if float(x['test_accuracy'])<.9]
    first=int(joint[0]['step']) if joint else None
    assert first==summary['first_joint_failure_step']
    measured=[x for x in rows if x['test_accuracy']]
    assert measured[0]['step']==str(start) and measured[-1]['step']==str(expected_end)
    assert summary['train_below90_states']==len(failing)
    assert summary['joint_below90_states']==len(joint)
    assert abs(min(float(x['train_accuracy']) for x in rows)-summary['minimum_train_accuracy'])<1e-14
    assert abs(min(float(x['test_accuracy']) for x in measured)-summary['minimum_measured_test_accuracy'])<1e-14
    if first is not None:
        assert (directory/'collapse.pt').is_file() and (directory/'previous.pt').is_file()
    return dict(name=directory.name,seed=summary['seed'],arm=summary['arm'],start=start,end=expected_end,
        updates=expected_end-start,event=first is not None,first_joint_failure_step=first,
        minimum_train_accuracy=summary['minimum_train_accuracy'],minimum_measured_test_accuracy=summary['minimum_measured_test_accuracy'],
        measured_test_states_below95=sum(float(x['test_accuracy'])<.95 for x in measured),
        final_test_accuracy=float(rows[-1]['test_accuracy']),final_train_accuracy=float(rows[-1]['train_accuracy']),
        source_checkpoint_sha256=summary['source_checkpoint_sha256'],path=str(directory.relative_to(HERE)),dense_endpoint_validated=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    plan=read(HERE/'main_suite_plan.json');rows=[];pending=[]
    jobs=[(HERE/'main_runs'/job['name'],job['end']) for job in plan['jobs']]
    jobs +=[(HERE/'projection_runs'/f'seed{seed}',20000) for seed in range(5)]
    for directory,end in jobs:
        if not (directory/'summary.json').exists():pending.append(str(directory.relative_to(HERE)));continue
        rows.append(inspect_branch(directory,end))
    if pending and not args.allow_partial:raise RuntimeError(f'Incomplete branches: {pending}')
    primary=[];specificity=[]
    for seed in range(5):
        event=read(HERE/'diagnostics_results'/f'seed{seed}'/'event.json')
        assert event['exact_replay'] and event['in_memory_source_state_unchanged']
        stock=next(x for x in event['gradient_swap_arms'] if x['mask']=='000')['metrics']
        assert stock['train_accuracy']<.9 and stock['test_accuracy']<.9
        accurate=next((x for x in rows if x['name']==f'seed{seed}_accurate6000'),None)
        primary.append(dict(seed=seed,stock_event_step=event['step']+1,stock_event_test_accuracy=stock['test_accuracy'],
            stock_event_train_accuracy=stock['train_accuracy'],stock_endpoint=True,accurate=accurate,
            stock_final=next((x for x in rows if x['name']==f'seed{seed}_stock_extension'),None)))
        specificity.append(dict(seed=seed,**{name:next((x for x in rows if x['path']==path),None) for name,path in [
            ('target_repair',f'main_runs/seed{seed}_target_repair15000'),
            ('accurate',f'main_runs/seed{seed}_stable3215000'),
            ('row_projection',f'projection_runs/seed{seed}')]}))
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    result=dict(complete=not pending,pending=pending,new_main_and_projection_updates=sum(x['updates'] for x in rows),
        validated_dense_branches=len(rows),expected_dense_branches=len(jobs),primary=primary,specificity=specificity,branches=rows,
        scope='Binary stock incidence uses verified saved positive events. Accurate monitoring checks every possible joint failure. Specificity horizon ends20000; primary horizon ends30000.',
        metadata_erratum='Runner metadata primary_endpoint says30000 even in20k arms. The prespecified protocols, execution end_step, and this aggregation use their actual end20000.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    if rows:
        with (out/'branches.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:result[k] for k in ['complete','pending','new_main_and_projection_updates','validated_dense_branches']}))

if __name__=='__main__':main()
