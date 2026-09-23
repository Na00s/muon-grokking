"""Count sampled post-grokking episodes in the archived matched five-seed study."""
from pathlib import Path
import csv
import json

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / 'muon-grokking'


def sustained_start(rows, threshold=.95, consecutive=6):
    for i in range(len(rows) - consecutive + 1):
        if all(float(r['test_accuracy']) >= threshold for r in rows[i:i + consecutive]):
            return int(rows[i]['step'])
    return None


def episodes(rows, predicate):
    result=[]
    active=[]
    for row in rows:
        if predicate(row):
            active.append(row)
        elif active:
            result.append(active)
            active=[]
    if active:
        result.append(active)
    return result


def event_info(part):
    both=sum(float(r['train_accuracy']) < .9 and float(r['test_accuracy']) < .9 for r in part)
    return dict(first_sample_step=int(part[0]['step']),last_sample_step=int(part[-1]['step']),affected_evaluations=len(part),joint_train_test_below90_evaluations=both,minimum_train_accuracy=min(float(r['train_accuracy']) for r in part),minimum_test_accuracy=min(float(r['test_accuracy']) for r in part),classification='contains_joint_train_test_below90' if both else 'test_only_below90')


def summarize(seed,regime):
    path=REPO/'runs'/f'seedstudy_{regime}_seed_{seed}.csv'
    rows=list(csv.DictReader(path.open()))
    steps=[int(r['step']) for r in rows]
    assert steps==list(range(0,100001,100))
    grok=sustained_start(rows)
    post=[r for r in rows if int(r['step'])>=grok]
    def tr90(r):return float(r['train_accuracy']) < .9
    def te90(r):return float(r['test_accuracy']) < .9
    def te95(r):return float(r['test_accuracy']) < .95
    def both(r):return tr90(r) and te90(r)
    def testonly(r):return te90(r) and not tr90(r)
    test_episodes=[event_info(part) for part in episodes(post,te90)]
    schedules=sorted(set(int(r['scheduled_freeze_step']) for r in rows if r['scheduled_freeze_step']))
    frozen=[int(r['step']) for r in rows if int(r['auxiliary_frozen'])]
    return dict(seed=seed,regime=regime,file=str(path.relative_to(REPO)),rows=len(rows),last_step=steps[-1],evaluation_interval=100,sustained95_start_six_evaluations=grok,post_grok_evaluations=len(post),scheduled_freeze_steps=schedules,first_evaluation_reporting_frozen=frozen[0] if frozen else None,minimum_post_grok_train_accuracy=min(float(r['train_accuracy']) for r in post),minimum_post_grok_test_accuracy=min(float(r['test_accuracy']) for r in post),counts=dict(train_below90_evaluations=sum(map(tr90,post)),joint_train_test_below90_evaluations=sum(map(both,post)),test_only_below90_evaluations=sum(map(testonly,post)),train_only_below90_evaluations=sum(tr90(r) and not te90(r) for r in post),all_test_below90_evaluations=sum(map(te90,post)),all_test_below95_evaluations=sum(map(te95,post)),joint_condition_contiguous_segments=len(episodes(post,both)),test_only_condition_contiguous_segments=len(episodes(post,testonly)),all_test_below90_episodes=len(test_episodes),test_below90_episodes_containing_joint_train_failure=sum(e['joint_train_test_below90_evaluations']>0 for e in test_episodes),test_below90_episodes_entirely_test_only=sum(e['joint_train_test_below90_evaluations']==0 for e in test_episodes),all_test_below95_episodes=len(episodes(post,te95))),test_below90_episodes=test_episodes)


results=[summarize(seed,regime) for seed in range(5) for regime in ['muon','stable_muon']]
pair_checks=[]
for seed in range(5):
    ordinary=list(csv.DictReader((REPO/'runs'/f'seedstudy_muon_seed_{seed}.csv').open()))
    stable=list(csv.DictReader((REPO/'runs'/f'seedstudy_stable_muon_seed_{seed}.csv').open()))
    freeze=next(int(r['scheduled_freeze_step']) for r in stable if r['scheduled_freeze_step'])
    fields=['train_loss','train_accuracy','test_loss','test_accuracy']
    pairs=[(a,b) for a,b in zip(ordinary,stable) if int(a['step'])<=freeze]
    pair_checks.append(dict(seed=seed,scheduled_freeze=freeze,paired_pre_freeze_evaluations=len(pairs),maximum_absolute_metric_difference=max(abs(float(a[k])-float(b[k])) for a,b in pairs for k in fields)))
totals={regime:{key:sum(r['counts'][key] for r in results if r['regime']==regime) for key in results[0]['counts']} for regime in ['muon','stable_muon']}
report=dict(provenance='Reanalysis of archived CSVs at repository commit ANONYMIZED_SOURCE_REVISION; this is historical context, not a new experiment.',definitions=dict(grok='First of six consecutive 100-step evaluations at test accuracy >=95%. Counts include that evaluation through step100000.',episode='A maximal contiguous block of sampled test accuracy <90%; consecutive affected samples are counted once. An episode with any simultaneous train accuracy <90% is classified as containing joint failure; otherwise it is entirely test-only.',affected_evaluation='One 100-step-grid observation. The joint/test-only evaluation categories are mutually exclusive; their counts sum to all test-below90 observations.',sampling_limitation='Episode boundaries and durations are censored by the 100-step grid. A gap at an observed healthy evaluation separates episodes; unsampled recoveries and failures cannot be detected.',freeze_condition='stable_muon freezes token embeddings, position embeddings, and unembedding; all hidden matrices continue under Muon. Current source defaults schedule the freeze 2000 steps after the first of five consecutive test>=95% evaluations. Counts here independently use the requested six-evaluation grok definition.'),configuration=dict(operation='addition',modulus=113,train_fraction=.3,d_model=128,d_mlp=512,num_heads=4,num_layers=1,steps=100000,hidden_optimizer='Muon lr.03 wd.1 momentum.95 NS5 Nesterov',embeddings='AdamW lr.001 wd1 beta(.9,.999)',readout='AdamW lr.00025 wd1 beta(.9,.999)'),paired_metric_checks=pair_checks,totals_by_regime=totals,runs=results)
(HERE/'archived_event_summary.json').write_text(json.dumps(report,indent=2))
for r in results:
    c=r['counts']
    print(json.dumps(dict(seed=r['seed'],regime=r['regime'],grok=r['sustained95_start_six_evaluations'],freeze=r['scheduled_freeze_steps'],joint_evals=c['joint_train_test_below90_evaluations'],testonly_evals=c['test_only_below90_evaluations'],test90_episodes=c['all_test_below90_episodes'],episodes_with_joint=c['test_below90_episodes_containing_joint_train_failure'],test95_evals=c['all_test_below95_evaluations'],test95_episodes=c['all_test_below95_episodes'],minimum_test=r['minimum_post_grok_test_accuracy'])))
print(json.dumps({'pair_checks':pair_checks}))
