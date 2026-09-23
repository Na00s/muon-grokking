"""Incorporate audited long-horizon outcomes after all six branches finish.

This template is for the zero-event outcome. A detected event raises before any
manuscript change so its interpretation can be revised from the actual evidence.
"""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'work/muon-grokking/paper/source'
LONG=ROOT/'work/long_horizon'
results=json.loads((LONG/'summary.json').read_text())
audit=json.loads((LONG/'verification.json').read_text())
assert results['status']=='completed' and audit['status']=='passed'
assert audit['passed']==audit['total'] and all(c['passed'] for c in audit['checks'])
records=results['records']
assert len(records)==6 and all(r['end_step']==100000 for r in records)
assert sorted((r['operation'], r['seed']) for r in records)==[('addition', seed) for seed in range(5)]+[('subtraction', 0)]
assert all(r['extension_updates']==70000 and r['arithmetic_start_step']==(6000 if r['operation']=='addition' else 0) for r in records)
assert all(r['first_joint_failure_step'] is None for r in records), 'A late failure occurred; revise the scientific conclusion from its saved event.'
assert all(r['final_train_accuracy']==r['final_test_accuracy']==1 for r in records if r['operation']=='addition'), 'Read and describe the actual final accuracies.'
addition=[r for r in records if r['operation']=='addition']
subtraction=next(r for r in records if r['operation']=='subtraction')
minimum=100*min(r['minimum_measured_test_accuracy'] for r in addition)
updates={}
def replace_once(text, old, new):
    assert text.count(old)==1, f'Expected exactly one integration anchor: {old!r}'
    return text.replace(old,new,1)

assert r'\label{tab:long-horizon}' not in (SRC/'followup_appendix.tex').read_text(), 'Horizon integration already applied; no files changed.'

p=SRC/'main.tex';s=p.read_text()
s=replace_once(s,'through step $30{,}000$ in five matched branches','through step $100{,}000$ in five matched branches')
s=replace_once(s,'Arithmetic correction supplies another successful intervention over a shorter horizon.', r'Arithmetic correction also reaches the original $100{,}000$-step budget (\S\ref{sec:arithmetic}).')
s=replace_once(s,r'accurate CE from the matched step-$6{,}000$ state yields $0/5$ through $30{,}000$.', r'accurate CE from the matched step-$6{,}000$ state yields $0/5$ through $100{,}000$. Original failures were captured before step $30{,}000$.')
s=replace_once(s,r'Through step $30{,}000$, none has a joint train/test failure below $90\%$; minimum measured test accuracy is $98.86\%$, and all finish at $100\%$.',
    rf'Through step $100{{,}}000$, the original training horizon, none has a joint train/test failure below $90\%$; minimum measured test accuracy is ${minimum:.2f}\%$, and all finish at $100\%$.')
s=replace_once(s,r'One matched subtraction pair reproduces prevention through $30{,}000$ steps.',r'The corrected subtraction run also remains free of joint failure through $100{,}000$ steps.')
s=replace_once(s,'Longer horizons and brief test-only excursions remain unresolved',r'Behavior beyond $100{,}000$ steps and brief test-only excursions remain unresolved')
updates[p]=s

p=SRC/'followup_appendix.tex';s=p.read_text()
anchor='Verification replays are excluded from these totals.'
s=replace_once(s,anchor,r'A subsequent horizon extension adds $420{,}000$ updates: each of the five corrected addition branches and the corrected subtraction run continues from step $30{,}000$ to $100{,}000$. '+anchor)
anchor=r'\subsection{Mean amplification and finite-step overshoot}'
rows=[]
for r in records:
    label=f'Addition, seed {r["seed"]}' if r['operation']=='addition' else 'Subtraction, seed 0'
    start=f'{r["monitoring_start_step"]:,}'.replace(',',r'{,}')
    rows.append(rf'{label} & ${start}$ & ${100*r["minimum_measured_test_accuracy"]:.3f}$ & ${100*r["final_test_accuracy"]:.3f}$ & none \\')
extension=r'''\textbf{Persistence through the original training horizon.} We resume each corrected step-$30{,}000$ state through step $100{,}000$, matching the original depth-$1$ horizon. All model parameters, optimizer buffers, and the PyTorch random state are restored. Twelve direct updates agree bitwise with both the monitored runner and a six-plus-six resumed continuation for all six sources. Each extension adds $70{,}000$ updates, with training accuracy recorded at every state and test accuracy every $100$ steps and whenever training accuracy is below $90\%$. The final and minimum-accuracy checkpoints reproduce their logged metrics exactly.

\begin{table}[htbp]
\centering\small
\caption{Corrected-loss runs through step $100{,}000$. Accuracies are percentages. Minima combine the earlier prefix with the extension, counting the shared step-$30{,}000$ state once. Addition is corrected from step $6{,}000$; subtraction uses corrected CE from initialization and its minima start at grokking confirmation.}
\label{tab:long-horizon}
\begin{tabular}{lrrrr}
\toprule
Run & Monitoring start & Min.\ test & Final test & Joint event \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}

All six extensions complete without a joint event. Thus correction suppresses the observed failures through the original training budget. This addresses postponement beyond step $30{,}000$ within that budget; behavior after $100{,}000$ and brief test-only excursions between scheduled evaluations remain outside the measurement. The original stock trajectories already have captured failures by $30{,}000$, which establishes their positive binary incidence at the longer horizon. Their historical monitoring supports no comparison of event counts or durations with these dense branches.

'''
assert anchor in s
s=replace_once(s,anchor,extension+anchor)
s=replace_once(s,r'Accurate CE prevents the joint event through the horizon and finishes at $99.966\%$ test accuracy.',r'At step $30{,}000$, accurate CE has no joint event and test accuracy is $99.966\%$. Its continuation through $100{,}000$ is reported in Table~\ref{tab:long-horizon}.')
updates[p]=s

p=SRC/'figures/build_followup_figures.py';s=p.read_text()
old='''    accurate_events = [p["accurate"]["event"] for p in primary]
    incidence = [sum(original_events), sum(accurate_events)]
    assert incidence == [5, 0]'''
new='''    extended = read("work/long_horizon/summary.json")
    extended_addition = sorted([r for r in extended["records"] if r["operation"] == "addition"], key=lambda r: r["seed"])
    assert extended["status"] == "completed" and [r["seed"] for r in extended_addition] == list(range(5))
    assert all(r["arithmetic_start_step"] == 6000 and r["end_step"] == 100000 for r in extended_addition)
    accurate_events = [r["first_joint_failure_step"] is not None for r in extended_addition]
    incidence = [sum(original_events), sum(accurate_events)]
    assert incidence[0] == 5'''
assert old in s
s=replace_once(s,old,new)
s=replace_once(s,'"6,000 to 30,000\\nupdates"','"6,000 to 100,000\\nupdates"')
s=replace_once(s,'"source_step": 6000, "end_step": 30000,','"source_step": 6000, "end_step": 100000,\n                         "original_events_observed_by_step": 30000,')
s=replace_once(s,'Paired seeds; original monitoring grids are mixed. Binary incidence only.','Paired seeds; original failures were observed by 30,000 under mixed monitoring. Accurate branches continue through 100,000. Binary incidence only.')
updates[p]=s
for path, text in updates.items():
    path.write_text(text)
print('Incorporated six audited zero-event continuations through 100,000 steps.')
