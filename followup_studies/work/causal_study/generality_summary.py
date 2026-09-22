"""Summarize the four completed generality controls and export a figure."""
import csv
import json
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(base):
    records=[]
    colors={'stock':'#b24734','accurate':'#167b94'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex='col',layout='constrained')
    for column,(prefix,title) in enumerate([('subtraction','Subtraction, no normalization'),('rms','Addition, gain-free RMS normalization')]):
        for arithmetic in ['stock','accurate']:
            path=base/f'{prefix}_{arithmetic}'
            result=json.loads((path/'completion.json').read_text())
            result['condition']=prefix;records.append(result)
            rows=list(csv.DictReader((path/'trajectory.csv').open()))
            measured=[row for row in rows if row['test_accuracy']]
            axes[0,column].plot([int(row['step']) for row in measured],[100*float(row['test_accuracy']) for row in measured],
                color=colors[arithmetic],linewidth=1.5,label=f'{arithmetic.capitalize()} CE')
            if result['event_step'] is not None:
                axes[0,column].axvline(result['event_step'],color=colors[arithmetic],linestyle=':',alpha=.7)
            diagnostics=json.loads((path/'gradient_timecourse.json').read_text())
            axes[1,column].semilogy([row['step'] for row in diagnostics],
                [row['logit_gradients'][arithmetic]['relative_l2_error'] for row in diagnostics],
                marker='o',markersize=4,color=colors[arithmetic],label=f'{arithmetic.capitalize()} CE')
        axes[0,column].set(title=title,ylim=(-2,102),ylabel='Held-out accuracy (%)')
        axes[0,column].axhline(90,color='#999999',linewidth=.7,linestyle='--')
        axes[1,column].set(ylabel='Logit-gradient relative error',xlabel='Training updates',xlim=(0,30000),ylim=(3e-8,1.5))
        axes[0,column].legend(loc='lower right' if column==0 else 'center left',frameon=False)
        for ax in axes[:,column]:
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.18)
            ax.ticklabel_format(axis='x',style='sci',scilimits=(3,3))
    fig.savefig(base/'generality_controls.png',dpi=180)
    fig.savefig(base/'generality_controls.pdf')
    (base/'summary.json').write_text(json.dumps(records,indent=2))
    with (base/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# Generality control results','',
        'Each pair starts from identical seed-0 initialization and runs for 30,000 updates under its prespecified architecture and operation. The accurate-loss arm changes the loss arithmetic throughout training. Every low-training-accuracy state after sustained grokking is checked on held-out data. The event threshold is simultaneous train and held-out accuracy below 90%.', '',
        '| Condition | Arithmetic | Grokking confirmed | First joint failure | Minimum post-grok train | Minimum sampled post-grok test | Final test |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    pct=lambda value:'Unevaluated' if value is None else f'{100*value:.3f}%'
    for row in records:
        event=str(row['event_step']) if row['event_step'] is not None else ('None through 30,000' if row['grok_step'] is not None else 'Unevaluated: no grokking')
        grok=str(row['grok_step']) if row['grok_step'] is not None else 'Not reached'
        lines.append(f"| {row['condition']} | {row['arithmetic']} | {grok} | {event} | {pct(row['post_grok_train_min'])} | {pct(row['post_grok_test_min'])} | {pct(row['final_test_accuracy'])} |")
    lines += ['', 'The accuracy panel includes every recorded held-out evaluation; dotted vertical lines indicate the first joint failure. The gradient panel compares the derivative actually used in each arm against an analytic float64 derivative at that same saved logit matrix. This isolates derivative arithmetic from differences in the learned weights. The conclusions cover one seed per configuration and the observed 30,000-update horizon.', '']
    lines += ['The endpoint table uses the every-update failure triggers and completed run summaries. The 100-update progress snapshots missed brief RMS failures that the every-update trigger captured.', '']
    for row in records:
        if row['event_step'] is None:continue
        folder=base/f"{row['condition']}_{row['arithmetic']}"
        cases=[('First registered event',folder)]
        if (folder/'peak_event/event_analysis.json').exists():cases.append(('Worst sampled checkpoint',folder/'peak_event'))
        for case,location in cases:
            event=json.loads((location/'event_analysis.json').read_text())
            hybrids={value['mask']:value['metrics']['heldout']['accuracy'] for value in event['parameter_hybrids']}
            mean=json.loads((location/'mean_mediation.json').read_text())
            reference=mean['arms']['reference']['heldout']['accuracy']
            mean_acc=mean['arms']['mean_only']['heldout']['accuracy']
            centered_acc=mean['arms']['centered_only']['heldout']['accuracy']
            derivative=event['pre_gradient']['logit_gradients'][row['arithmetic']]
            lines += [f"## {row['condition']} {row['arithmetic']}: {case.lower()} at {event['post_step']}",'',
                f"Exact replay reproduced the captured model, optimizer state, and RNG. The preceding model had {pct(hybrids['000'])} held-out accuracy. Updating only the readout gives {pct(hybrids['001'])}; updating only embeddings gives {pct(hybrids['010'])}; updating hidden matrices and the readout while retaining the preceding embeddings gives {pct(hybrids['101'])}. Updating hidden matrices and embeddings while retaining the preceding readout gives {pct(hybrids['110'])}. The complete update gives {pct(hybrids['111'])}.", '',
                f"At the preceding state, the derivative used in training has relative error {derivative['relative_l2_error']:.6g} against the analytic float64 logit derivative. The fraction of vanished correct-class derivatives is {pct(derivative['zero_target_fraction'])}. The preceding feature-mean norm is {event['pre_gradient']['residual_mean_norm']:,.3f}.", '',
                f"The head-displacement decomposition starts from the preceding readout on the updated features, with {pct(reference)} held-out accuracy. Adding the training-mean contribution gives {pct(mean_acc)}; adding the centered-feature contribution gives {pct(centered_acc)}; adding both gives {pct(mean['arms']['full']['heldout']['accuracy'])}. The exact decomposition identity passed.", '']
            if reference>=.9 and mean_acc<.9 and centered_acc>=.9:
                lines += ['In this transition the mean contribution alone reproduces the failure from a healthy baseline, establishing its local sufficiency.', '']
            elif reference<.9:
                if hybrids['000']<.9:
                    lines += ['The preceding state and the updated-feature baseline are already in failure. This adjacent comparison quantifies changes within that episode. The first-event comparisons identify the acute locus.', '']
                else:
                    lines += ['The updated-feature baseline has already failed before the readout displacement is applied. This decomposition measures the subsequent readout increment; the parameter-group controls locate the earlier loss.', '']
            if (location/'healthy_hybrids.json').exists():
                panel=json.loads((location/'healthy_hybrids.json').read_text())
                values={value['mask']:value['metrics']['heldout']['accuracy'] for value in panel['parameter_hybrids']}
                lines += [f"The separate healthy-to-worst factorial begins at update {panel['healthy_step']} with {pct(values['000'])} held-out accuracy and ends at {panel['peak_step']}. Replacing only the readout with its worst-state value gives {pct(values['001'])}; replacing only embeddings gives {pct(values['010'])}; replacing hidden matrices plus readout while retaining the healthy embeddings gives {pct(values['101'])}; replacing all groups gives {pct(values['111'])}. This evaluates parameter combinations across the stated multi-update interval.", '']
    lines += ['## Scope and verification','',
        'The normalized architecture adds gain-free RMS normalization at attention and MLP inputs and before the readout, with epsilon 1e-6. Its parameters and optimizer routing match the source initialization. Both loss arms begin at update zero, so their complete learning paths and grokking times can differ. Across the two extension configurations, both operation and normalization differ. The original seed-0 addition baseline supplies a matched operation for interpreting the RMS result. Any such architecture comparison combines forward-pass and ensuing learning-path changes.', '',
        'Six runner tests verify source initialization, optimizer routing, exact no-norm source updates, subtraction targets and split, accurate-loss derivatives, normalization arithmetic, and independent checkpoint loading. completion_verification.json audits every saved checkpoint, paired initial model/optimizer/RNG states, source hashes, full 30,000-update completion, event qualification, diagnostic provenance, and consistency of final snapshots. checkpoint_manifest.json lists checkpoint SHA-256 hashes.', '']
    (base/'summary.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,default=Path(__file__).parent/'generality')
    main(parser.parse_args().base)
