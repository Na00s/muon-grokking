"""Extract declared fresh-seed comparisons, retaining original training splits."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / 'experiments'


def pair(seed, suffix, first, second, role):
    out = HERE / f'seed{seed}_{suffix}'
    if not (out / 'completion.json').exists():
        subprocess.run([sys.executable, str(EXP / 'run_alignment.py'),
                        '--healthy', str(first), '--collapsed', str(second),
                        '--out', str(out)], check=True)
        metadata = json.loads((out / 'metadata.json').read_text())
        metadata.update(reference_role=role,
                        step_interval=metadata['collapsed_step'] - metadata['healthy_step'],
                        selection_note='Fresh seeds 1, 2, 3 declared together before training. Preceding-step references may already have degraded generalization.')
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=[1, 2, 3], required=True)
    p.add_argument('--phase', choices=['event', 'peak', 'control'], required=True)
    args = p.parse_args()
    seed = args.seed
    baseline = HERE / f'seed{seed}_baseline'
    event = json.loads((baseline / 'event.json').read_text())
    outputs = []
    if args.phase == 'event':
        outputs.append(pair(seed, 'first_event', baseline / 'healthy.pt',
                            baseline / 'collapse.pt', 'Latest measured jointly >=99% checkpoint'))
        outputs.append(pair(seed, 'adjacent_event', baseline / 'previous.pt',
                            baseline / 'collapse.pt', 'Immediately preceding update, irrespective of accuracy'))
    elif args.phase == 'peak':
        if not (baseline / 'completion.json').exists():
            raise RuntimeError('Peak selection requires completed 200-update follow-up')
        peak = json.loads((baseline / 'peak.json').read_text())
        if peak['step'] == event['step']:
            out = HERE / f'seed{seed}_first_event'
            (baseline / 'peak_pair_alias.json').write_text(json.dumps(dict(
                feature_pair=str(out.resolve()), reason='First event is also the minimum measured accuracy during follow-up; analyze this pair once.'), indent=2))
            outputs.append(out)
        else:
            outputs.append(pair(seed, 'peak_event', baseline / 'healthy.pt',
                                baseline / 'peak.pt', 'Latest measured jointly >=99% checkpoint before first joint collapse'))
    else:
        # Adaptive matched-duration control, fixed selection rule before extraction.
        # Match the first-event interval and choose the latest saved 1000-grid end
        # at least 1000 updates before its healthy reference with >=99% accuracy.
        lag = int(event['step']) - int(event['healthy_step'])
        rows = list(csv.DictReader((baseline / 'trajectory.csv').open()))
        candidates = [int(r['step']) for r in rows if int(r['step']) % 1000 == 0
                      and int(r['step']) <= event['healthy_step'] - 1000
                      and r['test_accuracy'] and float(r['test_accuracy']) >= .99
                      and float(r['train_accuracy']) >= .99
                      and int(r['step']) - lag >= 1000]
        if not candidates:
            raise RuntimeError('No matched healthy control end satisfies the declared rule')
        end = max(candidates)
        start = end - lag
        floor = start // 1000 * 1000
        start_path = baseline / f'step_{floor:06d}.pt'
        if start != floor:
            replay = HERE / f'seed{seed}_matched_control_replay'
            if not (replay / 'summary.json').exists():
                subprocess.run([sys.executable, str(EXP / 'run_precision_branches.py'),
                                '--checkpoint', str(start_path), '--out', str(replay),
                                '--arm', 'original32', '--steps', str(start-floor),
                                '--eval-every', '10', '--save-every', '1000'], check=True)
            start_path = replay / 'final.pt'
        out = pair(seed, 'matched_healthy', start_path, baseline / f'step_{end:06d}.pt',
                   'Adaptive matched-duration healthy control; latest eligible 1000-grid end at least 1000 updates before event reference')
        md = json.loads((out / 'metadata.json').read_text())
        md['control_selection'] = dict(interval=lag, start=start, end=end,
                                      eligible_end_steps=candidates,
                                      passes_endpoint_99_percent=min(md['native_healthy']['heldout']['accuracy'], md['native_collapsed']['heldout']['accuracy']) >= .99)
        (out / 'metadata.json').write_text(json.dumps(md, indent=2))
        outputs.append(out)
    print(json.dumps({'seed': seed, 'phase': args.phase, 'pairs': [str(o) for o in outputs]}), flush=True)


if __name__ == '__main__':
    main()
