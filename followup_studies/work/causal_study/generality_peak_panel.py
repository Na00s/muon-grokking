"""Factorial parameter comparison from the last sampled healthy state to a peak."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import generality_runner as g
import generality_analyze as a
from diagnostics import memberships


def run(path):
    before=a.load(path/'healthy.pt');after=a.load(path/'collapse.pt')
    data=g.generate_modular_addition_data(seed=before['seed'],operation=before['operation'])
    model,opts=a.build(before);groups=memberships(model,opts)
    result=dict(healthy_step=before['step'],peak_step=after['step'],
        healthy_checkpoint_sha256=a.sha256(path/'healthy.pt'),peak_checkpoint_sha256=a.sha256(path/'collapse.pt'),
        note='Evaluation-only parameter factorial across the stated multi-update interval; preceding-step interventions are reported separately.',
        parameter_hybrids=[])
    for bits in itertools.product([False,True],repeat=3):
        model,opts=a.build(before);state=copy.deepcopy(before['model_state_dict'])
        for group,flag in zip(groups,bits):
            if flag:
                for name in groups[group]:state[name]=after['model_state_dict'][name].clone()
        model.load_state_dict(state)
        result['parameter_hybrids'].append(dict(mask=''.join(str(int(flag)) for flag in bits),
            updated_groups=[group for group,flag in zip(groups,bits) if flag],metrics=a.model_metrics(model,data)))
    (path/'healthy_hybrids.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(healthy_step=before['step'],peak_step=after['step'],
        heldout_accuracy={row['mask']:row['metrics']['heldout']['accuracy'] for row in result['parameter_hybrids']}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--path',type=Path,required=True)
    args=parser.parse_args();g.configure_runtime();run(args.path)
