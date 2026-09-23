"""Replay a captured step and evaluate all eight three-group parameter hybrids.

The factorial intervention changes parameters only, followed by evaluation.
No optimizer step follows any hybrid. An optional separate control takes one
accurate-CE step from the same reconstructed pre-event model and optimizer state.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import sys

import torch
import torch.nn.functional as F

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = (SCRIPT_DIRECTORY if (SCRIPT_DIRECTORY / 'run_collapse.py').exists()
                        else SCRIPT_DIRECTORY.parent)
sys.path.insert(0, str(EXPERIMENT_DIRECTORY))
import run_collapse as original
from run_precision_branches import configure_runtime, file_sha256, make_arm

GROUPS = {'hidden': 'muon', 'embeddings': 'auxiliary_adamw', 'readout': 'unembedding_adamw'}


def assert_identical_tree(first, second, path='root'):
    if isinstance(first, torch.Tensor):
        if not (isinstance(second, torch.Tensor) and first.dtype == second.dtype
                and first.shape == second.shape and torch.equal(first, second)):
            raise AssertionError(f'Exact replay mismatch at {path}')
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError(f'Dictionary keys differ at {path}')
        for key in first:
            assert_identical_tree(first[key], second[key], f'{path}.{key}')
    elif isinstance(first, (list, tuple)):
        if type(first) != type(second) or len(first) != len(second):
            raise AssertionError(f'Sequence differs at {path}')
        for index, (a, b) in enumerate(zip(first, second)):
            assert_identical_tree(a, b, f'{path}[{index}]')
    elif first != second:
        raise AssertionError(f'Value differs at {path}')


@torch.no_grad()
def metrics(model, train_x, train_y, test_x, test_y):
    train = original.evaluate(model, train_x, train_y)
    test = original.evaluate(model, test_x, test_y)
    # Match the full-batch logits used for dense per-step monitoring as well.
    model.train()
    full_logits = model(train_x)
    return dict(
        train_loss=train['loss'], train_loss64=train['loss64'], train_accuracy=train['accuracy'],
        test_loss=test['loss'], test_loss64=test['loss64'], test_accuracy=test['accuracy'],
        native_fullbatch_train_loss=float(F.cross_entropy(full_logits, train_y)),
        native_fullbatch_train_accuracy=float((full_logits.argmax(-1) == train_y).double().mean()),
        maximum_absolute_logit=max(train['max_abs_logit'], test['max_abs_logit']),
    )


def relative(value, denominator):
    return value / denominator if denominator != 0 else None


def parameter_update_norms(pre, post, names):
    before = torch.cat([pre[name].double().reshape(-1) for name in names])
    after = torch.cat([post[name].double().reshape(-1) for name in names])
    delta = after - before
    before_norm = float(before.norm())
    delta_norm = float(delta.norm())
    return dict(
        parameter_names=names, parameter_count=before.numel(),
        pre_parameter_norm=before_norm, post_parameter_norm=float(after.norm()),
        update_norm=delta_norm, relative_update_norm=relative(delta_norm, before_norm),
        maximum_absolute_update=float(delta.abs().max()),
    )


def readout_centered_norms(pre, post, readout_name):
    before, after = pre[readout_name].double(), post[readout_name].double()
    delta = after - before
    before_centered = before - before.mean(dim=0, keepdim=True)
    after_centered = after - after.mean(dim=0, keepdim=True)
    common_delta = delta.mean(dim=0, keepdim=True).expand_as(delta)
    centered_delta = delta - common_delta
    delta_norm = float(delta.norm())
    centered_norm = float(centered_delta.norm())
    common_norm = float(common_delta.norm())
    return dict(
        definition='Class centering subtracts the mean across output-class rows, independently for each hidden coordinate.',
        pre_centered_parameter_norm=float(before_centered.norm()),
        post_centered_parameter_norm=float(after_centered.norm()),
        class_centered_update_norm=centered_norm,
        class_centered_relative_update_norm=relative(centered_norm, float(before_centered.norm())),
        common_class_update_norm=common_norm,
        common_class_fraction_of_update_norm=relative(common_norm, delta_norm),
        centered_fraction_of_update_norm=relative(centered_norm, delta_norm),
        pythagorean_relative_residual=relative(abs(delta_norm ** 2 - centered_norm ** 2 - common_norm ** 2),
                                              delta_norm ** 2),
    )


def run(checkpoint_path, collapse_path, out, threads=1, stable_ce_control=False):
    configure_runtime(threads)
    checkpoint_path, collapse_path, out = [Path(path).resolve() for path in (checkpoint_path, collapse_path, out)]
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    captured = torch.load(collapse_path, map_location='cpu', weights_only=False)
    event_step = int(captured['step'])
    pre_step = event_step - 1
    if int(source['step']) > pre_step:
        raise ValueError('Source checkpoint must precede the collapse step')
    if source['seed'] != captured['seed']:
        raise ValueError('Source and collapse seeds differ')
    model, optimizers, loss_fn = make_arm(source, 'original32')
    seed = int(source['seed'])
    data = original.generate_modular_addition_data(seed=seed)
    train_x, train_y, test_x, test_y = data
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    memberships = {label: [names_by_id[id(parameter)]
                           for group in optimizers[key].param_groups for parameter in group['params']]
                   for label, key in GROUPS.items()}
    all_names = [name for names in memberships.values() for name in names]
    if len(set(all_names)) != len(all_names) or set(all_names) != set(dict(model.named_parameters())):
        raise ValueError('Optimizer groups must partition all model parameters')
    if len(memberships['readout']) != 1:
        raise ValueError('Expected exactly one readout matrix')
    readout_name = memberships['readout'][0]
    replay_losses = []
    for current_step in range(int(source['step']), pre_step):
        value = original.train_step(model, optimizers, train_x, train_y, loss_fn=loss_fn)
        replay_losses.append(dict(pre_update_step=current_step, post_update_step=current_step + 1,
                                  native_train_loss=value))
    pre_state = original.snapshot(model, optimizers, pre_step, seed)
    pre_metrics = metrics(model, *data)
    final_loss = original.train_step(model, optimizers, train_x, train_y, loss_fn=loss_fn)
    replay_losses.append(dict(pre_update_step=pre_step, post_update_step=event_step, native_train_loss=final_loss))
    post_state = original.snapshot(model, optimizers, event_step, seed)
    for key in ['model_state_dict', 'optimizer_state_dicts', 'torch_rng_state']:
        assert_identical_tree(post_state[key], captured[key], key)
    post_metrics = metrics(model, *data)
    pre_parameters, post_parameters = pre_state['model_state_dict'], post_state['model_state_dict']
    group_norms = {label: parameter_update_norms(pre_parameters, post_parameters, names)
                   for label, names in memberships.items()}
    group_norms['readout'].update(readout_centered_norms(pre_parameters, post_parameters, readout_name))
    per_parameter_norms = {name: parameter_update_norms(pre_parameters, post_parameters, [name])
                           for name in all_names}

    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'run_one_step_attribution.py')
    provenance = dict(
        source_checkpoint=str(checkpoint_path), source_checkpoint_sha256=file_sha256(checkpoint_path),
        collapse_checkpoint=str(collapse_path), collapse_checkpoint_sha256=file_sha256(collapse_path),
        source_step=int(source['step']), pre_step=pre_step, post_step=event_step,
        seed=seed, source_repository_commit=source['repository_commit'],
        source_config=source['model_config'],
        threads=torch.get_num_threads(), torch_version=str(torch.__version__),
        created_utc=datetime.now(timezone.utc).isoformat(),
        exact_replay=dict(model_state=True, all_optimizer_states=True, rng_state=True),
        parameter_groups=memberships,
        selection='Post hoc attribution of one exploratory-captured joint failure in seed 4.',
        intervention='For every subset of the three optimizer parameter groups, replace its pre-step parameters with the exact post-step parameters and evaluate without further optimization.',
        relative_update_norm_definition='Euclidean/Frobenius norm of post minus pre, divided by pre norm; calculated in float64 from the stored float32 parameters.',
        script_sha256=file_sha256(__file__),
    )
    for state, phase in [(pre_state, 'pre_step'), (post_state, 'post_step')]:
        state['attribution_provenance'] = copy.deepcopy(provenance)
        state['attribution_phase'] = phase
        torch.save(state, out / f'{phase}.pt')
    with (out / 'replay_losses.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(replay_losses[0]))
        writer.writeheader()
        writer.writerows(replay_losses)

    factorial = []
    for hidden_post, embeddings_post, readout_post in itertools.product([False, True], repeat=3):
        selected = dict(hidden=hidden_post, embeddings=embeddings_post, readout=readout_post)
        hybrid_parameters = copy.deepcopy(pre_parameters)
        for label, use_post in selected.items():
            if use_post:
                for name in memberships[label]:
                    hybrid_parameters[name] = post_parameters[name].clone()
        model.load_state_dict(hybrid_parameters)
        hybrid_metrics = metrics(model, *data)
        row = dict(
            mask=''.join('1' if selected[label] else '0' for label in GROUPS),
            hidden_post=int(hidden_post), embeddings_post=int(embeddings_post), readout_post=int(readout_post),
            **hybrid_metrics,
        )
        if row['mask'] == '000':
            assert hybrid_metrics == pre_metrics
        if row['mask'] == '111':
            assert hybrid_metrics == post_metrics
        factorial.append(row)
        print(json.dumps(row), flush=True)
    with (out / 'factorial_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(factorial[0]))
        writer.writeheader()
        writer.writerows(factorial)

    results = dict(
        status='completed', provenance=provenance,
        pre_step_metrics=pre_metrics, post_step_metrics=post_metrics,
        group_update_norms=group_norms, per_parameter_update_norms=per_parameter_norms,
        factorial_results=factorial,
        mask_order=['hidden', 'embeddings', 'readout'],
        optional_stable_ce_control=None,
    )
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    if stable_ce_control:
        # Run only after finishing the requested eight parameter hybrids.
        control_model, control_optimizers, control_loss = make_arm(pre_state, 'stable32')
        control_training_loss = original.train_step(control_model, control_optimizers,
                                                     train_x, train_y, loss_fn=control_loss)
        control_state = original.snapshot(control_model, control_optimizers, event_step, seed)
        control_metrics = metrics(control_model, *data)
        control_norms = {label: parameter_update_norms(pre_parameters,
                         control_state['model_state_dict'], names) for label, names in memberships.items()}
        control_norms['readout'].update(readout_centered_norms(
            pre_parameters, control_state['model_state_dict'], readout_name))
        control_state['attribution_provenance'] = copy.deepcopy(provenance)
        control_state['attribution_phase'] = 'stable_ce_one_step'
        torch.save(control_state, out / 'stable_ce_one_step.pt')
        results['optional_stable_ce_control'] = dict(
            description='One accurate-cross-entropy float32 step from the identical step-16055 model and all three optimizer states; existing optimizer history is retained.',
            training_loss=control_training_loss, metrics=control_metrics, group_update_norms=control_norms,
        )
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print('STABLE_CE ' + json.dumps(results['optional_stable_ce_control']), flush=True)
    print('REPLAY_EXACT ' + json.dumps(provenance['exact_replay']), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--collapse', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--stable-ce-control', action='store_true')
    args = parser.parse_args()
    run(args.checkpoint, args.collapse, args.out, args.threads, args.stable_ce_control)


if __name__ == '__main__':
    main()
