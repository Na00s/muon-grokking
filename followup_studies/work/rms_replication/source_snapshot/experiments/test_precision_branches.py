"""Verify two-step original32 resume and inspect all four arm restorations."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import run_collapse as original
from numerical_controls import PrecisionMuon, accurate_cross_entropy
from run_precision_branches import (
    ARMS, configure_runtime, evaluate_branch, make_arm,
    optimizer_hyperparameters, run_branch,
)


def assert_tree_equal(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_tree_equal(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            assert_tree_equal(x, y, f'{path}[{index}]')
    else:
        assert a == b, path


def verify(checkpoint_path, out):
    configure_runtime(1)
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    seed = int(source['seed'])
    train_x, train_y, test_x, test_y = original.generate_modular_addition_data(seed=seed)
    baseline_model, baseline_opts = original.make_model_optimizers(seed, checkpoint=copy.deepcopy(source))
    torch.set_rng_state(source['torch_rng_state'].cpu())
    expected_initial_train = original.evaluate(baseline_model, train_x, train_y)
    expected_initial_test = original.evaluate(baseline_model, test_x, test_y)
    baseline_losses = []
    intermediate_states = []
    for _ in range(2):
        baseline_losses.append(original.train_step(baseline_model, baseline_opts, train_x, train_y))
        intermediate_states.append(original.snapshot(baseline_model, baseline_opts,
                                                     int(source['step']) + len(baseline_losses), seed))
    expected_final_train = original.evaluate(baseline_model, train_x, train_y)
    expected_final_test = original.evaluate(baseline_model, test_x, test_y)
    baseline_final = original.snapshot(baseline_model, baseline_opts, int(source['step']) + 2, seed)

    summary = run_branch(checkpoint_path, out, 'original32', steps=2, threads=1, eval_every=1, save_every=1)
    actual_initial = torch.load(out / 'start.pt', map_location='cpu', weights_only=False)
    actual_final = torch.load(out / 'final.pt', map_location='cpu', weights_only=False)
    for key in ['model_state_dict', 'optimizer_state_dicts', 'torch_rng_state']:
        assert_tree_equal(source[key], actual_initial[key], f'initial.{key}')
        assert_tree_equal(baseline_final[key], actual_final[key], f'final.{key}')
    for state in intermediate_states:
        saved = torch.load(out / f'step_{state["step"]:06d}.pt', map_location='cpu', weights_only=False)
        for key in ['model_state_dict', 'optimizer_state_dicts', 'torch_rng_state']:
            assert_tree_equal(state[key], saved[key], f'periodic.{key}')
    with (out / 'trajectory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['step']) for row in rows] == [int(source['step']) + i for i in range(3)]
    assert [float(row['last_training_loss']) for row in rows[1:]] == baseline_losses
    for row, tr, te in [(rows[0], expected_initial_train, expected_initial_test),
                        (rows[-1], expected_final_train, expected_final_test)]:
        for split, expected in [('train', tr), ('test', te)]:
            for field in ['loss', 'loss64', 'accuracy']:
                assert float(row[f'{split}_{field}']) == expected[field]
        assert float(row['max_abs_logit']) == max(tr['max_abs_logit'], te['max_abs_logit'])
    assert actual_initial['arm'] == actual_final['arm'] == 'original32'
    assert actual_initial['source_checkpoint_metadata']['step'] == source['step']
    assert actual_final['checkpoint_phase'] == 'final'
    assert actual_final['local_step'] == 2

    # Loading into every arm must preserve group membership and hyperparameters.
    # A small 16-example single step also catches dtype/backend integration errors.
    source_hyperparameters = optimizer_hyperparameters(baseline_opts)
    restorations = {}
    for arm, settings in ARMS.items():
        model, optimizers, loss_fn = make_arm(copy.deepcopy(source), arm)
        assert optimizer_hyperparameters(optimizers) == source_hyperparameters
        assert all(parameter.dtype == settings['dtype'] for parameter in model.parameters())
        assert isinstance(optimizers['muon'], PrecisionMuon) == settings['precision_muon']
        assert (loss_fn is accurate_cross_entropy) == settings['accurate_ce']
        assert (loss_fn is F.cross_entropy) != settings['accurate_ce']
        names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
        memberships = {key: [[names_by_id[id(parameter)] for parameter in group['params']]
                             for group in optimizer.param_groups]
                       for key, optimizer in optimizers.items()}
        state_dtypes = {}
        for name, optimizer in optimizers.items():
            for parameter, state in optimizer.state.items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and key != 'step':
                        assert value.dtype == settings['dtype'], (arm, name, key)
            state_dtypes[name] = sorted({str(value.dtype) for state in optimizer.state.values()
                                        for key, value in state.items()
                                        if isinstance(value, torch.Tensor) and key != 'step'})
        loss = original.train_step(model, optimizers, train_x[:16], train_y[:16], loss_fn=loss_fn)
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        restorations[arm] = dict(parameter_dtype=str(settings['dtype']),
                                optimizer_state_dtypes=state_dtypes,
                                optimizer_memberships=memberships,
                                finite_16_example_smoke_step_loss=loss)
    result = dict(status='passed', source_checkpoint=str(checkpoint_path.resolve()),
                  replay_steps=2, replay_start_step=int(source['step']),
                  original32_model_states_bitwise_identical=True,
                  original32_all_three_optimizer_states_bitwise_identical=True,
                  original32_rng_states_bitwise_identical=True,
                  original32_training_losses_and_initial_final_metrics_exact=True,
                  periodic_saved_states_exact=True,
                  source_hyperparameters_preserved_in_all_arms=True,
                  arm_restorations=restorations,
                  branch_summary_path=str(out / 'summary.json'))
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    verify(args.checkpoint, args.out)
