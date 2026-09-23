"""Two-step exact replay check using a healthy saved original checkpoint."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

import run_collapse as original
from run_dense_capture import run_capture
from run_precision_branches import configure_runtime
from test_precision_branches import assert_tree_equal


def verify(checkpoint_path, out):
    configure_runtime(1)
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    seed = int(source['seed'])
    train_x, train_y, _, _ = original.generate_modular_addition_data(seed=seed)
    model, optimizers = original.make_model_optimizers(seed, checkpoint=copy.deepcopy(source))
    torch.set_rng_state(source['torch_rng_state'].cpu())
    losses = [original.train_step(model, optimizers, train_x, train_y) for _ in range(2)]
    expected = original.snapshot(model, optimizers, int(source['step']) + 2, seed)
    summary = run_capture(checkpoint_path, out, steps=2, threads=1)
    actual_start = torch.load(out / 'start.pt', map_location='cpu', weights_only=False)
    actual_final = torch.load(out / 'final.pt', map_location='cpu', weights_only=False)
    for key in ['model_state_dict', 'optimizer_state_dicts', 'torch_rng_state']:
        assert_tree_equal(source[key], actual_start[key], f'initial.{key}')
        assert_tree_equal(expected[key], actual_final[key], f'final.{key}')
    assert summary['updates_completed'] == 2
    assert summary['train_accuracy_inspections'] == 3
    assert summary['final_step'] == source['step'] + 2
    assert summary['event_step'] is None
    assert summary['heldout_evaluation_count'] == 1
    assert summary['final_metrics']['heldout_measured'] is False
    assert actual_final['branch_metadata']['selection_status'] == 'exploratory_followup'
    result = dict(
        status='passed', checkpoint=str(checkpoint_path.resolve()), replay_steps=2,
        original_model_state_exact=True, all_three_optimizer_states_exact=True,
        rng_state_exact=True, initial_state_exact=True,
        heldout_evaluation_cadence_correct=True,
        explicit_unknown_final_test_metric=True,
        original_step_losses=losses,
        output_directory=str(out.resolve()),
    )
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    verify(args.checkpoint, args.out)
