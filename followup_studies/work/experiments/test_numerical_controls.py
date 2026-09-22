"""Focused CPU checks. Run directly with the research Python interpreter."""
from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F

from numerical_controls import (
    PrecisionMuon,
    accurate_cross_entropy,
    zeroth_power_newton_schulz_precision,
)
from optimizers.muon import Muon, zeroth_power_newton_schulz


def value_gradient(function, values, targets):
    x = values.detach().clone().requires_grad_(True)
    loss = function(x, targets)
    gradient, = torch.autograd.grad(loss, x)
    return loss.detach(), gradient.detach()


def run_checks():
    torch.set_num_threads(1)
    torch.manual_seed(943)
    results = {}

    # Moderate logits must agree with the usual implementation in both
    # value and derivative, including rows on both sides of the branch.
    errors = {}
    for dtype, tolerance in [(torch.float32, 3e-6), (torch.float64, 2e-14)]:
        values = torch.randn(31, 11, dtype=dtype) * 2
        targets = torch.randint(0, 11, (31,))
        targets[:12] = values[:12].argmax(dim=1)
        for reduction in ['sum', 'mean']:
            ordinary, ordinary_grad = value_gradient(
                lambda z, y: F.cross_entropy(z, y, reduction=reduction), values, targets)
            stable, stable_grad = value_gradient(
                lambda z, y: accurate_cross_entropy(z, y, reduction=reduction), values, targets)
            torch.testing.assert_close(stable, ordinary, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(stable_grad, ordinary_grad, rtol=tolerance, atol=tolerance)
            errors[f'{dtype}_{reduction}'] = {
                'loss_abs_error': float((stable - ordinary).abs()),
                'gradient_max_abs_error': float((stable_grad - ordinary_grad).abs().max()),
            }
        torch.testing.assert_close(
            accurate_cross_entropy(values, targets, reduction='none'),
            F.cross_entropy(values, targets, reduction='none'),
            rtol=tolerance, atol=tolerance)
    results['moderate_logits_equivalence'] = errors

    # Independent analytic reference: sum the non-target probabilities
    # directly so the correct-class gradient never subtracts from 1.
    tiny_cases = []
    for margin in [20., 50., 80.]:
        values = torch.tensor([[margin, 0., -1.]], dtype=torch.float32)
        targets = torch.tensor([0])
        tail_terms = [math.exp(-margin), math.exp(-margin - 1)]
        tail = math.fsum(tail_terms)
        reference_loss = math.log1p(tail)
        reference_grad = torch.tensor(
            [[-tail / (1 + tail), tail_terms[0] / (1 + tail), tail_terms[1] / (1 + tail)]],
            dtype=torch.float64)
        stable, stable_grad = value_gradient(accurate_cross_entropy, values, targets)
        ordinary, ordinary_grad = value_gradient(F.cross_entropy, values, targets)
        stable64, stable_grad64 = value_gradient(accurate_cross_entropy, values.double(), targets)
        torch.testing.assert_close(stable.double(), torch.tensor(reference_loss, dtype=torch.float64),
                                   rtol=2e-6, atol=0)
        torch.testing.assert_close(stable_grad.double(), reference_grad, rtol=2e-6, atol=0)
        torch.testing.assert_close(stable_grad64, reference_grad, rtol=3e-14, atol=0)
        torch.testing.assert_close(stable64, torch.tensor(reference_loss, dtype=torch.float64),
                                   rtol=3e-14, atol=0)
        assert stable_grad[0, 0] != 0
        assert ordinary_grad[0, 0] == 0
        tiny_cases.append({
            'margin': margin,
            'stable_f32_loss': float(stable),
            'analytic_loss': reference_loss,
            'ordinary_f32_loss': float(ordinary),
            'stable_f32_correct_gradient': float(stable_grad[0, 0]),
            'ordinary_f32_correct_gradient': float(ordinary_grad[0, 0]),
            'analytic_correct_gradient': float(reference_grad[0, 0]),
        })
    results['tiny_loss_gradient_reference'] = tiny_cases

    # Incorrect rows must not overflow through the unused stable branch.
    extreme = torch.tensor([[0., 1000., -1000.], [80., 0., -1.], [0., 0., 0.]])
    targets = torch.tensor([0, 0, 0])
    value, grad = value_gradient(accurate_cross_entropy, extreme, targets)
    assert torch.isfinite(value) and torch.isfinite(grad).all()
    torch.testing.assert_close(grad.sum(dim=1), torch.zeros(3), atol=3e-8, rtol=0)
    single_value, single_grad = value_gradient(
        accurate_cross_entropy, torch.tensor([[100.], [-100.]]), torch.tensor([0, 0]))
    assert single_value == 0 and (single_grad == 0).all()
    assert torch.autograd.gradcheck(
        accurate_cross_entropy,
        (torch.tensor([[2., .5, -.3], [-1., 2., 0.]], dtype=torch.float64,
                      requires_grad=True), torch.tensor([0, 0])))
    assert torch.autograd.gradgradcheck(
        accurate_cross_entropy,
        (torch.tensor([[2., .5, -.3], [-1., 2., 0.]], dtype=torch.float64,
                      requires_grad=True), torch.tensor([0, 0])))
    promoted_value, promoted_grad = value_gradient(
        lambda z, y: accurate_cross_entropy(z, y, compute_dtype=torch.float64),
        torch.tensor([[50., 0., -1.]]), torch.tensor([0]))
    assert promoted_value.dtype == torch.float64 and promoted_grad.dtype == torch.float32
    results['ce_edge_cases_and_first_second_derivatives'] = 'passed'

    # Float32 NS is exactly source-compatible for all shapes/orientations.
    for shape in [(3, 7), (7, 3), (5, 5)]:
        matrix = torch.randn(shape)
        for scale in [1., 1e-8, 1e-20]:
            torch.testing.assert_close(
                zeroth_power_newton_schulz_precision(matrix * scale),
                zeroth_power_newton_schulz(matrix * scale), rtol=0, atol=0)
    for dtype in [torch.float32, torch.float64]:
        zero = torch.zeros(7, 3, dtype=dtype)
        out = zeroth_power_newton_schulz_precision(zero)
        assert out.dtype == dtype and (out == 0).all()
    microscopic = torch.eye(3, dtype=torch.float64) * 1e-50
    control = zeroth_power_newton_schulz_precision(microscopic)
    source = zeroth_power_newton_schulz(microscopic)
    assert (source == 0).all() and (control.diag() > 0).all()
    expected_small_amplitude = 1e-50 / 1e-7 * (3.4445 ** 5)
    torch.testing.assert_close(control.diag(), torch.full((3,), expected_small_amplitude,
                              dtype=torch.float64), rtol=5e-15, atol=0)
    base = torch.randn(4, 9, dtype=torch.float64)
    perturbed = base.clone()
    perturbed[0, 0] += 1e-10
    assert torch.equal(zeroth_power_newton_schulz(base), zeroth_power_newton_schulz(perturbed))
    precision_delta = float((zeroth_power_newton_schulz_precision(base)
                            - zeroth_power_newton_schulz_precision(perturbed)).abs().max())
    assert precision_delta > 1e-13
    results['ns_precision'] = {
        'float32_bitwise_source_equivalence': True,
        'float64_sub_float32_range_output': float(control[0, 0]),
        'float64_sensitivity_to_1e_10_input_change': precision_delta,
    }

    # Verify that the copied update body has precisely one functional edit.
    original_body = inspect.getsource(Muon.step).replace(
        'zeroth_power_newton_schulz(', 'zeroth_power_newton_schulz_precision(')
    control_body = inspect.getsource(PrecisionMuon.step)
    assert ast.dump(ast.parse(textwrap.dedent(original_body))) == ast.dump(
        ast.parse(textwrap.dedent(control_body)))

    # Source/control steps and momentum remain bitwise-identical in float32.
    p = torch.nn.Parameter(torch.randn(7, 3))
    q = torch.nn.Parameter(p.detach().clone())
    ordinary_opt = Muon([p], learning_rate=.01, weight_decay=.1)
    control_opt = PrecisionMuon([q], learning_rate=.01, weight_decay=.1)
    for _ in range(4):
        grad = torch.randn_like(p) * 1e-8
        p.grad = grad.clone()
        q.grad = grad.clone()
        ordinary_opt.step()
        control_opt.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(ordinary_opt.state[p]['momentum_buffer'],
                                   control_opt.state[q]['momentum_buffer'], rtol=0, atol=0)
    resumed = torch.nn.Parameter(p.detach().double())
    resumed_opt = PrecisionMuon([resumed])
    resumed_opt.load_state_dict(copy.deepcopy(ordinary_opt.state_dict()))
    assert resumed_opt.state[resumed]['momentum_buffer'].dtype == torch.float64
    assert resumed_opt.param_groups[0]['learning_rate'] == .01
    resumed.grad = torch.randn_like(resumed) * 1e-12
    resumed_opt.step()
    assert torch.isfinite(resumed).all()

    aux = torch.nn.Parameter(torch.randn(2, 3))
    aux_opt = torch.optim.AdamW([aux])
    aux.grad = torch.randn_like(aux)
    aux_opt.step()
    resumed_aux = torch.nn.Parameter(aux.detach().double())
    resumed_aux_opt = torch.optim.AdamW([resumed_aux])
    resumed_aux_opt.load_state_dict(copy.deepcopy(aux_opt.state_dict()))
    assert resumed_aux_opt.state[resumed_aux]['exp_avg'].dtype == torch.float64
    assert resumed_aux_opt.state[resumed_aux]['exp_avg_sq'].dtype == torch.float64
    results['optimizer_compatibility'] = {
        'step_ast_exact_except_ns_function': True,
        'float32_updates_and_momentum_bitwise_equal': True,
        'muon_restored_momentum_dtype': str(resumed_opt.state[resumed]['momentum_buffer'].dtype),
        'adamw_restored_exp_avg_dtype': str(resumed_aux_opt.state[resumed_aux]['exp_avg'].dtype),
        'adamw_restored_step_dtype': str(resumed_aux_opt.state[resumed_aux]['step'].dtype),
    }
    return {'status': 'passed', 'torch_version': torch.__version__, 'threads': 1,
            'checks': results}


if __name__ == '__main__':
    result = run_checks()
    output = Path(__file__).with_name('numerical_controls_test_results.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
