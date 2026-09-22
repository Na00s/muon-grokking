"""Fixed-logit numerical diagnostics and exact next-update causal interventions."""
from __future__ import annotations
import argparse
import copy
import itertools
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'experiments'))
from run_precision_branches import configure_runtime, make_arm, file_sha256
from numerical_controls import accurate_cross_entropy
from run_one_step_attribution import assert_identical_tree, metrics
import run_collapse as original

GROUPS = {'hidden': 'muon', 'embeddings': 'auxiliary_adamw', 'readout': 'unembedding_adamw'}
HEAD = 'unembedding.weight'


def accurate_logit_gradient(logits, targets):
    """Analytic mean-CE derivative in double, avoiding p_y minus one cancellation."""
    values = logits.detach().double()
    shifted = values - values.max(1, keepdim=True).values
    exp_values = shifted.exp()
    normalizer = exp_values.sum(1, keepdim=True)
    gradient = exp_values / normalizer
    wrong_mass = exp_values.scatter(1, targets[:, None], 0).sum(1, keepdim=True)
    gradient.scatter_(1, targets[:, None], -wrong_mass / normalizer)
    return gradient / len(targets)


def compare(value, reference):
    a, b = value.double().reshape(-1), reference.double().reshape(-1)
    denominator = float(b.norm())
    return dict(norm=float(a.norm()), reference_norm=denominator,
                relative_l2_error=float((a-b).norm()) / denominator if denominator else None,
                cosine=float(F.cosine_similarity(a[None], b[None], eps=1e-300))
                if float(a.norm()) and denominator else None)


def memberships(model, optimizers):
    names = {id(p): n for n,p in model.named_parameters()}
    groups = {label: [names[id(p)] for group in optimizers[key].param_groups for p in group['params']]
              for label,key in GROUPS.items()}
    flat = sum(groups.values(), [])
    assert len(flat) == len(set(flat)) and set(flat) == set(dict(model.named_parameters()))
    return groups


def gradient_bundle(state, x, y):
    model, optimizers, _ = make_arm(state, 'original32')
    model.train()
    logits = model(x)
    names, parameters = zip(*model.named_parameters())
    groups = memberships(model, optimizers)
    gradients, logit_gradients, losses = {}, {}, {}
    for label, loss_fn in [('stock32', F.cross_entropy), ('accurate32', accurate_cross_entropy)]:
        loss = loss_fn(logits, y)
        dlogit, = torch.autograd.grad(loss, logits, retain_graph=True)
        upstream = torch.autograd.grad(logits, parameters, grad_outputs=dlogit, retain_graph=True)
        gradients[label] = {name: value.detach().clone() for name,value in zip(names,upstream)}
        logit_gradients[label] = dlogit.detach()
        losses[label] = float(loss.detach())
    analytic = accurate_logit_gradient(logits, y)
    upstream = torch.autograd.grad(logits, parameters, grad_outputs=analytic.to(logits.dtype))
    gradients['reference64_cast32'] = {name: value.detach().clone() for name,value in zip(names,upstream)}
    logit_gradients['reference64'] = analytic
    losses['reference64'] = float(accurate_cross_entropy(logits.detach().double(),y))
    report = dict(step=int(state['step']), seed=int(state['seed']), training_accuracy=float((logits.argmax(1)==y).double().mean()),
                  maximum_absolute_logit=float(logits.detach().abs().max()), losses=losses, logit_gradients={}, parameter_gradients={})
    for label, gradient in logit_gradients.items():
        target = gradient.gather(1,y[:,None])[:,0]
        report['logit_gradients'][label] = dict(
            **compare(gradient, analytic), zero_target_fraction=float((target==0).double().mean()),
            zero_target_with_nonzero_wrong_fraction=float(((target==0)&(gradient.abs().sum(1)>0)).double().mean()),
            row_sum_l2=float(gradient.double().sum(1).norm()),
            row_sum_relative_l2=float(gradient.double().sum(1).norm()/analytic.norm()),
            target_gradient_relative_l2=compare(target,analytic.gather(1,y[:,None])[:,0])['relative_l2_error'])
    for label, gradients_by_name in gradients.items():
        report['parameter_gradients'][label] = {}
        for group, group_names in groups.items():
            flat = torch.cat([gradients_by_name[n].reshape(-1) for n in group_names])
            reference = torch.cat([gradients['reference64_cast32'][n].reshape(-1) for n in group_names])
            report['parameter_gradients'][label][group] = compare(flat,reference)
        head = gradients_by_name[HEAD].double()
        common = head.mean(0,keepdim=True).expand_as(head)
        report['parameter_gradients'][label]['readout_class_common_fraction'] = float(common.norm()/head.norm())
    report['head_adam_decomposition'] = adam_decomposition(model.unembedding.weight, optimizers['unembedding_adamw'], gradients['stock32'][HEAD])[0]
    return report, gradients, groups


def adam_decomposition(parameter, optimizer, gradient):
    """Float64 formula using original current-gradient denominator for both parts."""
    state = optimizer.state[parameter]
    group = optimizer.param_groups[0]
    beta1,beta2 = group['betas']
    step = int(state.get('step', torch.tensor(0)).item()) + 1
    old_m = state.get('exp_avg', torch.zeros_like(parameter)).double()
    old_v = state.get('exp_avg_sq', torch.zeros_like(parameter)).double()
    g = gradient.double()
    variance = beta2 * old_v + (1-beta2) * g.square()
    denominator = variance.sqrt() / (1-beta2**step)**.5 + group['eps']
    factor = -group['lr'] / (1-beta1**step)
    historical = factor * beta1 * old_m / denominator
    current = factor * (1-beta1) * g / denominator
    decay = -group['lr'] * group['weight_decay'] * parameter.detach().double()
    update = historical + current + decay
    total = float(update.norm())
    parts = dict(historical=historical, current=current, decay=decay, total=update)
    report = dict(step=step, old_m_norm=float(old_m.norm()), current_g_norm=float(g.norm()),
                  norms={key: float(value.norm()) for key,value in parts.items()},
                  norm_ratios={key: float(value.norm())/total if total else None for key,value in parts.items()},
                  historical_current_cosine=compare(historical,current)['cosine'],
                  note='Component norms are not additive. Contributions use the original update denominator, including current-gradient squared.')
    return report, parts


def apply_update(state, gradients, groups, stable_groups=(), action=None):
    model, optimizers, _ = make_arm(state,'original32')
    names = dict(model.named_parameters())
    for group, group_names in groups.items():
        source = 'accurate32' if group in stable_groups else 'stock32'
        if action == 'reference': source = 'reference64_cast32'
        for name in group_names: names[name].grad = gradients[source][name].clone()
    head = model.unembedding.weight
    optimizer = optimizers['unembedding_adamw']
    _, parts = adam_decomposition(head, optimizer, head.grad)
    if action == 'zero_head_m': optimizer.state[head]['exp_avg'].zero_()
    if action == 'zero_head_g': head.grad.zero_()
    if action == 'reset_head_state': optimizer.state[head].clear()
    if action == 'center_head_g': head.grad.sub_(head.grad.mean(0,keepdim=True))
    skip = action in {'freeze_head','head_decay_only','head_historical_fixed_denominator','head_current_fixed_denominator'}
    for key,opt in optimizers.items():
        if skip and key == 'unembedding_adamw': continue
        opt.step()
    with torch.no_grad():
        if action == 'head_decay_only': head.mul_(1-optimizer.param_groups[0]['lr']*optimizer.param_groups[0]['weight_decay'])
        if action == 'head_historical_fixed_denominator': head.copy_((head.double()+parts['historical']+parts['decay']).float())
        if action == 'head_current_fixed_denominator': head.copy_((head.double()+parts['current']+parts['decay']).float())
    return model,optimizers


def event_paths(seed):
    if seed == 0:
        return WORK/'basis_study/seed0_preceding_step_replay/final.pt', WORK/'experiments/seed0_dense_capture/collapse.pt'
    if seed == 4:
        return WORK/'experiments/seed4_one_step_attribution/pre_step.pt', WORK/'experiments/seed4_dense_capture/collapse.pt'
    return WORK/f'basis_study/seed{seed}_baseline/previous.pt', WORK/f'basis_study/seed{seed}_baseline/collapse.pt'


def diagnose_event(seed, out):
    pre_path, post_path = event_paths(seed)
    pre,post = [torch.load(path, map_location='cpu', weights_only=False) for path in (pre_path,post_path)]
    assert pre['step']+1 == post['step']
    data = original.generate_modular_addition_data(seed=seed)
    report, gradients, groups = gradient_bundle(pre,*data[:2])
    stock_model,stock_opts = apply_update(pre,gradients,groups)
    replay = original.snapshot(stock_model,stock_opts,int(post['step']),seed)
    for key in ('model_state_dict','optimizer_state_dicts','torch_rng_state'):
        assert_identical_tree(replay[key],post[key],key)
    decomposition, parts = adam_decomposition(*_head_and_optimizer(pre),gradients['stock32'][HEAD])
    exact_delta = post['model_state_dict'][HEAD].double()-pre['model_state_dict'][HEAD].double()
    decomposition['formula_vs_actual_float32_update'] = compare(parts['total'],exact_delta)
    report.update(pre_checkpoint=str(pre_path), pre_checkpoint_sha256=file_sha256(pre_path),
                  post_checkpoint=str(post_path), post_checkpoint_sha256=file_sha256(post_path),
                  exact_replay=True, head_adam_decomposition=decomposition, gradient_swap_arms=[], state_arms=[], parameter_hybrids=[])
    pre_model,_,_ = make_arm(pre,'original32')
    report['pre_metrics'] = metrics(pre_model,*data)
    for bits in itertools.product([False,True],repeat=3):
        selected = tuple(label for label,bit in zip(groups,bits) if bit)
        model,_ = apply_update(pre,gradients,groups,selected)
        report['gradient_swap_arms'].append(dict(accurate_groups=list(selected), mask=''.join(str(int(bit)) for bit in bits),metrics=metrics(model,*data)))
    for action in ['reference','zero_head_m','zero_head_g','reset_head_state','center_head_g','freeze_head','head_decay_only','head_historical_fixed_denominator','head_current_fixed_denominator']:
        model,_ = apply_update(pre,gradients,groups,action=action)
        report['state_arms'].append(dict(action=action,metrics=metrics(model,*data)))
    # Accuracy depends only on class-centered head updates in exact arithmetic.
    for action, delta in [('head_common_only',exact_delta.mean(0,keepdim=True).expand_as(exact_delta)),
                          ('head_centered_only',exact_delta-exact_delta.mean(0,keepdim=True))]:
        model,_,_ = make_arm(post,'original32')
        with torch.no_grad(): model.unembedding.weight.copy_((pre['model_state_dict'][HEAD].double()+delta).float())
        report['state_arms'].append(dict(action=action,metrics=metrics(model,*data)))
    for bits in itertools.product([False,True],repeat=3):
        model,_,_ = make_arm(pre,'original32')
        new_state = copy.deepcopy(pre['model_state_dict'])
        for label,bit in zip(groups,bits):
            if bit:
                for name in groups[label]: new_state[name] = post['model_state_dict'][name].clone()
        model.load_state_dict(new_state)
        report['parameter_hybrids'].append(dict(mask=''.join(str(int(bit)) for bit in bits),metrics=metrics(model,*data)))
    assert_identical_tree(pre,torch.load(pre_path,map_location='cpu',weights_only=False),'unmodified_source')
    report['in_memory_source_state_unchanged'] = True
    out.write_text(json.dumps(report,indent=2)+'\n')
    return report


def _head_and_optimizer(state):
    model,optimizers,_ = make_arm(state,'original32')
    return model.unembedding.weight, optimizers['unembedding_adamw']


def timecourse_paths(seed):
    base = WORK/(f'experiments/seed{seed}_original' if seed in (0,4) else f'basis_study/seed{seed}_baseline')
    events = WORK/(f'experiments/seed{seed}_dense_capture' if seed in (0,4) else f'basis_study/seed{seed}_baseline')
    paths = [base/f'step_{step:06d}.pt' for step in (6000,10000,14000,15000)]
    paths += sorted((events/'event_window').glob('step_*.pt'))
    paths += list(event_paths(seed))
    unique = {}
    for path in paths:
        state = torch.load(path,map_location='cpu',weights_only=False)
        unique[int(state['step'])] = path
    return [unique[step] for step in sorted(unique)]


def run(seed,out):
    out.mkdir(parents=True,exist_ok=False)
    x,y,_,_ = original.generate_modular_addition_data(seed=seed)
    records=[]
    for path in timecourse_paths(seed):
        state=torch.load(path,map_location='cpu',weights_only=False)
        report,_,_=gradient_bundle(state,x,y)
        report['checkpoint']=str(path)
        records.append(report)
        print(json.dumps(dict(seed=seed,step=state['step'],stock_logit_relative_error=report['logit_gradients']['stock32']['relative_l2_error'])),flush=True)
    (out/'timecourse.json').write_text(json.dumps(records,indent=2)+'\n')
    event = diagnose_event(seed,out/'event.json')
    (out/'completion.json').write_text(json.dumps(dict(seed=seed,timecourse_checkpoints=len(records),exact_next_step_replay=event['exact_replay'],gradient_swap_arms=8,state_arms=len(event['state_arms']),parameter_hybrids=8),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();configure_runtime(1);run(args.seed,args.out)
