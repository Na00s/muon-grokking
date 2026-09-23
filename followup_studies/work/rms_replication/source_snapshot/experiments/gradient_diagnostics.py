"""Measure loss-gradient roundoff on frozen logits and its upstream effect."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from run_collapse import make_model_optimizers, generate_modular_addition_data
from numerical_controls import accurate_cross_entropy


def comparison(value, reference):
    value=value.double().flatten();reference=reference.double().flatten()
    norm=torch.linalg.vector_norm(reference)
    return dict(norm=float(torch.linalg.vector_norm(value)),reference_norm=float(norm),
        relative_l2_error=float(torch.linalg.vector_norm(value-reference)/norm) if norm>0 else None,
        cosine=float(F.cosine_similarity(value[None],reference[None],eps=1e-300)) if norm>0 and value.norm()>0 else None)


def diagnose(checkpoint):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model,_=make_model_optimizers(state['seed'],checkpoint=state)
    x,y,_,_=generate_modular_addition_data(seed=state['seed'])
    model.eval()
    with torch.no_grad(): frozen=model(x)
    results={};gradients={}
    for name,dtype,loss_fn in [('stock32',torch.float32,F.cross_entropy),('stable32',torch.float32,accurate_cross_entropy),
                               ('stock64',torch.float64,F.cross_entropy),('stable64',torch.float64,accurate_cross_entropy)]:
        logits=frozen.to(dtype).clone().requires_grad_()
        loss=loss_fn(logits,y)
        grad,=torch.autograd.grad(loss,logits)
        gradients[name]=grad
        correct=logits.argmax(-1)==y
        target_grad=grad.gather(1,y[:,None])[:,0]
        per_example_loss=loss_fn(logits.detach(),y,reduction='none')
        results[name]=dict(loss=float(loss.detach()),zero_target_gradient_fraction=float((target_grad==0).double().mean()),
            zero_loss_fraction=float((per_example_loss==0).double().mean()),
            zero_loss_and_zero_target_gradient_fraction=float(((per_example_loss==0)&(target_grad==0)).double().mean()),
            zero_target_but_nonzero_gradient_fraction=float(((target_grad==0)&(grad.abs().sum(-1)>0)).double().mean()),
            correct_examples=int(correct.sum()),max_absolute_row_gradient_sum=float(grad.double().sum(-1).abs().max()))
    for name,grad in gradients.items():results[name]['gradient_vs_stable64']=comparison(grad,gradients['stable64'])
    upstream={}
    for name,loss_fn in [('stock32',F.cross_entropy),('stable32',accurate_cross_entropy)]:
        model.zero_grad(set_to_none=True)
        loss_fn(model(x),y).backward()
        upstream[name]={n:p.grad.detach().clone() for n,p in model.named_parameters()}
    parameters={name:comparison(upstream['stock32'][name],gradient) for name,gradient in upstream['stable32'].items()}
    common_mode={}
    for name,gradients_by_name in upstream.items():
        readout_gradient=gradients_by_name['unembedding.weight'].double()
        common=readout_gradient.mean(dim=0,keepdim=True).expand_as(readout_gradient)
        common_mode[name]=dict(class_common_gradient_norm=float(common.norm()),
            total_gradient_norm=float(readout_gradient.norm()),
            class_common_gradient_fraction=float(common.norm()/readout_gradient.norm()))
    return dict(checkpoint=str(Path(checkpoint).resolve()),step=state['step'],seed=state['seed'],
        description='Both loss calculations see exactly the same frozen float32 logits. Float64 reference promotes these logits. Parameter gradients use separate deterministic float32 forward passes with identical weights.',
        loss_gradient=results,stock32_parameter_gradient_vs_stable32=parameters,
        readout_class_common_gradient=common_mode,
        common_mode_note='Exact softmax cross-entropy is invariant to a shared shift of all class logits, so its readout gradient has zero class-common component up to arithmetic error.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    result=diagnose(a.checkpoint);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
