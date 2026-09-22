"""Cheap fixed-logit checks for targeted CE derivative corrections.

This file performs no training. All corrections start from the identical stock
PyTorch derivative on a fixed logit matrix. Float64 accurate CE is the reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from numerical_controls import accurate_cross_entropy
from run_collapse import make_model_optimizers, generate_modular_addition_data


def metric(g: torch.Tensor, ref: torch.Tensor) -> dict:
    gd, rd = g.double(), ref.double()
    denom = float(rd.norm())
    common = gd.mean(-1, keepdim=True).expand_as(gd)
    return {
        "relative_l2_error": float((gd-rd).norm()) / denom if denom else 0.0,
        "gradient_norm": float(gd.norm()),
        "class_common_norm_fraction": float(common.norm()/gd.norm()) if gd.norm() else 0.0,
        "max_abs_row_sum": float(gd.sum(-1).abs().max()),
        "max_abs_error": float((gd-rd).abs().max()),
    }


def corrections(stock: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    wrong = stock.scatter(1, y[:, None], 0.0)
    repaired = stock.scatter(1, y[:, None], -wrong.double().sum(-1, keepdim=True).to(stock.dtype))
    projected = (stock.double()-stock.double().mean(-1, keepdim=True)).to(stock.dtype)
    return {"stock": stock, "target_repaired": repaired, "row_mean_projected": projected}


def inspect(logits: torch.Tensor, y: torch.Tensor) -> dict:
    stocks = logits.detach().float().clone().requires_grad_()
    exacts = logits.detach().double().clone().requires_grad_()
    stock_loss = F.cross_entropy(stocks, y)
    exact_loss = accurate_cross_entropy(exacts, y)
    stock, = torch.autograd.grad(stock_loss, stocks)
    ref, = torch.autograd.grad(exact_loss, exacts)
    stable_logits = logits.detach().float().clone().requires_grad_()
    stable_loss = accurate_cross_entropy(stable_logits, y)
    stable, = torch.autograd.grad(stable_loss, stable_logits)
    gs = corrections(stock, y)
    gs["accurate32"] = stable
    same_wrong = gs["target_repaired"].scatter(1,y[:,None],0.0)
    assert torch.equal(same_wrong, stock.scatter(1,y[:,None],0.0))
    return {
        "examples": len(y), "classes": logits.shape[-1],
        "accuracy": float((logits.argmax(-1)==y).double().mean()),
        "stock_loss": float(stock_loss.detach()), "accurate64_loss": float(exact_loss.detach()),
        "zero_stock_target_fraction": float((stock.gather(1,y[:,None])==0).double().mean()),
        "methods": {name: metric(g, ref) for name,g in gs.items()},
        "wrong_class_derivatives_bitwise_preserved": True,
    }


def run() -> dict:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cases = {}
    for gap in (0, 5, 10, 15, 20, 25, 30, 50, 80):
        z = torch.zeros(4, 113)
        z[:, 0] = gap
        y = torch.zeros(4, dtype=torch.long)
        cases[f"correct_equal_wrong_gap_{gap}"] = inspect(z, y)
    gen = torch.Generator().manual_seed(20260922)
    z = torch.randn(29, 113, generator=gen)*2
    y = torch.arange(29, dtype=torch.long)%113
    z[0,0] = 25
    cases["mixed_correct_incorrect"] = inspect(z, y)
    cases["all_wrong"] = inspect(torch.tensor([[25.0,0.0,-25.0]]), torch.tensor([1]))
    checkpoint_paths = [
        ROOT/"experiments/seed4_original/step_006000.pt",
        ROOT/"experiments/seed4_original/step_012000.pt",
        ROOT/"experiments/seed4_original/step_016000.pt",
    ]
    checkpoints = []
    for p in checkpoint_paths:
        state = torch.load(p, map_location="cpu", weights_only=False)
        model,_ = make_model_optimizers(int(state["seed"]), checkpoint=state)
        x,y,_,_ = generate_modular_addition_data(seed=int(state["seed"]))
        model.eval()
        with torch.no_grad(): z = model(x)
        checkpoints.append({"checkpoint":str(p),"step":int(state["step"]),"seed":int(state["seed"]),**inspect(z,y)})
    # A simple counterexample to GL(d) invariance of power rankings.
    c = torch.diag(torch.tensor([2.0,1.0],dtype=torch.float64))
    a = torch.diag(torch.tensor([0.1,10.0],dtype=torch.float64))
    before = c.square().sum(-1)
    after = (c@a).square().sum(-1)
    algebra = {"matrix":a.tolist(),"power_before":before.tolist(),"power_after":after.tolist(),
               "support_preserved":bool(torch.equal(before>0,after>0)),
               "ranking_reversed":bool(before.argmax()!=after.argmax())}
    assert algebra["support_preserved"] and algebra["ranking_reversed"]
    for case in cases.values():
        assert case["methods"]["target_repaired"]["relative_l2_error"] < 1e-5
    for case in checkpoints:
        assert case["methods"]["target_repaired"]["relative_l2_error"] < 1e-5
    return {"description":"Frozen identical logits; no optimization or parameter changes.",
            "synthetic_cases":cases,"checkpoints":checkpoints,"gl_power_counterexample":algebra,
            "tests_passed":len(cases)+len(checkpoints)+2}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(__file__).with_name("derivative_identity_check.json"))
    args=p.parse_args()
    result=run()
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"out":str(args.out),"tests_passed":result["tests_passed"],
                      "checkpoints":[{"step":c["step"],"methods":c["methods"]} for c in result["checkpoints"]]},indent=2))
