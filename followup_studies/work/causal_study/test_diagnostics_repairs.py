"""Fixed-logit verification of the targeted derivative controls."""
import sys
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from numerical_controls import accurate_cross_entropy
from diagnostics_repairs import ce_target_repair, ce_row_projection, target_repair_gradient, row_project_gradient


class DerivativeRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self):
        gen=torch.Generator().manual_seed(1348)
        z=torch.randn(9,113,generator=gen)*2
        y=torch.arange(9)
        z[:6]=0
        z[torch.arange(6),torch.arange(6)]=torch.tensor([5,15,20,25,30,50.])
        return z,y

    def grad(self,fn,z,y,reduction="mean",weights=None):
        x=z.detach().clone().requires_grad_()
        loss=fn(x,y,reduction=reduction)
        scalar=(loss*weights).sum() if weights is not None else loss.sum()
        gradient,=torch.autograd.grad(scalar,x)
        return loss.detach(),gradient

    def test_forward_is_bitwise_stock(self):
        z,y=self.data()
        for reduction in ("mean","sum","none"):
            baseline=F.cross_entropy(z,y,reduction=reduction)
            for fn in (ce_target_repair,ce_row_projection):
                self.assertTrue(torch.equal(fn(z,y,reduction=reduction),baseline))

    def test_target_repair_retains_wrong_entries(self):
        z,y=self.data()
        _,stock=self.grad(F.cross_entropy,z,y)
        _,repaired=self.grad(ce_target_repair,z,y)
        self.assertTrue(torch.equal(stock.scatter(1,y[:,None],0.),repaired.scatter(1,y[:,None],0.)))
        self.assertTrue(torch.equal(repaired,target_repair_gradient(stock,y)))

    def test_reduction_and_example_weights_are_respected(self):
        z,y=self.data()
        weights=torch.arange(1.,10.)
        for reduction,w in (("mean",None),("sum",None),("none",weights)):
            _,stock=self.grad(F.cross_entropy,z,y,reduction,w)
            for fn,transform in ((ce_target_repair,lambda g:target_repair_gradient(g,y)),(ce_row_projection,row_project_gradient)):
                _,corrected=self.grad(fn,z,y,reduction,w)
                self.assertTrue(torch.equal(corrected,transform(stock)))

    def test_projection_preserves_class_centered_direction(self):
        z,y=self.data()
        _,stock=self.grad(F.cross_entropy,z,y)
        _,projected=self.grad(ce_row_projection,z,y)
        double_centered=stock.double()-stock.double().mean(-1,keepdim=True)
        self.assertLess(float((projected.double()-double_centered).norm()/double_centered.norm()),1e-6)
        self.assertLess(float(projected.double().sum(-1).norm()/projected.double().norm()),1e-6)

    def test_target_repair_matches_accurate_reference(self):
        z,y=self.data()
        _,repaired=self.grad(ce_target_repair,z,y)
        _,reference=self.grad(accurate_cross_entropy,z.double(),y)
        self.assertLess(float((repaired.double()-reference).norm()/reference.norm()),1e-6)
        # Per-row checks retain sensitivity in high-confidence, tiny-gradient rows.
        per_row=(repaired.double()-reference).norm(dim=-1)/reference.norm(dim=-1)
        self.assertLess(float(per_row.max()),1e-5)

    def test_zero_stock_target_has_corrected_nonzero_derivative(self):
        z=torch.zeros(1,113);z[0,0]=30
        y=torch.tensor([0])
        _,stock=self.grad(F.cross_entropy,z,y)
        _,repaired=self.grad(ce_target_repair,z,y)
        self.assertEqual(float(stock[0,0]),0.)
        self.assertLess(float(repaired[0,0]),0.)
        self.assertGreater(float(stock[0,1:].sum()),0.)


if __name__=="__main__":
    unittest.main()
