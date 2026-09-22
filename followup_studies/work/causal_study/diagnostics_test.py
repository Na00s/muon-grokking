import unittest
import torch
from diagnostics import accurate_logit_gradient, adam_decomposition, compare
from numerical_controls import accurate_cross_entropy


class DiagnosticsTests(unittest.TestCase):
    def test_gradient_matches_autograd(self):
        torch.manual_seed(81)
        x=(torch.randn(11,7,dtype=torch.float64)*6).requires_grad_();y=torch.arange(11)%7
        expected,=torch.autograd.grad(accurate_cross_entropy(x,y),x)
        self.assertLess(compare(accurate_logit_gradient(x,y),expected)['relative_l2_error'],1e-14)

    def test_saturated_target_derivative(self):
        x=torch.tensor([[80.,0.,-10.]],dtype=torch.float32);y=torch.tensor([0])
        derivative=accurate_logit_gradient(x,y)
        self.assertLess(float(derivative[0,0]),0.)
        self.assertLess(abs(float(derivative.sum())),1e-48)

    def test_shift_invariance(self):
        x=torch.tensor([[3.,-2.,1.],[-1.,4.,3.]],dtype=torch.float64);y=torch.tensor([2,1])
        self.assertTrue(torch.allclose(accurate_logit_gradient(x,y),accurate_logit_gradient(x+1234,y),rtol=1e-13,atol=1e-16))

    def test_adam_component_reconstruction(self):
        torch.manual_seed(4)
        p=torch.nn.Parameter(torch.randn(7,3,dtype=torch.float64));o=torch.optim.AdamW([p],lr=.003,weight_decay=.5)
        for _ in range(4): p.grad=torch.randn_like(p);o.step()
        p.grad=torch.randn_like(p);before=p.detach().clone();report,parts=adam_decomposition(p,o,p.grad)
        o.step();delta=p.detach()-before
        self.assertLess(compare(parts['total'],delta)['relative_l2_error'],1e-12)
        self.assertTrue(torch.equal(parts['historical']+parts['current']+parts['decay'],parts['total']))

    def test_fixed_head_directional_curvature(self):
        torch.manual_seed(10)
        z=torch.randn(9,5,dtype=torch.float64);d=torch.randn_like(z);y=torch.arange(9)%5
        alpha=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        loss=accurate_cross_entropy(z+alpha*d,y)
        first,=torch.autograd.grad(loss,alpha,create_graph=True)
        second,=torch.autograd.grad(first,alpha)
        p=torch.softmax(z,1)
        computed_first=(accurate_logit_gradient(z,y)*d).sum()
        computed_second=(p*(d-(p*d).sum(1,keepdim=True)).square()).sum(1).mean()
        self.assertLess(abs(float(first.detach()-computed_first)),1e-14)
        self.assertLess(abs(float(second-computed_second)),1e-14)

    def test_adam_can_center_common_class_gradient(self):
        p=torch.nn.Parameter(torch.zeros(2,2,dtype=torch.float64))
        o=torch.optim.AdamW([p],lr=.1,weight_decay=0.)
        o.state[p].update(step=torch.tensor(10.),exp_avg=torch.zeros_like(p),
                          exp_avg_sq=torch.tensor([[1.,1.],[4.,4.]],dtype=torch.float64))
        common=torch.ones_like(p)
        _,parts=adam_decomposition(p,o,common)
        delta=parts['total'];centered=delta-delta.mean(0,keepdim=True)
        self.assertGreater(float(centered.norm()),1e-4)
        sgd=-.1*common
        self.assertEqual(float((sgd-sgd.mean(0,keepdim=True)).norm()),0.)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
