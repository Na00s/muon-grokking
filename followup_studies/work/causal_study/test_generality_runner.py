import copy
import json
from pathlib import Path
import time
import unittest
import torch
import torch.nn.functional as F
import generality_runner as g


def equal_tree(first, second):
    if isinstance(first,torch.Tensor): return torch.equal(first,second)
    if isinstance(first,dict): return first.keys()==second.keys() and all(equal_tree(first[k],second[k]) for k in first)
    if isinstance(first,(list,tuple)): return len(first)==len(second) and all(equal_tree(x,y) for x,y in zip(first,second))
    return first==second


class TestGenerality(unittest.TestCase):
    def test_source_model_initialization_and_optimizer_routing(self):
        original, oldopts=g.original_make(0)
        oldstate=g.original_snapshot(original,oldopts,0,0)
        for norm in ['none','rms']:
            model,opts=g.make_model_optimizers(0,norm)
            self.assertTrue(equal_tree(original.state_dict(),model.state_dict()))
            self.assertTrue(equal_tree(oldstate['optimizer_state_dicts'],{k:o.state_dict() for k,o in opts.items()}))
            names={id(p):n for n,p in model.named_parameters()}
            groups={k:[names[id(p)] for group in o.param_groups for p in group['params']] for k,o in opts.items()}
            self.assertEqual(groups['auxiliary_adamw'],['token_embedding.weight','position_embedding.weight'])
            self.assertEqual(groups['unembedding_adamw'],['unembedding.weight'])
            self.assertEqual(len(groups['muon']),4)
            covered=sum(groups.values(),[])
            self.assertEqual(len(covered),len(set(covered)))
            self.assertEqual(set(covered),set(names.values()))

    def test_original_update_exact(self):
        x,y,*_=g.generate_modular_addition_data(seed=0)
        original,oldopts=g.original_make(0)
        model,opts=g.make_model_optimizers(0,'none')
        for _ in range(3):
            g.original_step(original,oldopts,x,y)
            model.train()
            for opt in opts.values(): opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y);loss.backward()
            for opt in opts.values():opt.step()
        self.assertTrue(equal_tree(original.state_dict(),model.state_dict()))
        self.assertTrue(equal_tree({k:o.state_dict() for k,o in oldopts.items()},{k:o.state_dict() for k,o in opts.items()}))

    def test_task_split_and_labels(self):
        ax,ay,bx,by=g.generate_modular_addition_data(seed=0,operation='addition')
        sx,sy,tx,ty=g.generate_modular_addition_data(seed=0,operation='subtraction')
        self.assertTrue(torch.equal(ax,sx));self.assertTrue(torch.equal(bx,tx))
        self.assertEqual(len(sx),3830);self.assertEqual(len(tx),8939)
        self.assertTrue(torch.equal(sy,(sx[:,0]-sx[:,1])%113))
        self.assertTrue(torch.equal(ty,(tx[:,0]-tx[:,1])%113))

    def test_accurate_ce_independent_derivative(self):
        # Includes confidently correct, tied, and incorrect targets.
        logits=torch.tensor([[40.,0.,-2.],[1.,1.,-3.],[-4.,2.,0.]],dtype=torch.float64,requires_grad=True)
        targets=torch.tensor([0,0,0])
        loss=g.accurate_cross_entropy(logits,targets,reduction='sum')
        grad=torch.autograd.grad(loss,logits)[0]
        # Derive each true-target gradient as negative sum of wrong probabilities.
        prob=logits.detach().softmax(-1); expected=prob.clone()
        for row,target in enumerate(targets):
            mask=torch.arange(3)!=target
            expected[row,target]=-prob[row,mask].sum()
        torch.testing.assert_close(grad,expected,rtol=2e-15,atol=1e-17)
        self.assertLess(float(grad[0,0]),0.)

    def test_rms_formula_and_no_parameters(self):
        model,_=g.make_model_optimizers(0,'rms')
        values=torch.randn(3,7,128,dtype=torch.float64)
        expected=values/torch.sqrt((values*values).mean(-1,keepdim=True)+1e-6)
        torch.testing.assert_close(model.normalize(values),expected)
        self.assertEqual(len(list(model.parameters())),7)

    def test_checkpoint_load_is_independent(self):
        model,opts=g.make_model_optimizers(0,'none')
        x,y,*_=g.generate_modular_addition_data(seed=0)
        g.original_step(model,opts,x,y)
        cp=g.snapshot(model,opts,1,0,'addition','none','stock')
        before=copy.deepcopy(cp)
        restored,newopts=g.make_model_optimizers(0,'none',cp)
        g.original_step(restored,newopts,x,y)
        self.assertTrue(equal_tree(cp,before))


if __name__=='__main__':
    g.configure_runtime()
    unittest.main()
