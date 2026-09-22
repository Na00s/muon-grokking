import unittest
import numpy as np
import torch
from model_gauge import DepthModularAdditionTransformer,gauge_model,capture_sites,make_A

class ModelGaugeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def make_model(self,dtype=torch.float64):
        torch.manual_seed(719)
        return DepthModularAdditionTransformer(modulus=13,sequence_length=3,d_model=16,num_heads=4,d_mlp=32,num_layers=2).to(dtype=dtype)

    def test_full_network_gauge_preserves_all_sites_and_logits_float64(self):
        model=self.make_model();x=torch.tensor([[1,4,13],[2,7,13],[10,2,13],[12,12,13]])
        A=make_A(16,10)
        changed=gauge_model(model,A)
        old,logits0=capture_sites(model,x);new,logits1=capture_sites(changed,x)
        np.testing.assert_allclose(logits0,logits1,atol=3e-14,rtol=1e-13)
        for site in old:np.testing.assert_allclose(old[site]@A,new[site],atol=5e-14,rtol=1e-12)

    def test_full_network_gauge_float32_at_native_precision(self):
        model=self.make_model(torch.float32);x=torch.tensor([[1,4,13],[2,7,13],[10,2,13],[12,12,13]])
        changed=gauge_model(model,make_A(16,10))
        with torch.no_grad():
            old=model(x).numpy();new=changed(x).numpy()
        self.assertLess(np.linalg.norm(old-new)/np.linalg.norm(old),3e-6)

    def test_capture_matches_original_forward_exactly(self):
        model=self.make_model();x=torch.tensor([[1,4,13],[2,7,13],[10,2,13]])
        _,logits=capture_sites(model,x)
        with torch.no_grad():native=model(x).numpy()
        np.testing.assert_array_equal(logits,native)

    def test_original_model_is_unchanged(self):
        model=self.make_model();before={k:v.clone() for k,v in model.state_dict().items()}
        changed=gauge_model(model,make_A(16,3))
        for name,value in model.state_dict().items():self.assertTrue(torch.equal(before[name],value))
        self.assertFalse(torch.equal(model.token_embedding.weight,changed.token_embedding.weight))

    def test_leaving_old_readout_is_exact_mismatch_control(self):
        model=self.make_model();A=make_A(16,3);x=torch.tensor([[1,4,13],[2,7,13],[10,2,13]])
        changed=gauge_model(model,A,compensate_readout=False)
        old,_=capture_sites(model,x);new,logits=capture_sites(changed,x)
        W=model.unembedding.weight.detach().numpy().T
        expected=old['block1_after_mlp'][:,-1]@A@W
        np.testing.assert_allclose(logits,expected,atol=3e-14,rtol=1e-12)
        self.assertTrue(torch.equal(model.unembedding.weight,changed.unembedding.weight))

if __name__=='__main__':unittest.main()
