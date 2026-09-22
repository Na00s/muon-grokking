"""Meaningful synthetic checks for task decomposition and identifiability."""
import unittest
import numpy as np
from task_subspace import fourier_task, fit_task_coefficients, complete_invertible_backward, fit_ridge_map

class TaskSubspaceTests(unittest.TestCase):
    def test_fourier_basis_complete_orthonormal(self):
        p=13;T=fourier_task(np.arange(p),p)
        np.testing.assert_allclose(T.T@T/p,np.eye(p),atol=3e-15)

    def test_coefficients_equal_class_means_even_with_unequal_counts(self):
        rng=np.random.default_rng(51);p,d=13,20
        y=np.concatenate([np.repeat(k,3+k) for k in range(p)])
        H=rng.normal(size=(len(y),d));train=np.ones(len(y),dtype=bool)
        C=fit_task_coefficients(H,y,train,p)
        P=fourier_task(np.arange(p),p)@C
        means=np.stack([H[y==k].mean(0) for k in range(p)])
        np.testing.assert_allclose(P,means,atol=2e-15)

    def test_arbitrary_independent_task_encodings_admit_invertible_map(self):
        rng=np.random.default_rng(52);q,d=13,20
        C0=rng.normal(size=(q,d));C1=rng.normal(size=(q,d))
        B,metrics=complete_invertible_backward(C0,C1)
        np.testing.assert_allclose(C1@B,C0,atol=2e-14)
        self.assertEqual(np.linalg.matrix_rank(B),d)
        self.assertLess(metrics['relative_coefficient_error'],1e-14)

    def test_task_only_fit_fails_on_independent_nuisance(self):
        rng=np.random.default_rng(53);p,d,n=13,20,1300
        y=np.arange(n)%p;T=fourier_task(y,p)
        C0=rng.normal(size=(p,d));C1=rng.normal(size=(p,d))
        B,_=complete_invertible_backward(C0,C1)
        self.assertLess(np.linalg.norm(T@C1@B-T@C0),1e-10)
        H0=T@C0+rng.normal(size=(n,d));H1=T@C1+rng.normal(size=(n,d))
        self.assertGreater(np.linalg.norm(H1@B-H0)/np.linalg.norm(H0),0.4)

    def test_training_fit_is_independent_of_heldout_features(self):
        rng=np.random.default_rng(54);p,d,n=13,20,260
        y=np.arange(n)%p;H=rng.normal(size=(n,d));train=np.arange(n)<130
        C=fit_task_coefficients(H,y,train,p)
        H[~train]=1e9*rng.normal(size=(sum(~train),d))
        changed=fit_task_coefficients(H,y,train,p)
        np.testing.assert_array_equal(C,changed)

    def test_training_fit_is_independent_of_heldout_labels(self):
        rng=np.random.default_rng(55);p,d,n=13,20,260
        y=np.arange(n)%p;H=rng.normal(size=(n,d));train=np.arange(n)<130
        C=fit_task_coefficients(H,y,train,p)
        y[~train]=rng.integers(0,p,sum(~train))
        changed=fit_task_coefficients(H,y,train,p)
        np.testing.assert_array_equal(C,changed)

    def test_planted_global_map_generalizes_to_excluded_classes(self):
        rng=np.random.default_rng(56);p,d,n=13,20,1300
        y=np.arange(n)%p;H0=rng.normal(size=(n,d))
        U,_=np.linalg.qr(rng.normal(size=(d,d)));V,_=np.linalg.qr(rng.normal(size=(d,d)))
        A=U@np.diag(np.linspace(0.5,2,d))@V.T;H1=H0@A
        train=np.arange(n)<800;fit=train&(y%5!=0);heldout=(~train)&(y%5==0)
        B=fit_ridge_map(H1[fit],H0[fit],0)
        np.testing.assert_allclose(H1[heldout]@B,H0[heldout],atol=2e-14)

    def test_rank_deficient_coefficients_rejected(self):
        C=np.eye(5,8);C[0]=C[1]
        with self.assertRaises(ValueError): complete_invertible_backward(C,C)

if __name__=='__main__':unittest.main()
