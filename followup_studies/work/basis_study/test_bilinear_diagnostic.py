"""Mathematical controls for endpoint component attribution."""
import unittest
import numpy as np
from bilinear_diagnostic import analyze_bilinear


class BilinearTests(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        self.h0 = r.normal(size=(90, 8))
        self.h1 = self.h0 + r.normal(size=(90, 8))
        self.w0 = r.normal(size=(8, 5))
        self.w1 = self.w0 + r.normal(size=(8, 5))
        self.y = (self.h0 @ self.w0).argmax(1)
        self.train = np.arange(90) < 50

    def run_pair(self, h1=None, w1=None):
        return analyze_bilinear(self.h0, self.h1 if h1 is None else h1,
            self.w0, self.w1 if w1 is None else w1, self.y, self.train, ~self.train)

    def test_exact_identity_and_all_three_cross_terms(self):
        report, _ = self.run_pair()
        for d in report['splits'].values():
            self.assertLess(d['logit_identity_error_max_abs'], 1e-12)
            self.assertLess(d['squared_norm_identity_relative_error'], 1e-12)
            self.assertEqual(len(d['pairwise_cross_terms']), 3)
            terms = sum(c['squared_norm_over_total_change_squared'] for c in d['components'].values())
            terms += sum(c['normalized_by_total_change_squared'] for c in d['pairwise_cross_terms'].values())
            self.assertAlmostEqual(terms, 1.0, places=13)

    def test_readout_only_change(self):
        report, _ = self.run_pair(h1=self.h0)
        d = report['splits']['heldout']
        self.assertEqual(d['components']['feature_change']['class_centered_norm'], 0)
        self.assertEqual(d['components']['interaction']['class_centered_norm'], 0)
        self.assertAlmostEqual(d['components']['readout_change']['norm_over_total_change'], 1.0)
        self.assertEqual(d['identity_vs_linear_prediction']['identity_features_with_later_readout']['argmax_agreement_with_actual'], 1.0)

    def test_feature_only_change(self):
        report, _ = self.run_pair(w1=self.w0)
        d = report['splits']['heldout']
        self.assertEqual(d['components']['readout_change']['class_centered_norm'], 0)
        self.assertEqual(d['components']['interaction']['class_centered_norm'], 0)
        self.assertAlmostEqual(d['components']['feature_change']['norm_over_total_change'], 1.0)

    def test_exact_joint_basis_change_cancels_components(self):
        rng = np.random.default_rng(1)
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        report, _ = self.run_pair(h1=self.h0 @ q, w1=q.T @ self.w0)
        d = report['splits']['heldout']
        self.assertTrue(d['near_zero_total_change'])
        self.assertTrue(all(v['norm_over_total_change'] is None for v in d['components'].values()))
        self.assertEqual(d['counterfactuals_2x2']['actual_later']['classification']['accuracy'], 1.0)

    def test_no_heldout_leakage_in_linear_fit(self):
        _, before = self.run_pair()
        h0 = self.h0.copy()
        h1 = self.h1.copy()
        h0[~self.train] *= 10
        h1[~self.train] += 100
        _, after = analyze_bilinear(h0, h1, self.w0, self.w1, self.y[::-1], self.train, ~self.train)
        np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
    unittest.main()
