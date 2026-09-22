"""Deterministic mathematical controls for checkpoint geometry diagnostics."""
import unittest
import numpy as np
from geometry import analyze_pair, column_space_diagnostic, controls, factor, functional_decomposition, solve_factor, thresholds


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(37)
        self.h = self.rng.normal(size=(90, 7))
        self.w = self.rng.normal(size=(7, 4))
        self.y = (self.h @ self.w).argmax(1)
        self.train = np.arange(90) < 50
        self.test = ~self.train

    def analyze(self, h1):
        return analyze_pair(self.h, h1, self.w, self.w, self.y, self.train, self.test)

    def test_invertible_planted_map_exactly_recovered(self):
        q, _ = np.linalg.qr(self.rng.normal(size=(7, 7)))
        a = q @ np.diag(np.geomspace(1, 100, 7))
        report, maps = self.analyze(self.h @ a)
        np.testing.assert_allclose(maps['forward_linear_mapping'], a, atol=1e-11)
        np.testing.assert_allclose(maps['backward_linear_mapping'], np.linalg.inv(a), atol=1e-11)
        oracle = report['oracle_geometry']['uncentered']['by_rank_tolerance']['float64_solver']
        self.assertLess(oracle['y_outside_x_column_space_relative'], 1e-12)
        self.assertLess(oracle['x_outside_y_column_space_relative'], 1e-12)

    def test_heldout_values_cannot_change_fitted_maps(self):
        h1 = self.h + .1 * self.rng.normal(size=self.h.shape)
        _, before = self.analyze(h1)
        modified_h = self.h.copy()
        modified_h[self.test] = self.rng.normal(size=(self.test.sum(), 7)) * 100
        modified_h1 = h1.copy()
        modified_h1[self.test] = self.rng.normal(size=(self.test.sum(), 7)) * 100
        _, after = analyze_pair(modified_h, modified_h1, self.w, self.w,
                               self.y[::-1], self.train, self.test)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_affine_translation_is_distinct_from_linear_basis(self):
        report, _ = self.analyze(self.h + np.arange(1, 8))
        fwd = report['train_only_maps']['forward']
        linear = fwd['linear']['rank_sensitivity']['float64_solver']['reconstruction']['heldout']
        affine = fwd['affine']['rank_sensitivity']['float64_solver']['reconstruction']['heldout']
        self.assertGreater(linear['relative_error'], .5)
        self.assertLess(affine['relative_error'], 1e-12)

    def test_row_specific_changes_violate_shared_linear_map(self):
        h1 = self.h.copy()
        h1[:, 0] = np.sin(self.h[:, 0])
        r = column_space_diagnostic(self.h, h1)['by_rank_tolerance']['float64_solver']
        self.assertGreater(r['y_outside_x_column_space_relative'], .02)
        self.assertGreater(r['angle_max_degrees'], 10)

    def test_oracle_equals_explicit_least_squares_optimum(self):
        h1 = self.h + self.rng.normal(size=self.h.shape)
        a = solve_factor(factor(self.h), h1, thresholds(self.h.shape)['float64_solver'])
        explicit = np.linalg.norm(self.h @ a - h1) / np.linalg.norm(h1)
        oracle = column_space_diagnostic(self.h, h1)['by_rank_tolerance']['float64_solver']
        self.assertAlmostEqual(explicit, oracle['y_outside_x_column_space_relative'], places=13)

    def test_large_constant_component_does_not_hide_centered_error(self):
        self.h[:, 0] = 1e4
        h1 = self.h.copy()
        h1[:, 1:] = self.rng.normal(size=(90, 6))
        report, _ = self.analyze(h1)
        error = report['train_only_maps']['backward']['linear']['rank_sensitivity']['float64_solver']['reconstruction']['heldout']
        self.assertLess(error['relative_error'], 1e-3)
        self.assertGreater(error['error_over_centered_target_norm'], .9)

    def test_rank_deficient_representation_reported(self):
        self.h[:, -1] = self.h[:, 0] + self.h[:, 1]
        r = column_space_diagnostic(self.h, self.h)
        self.assertEqual(r['x_spectrum']['ranks']['float64_solver'], 6)
        self.assertLess(r['by_rank_tolerance']['float64_solver']['y_outside_x_column_space_relative'], 1e-12)

    def test_native_rounding_positive_and_permutation_negative_controls(self):
        r = controls(self.h, self.train, self.test)
        for planted in r['planted'].values():
            self.assertLess(planted['double_precision']['forward']['heldout']['relative_error'], 1e-12)
            self.assertLess(planted['float32_matmul']['forward']['heldout']['relative_error'], 1e-6)
        self.assertGreater(r['permuted_training_correspondence']['heldout']['relative_error'], .8)

    def test_functional_decomposition_identity_and_cross_term(self):
        a = self.rng.normal(size=(7, 7))
        h1 = self.h @ a + .2 * self.rng.normal(size=self.h.shape)
        w1 = self.w + self.rng.normal(size=self.w.shape)
        report = functional_decomposition(self.h, h1, self.w, w1, a,
                                          self.y, self.train, self.test)
        for split in report['splits'].values():
            self.assertLess(split['identity_error_relative_to_delta'], 1e-14)
            terms = [split[k] for k in ['basis_squared_norm_over_delta_squared',
                                       'residual_squared_norm_over_delta_squared',
                                       'normalized_cross_term']]
            self.assertAlmostEqual(sum(terms), 1.0, places=13)
            self.assertEqual(split['counterfactuals']['actual_later']['actual_flip_recall'], 1.0)

    def test_pure_planted_basis_functional_effect_recovered(self):
        a = self.rng.normal(size=(7, 7))
        report = functional_decomposition(self.h, self.h @ a, self.w,
                                          self.w, a, self.y, self.train, self.test)
        for split in report['splits'].values():
            self.assertEqual(split['residual_component_class_centered_norm'], 0.0)
            self.assertEqual(split['counterfactuals']['basis_predicted']['argmax_agreement_with_actual'], 1.0)
            self.assertEqual(split['counterfactuals']['residual_only']['flips_from_healthy'], 0)


if __name__ == '__main__':
    unittest.main()
