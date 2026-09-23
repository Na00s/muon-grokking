"""Small synthetic checks of decoder gradients, recoverability and data discipline."""

import json
import unittest

import numpy as np

from decoder_probe import _feature_preconditioner, _loss_and_gradient, fit_logistic_probe


class DecoderProbeTests(unittest.TestCase):
    def test_whitening_preserves_logits_and_gradient_mapping(self):
        rng = np.random.default_rng(20)
        H = rng.normal(size=(100, 6)) * np.geomspace(1.0, 1e-3, 6)
        P, inverse, audit = _feature_preconditioner(H, "whiten", 1e-6, "float32")
        C = rng.normal(size=(6, 3))
        y = rng.integers(3, size=len(H))
        W = P @ C
        transformed_loss, transformed_gradient = _loss_and_gradient(C.ravel(), H @ P, y, 3, 0.0)
        raw_loss, raw_gradient = _loss_and_gradient(W.ravel(), H, y, 3, 0.0)
        self.assertAlmostEqual(transformed_loss, raw_loss, places=12)
        np.testing.assert_allclose(transformed_gradient.reshape(6, 3), P.T @ raw_gradient.reshape(6, 3), atol=1e-12)
        np.testing.assert_allclose(inverse @ W, C, atol=1e-11)
        self.assertEqual(audit["floored_singular_values"], 0)
        self.assertLess(audit["transformed_covariance_relative_identity_error"], 1e-10)

    def test_whitening_heldout_independence_and_full_dimension(self):
        rng = np.random.default_rng(32)
        H = rng.normal(size=(260, 5)) * np.geomspace(1.0, 1e-4, 5)
        y = (H[:, -1] > 0).astype(int)
        train = np.arange(len(H)) < 160
        options = dict(n_classes=2, regularization_grid=(0.0, 1e-4), max_iter=40, refit_max_iter=60,
                       feature_transform="whiten", native_feature_dtype="float32")
        first = fit_logistic_probe(H, y, train, ~train, **options)
        changed = H.copy()
        changed[~train] = rng.normal(size=changed[~train].shape) * 100
        second = fit_logistic_probe(changed, y, train, ~train, **options)
        np.testing.assert_array_equal(first.readout, second.readout)
        np.testing.assert_array_equal(first.feature_matrix, second.feature_matrix)
        self.assertEqual(first.report["candidates"], second.report["candidates"])
        self.assertEqual(first.feature_matrix.shape, (5, 5))
        np.testing.assert_allclose(H @ first.readout, (H @ first.feature_matrix) @ first.scaled_readout, atol=1e-9)

    def test_saturated_target_retains_small_gradient(self):
        X = np.ones((1, 1))
        y = np.array([0])
        weights = np.array([0.0, -50.0])
        loss, gradient = _loss_and_gradient(weights, X, y, 2, 0.0)
        expected = np.exp(-50.0)
        self.assertGreater(loss, 0.0)
        self.assertAlmostEqual(loss / expected, 1.0, places=12)
        self.assertAlmostEqual(gradient[0] / -expected, 1.0, places=12)
        self.assertAlmostEqual(gradient[1] / expected, 1.0, places=12)

    def test_exact_gradient_with_regularization(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(23, 5))
        y = rng.integers(4, size=len(X))
        parameters = rng.normal(size=20)
        for penalty in (0.0, 0.2):
            _, gradient = _loss_and_gradient(parameters, X, y, 4, penalty)
            for _ in range(3):
                direction = rng.normal(size=20)
                direction /= np.linalg.norm(direction)
                epsilon = 1e-6
                plus = _loss_and_gradient(parameters + epsilon * direction, X, y, 4, penalty)[0]
                minus = _loss_and_gradient(parameters - epsilon * direction, X, y, 4, penalty)[0]
                self.assertAlmostEqual((plus - minus) / (2 * epsilon), float(gradient @ direction), places=7)

    def test_separable_recovery_and_destroyed_feature(self):
        rng = np.random.default_rng(42)
        H = rng.normal(size=(2600, 6))
        y = (H[:, 0] > 0).astype(int)
        train = np.arange(len(H)) < 500
        options = dict(n_classes=2, regularization_grid=(0.0, 1e-6, 1e-2), max_iter=60, refit_max_iter=100)
        healthy = fit_logistic_probe(H, y, train, ~train, **options)
        destroyed = H.copy()
        destroyed[:, 0] = 0
        collapsed = fit_logistic_probe(destroyed, y, train, ~train, **options)
        self.assertGreater(healthy.report["classification"]["heldout"]["accuracy"], 0.985)
        self.assertGreater(collapsed.report["classification"]["heldout"]["accuracy"], 0.44)
        self.assertLess(collapsed.report["classification"]["heldout"]["accuracy"], 0.56)
        self.assertEqual(healthy.report["initialization"], "zero")
        scale = healthy.report["feature_transform"]["original_training_feature_global_rms"]
        np.testing.assert_allclose(H @ healthy.readout, (H / scale) @ healthy.scaled_readout, atol=1e-10)
        json.dumps(healthy.report, allow_nan=False)
        json.dumps(collapsed.report, allow_nan=False)

    def test_heldout_data_do_not_change_training_or_selection(self):
        rng = np.random.default_rng(30)
        H = rng.normal(size=(400, 5))
        y = (H @ rng.normal(size=(5, 3))).argmax(axis=1)
        train = np.arange(len(H)) < 180
        options = dict(n_classes=3, regularization_grid=(1e-6, 1e-2), max_iter=30, refit_max_iter=40)
        first = fit_logistic_probe(H, y, train, ~train, **options)
        changed = H.copy()
        changed[~train] = rng.normal(size=changed[~train].shape) * 20
        changed_y = y.copy()
        changed_y[~train] = rng.integers(3, size=(~train).sum())
        second = fit_logistic_probe(changed, changed_y, train, ~train, **options)
        np.testing.assert_array_equal(first.readout, second.readout)
        self.assertEqual(first.report["candidates"], second.report["candidates"])
        self.assertEqual(first.report["refit"], second.report["refit"])

    def test_iteration_exhaustion_is_reported_honestly(self):
        rng = np.random.default_rng(9)
        H = rng.normal(size=(100, 8))
        y = (H @ rng.normal(size=(8, 4))).argmax(axis=1)
        train = np.arange(len(H)) < 70
        result = fit_logistic_probe(H, y, train, ~train, n_classes=4, regularization_grid=(0.0,), max_iter=1, refit_max_iter=1)
        self.assertFalse(result.report["refit"]["converged"])
        self.assertEqual(result.report["refit"]["iterations"], 1)
        self.assertNotEqual(result.report["refit"]["status"], 0)

    def test_invalid_validation_mask_rejected(self):
        H = np.ones((10, 3))
        y = np.arange(10) % 2
        train = np.arange(10) < 6
        with self.assertRaisesRegex(ValueError, "subset"):
            fit_logistic_probe(H, y, train, ~train, inner_validation_mask=~train)


if __name__ == "__main__":
    unittest.main(verbosity=2)
