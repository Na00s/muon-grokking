"""Synthetic sanity checks for the alignment protocol; no model repository needed."""

import json
import unittest

import numpy as np

from alignment_core import (
    analyze_alignment,
    classification_metrics,
    fit_orthogonal_forward,
    fit_ridge_map,
    logit_comparison_metrics,
    reconstruction_metrics,
)


def make_case(seed=17, n=1000, d=12, classes=5):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n, d))
    W = rng.normal(size=(d, classes))
    y = (H @ W).argmax(axis=1)
    train = np.arange(n) < n // 3
    return rng, H, W, y, train, ~train


class AlignmentSanityTests(unittest.TestCase):
    def test_identity(self):
        _, H, W, y, train, heldout = make_case()
        result = analyze_alignment(H, H, W, W, y, train, heldout)
        for name in ("orthogonal_transport", "linear_transport", "inverse_forward_transport"):
            self.assertEqual(result.report["classification"]["heldout"][name]["accuracy"], 1.0)
            self.assertLess(result.report["healthy_function_reconstruction"]["heldout"][name]["class_centered_relative_error"], 1e-12)
        np.testing.assert_allclose(result.transforms["orthogonal_forward"], np.eye(H.shape[1]), atol=1e-12)
        json.dumps(result.report, allow_nan=False)

    def test_planted_orthogonal_transform_and_coherent_gauge(self):
        rng, H, W, y, train, heldout = make_case()
        Q, _ = np.linalg.qr(rng.normal(size=(H.shape[1], H.shape[1])))
        H1 = H @ Q
        np.testing.assert_allclose(H1 @ (Q.T @ W), H @ W, atol=1e-12)
        result = analyze_alignment(H, H1, W, W, y, train, heldout)
        np.testing.assert_allclose(result.transforms["orthogonal_forward"], Q, atol=1e-12)
        self.assertLess(result.report["classification"]["heldout"]["collapsed_original"]["accuracy"], 0.5)
        self.assertEqual(result.report["classification"]["heldout"]["orthogonal_transport"]["accuracy"], 1.0)
        self.assertLess(result.report["reconstruction"]["heldout"]["orthogonal_forward"]["relative_error"], 1e-12)

    def test_planted_general_transforms(self):
        rng, H, W, y, train, heldout = make_case()
        U, _ = np.linalg.qr(rng.normal(size=(H.shape[1], H.shape[1])))
        V, _ = np.linalg.qr(rng.normal(size=(H.shape[1], H.shape[1])))
        for condition in (2.0, 10.0, 100.0):
            with self.subTest(condition=condition):
                A = (U * np.geomspace(1.0, condition, H.shape[1])) @ V.T
                H1 = H @ A
                result = analyze_alignment(H, H1, W, W, y, train, heldout)
                np.testing.assert_allclose(result.transforms["linear_forward"], A, atol=1e-10)
                for name in ("linear_transport", "inverse_forward_transport"):
                    self.assertEqual(result.report["classification"]["heldout"][name]["accuracy"], 1.0)
                    self.assertLess(result.report["healthy_function_reconstruction"]["heldout"][name]["class_centered_relative_error"], 1e-10)
                self.assertGreater(result.report["reconstruction"]["heldout"]["orthogonal_forward"]["relative_error"], 0.2)

    def test_shuffled_fitting_correspondence_fails_on_heldout(self):
        rng, H, W, y, train, heldout = make_case(n=1800)
        Q, _ = np.linalg.qr(rng.normal(size=(H.shape[1], H.shape[1])))
        H1 = H @ Q
        train_indices = np.flatnonzero(train)
        H1[train_indices] = H1[rng.permutation(train_indices)]
        result = analyze_alignment(H, H1, W, W, y, train, heldout)
        self.assertLess(result.report["classification"]["heldout"]["linear_transport"]["accuracy"], 0.5)
        self.assertGreater(result.report["reconstruction"]["heldout"]["linear_backward"]["relative_error"], 0.8)

    def test_destroyed_label_feature_cannot_be_rescued(self):
        rng = np.random.default_rng(23)
        H = rng.normal(size=(5000, 12))
        W = np.zeros((12, 2))
        W[0] = (-1.0, 1.0)
        y = (H @ W).argmax(axis=1)
        H1 = H.copy()
        H1[:, 0] = 0
        train = np.arange(len(H)) < 600
        result = analyze_alignment(H, H1, W, W, y, train, ~train)
        metrics = result.report["classification"]["heldout"]["linear_transport"]
        self.assertLess(metrics["accuracy"], 0.56)
        self.assertGreater(metrics["accuracy"], 0.44)
        self.assertFalse(result.report["inverse_transport"]["performed"])
        self.assertEqual(result.report["training_representation_spectra"]["collapsed"]["numerical_rank"], 11)
        self.assertNotIn("inverse_forward_transport", result.transported_readouts)
        json.dumps(result.report, allow_nan=False)

    def test_heldout_values_cannot_change_fit_or_penalty_selection(self):
        rng, H, W, y, train, heldout = make_case()
        H1 = H + rng.normal(scale=0.2, size=H.shape)
        initial = analyze_alignment(H, H1, W, W, y, train, heldout)
        changed_H, changed_H1, changed_y = H.copy(), H1.copy(), y.copy()
        changed_H[heldout] = rng.normal(scale=50, size=changed_H[heldout].shape)
        changed_H1[heldout] = rng.normal(scale=100, size=changed_H1[heldout].shape)
        changed_y[heldout] = rng.integers(W.shape[1], size=heldout.sum())
        changed = analyze_alignment(changed_H, changed_H1, W, W, changed_y, train, heldout)
        for name in initial.transforms:
            np.testing.assert_array_equal(initial.transforms[name], changed.transforms[name])
        self.assertEqual(initial.report["selection"], changed.report["selection"])

    def test_masks_prevent_leakage(self):
        _, H, W, y, train, heldout = make_case()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            analyze_alignment(H, H, W, W, y, train, train)
        with self.assertRaisesRegex(ValueError, "subset"):
            analyze_alignment(H, H, W, W, y, train, heldout, inner_validation_mask=heldout)
        with self.assertRaisesRegex(ValueError, "boolean"):
            analyze_alignment(H, H, W, W, y, np.flatnonzero(train), heldout)

    def test_rank_deficient_zero_source_stays_finite(self):
        rng = np.random.default_rng(4)
        source = np.zeros((20, 5))
        target = rng.normal(size=(20, 3))
        for penalty in (0.0, 1e-4, 1.0):
            np.testing.assert_array_equal(fit_ridge_map(source, target, penalty), np.zeros((5, 3)))

    def test_regularization_changes_ill_conditioned_fit(self):
        rng = np.random.default_rng(3)
        source = rng.normal(size=(100, 4))
        source[:, -1] *= 1e-8
        target = source[:, [-1]] * 1e8
        unregularized = fit_ridge_map(source, target, 0.0)
        regularized = fit_ridge_map(source, target, 1e-4)
        self.assertGreater(np.linalg.norm(unregularized), 1e7)
        self.assertLess(np.linalg.norm(regularized), 1)

    def test_metrics_handle_offsets_ties_and_large_logits(self):
        logits = np.array([[10000.0, 9998.0, 9999.0], [-9999.0, -9997.0, -9998.0], [5.0, 5.0, 4.0]])
        labels = np.array([0, 2, 1])
        metrics = classification_metrics(logits, labels)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["margin_mean"], 0.0)
        self.assertTrue(np.isfinite(metrics["cross_entropy"]))
        offset = logits + np.array([[1e4], [-5e3], [18.0]])
        comparison = logit_comparison_metrics(offset, logits)
        self.assertLess(comparison["class_centered_relative_error"], 1e-10)
        self.assertEqual(comparison["prediction_agreement"], 1.0)

    def test_centered_metrics_expose_constant_dominance(self):
        target = np.array([[1000.0, 1001.0], [1002.0, 999.0], [998.0, 1000.0]])
        prediction = np.broadcast_to(target.mean(axis=0), target.shape)
        metrics = reconstruction_metrics(prediction, target, prediction_train_mean=prediction.mean(axis=0), target_train_mean=target.mean(axis=0))
        self.assertLess(metrics["relative_error"], 0.01)
        self.assertAlmostEqual(metrics["centered_relative_error"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
