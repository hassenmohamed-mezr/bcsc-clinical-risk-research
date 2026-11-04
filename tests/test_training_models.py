"""Tests for the four-model training runner's pure helper functions.

These tests use synthetic data only. They do not load project splits, do not fit
project models, and never reference the Test split.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.training import train_models as tm


class CandidateGridTests(unittest.TestCase):
    def test_predeclared_grid_sizes_are_within_the_cap(self) -> None:
        expected = {
            "logistic_regression": 4,
            "random_forest": 8,
            "gradient_boosting": 24,
            "mlp": 18,
        }
        for model_key, size in expected.items():
            grid = tm._candidate_grid(model_key)
            self.assertEqual(len(grid), size)
            self.assertLessEqual(len(grid), 24)

    def test_every_candidate_contains_the_declared_parameters(self) -> None:
        for model_key in tm.MODEL_ORDER:
            grid = tm._candidate_grid(model_key)
            expected_names = set(tm.TUNING_SPACES[model_key])
            for candidate in grid:
                self.assertEqual(set(candidate), expected_names)


class ParameterNamespaceTests(unittest.TestCase):
    def test_plain_estimators_use_direct_parameters(self) -> None:
        prefixed = tm._prefixed_model_params("random_forest", {"n_estimators": 300})
        self.assertEqual(prefixed, {"n_estimators": 300})

    def test_mlp_parameters_go_to_the_classifier_step(self) -> None:
        prefixed = tm._prefixed_model_params("mlp", {"max_iter": 400, "alpha": 0.001})
        self.assertEqual(
            prefixed,
            {"classifier__max_iter": 400, "classifier__alpha": 0.001},
        )

    def test_calibration_prefix_wraps_the_estimator(self) -> None:
        prefixed = tm._prefixed_calibration_params("mlp", {"max_iter": 400})
        self.assertEqual(prefixed, {"estimator__classifier__max_iter": 400})

    def test_candidate_pipeline_is_unfitted_and_configured(self) -> None:
        pipeline = tm._build_candidate_pipeline(
            "mlp", {"hidden_layer_sizes": (32,), "alpha": 0.001, "max_iter": 200}
        )
        self.assertFalse(hasattr(pipeline, "classes_"))
        params = pipeline.get_params()
        self.assertEqual(params["classifier__hidden_layer_sizes"], (32,))
        self.assertEqual(params["classifier__alpha"], 0.001)
        self.assertEqual(params["classifier__max_iter"], 200)
        self.assertFalse(params["classifier__early_stopping"])


class SelectionTests(unittest.TestCase):
    def _summary(self, candidate_id, pr_auc, roc_auc, neg_brier):
        return {
            "candidate_id": candidate_id,
            "mean_pr_auc": pr_auc,
            "mean_roc_auc": roc_auc,
            "mean_neg_brier": neg_brier,
        }

    def test_selection_uses_pr_auc_first(self) -> None:
        summaries = [
            self._summary("a", 0.010, 0.70, -0.004),
            self._summary("b", 0.020, 0.60, -0.005),
        ]
        self.assertEqual(tm.select_best(summaries)["candidate_id"], "b")

    def test_ties_break_on_roc_auc_then_brier(self) -> None:
        summaries = [
            self._summary("a", 0.010, 0.65, -0.0040),
            self._summary("b", 0.010, 0.66, -0.0041),
            self._summary("c", 0.010, 0.66, -0.0042),
        ]
        # b and c tie on PR-AUC/ROC-AUC; b has the higher (better) negative Brier.
        self.assertEqual(tm.select_best(summaries)["candidate_id"], "b")


class CalibrationTableTests(unittest.TestCase):
    def test_perfectly_calibrated_constant_predictions_have_zero_gap(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=int)
        p = np.full(4, 0.5)
        w = np.ones(4, dtype=float)
        table, metrics = tm.calibration_table(y, p, w)
        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(metrics["mean_predicted"], 0.5)
        self.assertAlmostEqual(metrics["observed_rate"], 0.5)
        self.assertAlmostEqual(metrics["ece"], 0.0)
        self.assertAlmostEqual(metrics["mce"], 0.0)

    def test_ece_is_weighted(self) -> None:
        y = np.array([0, 1], dtype=int)
        p = np.array([0.0, 1.0])
        w = np.array([1.0, 9.0])
        table, metrics = tm.calibration_table(y, p, w)
        self.assertAlmostEqual(metrics["ece"], 0.0)
        self.assertEqual(int(table["rows"].sum()), 2)


if __name__ == "__main__":
    unittest.main()
