"""Tests for the Training-only weighted CV utilities.

All data are synthetic. The only estimator used here is a non-training spy that
records fit arguments and returns fixed probabilities; no project model is fitted
and no BCSC split is loaded.
"""
from __future__ import annotations

import unittest

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.training import cv
from src.training.pipeline import SampleWeightPipeline


class WeightedFoldTests(unittest.TestCase):
    def _synthetic(self, n_negative: int = 160, n_positive: int = 40):
        rng = np.random.default_rng(7)
        y = np.array([0] * n_negative + [1] * n_positive, dtype=int)
        w = rng.integers(1, 20, size=len(y)).astype(float)
        return y, w

    def test_folds_cover_every_row_once_without_expansion(self) -> None:
        y, w = self._synthetic()
        folds = cv.weighted_folds(y, w, n_splits=5, random_state=42)
        self.assertEqual(len(folds), 5)

        covered = np.concatenate([fold.validation_index for fold in folds])
        self.assertEqual(sorted(covered.tolist()), list(range(len(y))))
        self.assertEqual(int(w[covered].sum()), int(w.sum()))

        for fold in folds:
            self.assertEqual(
                len(np.intersect1d(fold.train_index, fold.validation_index)), 0
            )
            self.assertEqual(
                len(fold.train_index) + len(fold.validation_index), len(y)
            )
            self.assertGreater(len(fold.validation_index), 0)
            self.assertEqual(set(y[fold.validation_index].tolist()), {0, 1})

    def test_fold_diagnostics_report_weights_and_weighted_prevalence(self) -> None:
        y, w = self._synthetic()
        folds = cv.weighted_folds(y, w, n_splits=5, random_state=42)
        diagnostics = cv.fold_diagnostics(y, w, folds)
        self.assertEqual(len(diagnostics), 5)

        total_weight = float(w.sum())
        for fold, diag in zip(folds, diagnostics):
            idx = fold.validation_index
            self.assertEqual(diag.rows, len(idx))
            self.assertAlmostEqual(diag.weight_sum, float(w[idx].sum()))
            self.assertAlmostEqual(
                diag.positive_weight, float(w[idx][y[idx] == 1].sum())
            )
            self.assertAlmostEqual(
                diag.weighted_prevalence,
                diag.positive_weight / diag.weight_sum,
            )
            self.assertAlmostEqual(diag.weight_share, diag.weight_sum / total_weight)

        balance = cv.assess_fold_balance(diagnostics)
        self.assertTrue(balance["within_tolerance"])
        self.assertFalse(balance["material_imbalance"])
        self.assertEqual(balance["violations"], [])

    def test_material_imbalance_is_flagged(self) -> None:
        y = np.array([0] * 10 + [1] * 10, dtype=int)
        w = np.ones(20, dtype=float)
        folds = [
            cv.WeightedFold(0, np.arange(5, 20), np.arange(0, 5)),
            cv.WeightedFold(1, np.arange(0, 10), np.arange(10, 15)),
        ]
        diagnostics = cv.fold_diagnostics(y, w, folds)
        balance = cv.assess_fold_balance(diagnostics)
        self.assertTrue(balance["material_imbalance"])
        self.assertFalse(balance["within_tolerance"])
        self.assertTrue(balance["violations"])

    def test_sklearn_compatible_splitter(self) -> None:
        y, w = self._synthetic()
        splitter = cv.WeightedStratifiedKFold(
            n_splits=5, sample_weight=w, random_state=42
        )
        self.assertEqual(splitter.get_n_splits(), 5)
        splits = list(splitter.split(np.zeros((len(y), 1)), y))
        self.assertEqual(len(splits), 5)
        covered = np.concatenate([validation for _, validation in splits])
        self.assertEqual(sorted(covered.tolist()), list(range(len(y))))


class WeightedScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.y = np.array([0, 0, 1, 1], dtype=int)
        self.w = np.array([1.0, 1.0, 10.0, 10.0])
        self.p = np.array([0.1, 0.8, 0.7, 0.2])

    def test_weighted_metrics_match_sklearn_and_weighted_confusion(self) -> None:
        metrics = cv.weighted_metrics(self.y, self.p, self.w, threshold=0.5)
        self.assertEqual(metrics["tp_w"], 10.0)
        self.assertEqual(metrics["fp_w"], 1.0)
        self.assertEqual(metrics["fn_w"], 10.0)
        self.assertEqual(metrics["tn_w"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 10.0 / 11.0)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["weighted_prevalence"], 20.0 / 22.0)

        self.assertAlmostEqual(
            metrics["pr_auc"],
            average_precision_score(self.y, self.p, sample_weight=self.w),
        )
        self.assertAlmostEqual(
            metrics["roc_auc"],
            roc_auc_score(self.y, self.p, sample_weight=self.w),
        )
        self.assertAlmostEqual(
            metrics["brier"],
            brier_score_loss(self.y, self.p, sample_weight=self.w),
        )

    def test_weighted_scores_use_count_and_pr_auc_is_primary(self) -> None:
        self.assertEqual(cv.PRIMARY_SCORER, "pr_auc")
        scores = cv.weighted_scores(self.y, self.p, self.w)
        self.assertAlmostEqual(
            scores["pr_auc"],
            average_precision_score(self.y, self.p, sample_weight=self.w),
        )
        self.assertAlmostEqual(
            scores["neg_brier"],
            -brier_score_loss(self.y, self.p, sample_weight=self.w),
        )

    def test_unweighted_diagnostic_is_clearly_secondary(self) -> None:
        weighted = cv.weighted_metrics(self.y, self.p, self.w)
        row_equal = cv.row_equal_metrics(self.y, self.p)
        self.assertNotAlmostEqual(weighted["pr_auc"], row_equal["pr_auc"])
        self.assertEqual(row_equal["weight_sum"], float(len(self.y)))
        self.assertEqual(row_equal["rows"], len(self.y))


class _SpyModel(BaseEstimator):
    """Non-training double: records fit weights and returns fixed probabilities."""

    classes_ = np.array([0, 1])

    def __init__(self) -> None:
        self.fit_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def fit(self, X, y, sample_weight=None):
        self.fit_calls.append(
            (
                np.asarray(X).copy(),
                np.asarray(y).copy(),
                np.asarray(sample_weight).copy(),
            )
        )
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.linspace(0.1, 0.9, n)
        return np.column_stack([1.0 - p, p])


class WeightedCrossValidationTests(unittest.TestCase):
    def test_weights_reach_both_fit_and_weighted_scoring(self) -> None:
        rng = np.random.default_rng(11)
        n = 40
        y = np.array([0] * 30 + [1] * 10, dtype=int)
        w = rng.integers(1, 8, size=n).astype(float)
        X = np.arange(n * 3, dtype=float).reshape(n, 3)
        folds = cv.weighted_folds(y, w, n_splits=2, random_state=42)

        spies: list[_SpyModel] = []

        def factory() -> _SpyModel:
            spy = _SpyModel()
            spies.append(spy)
            return spy

        results = cv.cross_validate_weighted(
            factory, X, y, w, folds=folds, threshold=0.5
        )
        self.assertEqual(len(results), len(folds))
        self.assertEqual(len(spies), len(folds))

        for fold, result, spy in zip(folds, results, spies):
            train_idx = fold.train_index
            validation_idx = fold.validation_index
            self.assertEqual(len(spy.fit_calls), 1)
            fit_X, fit_y, fit_w = spy.fit_calls[0]
            np.testing.assert_array_equal(fit_X, X[train_idx])
            np.testing.assert_array_equal(fit_y, y[train_idx])
            np.testing.assert_array_equal(fit_w, w[train_idx])

            self.assertEqual(result.validation_rows, len(validation_idx))
            self.assertAlmostEqual(
                result.validation_weight_sum, float(w[validation_idx].sum())
            )
            expected_p = np.linspace(0.1, 0.9, len(validation_idx))
            expected = cv.weighted_metrics(
                y[validation_idx], expected_p, w[validation_idx]
            )
            self.assertAlmostEqual(result.metrics["pr_auc"], expected["pr_auc"])
            self.assertAlmostEqual(result.metrics["brier"], expected["brier"])

    def test_scaler_is_fitted_on_training_fold_only(self) -> None:
        rng = np.random.default_rng(19)
        n = 40
        y = np.array([0] * 30 + [1] * 10, dtype=int)
        w = rng.integers(1, 8, size=n).astype(float)
        X = rng.normal(size=(n, 4))
        folds = cv.weighted_folds(y, w, n_splits=2, random_state=42)

        pipelines: list[SampleWeightPipeline] = []

        def factory() -> SampleWeightPipeline:
            pipeline = SampleWeightPipeline(
                [("scaler", StandardScaler()), ("classifier", _SpyModel())]
            )
            pipelines.append(pipeline)
            return pipeline

        cv.cross_validate_weighted(factory, X, y, w, folds=folds)
        self.assertEqual(len(pipelines), len(folds))

        for fold, pipeline in zip(folds, pipelines):
            scaler = pipeline.named_steps["scaler"]
            # Fitted on the fold's training rows, exactly.
            np.testing.assert_allclose(
                scaler.mean_, X[fold.train_index].mean(axis=0)
            )
            np.testing.assert_allclose(
                scaler.var_, X[fold.train_index].var(axis=0)
            )
            # And not on the validation rows.
            if len(fold.validation_index) > 1:
                self.assertFalse(
                    np.allclose(scaler.mean_, X[fold.validation_index].mean(axis=0))
                )


if __name__ == "__main__":
    unittest.main()
