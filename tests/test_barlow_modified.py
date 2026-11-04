"""Tests for the Modified Barlow Architecture experiment.

All checks use small in-memory synthetic data or the frozen Train/Val *headers*
only. No project model is trained here; no split is written; ``test.csv`` is
never loaded.

Run: python -m unittest tests.test_barlow_modified -v
"""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from outputs.experiments.barlow_2006_modified import barlow_common as bc


def _synthetic(n: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=int),
            "menopaus": np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=int),
            "agegrp": rng.integers(1, 11, size=n),
            "brstproc": rng.choice([0, 1, 9], size=n),
            "nrelbc": rng.choice([0, 1, 2, 9], size=n),
            "density": rng.choice([1, 2, 3, 4, 9], size=n),
            "bmi": rng.choice([1, 2, 3, 4, 9], size=n),
            "agefirst": rng.choice([0, 1, 2, 9], size=n),
            "hrt": rng.choice([0, 1, 9], size=n),
            "surgmeno": rng.choice([0, 1, 9], size=n),
            "lastmamm": rng.choice([0, 1, 9], size=n),
            "cancer": np.array([0] * (n - 10) + [1] * 10, dtype=int),
            "count": rng.integers(1, 9, size=n).astype(float),
        }
    )


class BarlowFeatureTests(unittest.TestCase):
    def test_premenopausal_features_exact(self) -> None:
        self.assertEqual(
            bc.PRE_FEATURES, ["agegrp", "brstproc", "nrelbc", "density"]
        )

    def test_postmenopausal_features_exact(self) -> None:
        self.assertEqual(
            bc.POST_FEATURES,
            ["agegrp", "bmi", "agefirst", "brstproc", "nrelbc", "hrt",
             "surgmeno", "lastmamm", "density"],
        )

    def test_no_forbidden_features_in_lists(self) -> None:
        for features in (bc.PRE_FEATURES, bc.POST_FEATURES):
            for forbidden in ("race", "Hispanic", "menopaus", "count", "cancer",
                              "training", "invasive"):
                self.assertNotIn(forbidden, features)

    def test_menopa_is_routing_only_not_a_feature(self) -> None:
        self.assertNotIn("menopaus", bc.PRE_FEATURES)
        self.assertNotIn("menopaus", bc.POST_FEATURES)

    def test_validate_feature_list_rejects_forbidden(self) -> None:
        with self.assertRaises(ValueError):
            bc.validate_feature_list(["race"])
        with self.assertRaises(ValueError):
            bc.validate_feature_list(["menopaus"])
        with self.assertRaises(ValueError):
            bc.validate_feature_list(["count"])

    def test_no_class_weighting_no_penalty(self) -> None:
        self.assertIsNone(bc.MODEL_PARAMS["penalty"])
        self.assertIsNone(bc.MODEL_PARAMS["class_weight"])
        model = bc.build_lr()
        self.assertEqual(model.class_weight, None)
        # sklearn >=1.8 deprecates penalty=None (default L2); we must assert
        # the estimator is constructed unpenalized, not defaulted to L2.
        if hasattr(model, "penalty"):
            self.assertIsNone(model.penalty)

    def test_reference_categories_are_first_codebook_levels(self) -> None:
        for feature, order in bc.CATEGORY_ORDERS.items():
            self.assertEqual(bc.REFERENCE_CATEGORIES[feature], order[0])
        self.assertEqual(bc.REFERENCE_CATEGORIES["agegrp"], 1)
        self.assertEqual(bc.REFERENCE_CATEGORIES["density"], 1)
        self.assertEqual(bc.REFERENCE_CATEGORIES["nrelbc"], 0)
        self.assertEqual(bc.REFERENCE_CATEGORIES["brstproc"], 0)

    def test_encoder_reference_first_shape(self) -> None:
        df = _synthetic()
        encoder = bc.build_encoder(bc.PRE_FEATURES)
        Z = encoder.fit_transform(df[bc.PRE_FEATURES].to_numpy())
        expected_cols = sum(len(bc.CATEGORY_ORDERS[f]) - 1 for f in bc.PRE_FEATURES)
        self.assertEqual(Z.shape[1], expected_cols)


class BarlowDataTests(unittest.TestCase):
    def test_menopa9_excluded_from_both_subgroups(self) -> None:
        df = _synthetic()
        df.loc[df.index[:5], "menopaus"] = 9
        pre = bc.get_subgroup(df, 0)
        post = bc.get_subgroup(df, 1)
        self.assertTrue((pre["menopaus"] == 0).all())
        self.assertTrue((post["menopaus"] == 1).all())
        excluded = bc.excluded_unknown(df)
        self.assertTrue((excluded["menopaus"] == 9).all())
        self.assertEqual(len(pre) + len(post) + len(excluded), len(df))

    def test_subgroup_rows_preserve_original_order(self) -> None:
        df = _synthetic(n=50)
        pre = bc.get_subgroup(df, 0)
        self.assertTrue(pre["row_id"].is_monotonic_increasing)

    def test_no_row_expansion_and_weights_unchanged(self) -> None:
        df = _synthetic()
        pre = bc.get_subgroup(df, 0)
        self.assertEqual(len(pre), int((df["menopaus"] == 0).sum()))
        self.assertAlmostEqual(
            float(pre["count"].sum()),
            float(df.loc[df["menopaus"] == 0, "count"].sum()),
        )

    def test_split_loader_rejects_non_train_val(self) -> None:
        with self.assertRaises(ValueError):
            bc.load_split_frame("test")
        with self.assertRaises(ValueError):
            bc.load_split_frame("validation")

    def test_no_test_split_reference_in_experiment_code(self) -> None:
        root = Path(__file__).resolve().parent.parent
        for name in ("train_evaluate.py", "barlow_common.py"):
            source = (root / "outputs" / "experiments" / "barlow_2006_modified"
                      / name).read_text(encoding="utf-8")
            lowered = source.lower()
            # No code path may open a test split.
            self.assertNotIn("load_split_frame(\"test\")", source)
            self.assertNotIn("load_split('test')", source)
            self.assertNotIn('04_splits/test', source)

    def test_weighted_metrics_with_reference_baseline(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=int)
        p = np.array([0.1, 0.8, 0.7, 0.2])
        w = np.array([1.0, 1.0, 10.0, 10.0])
        metrics = bc.compute_metrics(y, p, w, train_prevalence=0.99)
        self.assertAlmostEqual(metrics["weighted_prevalence"], 20.0 / 22.0)
        self.assertIn("brier_baseline", metrics)
        self.assertIn("log_loss", metrics)


def _train_sub_vs_val_sub_leakage_check() -> None:
    """Guard: fit on train subgroup, evaluate on val — no val data in fit inputs."""
    train = _synthetic(n=60, seed=1)
    val = _synthetic(n=30, seed=2)
    encoder = bc.build_encoder(bc.PRE_FEATURES)
    Z_train_sub = encoder.fit_transform(
        bc.get_subgroup(train, 0)[bc.PRE_FEATURES].to_numpy()
    )
    fit_source = inspect.getsource(bc.build_lr)
    assert "val" not in fit_source.lower()  # factory has no data access
    assert Z_train_sub.shape[0] == len(bc.get_subgroup(train, 0))
    assert len(bc.get_subgroup(val, 0)) > 0


if __name__ == "__main__":
    _train_sub_vs_val_sub_leakage_check()
    unittest.main()
