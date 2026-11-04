"""Read-only tests for the unified training configuration.

No project model is fitted, no encoder is refitted, and the Test split is never
loaded. The MLP/calibration tests construct and serialize unfitted artifacts.
"""
from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from src.config import N_JOBS, RANDOM_STATE
from src.preprocessing.pipeline import load_pipeline
from src.training import config
from src.training.cv import WeightedStratifiedKFold
from src.training.pipeline import (
    SampleWeightPipeline,
    load_model_artifact,
    save_model_artifact,
)


class TrainingConfigTests(unittest.TestCase):
    def test_features_never_include_weight_or_target(self) -> None:
        self.assertNotIn(config.WEIGHT_COLUMN, config.FEATURE_ORDER)
        self.assertNotIn(config.TARGET_COLUMN, config.FEATURE_ORDER)
        self.assertFalse(config.FORBIDDEN_MODEL_COLUMNS.intersection(config.FEATURE_ORDER))

    def test_required_models_and_encoder_widths(self) -> None:
        for key in config.REQUIRED_MODEL_KEYS:
            self.assertIn(key, config.MODEL_SPECS)
            spec = config.MODEL_SPECS[key]
            info = config.encoder_info(spec.encoder)
            self.assertEqual(info["n_features_in"], len(config.FEATURE_ORDER))

        self.assertEqual(config.encoder_info("onehot")["n_features_out"], 43)
        self.assertEqual(config.encoder_info("ordinal_tree")["n_features_out"], 26)

    def test_estimators_construct_with_central_seed(self) -> None:
        for key in config.REQUIRED_MODEL_KEYS:
            estimator = config.build_estimator(key)
            params = estimator.get_params()
            self.assertEqual(params["random_state"], RANDOM_STATE)
            self.assertTrue(hasattr(estimator, "predict_proba"))
            self.assertFalse(hasattr(estimator, "classes_"))
            if config.MODEL_SPECS[key].supports_n_jobs:
                self.assertEqual(params["n_jobs"], N_JOBS)

    def test_model_artifacts_are_unfitted_and_accept_sample_weight(self) -> None:
        for key in config.REQUIRED_MODEL_KEYS:
            model = config.build_model_pipeline(key)
            self.assertIn("sample_weight", inspect.signature(model.fit).parameters)
            self.assertTrue(hasattr(model, "predict_proba"))
            self.assertFalse(hasattr(model, "classes_"))
            clone(model)  # sklearn-compatible and serializable structure

    def test_mlp_uses_training_only_standard_scaler(self) -> None:
        scaler = config.build_scaler("mlp")
        self.assertIsNotNone(scaler)
        self.assertEqual(type(scaler).__name__, "StandardScaler")
        self.assertIn("CV fold", config.SCALERS["standard"].fit_scope)
        self.assertIn("fold's Training rows", config.SCALERS["standard"].fit_scope)
        self.assertIsNone(config.build_scaler("logistic_regression"))

    def test_mlp_pipeline_bundles_scaler_and_disables_early_stopping(self) -> None:
        pipeline = config.build_model_pipeline("mlp")
        self.assertIsInstance(pipeline, SampleWeightPipeline)
        self.assertEqual([name for name, _ in pipeline.steps], ["scaler", "classifier"])

        scaler = pipeline.named_steps["scaler"]
        classifier = pipeline.named_steps["classifier"]
        self.assertEqual(type(scaler).__name__, "StandardScaler")
        self.assertFalse(hasattr(scaler, "mean_"))
        self.assertEqual(type(classifier).__name__, "MLPClassifier")

        params = classifier.get_params()
        self.assertFalse(params["early_stopping"])
        self.assertNotIn("validation_fraction", config.MODEL_SPECS["mlp"].params)
        self.assertNotIn("n_iter_no_change", config.MODEL_SPECS["mlp"].params)
        self.assertIn("max_iter", config.MODEL_SPECS["mlp"].params)
        self.assertIn("max_iter", config.TUNING_SPACES["mlp"])

    def test_mlp_pipeline_serializes_scaler_and_model_together(self) -> None:
        pipeline = config.build_model_pipeline("mlp")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mlp_model.joblib"
            saved = save_model_artifact(pipeline, path)
            self.assertTrue(saved.is_file())

            loaded = load_model_artifact(saved)
            self.assertIsInstance(loaded, SampleWeightPipeline)
            self.assertEqual(
                [name for name, _ in loaded.steps], ["scaler", "classifier"]
            )
            self.assertEqual(
                type(loaded.named_steps["scaler"]).__name__, "StandardScaler"
            )
            self.assertEqual(
                type(loaded.named_steps["classifier"]).__name__, "MLPClassifier"
            )
            self.assertFalse(
                loaded.named_steps["classifier"].get_params()["early_stopping"]
            )

    def test_calibration_is_training_only_weighted_and_sigmoid(self) -> None:
        policy = config.CALIBRATION_POLICY
        self.assertTrue(policy["required"])
        self.assertEqual(policy["method"], "sigmoid")
        self.assertFalse(policy["ensemble"])
        self.assertEqual(policy["data"], "Training only")
        self.assertEqual(policy["test_use"], "forbidden")
        self.assertIn("never", policy["validation_use"])
        self.assertIn(
            "sample_weight", inspect.signature(CalibratedClassifierCV.fit).parameters
        )

        calibrated = config.build_calibrated_model(
            "mlp", cv=config.CV_POLICY["n_splits"]
        )
        self.assertIsInstance(calibrated, CalibratedClassifierCV)
        params = calibrated.get_params()
        self.assertEqual(params["method"], "sigmoid")
        self.assertFalse(params["ensemble"])
        self.assertIsInstance(params["estimator"], SampleWeightPipeline)
        self.assertFalse(hasattr(calibrated, "classes_"))

        with self.assertRaises(ValueError):
            config.build_calibrated_model("mlp", cv=None)

    def test_training_calibration_helper_uses_weighted_training_folds(self) -> None:
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        w = np.ones(len(y), dtype=float)

        calibrated = config.build_training_calibrated_model("mlp", y, w)
        splitter = calibrated.get_params()["cv"]
        self.assertIsInstance(splitter, WeightedStratifiedKFold)
        self.assertEqual(splitter.get_n_splits(), config.CV_POLICY["n_splits"])

        for forbidden_split in ("val", "test"):
            with self.assertRaises(ValueError):
                config.build_training_calibrated_model(
                    "mlp", y, w, split=forbidden_split
                )
        with self.assertRaises(ValueError):
            config.build_training_calibrated_model("mlp", y, w[:-1])

    def test_code_9_remains_one_category_for_surgmeno_and_hrt(self) -> None:
        policy = config.CODE_9_POLICY
        self.assertEqual(policy["treatment"], "single category")
        self.assertFalse(policy["split_into_unknown_and_not_applicable"])
        self.assertFalse(policy["raw_data_change"])
        self.assertFalse(policy["mapping_change"])

        for family in ("onehot", "ordinal_tree"):
            names = config.encoder_info(family)["feature_names"]
            for feature in ("surgmeno", "hrt"):
                self.assertEqual(names.count(f"{feature}_9"), 1)
                invented = [
                    name
                    for name in names
                    if name.startswith(f"{feature}_") and "unknown" in name.lower()
                ]
                self.assertEqual(invented, [])

        # The saved one-hot encoder also carries a single category 9.
        onehot = load_pipeline(config.ENCODERS["onehot"].artifact)
        categories = onehot.named_transformers_["categorical"].categories_
        for feature in ("surgmeno", "hrt"):
            feature_position = config.FEATURE_ORDER.index(feature)
            self.assertEqual(list(categories[feature_position]).count(9), 1)

    def test_test_split_is_not_exposed(self) -> None:
        self.assertNotIn("test", config.SPLIT_REFERENCES)
        with self.assertRaises(ValueError):
            config.load_encoded_split("mlp", "test")

    def test_encoded_matrix_matches_model_encoder(self) -> None:
        for key, expected_width in (
            ("logistic_regression", 43),
            ("random_forest", 26),
            ("gradient_boosting", 26),
            ("mlp", 43),
        ):
            Z, y, w = config.load_encoded_split(key, config.VALIDATION_SPLIT)
            self.assertEqual(Z.shape[1], expected_width)
            self.assertEqual(Z.shape[0], len(y))
            self.assertEqual(Z.shape[0], len(w))

    def test_full_configuration_validates(self) -> None:
        result = config.validate_config(check_artifacts=True, check_split_data=False)
        self.assertFalse(result["validation"]["test_accessed"])
        self.assertFalse(result["validation"]["models_trained"])
        self.assertGreaterEqual(result["validation"]["checks_passed"], 14)
        self.assertEqual(result["code_9"]["treatment"], "single category")
        self.assertEqual(result["cross_validation"]["primary_scoring"], "pr_auc")
        self.assertEqual(result["calibration"]["method"], "sigmoid")
        self.assertEqual(
            result["models"]["mlp"]["pipeline_steps"], ["scaler", "classifier"]
        )
        self.assertFalse(
            result["models"]["mlp"]["resolved_params"]["early_stopping"]
        )


if __name__ == "__main__":
    unittest.main()
