"""Read-only tests for the unified training configuration.

No model is fitted, no encoder is refitted, and the Test split is never loaded.
"""
from __future__ import annotations

import unittest

from src.config import N_JOBS, RANDOM_STATE
from src.training import config


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

    def test_mlp_uses_training_only_standard_scaler(self) -> None:
        scaler = config.build_scaler("mlp")
        self.assertIsNotNone(scaler)
        self.assertEqual(type(scaler).__name__, "StandardScaler")
        self.assertIn("Training", config.SCALERS["standard"].fit_scope)
        self.assertIsNone(config.build_scaler("logistic_regression"))

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
        self.assertGreaterEqual(result["validation"]["checks_passed"], 11)


if __name__ == "__main__":
    unittest.main()
