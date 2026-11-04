"""Configuration for the final BCSC inference system (SUPMT).

This file is intentionally separate from src/training/config.py.  It contains
only runtime/inference settings and must not be used for training or tuning.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Artifact locations.
MODEL_DIR = PROJECT_ROOT / "models" / "trained"
PREPROCESSING_DIR = PROJECT_ROOT / "models" / "preprocessing"

# Independent model weights.  Edit these values here only; do not edit SUPMT.py.
# All weights must be finite, non-negative, and not all zero.
MODEL_WEIGHTS = {
    "logistic_regression": 1.0,
    "gradient_boosting": 1.0,
    "mlp": 1.0,
    "random_forest": 1.0,
}

# Expected library version recorded in the saved training metadata.
# A mismatch produces a warning, not a silent failure.
EXPECTED_SKLEARN_VERSION = "1.9.1"

# Model artifact file names (verified against models/trained).
MODEL_FILES = {
    "logistic_regression": "logistic_regression_calibrated.joblib",
    "gradient_boosting": "gradient_boosting_calibrated.joblib",
    "mlp": "mlp_calibrated.joblib",
    "random_forest": "random_forest_calibrated.joblib",
}

MODEL_METADATA_FILES = {
    "logistic_regression": "logistic_regression_metadata.json",
    "gradient_boosting": "gradient_boosting_metadata.json",
    "mlp": "mlp_metadata.json",
    "random_forest": "random_forest_metadata.json",
}