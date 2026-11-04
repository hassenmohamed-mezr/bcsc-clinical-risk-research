"""Final inference system for BCSC risk estimation.

Run:
    python SUPMT.py

Programmatic use:
    from SUPMT import predict_patient

    result = predict_patient({
        "age": 42,
        "bmi": 27.5,
        "has_given_birth": True,
        "age_at_first_birth": 26,
        "is_postmenopausal": False,
        "first_degree_relatives_with_breast_cancer": 1,
        "breast_density": "heterogeneously dense",
        "prior_breast_procedure": False,
        "last_mammogram_result": "negative",
    })

This module reuses the existing preprocessing, mapping, encoders, and trained
calibrated models.  It does not train, refit, or modify any artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import SUPMT_CONFIG as CONFIG
from src.data.load_splits import FEATURES
from src.preprocessing.pipeline import load_pipeline
from src.preprocessing.raw_input_mapping import (
    HUMAN_FIELDS,
    InputMappingError,
    RawInputMapper,
    human_to_interface_frame,
)
from src.training.cv import positive_class_probability
from src.training.pipeline import load_model_artifact

MODEL_ORDER = (
    "logistic_regression",
    "gradient_boosting",
    "mlp",
    "random_forest",
)

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "MLP",
    "random_forest": "Random Forest",
}

# Encoder family required by each trained model (verified from metadata).
REQUIRED_ENCODER = {
    "logistic_regression": "onehot",
    "gradient_boosting": "ordinal_tree",
    "mlp": "onehot",
    "random_forest": "ordinal_tree",
}

_SYSTEM: dict[str, Any] | None = None


class SUPMTError(RuntimeError):
    """Raised when the inference system cannot be initialized or run."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_weights(weights: Mapping[str, Any]) -> dict[str, float]:
    missing = [key for key in MODEL_ORDER if key not in weights]
    extra = [key for key in weights if key not in MODEL_ORDER]
    if missing or extra:
        raise SUPMTError(
            f"MODEL_WEIGHTS must contain exactly {list(MODEL_ORDER)}; "
            f"missing={missing}, extra={extra}"
        )

    validated: dict[str, float] = {}
    for key in MODEL_ORDER:
        value = weights[key]
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SUPMTError(f"MODEL_WEIGHTS[{key!r}] must be a number") from exc
        if not np.isfinite(number):
            raise SUPMTError(f"MODEL_WEIGHTS[{key!r}] must be finite")
        if number < 0:
            raise SUPMTError(f"MODEL_WEIGHTS[{key!r}] must not be negative")
        validated[key] = number

    total = sum(validated.values())
    if not np.isfinite(total) or total <= 0:
        raise SUPMTError("sum of MODEL_WEIGHTS must be greater than zero")
    return validated


def _load_system() -> dict[str, Any]:
    global _SYSTEM
    if _SYSTEM is not None:
        return _SYSTEM

    try:
        mapper = RawInputMapper.load(CONFIG.PREPROCESSING_DIR / "raw_input_mapping.json")
    except Exception as exc:  # noqa: BLE001 - preserve and annotate load failures
        raise SUPMTError(
            f"could not load raw input mapping: {CONFIG.PREPROCESSING_DIR / 'raw_input_mapping.json'}: {exc}"
        ) from exc

    models: dict[str, Any] = {}
    pipelines: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    for key in MODEL_ORDER:
        model_path = CONFIG.MODEL_DIR / CONFIG.MODEL_FILES[key]
        model_meta_path = CONFIG.MODEL_DIR / CONFIG.MODEL_METADATA_FILES[key]

        if not model_path.is_file():
            raise SUPMTError(f"missing model artifact: {model_path}")
        if not model_meta_path.is_file():
            raise SUPMTError(f"missing model metadata: {model_meta_path}")

        try:
            model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SUPMTError(f"could not read model metadata {model_meta_path}: {exc}") from exc
        metadata[key] = model_meta

        # Verify feature order from metadata.
        if list(model_meta.get("feature_order", [])) != list(FEATURES):
            raise SUPMTError(
                f"{key}: metadata feature_order does not match required FEATURES order"
            )

        # Load and verify the trained/calibrated model.
        try:
            model = load_model_artifact(model_path)
        except Exception as exc:  # noqa: BLE001
            raise SUPMTError(f"could not load model artifact {model_path}: {exc}") from exc

        if not hasattr(model, "classes_"):
            raise SUPMTError(f"{key}: loaded model has no classes_ attribute")
        classes = np.asarray(getattr(model, "classes_"))
        if not np.any(classes == 1):
            raise SUPMTError(f"{key}: model classes_ does not contain positive class 1")
        models[key] = model

        # Verify artifact hash when metadata provides it.
        expected_model_sha = model_meta.get("artifact_sha256")
        if expected_model_sha is not None:
            actual_model_sha = _sha256(model_path)
            if actual_model_sha != expected_model_sha:
                raise SUPMTError(
                    f"{key}: model artifact SHA-256 mismatch "
                    f"(expected {expected_model_sha}, got {actual_model_sha})"
                )

        # Resolve and load the preprocessing pipeline from metadata.
        encoder_family = model_meta.get("encoder")
        if encoder_family != REQUIRED_ENCODER[key]:
            raise SUPMTError(
                f"{key}: metadata encoder {encoder_family!r} does not match "
                f"required {REQUIRED_ENCODER[key]!r}"
            )
        if encoder_family not in ("onehot", "ordinal_tree"):
            raise SUPMTError(f"{key}: unknown encoder family {encoder_family!r}")

        encoder_artifact_raw = model_meta.get("encoder_artifact")
        if not encoder_artifact_raw:
            raise SUPMTError(f"{key}: metadata is missing encoder_artifact")
        encoder_path = Path(encoder_artifact_raw)
        if not encoder_path.is_absolute():
            encoder_path = PROJECT_ROOT / encoder_path
        if not encoder_path.is_file():
            raise SUPMTError(f"{key}: missing preprocessing artifact: {encoder_path}")

        expected_encoder_sha = model_meta.get("encoder_sha256")
        if expected_encoder_sha is not None:
            actual_encoder_sha = _sha256(encoder_path)
            if actual_encoder_sha != expected_encoder_sha:
                raise SUPMTError(
                    f"{key}: preprocessing artifact SHA-256 mismatch "
                    f"(expected {expected_encoder_sha}, got {actual_encoder_sha})"
                )

        try:
            pipeline = load_pipeline(encoder_path)
        except Exception as exc:  # noqa: BLE001
            raise SUPMTError(
                f"{key}: could not load preprocessing pipeline {encoder_path}: {exc}"
            ) from exc

        feature_names_in = getattr(pipeline, "feature_names_in_", None)
        if feature_names_in is None or list(feature_names_in) != list(FEATURES):
            raise SUPMTError(
                f"{key}: preprocessing pipeline feature order does not match FEATURES"
            )
        pipelines[key] = pipeline

        # Version compatibility check (warning, not silent).
        saved_version = model_meta.get("sklearn_version") or model_meta.get("sklearn")
        if saved_version is not None and saved_version != sklearn.__version__:
            warnings.warn(
                f"{key}: scikit-learn version mismatch "
                f"(saved {saved_version}, installed {sklearn.__version__})",
                RuntimeWarning,
                stacklevel=2,
            )

    weights = _validate_weights(CONFIG.MODEL_WEIGHTS)

    # Global version note from preprocessing metadata.
    preprocessing_meta_path = CONFIG.PREPROCESSING_DIR / "preprocessing_metadata.json"
    if preprocessing_meta_path.is_file():
        try:
            preprocessing_meta = json.loads(preprocessing_meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SUPMTError(
                f"could not read preprocessing metadata {preprocessing_meta_path}: {exc}"
            ) from exc
        saved_sklearn = preprocessing_meta.get("sklearn_version")
        if saved_sklearn is not None and saved_sklearn != sklearn.__version__:
            warnings.warn(
                f"preprocessing: scikit-learn version mismatch "
                f"(saved {saved_sklearn}, installed {sklearn.__version__})",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        preprocessing_meta = {}

    _SYSTEM = {
        "mapper": mapper,
        "models": models,
        "pipelines": pipelines,
        "metadata": metadata,
        "preprocessing_metadata": preprocessing_meta,
        "weights": weights,
    }
    return _SYSTEM


def get_system() -> dict[str, Any]:
    """Return the lazily loaded (and cached) inference system."""
    return _load_system()


def _normalize_patient(patient: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(patient, Mapping):
        raise InputMappingError(f"patient must be a mapping, got {type(patient).__name__}")
    return {field: patient.get(field, None) for field in HUMAN_FIELDS}


def predict_patient(patient: Mapping[str, Any]) -> dict[str, float]:
    """Validate, map, preprocess, predict, and print the five required probabilities."""
    system = get_system()
    normalized = _normalize_patient(patient)

    try:
        interface_frame = human_to_interface_frame(normalized)
    except InputMappingError as exc:
        raise InputMappingError(f"invalid patient input: {exc}") from exc

    mapper = system["mapper"]
    predictions: dict[str, float] = {}

    for key in MODEL_ORDER:
        pipeline = system["pipelines"][key]
        model = system["models"][key]

        try:
            from src.preprocessing.pipeline import encode_user_inputs

            encoded = encode_user_inputs(mapper, pipeline, interface_frame)
        except Exception as exc:  # noqa: BLE001
            raise SUPMTError(
                f"{key}: preprocessing failed: {exc}"
            ) from exc

        try:
            probabilities = positive_class_probability(model, encoded)
        except Exception as exc:  # noqa: BLE001
            raise SUPMTError(f"{key}: prediction failed: {exc}") from exc

        if probabilities.shape != (len(interface_frame),):
            # Single-patient inference is the supported use case.
            probabilities = np.ravel(probabilities)
        if len(probabilities) != 1:
            raise SUPMTError(
                f"{key}: expected one probability for one patient, got {len(probabilities)}"
            )
        value = float(probabilities[0])
        if not np.isfinite(value) or value < 0 or value > 1:
            raise SUPMTError(f"{key}: model returned an invalid probability: {value}")
        predictions[key] = value

    weights = system["weights"]
    weighted_numerator = sum(predictions[key] * weights[key] for key in MODEL_ORDER)
    weighted_denominator = sum(weights[key] for key in MODEL_ORDER)
    if weighted_denominator <= 0:
        raise SUPMTError("sum of MODEL_WEIGHTS must be greater than zero")
    weighted_average = weighted_numerator / weighted_denominator

    print(f"Logistic Regression: {predictions['logistic_regression'] * 100:.2f}%")
    print(f"Gradient Boosting: {predictions['gradient_boosting'] * 100:.2f}%")
    print(f"MLP: {predictions['mlp'] * 100:.2f}%")
    print(f"Random Forest: {predictions['random_forest'] * 100:.2f}%")
    print()
    print(f"Weighted Average: {weighted_average * 100:.2f}%")

    return {
        "logistic_regression": predictions["logistic_regression"],
        "gradient_boosting": predictions["gradient_boosting"],
        "mlp": predictions["mlp"],
        "random_forest": predictions["random_forest"],
        "weighted_average": weighted_average,
    }


def main() -> None:
    try:
        system = get_system()
    except Exception as exc:  # noqa: BLE001
        print(f"SUPMT initialization failed: {exc}", file=sys.stderr)
        raise

    print("SUPMT initialized successfully.")
    print(f"Loaded models: {', '.join(MODEL_ORDER)}")
    print(f"Feature order: {', '.join(FEATURES)}")
    print(f"Preprocessing artifacts: {CONFIG.PREPROCESSING_DIR}")


if __name__ == "__main__":
    main()