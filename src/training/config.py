"""Unified training configuration for the four planned BCSC models.

This module is the single source of truth for *training-time* decisions:
feature order, target/positive class, frequency weight, split references,
model-to-encoder mapping, initial estimator settings, the shared metric policy,
the tuning policy, the central seed, and output locations.

It deliberately does **not** fit, transform, or score anything on import. The
saved preprocessing encoders remain the only source of the encoded matrix; this
module only references them by family and artifact path. `count` is the
frequency weight and is never a model feature.

Imports from existing single sources of truth
---------------------------------------------
* ``src.data.load_splits`` -> ``FEATURES``, ``TARGET``, ``WEIGHT``,
  ``PROJECT_ROOT``, ``SPLIT_PATHS``
* ``src.config`` -> ``RANDOM_STATE``, ``N_JOBS``, ``DEFAULT_THRESHOLD``

Model-specific settings live in :data:`MODEL_SPECS`. Shared rules live in
:data:`EVALUATION_POLICY`, :data:`CV_POLICY`, :data:`TUNING_POLICY`,
:data:`CALIBRATION_POLICY`, :data:`SEED_POLICY`, and
:data:`OUTPUT_DIRECTORIES`, so the four models are compared under one protocol
without repeating common values in each model entry.

The MLP model artifact is a :class:`src.training.pipeline.SampleWeightPipeline`
containing an unfitted ``StandardScaler`` and ``MLPClassifier``. Fitting that
pipeline inside a CV fold fits the scaler on that fold's training rows only, and
serializing it saves the scaler and model together.

Usage
-----
    from src.training.config import MODEL_SPECS, build_estimator, load_encoded_split

    estimator = build_estimator("random_forest")      # constructed, not fitted
    Z_train, y_train, w_train = load_encoded_split("random_forest", "train")

Validate the whole configuration without training:

    python -m src.training.config
    python -m src.training.config --with-splits
"""
from __future__ import annotations

import importlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config import DEFAULT_THRESHOLD, N_JOBS, RANDOM_STATE
from src.data.load_splits import FEATURES, PROJECT_ROOT, SPLIT_PATHS, TARGET, WEIGHT
from src.preprocessing.pipeline import FAMILIES
from src.training.pipeline import SampleWeightPipeline

# ---- identifiers (imported, never redefined) ------------------------------
FEATURE_ORDER: tuple[str, ...] = tuple(FEATURES)
TARGET_COLUMN: str = TARGET
WEIGHT_COLUMN: str = WEIGHT
POSITIVE_CLASS: int = 1
NEGATIVE_CLASS: int = 0
TRAIN_SPLIT: str = "train"
VALIDATION_SPLIT: str = "val"

# Columns that must never enter a model matrix. `training` is split metadata;
# the other three are excluded by the approved feature policy.
EXCLUDED_COLUMNS: tuple[str, ...] = ("invasive", "race", "Hispanic", "training")
FORBIDDEN_MODEL_COLUMNS: frozenset[str] = frozenset(
    (TARGET_COLUMN, WEIGHT_COLUMN, *EXCLUDED_COLUMNS)
)

# ---- filesystem locations -------------------------------------------------
PREPROCESSING_DIR: Path = PROJECT_ROOT / "models" / "preprocessing"
PREPROCESSING_METADATA_PATH: Path = PREPROCESSING_DIR / "preprocessing_metadata.json"
OUTPUT_ROOT: Path = PROJECT_ROOT / "outputs" / "training"


# ---- split references -----------------------------------------------------
@dataclass(frozen=True)
class SplitReference:
    """A frozen split file and the checks that guard against silent drift."""

    name: str
    path: Path
    role: str
    expected_rows: int
    expected_weight_sum: int
    expected_positive_weight: int
    expected_cancer_rate: float


SPLIT_REFERENCES: dict[str, SplitReference] = {
    TRAIN_SPLIT: SplitReference(
        name=TRAIN_SPLIT,
        path=SPLIT_PATHS[TRAIN_SPLIT],
        role="fit models; fit any Training-only scaler; internal CV",
        expected_rows=152_149,
        expected_weight_sum=1_525_868,
        expected_positive_weight=7_452,
        expected_cancer_rate=0.004883777626898264,
    ),
    VALIDATION_SPLIT: SplitReference(
        name=VALIDATION_SPLIT,
        path=SPLIT_PATHS[VALIDATION_SPLIT],
        role="model comparison and threshold-independent / t=0.5 reporting",
        expected_rows=28_316,
        expected_weight_sum=269_271,
        expected_positive_weight=1_315,
        expected_cancer_rate=0.004883555971493403,
    ),
}


# ---- encoders (artifacts are owned by the preprocessing stage) -------------
@dataclass(frozen=True)
class EncoderSpec:
    family: str
    artifact: Path
    description: str


_ENCODER_DESCRIPTIONS: dict[str, str] = {
    "onehot": "One-hot for all features; Unknown (9) is its own category.",
    "ordinal_tree": (
        "Ordinal ranks for ordered features plus Unknown flags; "
        "one-hot for nominal features."
    ),
}

# Family names come from the preprocessing stage; artifact names follow its
# ``<family>_pipeline.joblib`` convention.
ENCODERS: dict[str, EncoderSpec] = {
    family: EncoderSpec(
        family=family,
        artifact=PREPROCESSING_DIR / f"{family}_pipeline.joblib",
        description=_ENCODER_DESCRIPTIONS[family],
    )
    for family in FAMILIES
}


# ---- optional post-encoder scaling ----------------------------------------
@dataclass(frozen=True)
class ScalerSpec:
    name: str
    estimator: str
    params: Mapping[str, Any]
    fit_scope: str


SCALERS: dict[str, ScalerSpec] = {
    "standard": ScalerSpec(
        name="standard",
        estimator="sklearn.preprocessing.StandardScaler",
        params={"with_mean": True, "with_std": True},
        fit_scope=(
            "Fitted per CV fold on that fold's Training rows only inside the MLP "
            "model pipeline; reused unchanged for the fold's validation rows; "
            "refit on the full Training split for the final artifact."
        ),
    )
}


# ---- special-code policy -------------------------------------------------
# The BCSC source combines Unknown and Not Applicable in code 9 for these two
# features. They remain one category; no raw-data or mapping change is allowed
# without reliable source documentation.
CODE_9_POLICY: dict[str, Any] = {
    "code": 9,
    "features": ("surgmeno", "hrt"),
    "treatment": "single category",
    "split_into_unknown_and_not_applicable": False,
    "source_meaning": "combined Unknown / Not Applicable in the BCSC source",
    "rationale": (
        "The source does not provide a reliable row-level distinction. Splitting "
        "code 9 would invent information. Both encoder families keep code 9 as "
        "one explicit category (surgmeno_9 and hrt_9)."
    ),
    "raw_data_change": False,
    "mapping_change": False,
}


# ---- model specifications -------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Model-specific choices. Shared seeds/rules are applied at resolution."""

    key: str
    display_name: str
    estimator: str
    encoder: str
    scaler: str | None
    params: Mapping[str, Any]
    supports_n_jobs: bool
    notes: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        display_name="Logistic Regression",
        estimator="sklearn.linear_model.LogisticRegression",
        encoder="onehot",
        scaler=None,
        params={
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": None,
        },
        supports_n_jobs=True,
        notes="Linear baseline on the one-hot matrix; no scaling required.",
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        display_name="Random Forest",
        estimator="sklearn.ensemble.RandomForestClassifier",
        encoder="ordinal_tree",
        scaler=None,
        params={
            "n_estimators": 300,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": None,
        },
        supports_n_jobs=True,
        notes="Tree baseline on the compact ordinal matrix; no scaling.",
    ),
    "gradient_boosting": ModelSpec(
        key="gradient_boosting",
        display_name="Gradient Boosting",
        estimator="sklearn.ensemble.GradientBoostingClassifier",
        encoder="ordinal_tree",
        scaler=None,
        params={
            "loss": "log_loss",
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_depth": 3,
        },
        supports_n_jobs=False,
        notes=(
            "Initial implementation uses scikit-learn GradientBoostingClassifier "
            "on the numeric ordinal matrix. HistGradientBoostingClassifier or an "
            "external library is a separate, documented experiment."
        ),
    ),
    "mlp": ModelSpec(
        key="mlp",
        display_name="MLP Neural Network",
        estimator="sklearn.neural_network.MLPClassifier",
        encoder="onehot",
        scaler="standard",
        params={
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": 1024,
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "shuffle": True,
            "early_stopping": False,
        },
        supports_n_jobs=False,
        notes=(
            "scikit-learn MLP on the one-hot matrix. The model artifact is a "
            "SampleWeightPipeline(StandardScaler, MLPClassifier); the scaler is "
            "fitted on each CV training fold only and on the full Training split "
            "for the final model. Early stopping is disabled; max_iter is tuned "
            "with Training-only CV. External Validation is never used for fitting "
            "or early stopping."
        ),
    ),
}

REQUIRED_MODEL_KEYS: tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "mlp",
)


# ---- evaluation policy ----------------------------------------------------
@dataclass(frozen=True)
class MetricSpec:
    key: str
    name: str
    function: str
    threshold_dependent: bool
    weighted: bool
    direction: str
    denominator: str
    notes: str = ""


METRICS: dict[str, MetricSpec] = {
    "roc_auc": MetricSpec(
        key="roc_auc",
        name="ROC-AUC",
        function="sklearn.metrics.roc_auc_score",
        threshold_dependent=False,
        weighted=True,
        direction="higher",
        denominator="weighted TPR/FPR across thresholds",
        notes="Ranking only; interpret alongside weighted prevalence.",
    ),
    "pr_auc": MetricSpec(
        key="pr_auc",
        name="PR-AUC (average precision)",
        function="sklearn.metrics.average_precision_score",
        threshold_dependent=False,
        weighted=True,
        direction="higher",
        denominator="weighted positive weight; precision uses predicted-positive weight",
        notes="No-skill baseline is weighted prevalence, not 0.5.",
    ),
    "recall": MetricSpec(
        key="recall",
        name="Recall / Sensitivity",
        function="weighted confusion counts",
        threshold_dependent=True,
        weighted=True,
        direction="higher",
        denominator="W+ (positive weight)",
    ),
    "specificity": MetricSpec(
        key="specificity",
        name="Specificity",
        function="weighted confusion counts",
        threshold_dependent=True,
        weighted=True,
        direction="higher",
        denominator="W- (negative weight)",
    ),
    "precision": MetricSpec(
        key="precision",
        name="Precision / PPV",
        function="weighted confusion counts",
        threshold_dependent=True,
        weighted=True,
        direction="higher",
        denominator="predicted-positive weight",
        notes="N/A when no weight is predicted positive at the threshold.",
    ),
    "brier": MetricSpec(
        key="brier",
        name="Brier Score",
        function="sklearn.metrics.brier_score_loss",
        threshold_dependent=False,
        weighted=True,
        direction="lower",
        denominator="W (total weight)",
    ),
}

EVALUATION_POLICY: dict[str, Any] = {
    "positive_class": POSITIVE_CLASS,
    "negative_class": NEGATIVE_CLASS,
    "threshold": DEFAULT_THRESHOLD,
    "threshold_rule": "p >= threshold",
    "probability_source": (
        "predict_proba; select the column whose classes_ entry equals the positive class"
    ),
    "weight_column": WEIGHT_COLUMN,
    "weighting": (
        "count is a frequency weight; pass as sample_weight to fitting and to every "
        "metric. No row expansion, dropping, or resampling."
    ),
    "evaluation_split": VALIDATION_SPLIT,
    "comparison_scope": (
        "Fit on Training, compare on Validation only. Test is evaluated once after "
        "the model, encoder, threshold rule, and protocol are frozen."
    ),
    "test_available_to_training": False,
    "metrics": METRICS,
    "required_metrics": tuple(METRICS),
    "confusion_counts": ("tp_w", "fp_w", "tn_w", "fn_w"),
    "context_statistics": ("weight_sum", "positive_weight", "negative_weight", "weighted_prevalence"),
    "undefined_values": "Report as N/A with a reason; never substitute 0, 0.5, or 1.",
    "row_coverage": "Every Validation row must be scored by every model.",
}


# ---- cross-validation policy ---------------------------------------------
CV_POLICY: dict[str, Any] = {
    "data": "Training only",
    "test_available": False,
    "validation_use": "final evaluation only; never fold construction, fitting, tuning, or calibration",
    "n_splits": 5,
    "splitter": "WeightedStratifiedKFold",
    "splitter_implementation": "src.training.cv.WeightedStratifiedKFold",
    "fold_construction": (
        "Deterministic greedy weighted class-stratified assignment: within each "
        "cancer class, rows are ordered by descending count (seeded tie-break) "
        "and assigned to the fold with the smallest accumulated class weight. "
        "No row expansion, duplication, dropping, or resampling."
    ),
    "stratification_target": "cancer (positive class 1)",
    "seed": RANDOM_STATE,
    "weight_column": WEIGHT_COLUMN,
    "fit_weighting": "count passed as sample_weight to every fold's fit call",
    "score_weighting": (
        "count of the validation-fold rows passed as sample_weight to every "
        "weighted scorer; weights are never passed to fit only"
    ),
    "row_expansion": False,
    "primary_scoring": "pr_auc",
    "primary_scoring_function": "sklearn.metrics.average_precision_score",
    "supporting_scoring": ("roc_auc", "neg_brier"),
    "unweighted_diagnostics": (
        "Optional secondary analysis only: treat each aggregated row equally and "
        "label it clearly. Never used for selection."
    ),
    "fold_diagnostics": (
        "rows",
        "positive_rows",
        "weight_sum",
        "positive_weight",
        "negative_weight",
        "weighted_prevalence",
    ),
    "balance_tolerance": {
        "weight_relative_deviation": 0.05,
        "prevalence_relative_deviation": 0.10,
    },
    "balance_rule": (
        "Check each fold's represented weight and weighted prevalence against the "
        "Training reference. Material imbalance means the fold construction must "
        "be reviewed before model comparison."
    ),
    "fallback": (
        "If the weighted splitter is unavailable, use StratifiedKFold(shuffle=True, "
        "random_state=RANDOM_STATE) on Training only and document that it balances "
        "rows, not represented weight."
    ),
}


# ---- probability calibration policy --------------------------------------
CALIBRATION_POLICY: dict[str, Any] = {
    "required": True,
    "method": "sigmoid",
    "method_rationale": (
        "Platt/sigmoid calibration is more stable than isotonic regression for a "
        "rare positive class (~0.49% weighted prevalence)."
    ),
    "ensemble": False,
    "ensemble_rationale": (
        "ensemble=False fits the base pipeline on the full Training split and "
        "fits the calibrator on weighted out-of-fold probabilities. This yields a "
        "single deployable base model plus one calibrator."
    ),
    "cv": "CV_POLICY weighted cancer-stratified folds on Training only",
    "cv_n_splits": CV_POLICY["n_splits"],
    "data": "Training only",
    "weighting": (
        "count is passed as sample_weight to every calibration fold fit and to "
        "the calibrator. Weights are never silently dropped."
    ),
    "final_fit": (
        "Fit the base model pipeline on the full Training split with count; fit "
        "the calibrator on weighted out-of-fold probabilities from Training-only "
        "CV. Do not refit the calibrator on Validation or Test."
    ),
    "validation_use": "evaluation only; never used to fit, tune, or select the calibrator",
    "test_use": "forbidden",
    "artifact": (
        "One joblib artifact: CalibratedClassifierCV containing the fitted model "
        "pipeline and its calibrator."
    ),
    "sample_weight_compatibility": (
        "CalibratedClassifierCV.fit accepts sample_weight and forwards it to the "
        "base estimator fit and to the calibrator. The MLP SampleWeightPipeline "
        "exposes sample_weight explicitly so it is not dropped."
    ),
    "fallback": (
        "If weighted calibration is not supported, do not drop weights silently: "
        "stop and implement an explicit weighted calibrator or document the "
        "limitation before fitting."
    ),
}


# ---- tuning policy --------------------------------------------------------
TUNING_POLICY: dict[str, Any] = {
    "method": "exhaustive search over a pre-declared grid",
    "data": "Training only",
    "cross_validation": {
        "policy": "CV_POLICY",
        "splitter": CV_POLICY["splitter"],
        "n_splits": CV_POLICY["n_splits"],
        "shuffle": True,
        "random_state": CV_POLICY["seed"],
        "row_expansion": CV_POLICY["row_expansion"],
    },
    "weighting": CV_POLICY["score_weighting"],
    "primary_scoring": CV_POLICY["primary_scoring"],
    "supporting_scoring": CV_POLICY["supporting_scoring"],
    "max_candidates_per_model": 24,
    "refit": "Refit the selected candidate on the full Training split with count.",
    "validation_use": "Compare final candidates once; never use Validation for search.",
    "test_use": "Forbidden until the model, encoder, threshold, and protocol are frozen.",
    "n_jobs": N_JOBS,
    "recording": (
        "Record the grid, fold definition, seed, every fold's rows/weight_sum/"
        "weighted prevalence, every candidate score, and the selected parameters; "
        "do not record only the winner."
    ),
    "no_post_hoc_metric": True,
}

TUNING_SPACES: dict[str, dict[str, tuple[Any, ...]]] = {
    "logistic_regression": {
        "C": (0.01, 0.1, 1.0, 10.0),
        "solver": ("lbfgs",),
    },
    "random_forest": {
        "n_estimators": (300, 600),
        "max_depth": (None, 16),
        "min_samples_leaf": (1, 10),
        "max_features": ("sqrt",),
    },
    "gradient_boosting": {
        "learning_rate": (0.03, 0.05, 0.1),
        "n_estimators": (100, 200),
        "max_depth": (2, 3),
        "subsample": (0.8, 1.0),
    },
    "mlp": {
        "hidden_layer_sizes": ((32,), (64, 32), (128, 64)),
        "alpha": (0.0001, 0.001),
        "learning_rate_init": (0.001, 0.01),
    },
}


# ---- seed policy ----------------------------------------------------------
SEED_POLICY: dict[str, Any] = {
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
    "default_threshold": DEFAULT_THRESHOLD,
    "global_seeding": "src.config.set_global_seeds() before model construction",
    "launch_environment": "Set PYTHONHASHSEED and BLAS/OpenMP thread variables before Python starts.",
    "rule": (
        "Every estimator receives random_state=RANDOM_STATE; no training code may "
        "define a local seed. A different seed is a new experiment."
    ),
}


# ---- output locations -----------------------------------------------------
OUTPUT_DIRECTORIES: dict[str, Path] = {
    "models": PROJECT_ROOT / "models" / "trained",
    "experiments": OUTPUT_ROOT / "experiments",
    "predictions": OUTPUT_ROOT / "predictions",
    "metrics": OUTPUT_ROOT / "metrics",
    "logs": OUTPUT_ROOT / "logs",
    "reports": OUTPUT_ROOT / "reports",
}

EXPERIMENT_REGISTRY: Path = OUTPUT_DIRECTORIES["experiments"] / "registry.csv"

RUN_ID_FORMAT: str = "YYYYMMDDTHHMMSS_{model}_{encoder}_seed{seed}"

RUN_ARTIFACTS: dict[str, str] = {
    "config": "config.json",
    "environment": "environment.json",
    "splits": "splits.json",
    "metrics": "metrics.json",
    "predictions": "predictions_val.csv",
    "model": "model.joblib",
    "log": "log.txt",
    "notes": "config_notes.md",
}


# ---- helpers --------------------------------------------------------------
def _resolve_class(path: str) -> type:
    """Import a class from its fully qualified path without constructing it."""
    module_name, separator, class_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"expected a fully qualified class path, got {path!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"{class_name!r} not found in module {module_name!r}") from exc


def _resolved_model_params(spec: ModelSpec) -> dict[str, Any]:
    """Apply shared seed/thread settings to model-specific parameters."""
    params = dict(spec.params)
    params["random_state"] = RANDOM_STATE
    if spec.supports_n_jobs:
        params["n_jobs"] = N_JOBS
    return params


def get_model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as exc:
        raise KeyError(
            f"unknown model {model_key!r}; available: {sorted(MODEL_SPECS)}"
        ) from exc


def build_estimator(model_key: str):
    """Construct (but never fit) one configured estimator."""
    spec = get_model_spec(model_key)
    estimator_class = _resolve_class(spec.estimator)
    return estimator_class(**_resolved_model_params(spec))


def build_scaler(model_key: str):
    """Construct (but never fit) the configured post-encoder scaler, if any."""
    spec = get_model_spec(model_key)
    if spec.scaler is None:
        return None
    scaler = SCALERS[spec.scaler]
    return _resolve_class(scaler.estimator)(**dict(scaler.params))


def load_preprocessing_metadata() -> dict[str, Any]:
    if not PREPROCESSING_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"preprocessing metadata not found: {PREPROCESSING_METADATA_PATH}"
        )
    return json.loads(PREPROCESSING_METADATA_PATH.read_text(encoding="utf-8"))


def encoder_info(family: str) -> dict[str, Any]:
    """Return the saved encoder's input/output contract from its metadata."""
    if family not in ENCODERS:
        raise KeyError(f"unknown encoder {family!r}; available: {sorted(ENCODERS)}")
    metadata = load_preprocessing_metadata()

    if list(metadata.get("features", [])) != list(FEATURE_ORDER):
        raise ValueError("preprocessing metadata feature order does not match FEATURES")
    if metadata.get("target") != TARGET_COLUMN:
        raise ValueError("preprocessing metadata target does not match TARGET")
    if metadata.get("frequency_weight") != WEIGHT_COLUMN:
        raise ValueError("preprocessing metadata weight does not match WEIGHT")

    try:
        entry = metadata["families"][family]
    except KeyError as exc:
        raise KeyError(f"encoder family {family!r} missing from preprocessing metadata") from exc

    names = tuple(entry["feature_names"])
    width = int(entry["n_features_out"])
    if len(names) != width:
        raise ValueError(
            f"encoder {family!r}: {len(names)} feature names but n_features_out={width}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"encoder {family!r}: duplicate encoded feature names")

    return {
        "family": family,
        "artifact": ENCODERS[family].artifact,
        "n_features_in": int(entry["n_features_in"]),
        "n_features_out": width,
        "feature_names": names,
    }


def load_encoded_split(model_key: str, split: str = TRAIN_SPLIT):
    """Load one split, apply the model's saved encoder, and return ``Z, y, w``.

    Only the frozen Training and Validation splits are exposed. The saved
    encoder is applied with ``transform`` only; it is never refitted. The MLP
    scaler is *not* applied here because it must be fitted on the Training
    output by the training code.
    """
    if split not in SPLIT_REFERENCES:
        raise ValueError(
            f"split {split!r} is not available to training; allowed: {sorted(SPLIT_REFERENCES)}"
        )

    import numpy as np

    from src.data.load_splits import load_split
    from src.preprocessing.pipeline import load_pipeline, transform_features

    spec = get_model_spec(model_key)
    X, y, w = load_split(split)
    pipeline = load_pipeline(ENCODERS[spec.encoder].artifact)
    Z = transform_features(pipeline, X)

    info = encoder_info(spec.encoder)
    if Z.shape != (len(X), info["n_features_out"]):
        raise ValueError(
            f"encoded matrix shape {Z.shape} does not match "
            f"({len(X)}, {info['n_features_out']}) for {model_key!r}"
        )
    if not np.isfinite(Z).all():
        raise ValueError(f"encoded matrix for {model_key!r} contains NaN or infinite values")
    return Z, y, w


def ensure_output_directories() -> dict[str, Path]:
    """Create the configured output directories. Never called on import."""
    for path in OUTPUT_DIRECTORIES.values():
        path.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIRECTORIES


def config_summary() -> dict[str, Any]:
    """Return a JSON-friendly summary used by the report and experiment records."""
    encoders: dict[str, Any] = {}
    for family, spec in ENCODERS.items():
        info = encoder_info(family)
        encoders[family] = {
            "artifact": str(spec.artifact),
            "description": spec.description,
            "n_features_in": info["n_features_in"],
            "n_features_out": info["n_features_out"],
        }

    models: dict[str, Any] = {}
    for key, spec in MODEL_SPECS.items():
        models[key] = {
            "display_name": spec.display_name,
            "estimator": spec.estimator,
            "encoder": spec.encoder,
            "encoder_width": encoders[spec.encoder]["n_features_out"],
            "scaler": spec.scaler,
            "resolved_params": _resolved_model_params(spec),
            "notes": spec.notes,
        }

    return {
        "features": list(FEATURE_ORDER),
        "target": TARGET_COLUMN,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "weight": WEIGHT_COLUMN,
        "forbidden_model_columns": sorted(FORBIDDEN_MODEL_COLUMNS),
        "splits": {
            name: {
                "path": str(ref.path),
                "role": ref.role,
                "expected_rows": ref.expected_rows,
                "expected_weight_sum": ref.expected_weight_sum,
                "expected_positive_weight": ref.expected_positive_weight,
                "expected_cancer_rate": ref.expected_cancer_rate,
            }
            for name, ref in SPLIT_REFERENCES.items()
        },
        "encoders": encoders,
        "models": models,
        "metrics": [
            {
                "key": metric.key,
                "name": metric.name,
                "function": metric.function,
                "threshold_dependent": metric.threshold_dependent,
                "weighted": metric.weighted,
                "direction": metric.direction,
                "denominator": metric.denominator,
            }
            for metric in METRICS.values()
        ],
        "evaluation": {
            "threshold": EVALUATION_POLICY["threshold"],
            "threshold_rule": EVALUATION_POLICY["threshold_rule"],
            "evaluation_split": EVALUATION_POLICY["evaluation_split"],
            "weight_column": EVALUATION_POLICY["weight_column"],
            "test_available_to_training": EVALUATION_POLICY["test_available_to_training"],
        },
        "tuning": {
            "method": TUNING_POLICY["method"],
            "cross_validation": TUNING_POLICY["cross_validation"],
            "primary_scoring": TUNING_POLICY["primary_scoring"],
            "max_candidates_per_model": TUNING_POLICY["max_candidates_per_model"],
            "spaces": {key: {name: list(values) for name, values in space.items()}
                       for key, space in TUNING_SPACES.items()},
        },
        "seed": dict(SEED_POLICY),
        "outputs": {name: str(path) for name, path in OUTPUT_DIRECTORIES.items()},
        "experiment_registry": str(EXPERIMENT_REGISTRY),
        "run_id_format": RUN_ID_FORMAT,
        "run_artifacts": dict(RUN_ARTIFACTS),
    }


def validate_config(
    *,
    check_artifacts: bool = True,
    check_split_data: bool = False,
) -> dict[str, Any]:
    """Read-only validation of the complete training configuration.

    Raises ``ValueError`` with every detected problem. It never fits a model,
    never refits an encoder, and never exposes the Test split.
    """
    errors: list[str] = []
    checks: list[str] = []

    def fail(message: str) -> None:
        errors.append(message)

    # 1. Features, target, weight.
    if len(FEATURE_ORDER) == 0:
        fail("FEATURES is empty")
    if len(set(FEATURE_ORDER)) != len(FEATURE_ORDER):
        fail("FEATURES contains duplicate names")
    forbidden_present = sorted(FORBIDDEN_MODEL_COLUMNS.intersection(FEATURE_ORDER))
    if forbidden_present:
        fail(f"forbidden columns present in FEATURES: {forbidden_present}")
    if WEIGHT_COLUMN in FEATURE_ORDER:
        fail(f"frequency weight {WEIGHT_COLUMN!r} must never be a model feature")
    if TARGET_COLUMN in FEATURE_ORDER:
        fail(f"target {TARGET_COLUMN!r} must never be a model feature")
    checks.append("features/target/weight")

    # 2. Split references.
    if set(SPLIT_REFERENCES) != {TRAIN_SPLIT, VALIDATION_SPLIT}:
        fail(f"SPLIT_REFERENCES must contain exactly train/val, got {sorted(SPLIT_REFERENCES)}")
    for name, ref in SPLIT_REFERENCES.items():
        if ref.path != SPLIT_PATHS[name]:
            fail(f"split {name!r} path does not match load_splits.SPLIT_PATHS")
        if ref.expected_rows <= 0 or ref.expected_weight_sum <= 0:
            fail(f"split {name!r} has non-positive expected reference values")
    checks.append("split references")

    # 3. Encoders and their saved metadata.
    if set(ENCODERS) != set(FAMILIES):
        fail(f"ENCODERS families {sorted(ENCODERS)} do not match pipeline FAMILIES {sorted(FAMILIES)}")
    for family, spec in ENCODERS.items():
        if not spec.artifact.is_file():
            fail(f"encoder artifact missing for {family!r}: {spec.artifact}")
        try:
            info = encoder_info(family)
        except Exception as exc:  # noqa: BLE001 - collect all configuration errors
            fail(f"encoder {family!r} metadata invalid: {type(exc).__name__}: {exc}")
            continue
        if info["n_features_in"] != len(FEATURE_ORDER):
            fail(
                f"encoder {family!r} expects {info['n_features_in']} inputs, "
                f"FEATURES has {len(FEATURE_ORDER)}"
            )
        if info["n_features_out"] <= 0:
            fail(f"encoder {family!r} has non-positive output width")
        encoded_forbidden = sorted(FORBIDDEN_MODEL_COLUMNS.intersection(info["feature_names"]))
        if encoded_forbidden:
            fail(f"encoder {family!r} emits forbidden names: {encoded_forbidden}")

        if check_artifacts and spec.artifact.is_file():
            try:
                from src.preprocessing.pipeline import feature_names as pipeline_feature_names
                from src.preprocessing.pipeline import load_pipeline

                pipeline = load_pipeline(spec.artifact)
                if getattr(pipeline, "n_features_in_", None) != len(FEATURE_ORDER):
                    fail(f"encoder {family!r} artifact n_features_in_ mismatch")
                if list(getattr(pipeline, "feature_names_in_", [])) != list(FEATURE_ORDER):
                    fail(f"encoder {family!r} artifact feature order mismatch")
                if tuple(pipeline_feature_names(pipeline)) != info["feature_names"]:
                    fail(f"encoder {family!r} artifact output names differ from metadata")
            except Exception as exc:  # noqa: BLE001
                fail(f"encoder {family!r} artifact could not be loaded: {type(exc).__name__}: {exc}")
    checks.append("encoder metadata/artifacts")

    # 4. Models.
    missing_models = [key for key in REQUIRED_MODEL_KEYS if key not in MODEL_SPECS]
    if missing_models:
        fail(f"required models missing from MODEL_SPECS: {missing_models}")
    for key, spec in MODEL_SPECS.items():
        if spec.key != key:
            fail(f"model {key!r} has mismatched key {spec.key!r}")
        if spec.encoder not in ENCODERS:
            fail(f"model {key!r} references unknown encoder {spec.encoder!r}")
        if spec.scaler is not None and spec.scaler not in SCALERS:
            fail(f"model {key!r} references unknown scaler {spec.scaler!r}")
        if "random_state" in spec.params:
            fail(f"model {key!r} must not store random_state locally; it is injected centrally")
        if "n_jobs" in spec.params:
            fail(f"model {key!r} must not store n_jobs locally; it is injected centrally")

        try:
            estimator_class = _resolve_class(spec.estimator)
        except Exception as exc:  # noqa: BLE001
            fail(f"model {key!r} estimator cannot be imported: {type(exc).__name__}: {exc}")
            continue

        try:
            estimator = build_estimator(key)
        except Exception as exc:  # noqa: BLE001
            fail(f"model {key!r} cannot be constructed: {type(exc).__name__}: {exc}")
            continue

        if not hasattr(estimator, "predict_proba"):
            fail(f"model {key!r} does not expose predict_proba")
        if hasattr(estimator, "classes_"):
            fail(f"model {key!r} was unexpectedly fitted during configuration")
        if "sample_weight" not in inspect.signature(estimator.fit).parameters:
            fail(f"model {key!r} fit() does not accept sample_weight")
        resolved = estimator.get_params()
        if resolved.get("random_state") != RANDOM_STATE:
            fail(f"model {key!r} did not receive the central random_state")
        if spec.supports_n_jobs and resolved.get("n_jobs") != N_JOBS:
            fail(f"model {key!r} did not receive the central n_jobs")
        checks.append(f"model {key}")
    checks.append("model specs")

    # 5. Evaluation policy.
    if EVALUATION_POLICY["positive_class"] != POSITIVE_CLASS:
        fail("evaluation positive class mismatch")
    if EVALUATION_POLICY["threshold"] != DEFAULT_THRESHOLD:
        fail("evaluation threshold does not match src.config.DEFAULT_THRESHOLD")
    if EVALUATION_POLICY["weight_column"] != WEIGHT_COLUMN:
        fail("evaluation weight column mismatch")
    if EVALUATION_POLICY["evaluation_split"] != VALIDATION_SPLIT:
        fail("evaluation split must be Validation")
    if set(METRICS) != set(EVALUATION_POLICY["required_metrics"]):
        fail("METRICS and required_metrics differ")
    for key, metric in METRICS.items():
        if not metric.weighted:
            fail(f"metric {key!r} is not weighted; the shared policy requires weighted metrics")
    required_metric_keys = {"roc_auc", "pr_auc", "recall", "specificity", "precision", "brier"}
    if not required_metric_keys.issubset(METRICS):
        fail(f"missing required metrics: {sorted(required_metric_keys - set(METRICS))}")
    checks.append("evaluation policy")

    # 6. Tuning and seed policy.
    if TUNING_POLICY["cross_validation"]["random_state"] != RANDOM_STATE:
        fail("tuning CV does not use the central random_state")
    if TUNING_POLICY["data"] != "Training only":
        fail("tuning must use Training only")
    if TUNING_POLICY["n_jobs"] != N_JOBS:
        fail("tuning n_jobs does not match the central N_JOBS")
    if set(TUNING_SPACES) != set(MODEL_SPECS):
        fail(
            f"TUNING_SPACES keys {sorted(TUNING_SPACES)} do not match "
            f"MODEL_SPECS keys {sorted(MODEL_SPECS)}"
        )
    for key, space in TUNING_SPACES.items():
        if not space:
            fail(f"tuning space for {key!r} is empty")
    if SEED_POLICY["random_state"] != RANDOM_STATE:
        fail("SEED_POLICY random_state does not match src.config.RANDOM_STATE")
    checks.append("tuning/seed policy")

    # 7. Outputs.
    required_outputs = {"models", "experiments", "predictions", "metrics", "logs", "reports"}
    if not required_outputs.issubset(OUTPUT_DIRECTORIES):
        fail(f"missing output directories: {sorted(required_outputs - set(OUTPUT_DIRECTORIES))}")
    if not EXPERIMENT_REGISTRY.parent == OUTPUT_DIRECTORIES["experiments"]:
        fail("experiment registry is not inside the experiments directory")
    checks.append("output directories")

    # 8. Optional live split check (Train/Validation only; Test is never loaded).
    if check_split_data:
        from src.data.load_splits import load_split

        for name, ref in SPLIT_REFERENCES.items():
            X, y, w = load_split(name)
            if list(X.columns) != list(FEATURE_ORDER):
                fail(f"split {name!r} columns do not match FEATURE_ORDER")
            if len(X) != ref.expected_rows:
                fail(f"split {name!r} row count {len(X)} != expected {ref.expected_rows}")
            total = float(w.sum())
            positive = float(w[y == 1].sum())
            if int(total) != ref.expected_weight_sum:
                fail(f"split {name!r} weight sum {int(total)} != expected {ref.expected_weight_sum}")
            if int(positive) != ref.expected_positive_weight:
                fail(
                    f"split {name!r} positive weight {int(positive)} != "
                    f"expected {ref.expected_positive_weight}"
                )
            rate = positive / total
            if not math.isclose(rate, ref.expected_cancer_rate, rel_tol=1e-12, abs_tol=1e-15):
                fail(f"split {name!r} weighted cancer rate {rate!r} != reference")
        checks.append("live Train/Validation data")

    if errors:
        raise ValueError(
            "training configuration is invalid:\n- " + "\n- ".join(errors)
        )

    summary = config_summary()
    summary["validation"] = {
        "checks_passed": len(checks),
        "check_names": checks,
        "artifacts_loaded": check_artifacts,
        "split_data_loaded": check_split_data,
        "test_accessed": False,
        "models_trained": False,
    }
    return summary


if __name__ == "__main__":
    import sys

    result = validate_config(
        check_artifacts=True,
        check_split_data="--with-splits" in sys.argv[1:],
    )
    print(json.dumps(result, indent=2, default=str))
