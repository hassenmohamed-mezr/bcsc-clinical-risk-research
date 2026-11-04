"""Shared definitions for the Modified Barlow Architecture experiment.

This is a *modified* architecture inspired by Barlow et al. (2006), not a
reproduction of the paper: ``race`` and ``Hispanic`` are excluded entirely,
as instructed. ``menopaus`` is a routing variable only, never a feature.
``count`` is used only as a frequency weight (``sample_weight``).

Only the frozen ``train`` and ``val`` splits are loadable through this module.
There is intentionally no API to load any other split.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_THRESHOLD, RANDOM_STATE  # noqa: E402
from src.training.cv import weighted_metrics  # noqa: E402

TARGET = "cancer"
WEIGHT = "count"
ROUTING_COLUMN = "menopaus"

PRE_FEATURES: list[str] = ["agegrp", "brstproc", "nrelbc", "density"]
POST_FEATURES: list[str] = [
    "agegrp",
    "bmi",
    "agefirst",
    "brstproc",
    "nrelbc",
    "hrt",
    "surgmeno",
    "lastmamm",
    "density",
]

# BCSC codebook category order (numeric ascending, including code 9 where
# documented). The first element is the reference category (SAS ref=first).
CATEGORY_ORDERS: dict[str, list[int]] = {
    "agegrp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "bmi": [1, 2, 3, 4, 9],
    "agefirst": [0, 1, 2, 9],
    "brstproc": [0, 1, 9],
    "nrelbc": [0, 1, 2, 9],
    "density": [1, 2, 3, 4, 9],
    "hrt": [0, 1, 9],
    "surgmeno": [0, 1, 9],
    "lastmamm": [0, 1, 9],
}

REFERENCE_CATEGORIES: dict[str, int] = {
    feature: order[0] for feature, order in CATEGORY_ORDERS.items()
}

MODEL_PARAMS: dict[str, Any] = {
    # Unpenalized, as in the original SAS PROC LOGISTIC fit.
    # penalty=None is supported in sklearn 1.9 (deprecated, still valid).
    "penalty": None,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "class_weight": None,
}

SPLIT_PATHS = {
    "train": PROJECT_ROOT / "data" / "04_splits" / "train.csv",
    "val": PROJECT_ROOT / "data" / "04_splits" / "val.csv",
}


def load_split_frame(split: str) -> pd.DataFrame:
    """Load a frozen split frame. Only 'train' and 'val' are allowed."""
    if split not in SPLIT_PATHS:
        raise ValueError(
            f"unknown split {split!r}; allowed values: {sorted(SPLIT_PATHS)}"
        )
    path = SPLIT_PATHS[split]
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    df = pd.read_csv(path)
    # Preserve the original file row order with an explicit row_id.
    df = df.copy()
    df.insert(0, "row_id", np.arange(len(df), dtype=int))
    return df


def get_subgroup(df: pd.DataFrame, menopa_value: int) -> pd.DataFrame:
    """Rows routed to one subgroup model, preserving original row_id order."""
    if menopa_value not in (0, 1):
        raise ValueError(f"menopaus routing value must be 0 or 1, got {menopa_value!r}")
    return df.loc[df[ROUTING_COLUMN] == menopa_value].copy()


def excluded_unknown(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with menopa == 9. Excluded from both models and combined eval."""
    return df.loc[df[ROUTING_COLUMN] == 9].copy()


def validate_feature_list(features: list[str]) -> None:
    forbidden = {"race", "Hispanic", "menopaus", "count", "cancer", "training", "invasive"}
    clash = forbidden.intersection(features)
    if clash:
        raise ValueError(f"forbidden columns in feature list: {sorted(clash)}")
    missing = [f for f in features if f not in CATEGORY_ORDERS]
    if missing:
        raise ValueError(f"features without codebook order: {missing}")


def build_encoder(features: list[str]) -> OneHotEncoder:
    """Reference-first dummy encoding (SAS param=ref / ref=first equivalent).

    Categories follow the BCSC codebook numeric order, not alphabetical order.
    ``drop='first'`` drops the first (reference) category of each feature.
    """
    validate_feature_list(features)
    return OneHotEncoder(
        categories=[CATEGORY_ORDERS[f] for f in features],
        drop="first",
        handle_unknown="error",
        sparse_output=False,
        dtype=float,
    )


def build_lr() -> LogisticRegression:
    """Unpenalized logistic regression, no class weighting."""
    return LogisticRegression(**MODEL_PARAMS)


def subgroup_descriptive(
    df_sub: pd.DataFrame, y: np.ndarray, w: np.ndarray
) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    total = float(w.sum())
    pos_w = float(w[y == 1].sum())
    return {
        "rows": int(len(y)),
        "weight_sum": total,
        "positive_rows": int((y == 1).sum()),
        "negative_rows": int((y == 0).sum()),
        "positive_weight": pos_w,
        "negative_weight": float(total - pos_w),
        "weighted_prevalence": float(pos_w / total) if total > 0 else None,
        "row_id_min": int(df_sub["row_id"].min()) if len(df_sub) else None,
        "row_id_max": int(df_sub["row_id"].max()) if len(df_sub) else None,
    }


def _weighted_log_loss(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    p_clip = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    proba = np.column_stack([1.0 - p_clip, p_clip])
    return float(log_loss(y, proba, labels=[0, 1], sample_weight=w))


def _constant_baseline(pi: float, y: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Brier / log-loss of a constant predictor p = pi (training prevalence)."""
    pi = float(pi)
    p_const = np.full(len(y), pi, dtype=float)
    brier = float(np.average((p_const - y) ** 2, weights=w))
    ll = _weighted_log_loss(y, p_const, w)
    return {"brier_baseline": brier, "log_loss_baseline": ll}


def compute_metrics(
    y,
    p,
    w,
    *,
    train_prevalence: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """All requested weighted metrics plus constant-prevalence baselines."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    metrics = weighted_metrics(y_arr, p_arr, w_arr, threshold=threshold)
    baseline = _constant_baseline(train_prevalence, y_arr, w_arr)
    brier = metrics["brier"]
    skill = (1.0 - brier / baseline["brier_baseline"]) if baseline["brier_baseline"] else None
    return {
        "threshold": metrics["threshold"],
        "rows": metrics["rows"],
        "weight_sum": metrics["weight_sum"],
        "positive_weight": metrics["positive_weight"],
        "negative_weight": metrics["negative_weight"],
        "weighted_prevalence": metrics["weighted_prevalence"],
        "train_prevalence_baseline": float(train_prevalence),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "brier": brier,
        "log_loss": _weighted_log_loss(y_arr, p_arr, w_arr),
        "brier_baseline": baseline["brier_baseline"],
        "log_loss_baseline": baseline["log_loss_baseline"],
        "brier_skill_score": skill,
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "precision": metrics["precision"],
        "tp_w": metrics["tp_w"],
        "fp_w": metrics["fp_w"],
        "tn_w": metrics["tn_w"],
        "fn_w": metrics["fn_w"],
        "predicted_positive_weight": metrics["predicted_positive_weight"],
        "mean_predicted_probability": float(np.average(p_arr, weights=w_arr)),
        "undefined": metrics["undefined"],
    }


def positive_probability(model, X_encoded: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X_encoded), dtype=float)
    classes = np.asarray(getattr(model, "classes_"))
    match = np.flatnonzero(classes == 1)
    if len(match) != 1:
        raise ValueError(f"model classes must contain exactly one class 1, got {classes.tolist()}")
    return proba[:, int(match[0])]


def save_joblib(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_joblib(path: str | Path) -> Any:
    return joblib.load(Path(path))
