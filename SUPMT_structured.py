"""Structured (non-printing) wrapper around SUPMT inference."""

from __future__ import annotations

import contextlib
import io
import math
from typing import Any

from src.data.load_splits import load_split

import SUPMT


def _train_baseline_prevalence() -> float:
    X, y, w = load_split("train")
    y_arr = y.to_numpy(dtype=float) if hasattr(y, "to_numpy") else __import__("numpy").asarray(y, dtype=float)
    w_arr = w.to_numpy(dtype=float) if hasattr(w, "to_numpy") else __import__("numpy").asarray(w, dtype=float)
    return float((__import__("numpy").asarray(y_arr) * __import__("numpy").asarray(w_arr)).sum() / __import__("numpy").asarray(w_arr).sum())


def predict_patient_structured(patient: Any) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = SUPMT.predict_patient(patient)

    probs = {
        "logistic_regression": float(result["logistic_regression"]),
        "gradient_boosting": float(result["gradient_boosting"]),
        "mlp": float(result["mlp"]),
        "random_forest": float(result["random_forest"]),
    }
    weighted_average = float(result["weighted_average"])
    weights_used = dict(SUPMT.get_system()["weights"])
    baseline_prevalence = _train_baseline_prevalence()
    relative_risk = weighted_average / baseline_prevalence if baseline_prevalence > 0 else float("inf")
    spread_min = min(probs.values())
    spread_max = max(probs.values())

    warnings_list = []
    if (spread_max - spread_min) > 0.01:
        warnings_list.append("model spread is large")
    # Count unknown fields in normalized patient
    try:
        normalized = SUPMT._normalize_patient(patient)
        unknown_count = sum(1 for v in normalized.values() if v is None)
        if unknown_count >= 5:
            warnings_list.append("input has many unknown fields")
    except Exception:
        pass

    return {
        "probabilities": probs,
        "weighted_average": weighted_average,
        "weights_used": weights_used,
        "baseline_prevalence": baseline_prevalence,
        "relative_risk_vs_baseline": relative_risk,
        "model_spread": {"min": spread_min, "max": spread_max},
        "warnings": warnings_list,
    }