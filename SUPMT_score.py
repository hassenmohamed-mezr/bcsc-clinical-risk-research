"""Risk score wrapper on top of SUPMT (additive, read-only)."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import SUPMT
from risk_score.score import BANDS, load_parameters, probability_to_score, risk_band


def predict_risk_score(patient: Any, include_gb_check: bool = True) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = SUPMT.predict_patient(patient)

    params = load_parameters()
    lr_params = params["logistic_regression"]
    p_lr = float(result["logistic_regression"])
    p_weighted = float(result["weighted_average"])
    score_raw = probability_to_score(p_lr, lr_params)
    score = round(score_raw, 1)
    band_key, band_label = risk_band(score)

    warnings_list = []
    normalized = SUPMT._normalize_patient(patient)
    unknown_count = sum(1 for v in normalized.values() if v is None)
    if unknown_count >= 5:
        warnings_list.append("input has many unknown fields")

    out = {
        "risk_score": score,
        "risk_band_key": band_key,
        "risk_band_label": band_label,
        "lr_probability": p_lr,
        "lr_probability_pct": p_lr * 100.0,
        "weighted_average": p_weighted,
        "weighted_average_pct": p_weighted * 100.0,
        "unknown_field_count": unknown_count,
        "reference_prevalence_pct": params["weighted_prevalence"] * 100.0,
        "relative_to_reference": p_weighted / params["weighted_prevalence"] if params["weighted_prevalence"] > 0 else float("inf"),
        "interpretation": (
            f"The relative score places this case above {score:.0f}% of reference cases. "
            f"The estimated model probability is {p_weighted * 100:.2f}% "
            f"versus {params['weighted_prevalence'] * 100:.2f}% reference prevalence. "
            "The relative score is not a probability."
        ),
        "warnings": warnings_list,
        "reference": {
            "n_rows": params["n_rows"],
            "weight_sum": params["weight_sum"],
            "source_sha256": params["source_sha256"],
            "created_utc": params["created_utc"],
            "mu": lr_params["mu"],
            "sigma": lr_params["sigma"],
        },
    }

    if include_gb_check:
        p_gb = float(result["gradient_boosting"])
        gb_params = params["gradient_boosting"]
        score_gb = probability_to_score(p_gb, gb_params)
        gap = abs(score_raw - score_gb)
        disagree = gap > params["gb_gap_threshold"]
        if disagree:
            warnings_list.append("Logistic Regression and Gradient Boosting rankings differ")
        out["gb_check"] = {
            "gb_probability": p_gb,
            "gb_score": round(score_gb, 1),
            "gap": gap,
            "disagree": disagree,
        }

    return out