"""Log-normal CDF risk score derived from Logistic Regression probability."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from math import log, sqrt
from scipy.special import ndtr
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_CSV = ROOT / "outputs" / "training" / "predictions" / "val_predictions_all.csv"
PARAMS_JSON = ROOT / "risk_score" / "lr_score_params.json"
P_MIN, P_MAX = 1e-9, 1 - 1e-9


def probability_to_score(prob, params):
    """Log-normal CDF of the LR probability: smooth, strictly increasing, in (0,100)."""
    p = min(max(float(prob), P_MIN), P_MAX)
    z = (log(p) - params["mu"]) / params["sigma"]
    return 100.0 * float(ndtr(z))


BANDS = (
    (25.0, "very_low", "ضئيل"),
    (60.0, "average", "متوسط"),
    (85.0, "above_average", "أعلى من المتوسط"),
    (100.0000001, "high", "مرتفع - يستدعي إعادة التقييم"),
)


def risk_band(score):
    for upper, key, label in BANDS:
        if score < upper:
            return key, label
    return BANDS[-1][1], BANDS[-1][2]


def _weighted_median(values, weights):
    order = np.argsort(values)
    vals = np.asarray(values)[order]
    ws = np.asarray(weights)[order]
    cum = np.cumsum(ws)
    total = cum[-1]
    idx = np.searchsorted(cum, total / 2.0, side="left")
    idx = min(idx, len(vals) - 1)
    return float(vals[idx])


def _weighted_percentile(values, weights, q):
    order = np.argsort(values)
    vals = np.asarray(values)[order]
    ws = np.asarray(weights)[order]
    cum = np.cumsum(ws)
    total = cum[-1]
    target = total * q
    idx = np.searchsorted(cum, target, side="left")
    idx = min(idx, len(vals) - 1)
    return float(vals[idx])


def fit_parameters():
    df = pd.read_csv(PREDICTIONS_CSV)
    lr = df[df["model"] == "logistic_regression"][["row_id", "y_true", "count", "probability"]].copy()
    gb = df[df["model"] == "gradient_boosting"][["row_id", "y_true", "count", "probability"]].copy()

    if set(lr["row_id"]) != set(gb["row_id"]):
        raise ValueError("LR and GB row_ids differ")
    lr = lr.sort_values("row_id").reset_index(drop=True)
    gb = gb.sort_values("row_id").reset_index(drop=True)
    if not lr["y_true"].equals(gb["y_true"]) or not lr["count"].equals(gb["count"]):
        raise ValueError("LR and GB y_true/count differ")

    # Reproduction checks
    y = lr["y_true"].to_numpy(dtype=float)
    w = lr["count"].to_numpy(dtype=float)
    p = lr["probability"].to_numpy(dtype=float)
    roc = roc_auc_score(y, p, sample_weight=w)
    prevalence = float(np.average(y, weights=w))
    if abs(roc - 0.6242) > 1e-4:
        raise ValueError(f"ROC-AUC reproduction failed: {roc}")
    if abs(prevalence - 0.004884) > 1e-6:
        raise ValueError(f"prevalence reproduction failed: {prevalence}")

    # Fit LR
    logp = np.log(np.clip(p, 1e-12, 1 - 1e-12))
    mu_lr = float(np.average(logp, weights=w))
    var_lr = float(np.average((logp - mu_lr) ** 2, weights=w))
    sigma_lr = float(np.sqrt(var_lr))
    if not (sigma_lr > 0 and np.isfinite(sigma_lr)):
        raise ValueError("invalid LR sigma")

    # Fit GB
    pgb = gb["probability"].to_numpy(dtype=float)
    logpgb = np.log(np.clip(pgb, 1e-12, 1 - 1e-12))
    mu_gb = float(np.average(logpgb, weights=w))
    var_gb = float(np.average((logpgb - mu_gb) ** 2, weights=w))
    sigma_gb = float(np.sqrt(var_gb))

    from math import erf as _erf, sqrt as _sqrt

    def score_vec(prob, mu, sigma):
        pp = np.clip(prob, 1e-9, 1 - 1e-9)
        z = (np.log(pp) - mu) / sigma
        return 100.0 * 0.5 * (1.0 + np.vectorize(_erf)(z / _sqrt(2.0)))

    score_lr = score_vec(p, mu_lr, sigma_lr)
    score_gb = score_vec(pgb, mu_gb, sigma_gb)
    gap = np.abs(score_lr - score_gb)
    gb_gap_threshold = float(_weighted_percentile(gap, w, 0.95))

    # Reproduction checks mu/sigma/median
    median_prob = _weighted_median(p, w)
    median_pct = median_prob * 100.0
    if abs(mu_lr - (-5.408)) > 0.005:
        raise ValueError(f"mu reproduction failed: {mu_lr}")
    if abs(sigma_lr - 0.475) > 0.005:
        raise ValueError(f"sigma reproduction failed: {sigma_lr}")
    if abs(median_pct - 0.448) > 0.005:
        raise ValueError(f"median probability reproduction failed: {median_pct}")

    sha = hashlib.sha256(PREDICTIONS_CSV.read_bytes()).hexdigest()
    params = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "function": "log-normal CDF of LR probability",
        "source_csv": "outputs/training/predictions/val_predictions_all.csv",
        "source_sha256": sha,
        "population": "validation split, out-of-sample predictions, count-weighted",
        "n_rows": int(len(lr)),
        "weight_sum": float(w.sum()),
        "weighted_prevalence": prevalence,
        "weighted_median_probability": float(median_prob),
        "logistic_regression": {"mu": mu_lr, "sigma": sigma_lr},
        "gradient_boosting": {"mu": mu_gb, "sigma": sigma_gb},
        "gb_gap_threshold": gb_gap_threshold,
    }
    PARAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_JSON.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


def load_parameters():
    if not PARAMS_JSON.is_file():
        raise FileNotFoundError(f"missing parameters file: {PARAMS_JSON}")
    params = json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    lr = params["logistic_regression"]
    gb = params["gradient_boosting"]
    for block in (lr, gb):
        if not (block["sigma"] > 0 and np.isfinite(block["mu"]) and np.isfinite(block["sigma"])):
            raise ValueError("invalid parameters")
    if not np.isfinite(params["gb_gap_threshold"]):
        raise ValueError("invalid gb_gap_threshold")
    return params