"""Validate risk-score bands on the validation LR predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from risk_score.score import load_parameters, probability_to_score, risk_band

PRED = ROOT / "outputs" / "training" / "predictions" / "val_predictions_all.csv"
OUT_BANDS = ROOT / "outputs" / "risk_score" / "band_validation.csv"
OUT_MAP = ROOT / "outputs" / "risk_score" / "mapping_examples.csv"
OUT_BANDS.parent.mkdir(parents=True, exist_ok=True)


def weighted_bootstrap_ci(values, weights, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vb = values[idx]
        wb = weights[idx]
        if wb.sum() <= 0:
            stats.append(np.nan)
            continue
        stats.append(float(np.average(vb, weights=wb)))
    arr = np.asarray(stats, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main():
    params = load_parameters()
    lr = params["logistic_regression"]
    df = pd.read_csv(PRED)
    lrdf = df[df["model"] == "logistic_regression"].sort_values("row_id").reset_index(drop=True)
    p = lrdf["probability"].to_numpy(float)
    w = lrdf["count"].to_numpy(float)
    y = lrdf["y_true"].to_numpy(float)
    scores = np.array([probability_to_score(float(x), lr) for x in p])
    bands = np.array([risk_band(float(s))[0] for s in scores])
    val_prev = float(np.average(y, weights=w))

    rows = []
    for key in ["very_low", "average", "above_average", "high"]:
        mask = bands == key
        bw = w[mask]
        by = y[mask]
        bp = p[mask]
        bs = scores[mask]
        obs_rate = float(np.average(by, weights=bw)) if bw.sum() > 0 else float("nan")
        mean_pred = float(np.average(bs, weights=bw)) if bw.sum() > 0 else float("nan")
        ci_lo, ci_hi = weighted_bootstrap_ci(by, bw, n_boot=1000, seed=42)
        rows.append({
            "band": key,
            "weighted_share_pct": 100.0 * float(bw.sum()) / float(w.sum()),
            "weighted_observed_rate": obs_rate,
            "ratio_to_prevalence": obs_rate / val_prev if val_prev > 0 else float("nan"),
            "weighted_mean_score": mean_pred,
            "observed_rate_ci_lower": ci_lo,
            "observed_rate_ci_upper": ci_hi,
            "n_rows": int(mask.sum()),
            "weight_sum": float(bw.sum()),
        })
    band_df = pd.DataFrame(rows)
    band_df.to_csv(OUT_BANDS, index=False)

    # Stop condition
    obs_rates = band_df["weighted_observed_rate"].to_list()
    if any(obs_rates[i+1] < obs_rates[i] for i in range(len(obs_rates) - 1)):
        print("STOP: observed rate not non-decreasing across bands")
        print(band_df.to_string(index=False))
        raise SystemExit(1)

    # Mapping examples
    # empirical percentile for a probability
    order = np.argsort(p)
    ps = p[order]
    ws = w[order]
    cum = np.cumsum(ws)
    total = cum[-1]
    before = np.concatenate(([0.0], cum[:-1]))
    emp_pct_at = {}
    for prob in [0.002, 0.0035, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]:
        mask = p <= prob
        emp = float(np.sum(w[mask]) / total * 100.0) if mask.any() else 0.0
        emp_pct_at[prob] = emp

    map_rows = []
    for prob in [0.002, 0.0035, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]:
        sc = probability_to_score(prob, lr)
        map_rows.append({
            "lr_probability": prob,
            "lr_probability_pct": prob * 100.0,
            "smooth_score": sc,
            "smooth_band": risk_band(sc)[0],
            "empirical_percentile_pct": emp_pct_at[prob],
        })
    pd.DataFrame(map_rows).to_csv(OUT_MAP, index=False)
    print(band_df.to_string(index=False))
    print("saved", OUT_BANDS, OUT_MAP)


if __name__ == "__main__":
    main()