"""Weighted bootstrap utilities for validation predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def weighted_brier(y, prob, w):
    return float(np.average((prob - y) ** 2, weights=w))


def weighted_bss(y, prob, w, baseline_prevalence):
    brier = weighted_brier(y, prob, w)
    baseline_brier = baseline_prevalence * (1.0 - baseline_prevalence)
    if baseline_brier == 0:
        return float("nan")
    return 1.0 - brier / baseline_brier


def bootstrap_metric_ci(y, prob, w, metric, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = prob[idx]
        wb = w[idx]
        if len(np.unique(yb)) < 2:
            stats.append(np.nan)
            continue
        try:
            if metric == "roc_auc":
                stats.append(roc_auc_score(yb, pb, sample_weight=wb))
            elif metric == "pr_auc":
                stats.append(average_precision_score(yb, pb, sample_weight=wb))
            elif metric == "brier":
                stats.append(np.average((pb - yb) ** 2, weights=wb))
            elif metric == "bss":
                prev = np.average(yb, weights=wb)
                base = prev * (1 - prev)
                brier = np.average((pb - yb) ** 2, weights=wb)
                stats.append(1 - brier / base if base > 0 else np.nan)
            else:
                raise ValueError(metric)
        except Exception:
            stats.append(np.nan)
    arr = np.asarray(stats, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float(np.nanmedian(arr))


def paired_bootstrap_diff(y, prob_a, prob_b, w, metric, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            diffs.append(np.nan)
            continue
        wa = w[idx]
        pa = prob_a[idx]
        pb = prob_b[idx]
        try:
            if metric == "roc_auc":
                diffs.append(roc_auc_score(yb, pa, sample_weight=wa) - roc_auc_score(yb, pb, sample_weight=wa))
            elif metric == "pr_auc":
                diffs.append(average_precision_score(yb, pa, sample_weight=wa) - average_precision_score(yb, pb, sample_weight=wa))
            else:
                raise ValueError(metric)
        except Exception:
            diffs.append(np.nan)
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float(np.nanmedian(arr))