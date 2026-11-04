"""Phase 2: uncertainty and calibration analysis on validation predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "outputs" / "training" / "predictions" / "val_predictions_all.csv"
OUT_METRICS = ROOT / "outputs" / "evaluation" / "metrics"
OUT_PLOTS = ROOT / "outputs" / "evaluation" / "plots"
OUT_METRICS.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

TRAIN_PREVALENCE = 0.004883777626898264


def main():
    df = pd.read_csv(PRED)
    pivot = df.pivot(index="row_id", columns="model", values="probability")
    meta = df.drop_duplicates("row_id").set_index("row_id")
    y = meta["y_true"].to_numpy(dtype=float)
    w = meta["count"].to_numpy(dtype=float)
    val_prev = float(np.average(y, weights=w))

    model_cols = ["logistic_regression", "gradient_boosting", "mlp", "random_forest"]
    ensembles = {
        "all4": pivot[model_cols].mean(axis=1).to_numpy(),
        "lr_gb": pivot[["logistic_regression", "gradient_boosting"]].mean(axis=1).to_numpy(),
        "lr_gb_mlp": pivot[["logistic_regression", "gradient_boosting", "mlp"]].mean(axis=1).to_numpy(),
        "lr_only": pivot["logistic_regression"].to_numpy(),
    }
    all_candidates = {}
    for m in model_cols:
        all_candidates[m] = pivot[m].to_numpy()
    all_candidates.update(ensembles)

    # Reproduction check
    all4_roc, all4_pr, _ = (roc_auc_score(y, ensembles["all4"], sample_weight=w),
                           average_precision_score(y, ensembles["all4"], sample_weight=w),
                           None)
    lr_gb_roc, lr_gb_pr, _ = (roc_auc_score(y, ensembles["lr_gb"], sample_weight=w),
                             average_precision_score(y, ensembles["lr_gb"], sample_weight=w),
                             None)
    assert abs(all4_roc - 0.6214) < 1e-4, all4_roc
    assert abs(all4_pr - 0.00779) < 1e-4, all4_pr
    assert abs(lr_gb_roc - 0.6266) < 1e-4, lr_gb_roc
    assert abs(lr_gb_pr - 0.00792) < 1e-4, lr_gb_pr

    from src.evaluation.bootstrap import (
        bootstrap_metric_ci,
        paired_bootstrap_diff,
        weighted_brier,
        weighted_bss,
    )

    # Bootstrap CIs
    rows = []
    for name, prob in all_candidates.items():
        for metric in ["roc_auc", "pr_auc", "brier", "bss"]:
            lo, hi, med = bootstrap_metric_ci(y, prob, w, metric, n_boot=1000, seed=42)
            rows.append({
                "candidate": name,
                "metric": metric,
                "median": med,
                "ci_lower": lo,
                "ci_upper": hi,
                "point_estimate": (
                    roc_auc_score(y, prob, sample_weight=w) if metric == "roc_auc" else
                    average_precision_score(y, prob, sample_weight=w) if metric == "pr_auc" else
                    weighted_brier(y, prob, w) if metric == "brier" else
                    weighted_bss(y, prob, w, val_prev)
                ),
            })
    pd.DataFrame(rows).to_csv(OUT_METRICS / "bootstrap_ci.csv", index=False)

    # Paired bootstrap diffs vs all4
    paired_rows = []
    for name in ["lr_gb", "lr_gb_mlp", "lr_only"]:
        for metric in ["roc_auc", "pr_auc"]:
            lo, hi, med = paired_bootstrap_diff(y, all_candidates[name], all_candidates["all4"], w, metric, n_boot=1000, seed=42)
            paired_rows.append({
                "candidate": name,
                "vs": "all4",
                "metric": metric,
                "median_diff": med,
                "ci_lower": lo,
                "ci_upper": hi,
            })
    pd.DataFrame(paired_rows).to_csv(OUT_METRICS / "paired_bootstrap_ci.csv", index=False)

    # BSS against both baselines
    bss_rows = []
    for name, prob in all_candidates.items():
        brier = weighted_brier(y, prob, w)
        for baseline_name, baseline_prev in [("train", TRAIN_PREVALENCE), ("validation", val_prev)]:
            baseline_brier = baseline_prev * (1 - baseline_prev)
            bss = 1 - brier / baseline_brier
            bss_rows.append({
                "candidate": name,
                "baseline": baseline_name,
                "baseline_prevalence": baseline_prev,
                "model_brier": brier,
                "baseline_brier": baseline_brier,
                "bss": bss,
            })
    pd.DataFrame(bss_rows).to_csv(OUT_METRICS / "bss_baselines.csv", index=False)

    # Decile table for LR, GB, best candidate (by PR-AUC among candidates)
    best_candidate = max(ensembles.keys(), key=lambda k: average_precision_score(y, ensembles[k], sample_weight=w))
    # Actually best among all candidates
    best_candidate = max(all_candidates.keys(), key=lambda k: average_precision_score(y, all_candidates[k], sample_weight=w))

    def weighted_deciles(prob, n_bins=10):
        order = np.argsort(prob)
        y_sorted = y[order]
        w_sorted = w[order]
        p_sorted = prob[order]
        cumw = np.cumsum(w_sorted)
        total = cumw[-1]
        thresholds = np.array([total * i / n_bins for i in range(1, n_bins)])
        # assign decile by cumulative weight
        decile = np.searchsorted(thresholds, cumw, side="right")
        decile = np.clip(decile, 0, n_bins - 1)
        rows = []
        for d in range(n_bins):
            mask = decile == d
            if not mask.any():
                continue
            bin_w = w_sorted[mask]
            bin_y = y_sorted[mask]
            bin_p = p_sorted[mask]
            obs_rate = float(np.average(bin_y, weights=bin_w))
            mean_pred = float(np.average(bin_p, weights=bin_w))
            # bootstrap CI on observed rate within bin
            bin_indices = np.where(mask)[0]
            boot_obs = []
            rng = np.random.default_rng(42)
            for _ in range(1000):
                idx = rng.choice(bin_indices, size=len(bin_indices), replace=True)
                wb = w_sorted[idx]
                yb = y_sorted[idx]
                if wb.sum() <= 0:
                    continue
                boot_obs.append(float(np.average(yb, weights=wb)))
            ci_lo = float(np.percentile(boot_obs, 2.5))
            ci_hi = float(np.percentile(boot_obs, 97.5))
            rows.append({
                "decile": d + 1,
                "n_rows": int(mask.sum()),
                "weight_sum": float(bin_w.sum()),
                "mean_predicted": mean_pred,
                "observed_rate": obs_rate,
                "observed_expected_ratio": obs_rate / mean_pred if mean_pred > 0 else float("nan"),
                "lift_vs_prevalence": obs_rate / val_prev if val_prev > 0 else float("nan"),
                "observed_rate_ci_lower": ci_lo,
                "observed_rate_ci_upper": ci_hi,
            })
        return pd.DataFrame(rows)

    decile_parts = []
    for name in ["logistic_regression", "gradient_boosting", best_candidate]:
        dtab = weighted_deciles(all_candidates[name] if name in all_candidates else pivot[name].to_numpy())
        dtab.insert(0, "model", name)
        decile_parts.append(dtab)
    decile_df = pd.concat(decile_parts, ignore_index=True)
    decile_df.to_csv(OUT_METRICS / "decile_table.csv", index=False)

    # Plots
    for name in ["logistic_regression", "gradient_boosting", best_candidate]:
        sub = decile_df[decile_df["model"] == name]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sub["decile"], sub["mean_predicted"], marker="o", label="Mean predicted")
        ax.plot(sub["decile"], sub["observed_rate"], marker="s", label="Observed rate")
        ax.set_xlabel("Weighted risk decile")
        ax.set_ylabel("Risk")
        ax.set_title(f"Calibration by decile: {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_PLOTS / f"calibration_deciles_{name}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(sub["decile"], sub["lift_vs_prevalence"])
        ax.axhline(1.0, linestyle="--", color="grey")
        ax.set_xlabel("Weighted risk decile")
        ax.set_ylabel("Lift vs prevalence")
        ax.set_title(f"Lift by decile: {name}")
        fig.tight_layout()
        fig.savefig(OUT_PLOTS / f"lift_{name}.png")
        plt.close(fig)

    # Distribution check
    dist_rows = []
    for name, prob in all_candidates.items():
        arr = np.asarray(prob, dtype=float)
        dist_rows.append({
            "model": name,
            "min": float(np.min(arr)),
            "p1": float(np.percentile(arr, 1)),
            "p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        })
    pd.DataFrame(dist_rows).to_csv(OUT_METRICS / "distribution_check.csv", index=False)

    # Save best candidate name
    (OUT_METRICS / "best_candidate.txt").write_text(best_candidate, encoding="utf-8")
    print("best_candidate", best_candidate)
    print("done")


if __name__ == "__main__":
    main()