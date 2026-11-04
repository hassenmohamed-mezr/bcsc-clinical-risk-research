"""Descriptive threshold-sensitivity analysis on the Validation split.

This is a **development-only** supplement. It never selects a threshold: the
approved primary comparison remains the fixed ``t = 0.5`` rule. The grid below
is pre-specified before looking at results and is reported for every model at
every threshold so no single operating point is presented as optimised or
unbiased.

Usage
-----
    python -m src.training.threshold_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DEFAULT_THRESHOLD
from src.training.config import MODEL_SPECS, OUTPUT_DIRECTORIES
from src.training.cv import weighted_metrics

THRESHOLDS: tuple[float, ...] = (
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
)

COLORS = {
    "logistic_regression": "#1f77b4",
    "random_forest": "#2ca02c",
    "gradient_boosting": "#d62728",
    "mlp": "#9467bd",
}


def main() -> None:
    predictions_path = OUTPUT_DIRECTORIES["predictions"] / "val_predictions_all.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"{predictions_path} not found; run src.training.train_models first"
        )
    predictions = pd.read_csv(predictions_path)
    metrics_dir = OUTPUT_DIRECTORIES["metrics"]
    plots_dir = OUTPUT_DIRECTORIES["reports"].parent / "plots"

    rows = []
    for model, frame in predictions.groupby("model", sort=False):
        y = frame["y_true"].to_numpy(dtype=int)
        p = frame["probability"].to_numpy(dtype=float)
        w = frame["count"].to_numpy(dtype=float)
        for threshold in THRESHOLDS:
            metrics = weighted_metrics(y, p, w, threshold=threshold)
            rows.append(
                {
                    "model": model,
                    "display_name": MODEL_SPECS[model].display_name,
                    "threshold": threshold,
                    "is_primary_threshold": threshold == DEFAULT_THRESHOLD,
                    "recall": metrics["recall"],
                    "specificity": metrics["specificity"],
                    "precision": metrics["precision"],
                    "predicted_positive_weight": metrics["predicted_positive_weight"],
                    "tp_w": metrics["tp_w"],
                    "fp_w": metrics["fp_w"],
                    "tn_w": metrics["tn_w"],
                    "fn_w": metrics["fn_w"],
                    "note": (
                        "primary approved threshold" if threshold == DEFAULT_THRESHOLD
                        else "descriptive development analysis only; not used for selection"
                    ),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(metrics_dir / "threshold_sensitivity_val.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for model, frame in table.groupby("model", sort=False):
        axes[0].plot(
            frame["threshold"],
            frame["recall"],
            marker="o",
            ms=3,
            color=COLORS.get(model, "grey"),
            label=MODEL_SPECS[model].display_name,
        )
        axes[1].plot(
            frame["threshold"],
            frame["specificity"],
            marker="o",
            ms=3,
            color=COLORS.get(model, "grey"),
        )
        precision = pd.to_numeric(frame["precision"], errors="coerce")
        axes[2].plot(
            frame["threshold"],
            precision,
            marker="o",
            ms=3,
            color=COLORS.get(model, "grey"),
        )
    for ax, title in zip(
        axes, ["Weighted recall / sensitivity", "Weighted specificity", "Weighted precision / PPV"]
    ):
        ax.axvline(DEFAULT_THRESHOLD, ls="--", color="black", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Threshold (log scale)")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Validation threshold sensitivity (development-only; dashed line = approved t=0.5)"
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "threshold_sensitivity_val.png")
    plt.close(fig)

    summary = {
        "scope": "descriptive development analysis only",
        "selection": "no threshold is selected or recommended from Validation",
        "primary_threshold": DEFAULT_THRESHOLD,
        "grid": list(THRESHOLDS),
        "table": str(metrics_dir / "threshold_sensitivity_val.csv"),
        "plot": str(plots_dir / "threshold_sensitivity_val.png"),
    }
    (metrics_dir / "threshold_sensitivity_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
