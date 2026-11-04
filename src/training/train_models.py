"""Train, tune, calibrate, and evaluate the four BCSC clinical risk models.

This is the execution entry point for the model-training phase. It respects the
frozen experiment decisions in ``outputs/training/reports/07_final_experiment_decisions.md``:

* Training-only, five-fold, ``count``-weighted, cancer-stratified CV.
* Weighted PR-AUC is the primary hyperparameter-selection metric.
* ``count`` is passed to every fold fit and to every weighted scorer.
* Sigmoid (Platt) calibration with ``ensemble=False`` on Training-only folds.
* Validation is used for final comparison only, never for tuning or calibration.
* ``test.csv`` is never referenced or opened.

The saved preprocessing encoders are applied with ``transform`` only. The MLP
model artifact is a ``SampleWeightPipeline(StandardScaler, MLPClassifier)`` so
the scaler is fitted inside each CV training fold and serialized with the MLP.

Usage
-----
    python -m src.training.train_models                 # all four models, resume
    python -m src.training.train_models --models mlp    # one model
    python -m src.training.train_models --no-resume     # ignore candidate cache
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_THRESHOLD,
    N_JOBS,
    RANDOM_STATE,
    environment_snapshot,
    set_global_seeds,
)
from src.training.config import (
    CALIBRATION_POLICY,
    CV_POLICY,
    ENCODERS,
    FEATURE_ORDER,
    MODEL_SPECS,
    OUTPUT_DIRECTORIES,
    POSITIVE_CLASS,
    PREPROCESSING_METADATA_PATH,
    TARGET_COLUMN,
    TUNING_SPACES,
    WEIGHT_COLUMN,
    build_model_pipeline,
    build_training_calibrated_model,
    encoder_info,
    ensure_output_directories,
    load_encoded_split,
    load_preprocessing_metadata,
)
from src.training.cv import (
    assess_fold_balance,
    cross_validate_weighted,
    fold_diagnostics,
    positive_class_probability,
    row_equal_metrics,
    weighted_folds,
    weighted_metrics,
)
from src.training.pipeline import load_model_artifact, save_model_artifact

MODEL_ORDER: tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "mlp",
)

MODEL_COLORS: dict[str, str] = {
    "logistic_regression": "#1f77b4",
    "random_forest": "#2ca02c",
    "gradient_boosting": "#d62728",
    "mlp": "#9467bd",
}

METRIC_COLUMNS: tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "recall",
    "specificity",
    "precision",
    "brier",
)

SEARCH_DIR = OUTPUT_DIRECTORIES["experiments"] / "search"
PREDICTIONS_DIR = OUTPUT_DIRECTORIES["predictions"]
METRICS_DIR = OUTPUT_DIRECTORIES["metrics"]
PLOTS_DIR = OUTPUT_DIRECTORIES["reports"].parent / "plots"
LOGS_DIR = OUTPUT_DIRECTORIES["logs"]
MODELS_DIR = OUTPUT_DIRECTORIES["models"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, default=_json_default, sort_keys=True)


def _prefixed_model_params(model_key: str, params: dict[str, Any]) -> dict[str, Any]:
    """Map candidate parameters onto a model pipeline's estimator namespace."""
    spec = MODEL_SPECS[model_key]
    prefix = "classifier__" if spec.scaler is not None else ""
    return {f"{prefix}{name}": value for name, value in params.items()}


def _prefixed_calibration_params(model_key: str, params: dict[str, Any]) -> dict[str, Any]:
    """Map candidate parameters onto ``CalibratedClassifierCV(estimator=...)``."""
    nested = _prefixed_model_params(model_key, params)
    return {f"estimator__{name}": value for name, value in nested.items()}


def _build_candidate_pipeline(model_key: str, params: dict[str, Any]):
    pipeline = build_model_pipeline(model_key)
    pipeline.set_params(**_prefixed_model_params(model_key, params))
    return pipeline


def _candidate_grid(model_key: str) -> list[dict[str, Any]]:
    space = TUNING_SPACES[model_key]
    names = list(space)
    candidates = []
    for values in itertools.product(*(space[name] for name in names)):
        candidates.append(dict(zip(names, values)))
    if len(candidates) > 24:  # guard rail from TUNING_POLICY
        raise ValueError(
            f"{model_key}: {len(candidates)} candidates exceeds the configured maximum of 24"
        )
    return candidates


def _candidate_id(model_key: str, index: int) -> str:
    return f"{model_key}_{index:03d}"


def _mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None, None
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def _log(message: str, log_path: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------
def _fold_rows(model_key: str, candidate_id: str, params: dict[str, Any], results) -> list[dict[str, Any]]:
    rows = []
    for fold_result in results:
        metrics = fold_result.metrics
        rows.append(
            {
                "model": model_key,
                "candidate_id": candidate_id,
                "params": _params_json(params),
                "fold": fold_result.fold,
                "train_rows": fold_result.train_rows,
                "validation_rows": fold_result.validation_rows,
                "train_weight_sum": fold_result.train_weight_sum,
                "validation_weight_sum": fold_result.validation_weight_sum,
                "threshold": metrics["threshold"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "neg_brier": None if metrics["brier"] is None else -metrics["brier"],
                "brier": metrics["brier"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "tp_w": metrics["tp_w"],
                "fp_w": metrics["fp_w"],
                "tn_w": metrics["tn_w"],
                "fn_w": metrics["fn_w"],
            }
        )
    return rows


def _summarise_candidate(
    model_key: str,
    candidate_id: str,
    params: dict[str, Any],
    rows: list[dict[str, Any]],
    duration: float,
    warning_messages: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": model_key,
        "candidate_id": candidate_id,
        "params": params,
        "params_json": _params_json(params),
        "n_folds": len(rows),
        "duration_seconds": duration,
        "warnings": sorted(set(warning_messages)),
    }
    for metric in ("pr_auc", "roc_auc", "neg_brier", "brier", "recall", "specificity", "precision"):
        mean, std = _mean_std(row[metric] for row in rows)
        summary[f"mean_{metric}"] = mean
        summary[f"std_{metric}"] = std
    return summary


def run_search(
    model_key: str,
    Z,
    y: np.ndarray,
    w: np.ndarray,
    folds,
    *,
    resume: bool,
    log_path: Path,
) -> list[dict[str, Any]]:
    """Evaluate every pre-declared candidate with weighted Training-only CV."""
    candidates = _candidate_grid(model_key)
    model_search_dir = SEARCH_DIR / model_key
    model_search_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for index, params in enumerate(candidates):
        candidate_id = _candidate_id(model_key, index)
        cache_path = model_search_dir / f"{candidate_id}.json"
        if resume and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("n_folds") == CV_POLICY["n_splits"]:
                _log(f"{model_key}: candidate {index + 1}/{len(candidates)} cached ({candidate_id})", log_path)
                summaries.append(cached)
                continue

        _log(
            f"{model_key}: candidate {index + 1}/{len(candidates)} {candidate_id} {_params_json(params)}",
            log_path,
        )
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = cross_validate_weighted(
                lambda: _build_candidate_pipeline(model_key, params),
                Z,
                y,
                w,
                folds=folds,
            )
        duration = time.perf_counter() - start
        rows = _fold_rows(model_key, candidate_id, params, results)
        summary = _summarise_candidate(
            model_key,
            candidate_id,
            params,
            rows,
            duration,
            [str(item.message) for item in caught],
        )
        summary["folds"] = rows
        cache_path.write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )
        summaries.append(summary)
        _log(
            f"{model_key}: {candidate_id} mean PR-AUC={summary['mean_pr_auc']:.6f} "
            f"ROC-AUC={summary['mean_roc_auc']:.6f} "
            f"neg-Brier={summary['mean_neg_brier']:.8f} in {duration:.1f}s",
            log_path,
        )

    return summaries


def select_best(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by mean weighted PR-AUC; tie-break on ROC-AUC, then Brier."""
    if not summaries:
        raise ValueError("no candidate summaries to select from")

    def key(summary: dict[str, Any]):
        return (
            -(summary["mean_pr_auc"] if summary["mean_pr_auc"] is not None else -math.inf),
            -(summary["mean_roc_auc"] if summary["mean_roc_auc"] is not None else -math.inf),
            -(summary["mean_neg_brier"] if summary["mean_neg_brier"] is not None else -math.inf),
            summary["candidate_id"],
        )

    return sorted(summaries, key=key)[0]


# ---------------------------------------------------------------------------
# Final fit, calibration, and validation evaluation
# ---------------------------------------------------------------------------
def fit_final_calibrated(
    model_key: str,
    params: dict[str, Any],
    Z,
    y: np.ndarray,
    w: np.ndarray,
    *,
    log_path: Path,
):
    """Fit the selected candidate on all Training rows and calibrate it."""
    calibrated = build_training_calibrated_model(model_key, y, w, split="train")
    calibrated.set_params(**_prefixed_calibration_params(model_key, params))
    if calibrated.get_params()["ensemble"] is not False:
        raise RuntimeError("calibration must use ensemble=False")
    if calibrated.get_params()["method"] != CALIBRATION_POLICY["method"]:
        raise RuntimeError("calibration method does not match the approved policy")

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrated.fit(Z, y, sample_weight=w)
    duration = time.perf_counter() - start
    _log(f"{model_key}: final calibrated fit completed in {duration:.1f}s", log_path)
    return calibrated, duration, [str(item.message) for item in caught]


def evaluate_on_validation(model_key: str, model, *, log_path: Path) -> dict[str, Any]:
    Z_val, y_val, w_val = load_encoded_split(model_key, "val")
    start = time.perf_counter()
    probabilities = positive_class_probability(model, Z_val)
    inference_seconds = time.perf_counter() - start

    y_arr = np.asarray(y_val, dtype=int)
    w_arr = np.asarray(w_val, dtype=float)
    metrics = weighted_metrics(y_arr, probabilities, w_arr, threshold=DEFAULT_THRESHOLD)
    row_equal = row_equal_metrics(y_arr, probabilities, threshold=DEFAULT_THRESHOLD)

    predictions = pd.DataFrame(
        {
            "row_id": np.arange(len(y_arr), dtype=int),
            "model": model_key,
            "y_true": y_arr,
            "count": w_arr,
            "probability": probabilities,
            "predicted_positive_at_0.5": (probabilities >= DEFAULT_THRESHOLD).astype(int),
        }
    )
    predictions_path = PREDICTIONS_DIR / f"val_predictions_{model_key}.csv"
    predictions.to_csv(predictions_path, index=False)

    _log(
        f"{model_key}: validation PR-AUC={metrics['pr_auc']:.6f} "
        f"ROC-AUC={metrics['roc_auc']:.6f} Brier={metrics['brier']:.8f}",
        log_path,
    )
    return {
        "model": model_key,
        "rows": int(len(y_arr)),
        "weight_sum": float(w_arr.sum()),
        "positive_weight": float(w_arr[y_arr == 1].sum()),
        "negative_weight": float(w_arr[y_arr == 0].sum()),
        "weighted_prevalence": metrics["weighted_prevalence"],
        "metrics": metrics,
        "row_equal": row_equal,
        "probabilities": probabilities,
        "y_true": y_arr,
        "count": w_arr,
        "predictions_path": str(predictions_path),
        "inference_seconds": inference_seconds,
        "inference_seconds_per_row": inference_seconds / len(y_arr),
        "inference_seconds_per_weighted_case": inference_seconds / float(w_arr.sum()),
    }


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------
def calibration_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Weighted quantile-binned reliability table plus ECE/MCE."""
    edges = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([probabilities.min(), probabilities.max() + 1e-12])
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, len(edges) - 2)

    total_weight = float(sample_weight.sum())
    rows = []
    ece = 0.0
    mce = 0.0
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        bin_weight = float(sample_weight[mask].sum())
        mean_pred = float(np.average(probabilities[mask], weights=sample_weight[mask]))
        observed = float(np.average(y_true[mask], weights=sample_weight[mask]))
        gap = abs(observed - mean_pred)
        ece += gap * bin_weight / total_weight
        mce = max(mce, gap)
        rows.append(
            {
                "bin": bin_id,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "rows": int(mask.sum()),
                "weight": bin_weight,
                "mean_predicted": mean_pred,
                "observed_rate": observed,
                "absolute_gap": gap,
            }
        )

    brier = float(np.average((probabilities - y_true) ** 2, weights=sample_weight))
    prevalence = float(np.average(y_true, weights=sample_weight))
    brier_reference = prevalence * (1.0 - prevalence)
    return pd.DataFrame(rows), {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "brier_reference": brier_reference,
        "brier_skill_score": (
            None if brier_reference == 0 else 1.0 - brier / brier_reference
        ),
        "mean_predicted": float(np.average(probabilities, weights=sample_weight)),
        "observed_rate": prevalence,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def plot_roc_curves(evaluations: dict[str, dict[str, Any]]) -> Path:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for key, evaluation in evaluations.items():
        fpr, tpr, _ = roc_curve(
            evaluation["y_true"], evaluation["probabilities"], sample_weight=evaluation["count"]
        )
        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS[key],
            lw=1.8,
            label=f"{MODEL_SPECS[key].display_name} (AUC={evaluation['metrics']['roc_auc']:.4f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Chance")
    ax.set_xlabel("Weighted false-positive rate")
    ax.set_ylabel("Weighted true-positive rate")
    ax.set_title("Validation ROC curves (count-weighted)")
    ax.legend(loc="lower right", fontsize=8)
    path = PLOTS_DIR / "roc_curves_val.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pr_curves(evaluations: dict[str, dict[str, Any]]) -> Path:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    prevalence = None
    for key, evaluation in evaluations.items():
        precision, recall, _ = precision_recall_curve(
            evaluation["y_true"],
            evaluation["probabilities"],
            sample_weight=evaluation["count"],
        )
        ap = average_precision_score(
            evaluation["y_true"],
            evaluation["probabilities"],
            sample_weight=evaluation["count"],
        )
        prevalence = evaluation["metrics"]["weighted_prevalence"]
        ax.plot(
            recall,
            precision,
            color=MODEL_COLORS[key],
            lw=1.8,
            label=f"{MODEL_SPECS[key].display_name} (AP={ap:.4f})",
        )
    if prevalence is not None:
        ax.axhline(prevalence, ls="--", color="grey", lw=1, label=f"No-skill (π={prevalence:.4f})")
    ax.set_xlabel("Weighted recall / sensitivity")
    ax.set_ylabel("Weighted precision / PPV")
    ax.set_title("Validation precision-recall curves (count-weighted)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    path = PLOTS_DIR / "pr_curves_val.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_calibration_curves(
    evaluations: dict[str, dict[str, Any]],
    calibration_tables: dict[str, pd.DataFrame],
) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for key, table in calibration_tables.items():
        ax.plot(
            table["mean_predicted"],
            table["observed_rate"],
            marker="o",
            ms=4,
            lw=1.5,
            color=MODEL_COLORS[key],
            label=MODEL_SPECS[key].display_name,
        )
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (weighted, 10 quantile bins)")
    ax.set_ylabel("Observed weighted cancer rate")
    ax.set_title("Validation reliability curves (sigmoid-calibrated)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    path = PLOTS_DIR / "calibration_curves_val.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_probability_distributions(evaluations: dict[str, dict[str, Any]]) -> Path:
    bins = np.linspace(0.0, 1.0, 51)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), sharex=True, sharey=True)
    for ax, key in zip(axes.ravel(), evaluations):
        evaluation = evaluations[key]
        weights = evaluation["count"]
        ax.hist(
            evaluation["probabilities"],
            bins=bins,
            weights=weights,
            color=MODEL_COLORS[key],
            alpha=0.85,
        )
        ax.set_yscale("log")
        ax.axvline(DEFAULT_THRESHOLD, ls="--", color="black", lw=1)
        ax.set_title(f"{MODEL_SPECS[key].display_name}", fontsize=9)
        ax.set_ylabel("Represented weight (log)")
    for ax in axes[-1]:
        ax.set_xlabel("Predicted P(cancer=1)")
    fig.suptitle("Validation predicted-probability distributions (count-weighted, t=0.5 dashed)")
    path = PLOTS_DIR / "probability_distributions_val.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metric_comparison(comparison: pd.DataFrame) -> Path:
    metrics = ["pr_auc", "roc_auc", "recall", "specificity", "precision", "brier"]
    labels = ["PR-AUC", "ROC-AUC", "Recall", "Specificity", "Precision", "Brier"]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    for ax, metric, label in zip(axes.ravel(), metrics, labels):
        values = comparison.set_index("model")[f"val_{metric}"]
        colors = [MODEL_COLORS[key] for key in values.index]
        numeric = [
            0.0
            if value is None or (isinstance(value, float) and not math.isfinite(value))
            else float(value)
            for value in values
        ]
        text_labels = [
            "N/A"
            if value is None or (isinstance(value, float) and not math.isfinite(value))
            else f"{float(value):.3f}"
            for value in values
        ]
        bars = ax.bar(range(len(values)), numeric, color=colors, alpha=0.9)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(
            [MODEL_SPECS[key].display_name.replace(" ", "\n") for key in values.index],
            fontsize=7,
        )
        ax.set_title(label)
        ax.set_ylim(0, max(1.0, max(numeric) * 1.12))
        for bar, text_label in zip(bars, text_labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text_label,
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.suptitle("Validation comparison at t=0.5 (Brier is lower-is-better)")
    path = PLOTS_DIR / "metric_comparison_val.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrices(evaluations: dict[str, dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6))
    for ax, key in zip(axes.ravel(), evaluations):
        metrics = evaluations[key]["metrics"]
        matrix = np.array(
            [
                [metrics["tn_w"], metrics["fp_w"]],
                [metrics["fn_w"], metrics["tp_w"]],
            ],
            dtype=float,
        )
        row_sums = matrix.sum(axis=1, keepdims=True)
        shares = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        labels = [
            [f"{int(matrix[r, c]):,}\n({shares[r, c]:.2%})" for c in range(2)]
            for r in range(2)
        ]
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(
                    c,
                    r,
                    labels[r][c],
                    ha="center",
                    va="center",
                    color="black" if shares[r, c] < 0.6 else "white",
                    fontsize=9,
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"{MODEL_SPECS[key].display_name} (t=0.5)", fontsize=9)
        ax.grid(False)
    fig.suptitle("Validation weighted confusion matrices (row-normalised shares)")
    path = PLOTS_DIR / "confusion_matrices_val.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cv_vs_validation(comparison: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(comparison))
    width = 0.36
    ax.bar(
        x - width / 2,
        comparison["cv_mean_pr_auc"],
        width,
        yerr=comparison["cv_std_pr_auc"],
        capsize=3,
        label="CV mean PR-AUC",
        color="#8c8c8c",
    )
    ax.bar(
        x + width / 2,
        comparison["val_pr_auc"],
        width,
        label="Validation PR-AUC",
        color=[MODEL_COLORS[key] for key in comparison["model"]],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_SPECS[key].display_name.replace(" ", "\n") for key in comparison["model"]],
        fontsize=8,
    )
    ax.set_ylabel("Weighted PR-AUC")
    ax.set_title("Training CV vs external Validation PR-AUC")
    ax.legend(fontsize=8)
    path = PLOTS_DIR / "cv_vs_validation_pr_auc.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_candidate_scores(
    summaries_by_model: dict[str, list[dict[str, Any]]],
    selected: dict[str, dict[str, Any]],
) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.2))
    for ax, key in zip(axes, MODEL_ORDER):
        summaries = summaries_by_model.get(key, [])
        if not summaries:
            ax.set_visible(False)
            continue
        values = [summary["mean_pr_auc"] for summary in summaries]
        positions = np.arange(len(values))
        ax.plot(positions, values, marker="o", ms=3, lw=1, color=MODEL_COLORS[key])
        best = selected.get(key)
        if best is not None:
            best_index = [s["candidate_id"] for s in summaries].index(best["candidate_id"])
            ax.scatter([best_index], [best["mean_pr_auc"]], color="black", zorder=5, s=30)
        ax.set_title(MODEL_SPECS[key].display_name, fontsize=9)
        ax.set_xlabel("Candidate index")
        ax.set_ylabel("CV mean weighted PR-AUC")
    fig.suptitle("Training-only CV candidate scores (black marker = selected by PR-AUC)")
    path = PLOTS_DIR / "cv_candidate_pr_auc.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------
def _write_search_tables(model_key: str, summaries: list[dict[str, Any]]) -> None:
    fold_rows = []
    summary_rows = []
    for summary in summaries:
        for row in summary["folds"]:
            fold_rows.append(dict(row))
        summary_rows.append(
            {
                key: value
                for key, value in summary.items()
                if key != "folds" and key != "warnings"
            }
        )
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIRECTORIES["experiments"] / f"{model_key}_search_results.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIRECTORIES["experiments"] / f"{model_key}_search_summary.csv", index=False
    )


def _append_registry(run_id: str, comparison: pd.DataFrame, log_path: Path) -> Path:
    """Append one row per model to the experiment registry, replacing a rerun."""
    path = OUTPUT_DIRECTORIES["experiments"] / "registry.csv"
    rows = []
    for record in comparison.to_dict(orient="records"):
        rows.append(
            {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": record["model"],
                "encoder": record["encoder"],
                "selected_candidate": record["selected_candidate"],
                "selected_params": record["selected_params"],
                "cv_mean_pr_auc": record["cv_mean_pr_auc"],
                "cv_mean_roc_auc": record["cv_mean_roc_auc"],
                "val_pr_auc": record["val_pr_auc"],
                "val_roc_auc": record["val_roc_auc"],
                "val_brier": record["val_brier"],
                "artifact_path": record["artifact_path"],
                "test_accessed": False,
            }
        )
    new = pd.DataFrame(rows)
    if path.is_file():
        existing = pd.read_csv(path)
        existing = existing[existing["run_id"] != run_id]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(path, index=False)
    _log(f"registry updated: {path}", log_path)
    return path


def _write_manifest(
    run_id: str,
    model_keys: list[str],
    grids: dict[str, list[dict[str, Any]]],
    selected: dict[str, dict[str, Any]],
    log_path: Path,
) -> Path:
    manifest = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Training-only hyperparameter search, final fit, sigmoid calibration, Validation comparison",
        "test_accessed": False,
        "validation_used_for_tuning": False,
        "validation_used_for_calibration": False,
        "selection_rule": (
            "Highest mean count-weighted PR-AUC across the five Training-only weighted "
            "cancer-stratified folds; ties break on mean weighted ROC-AUC, then mean "
            "negative Brier, then candidate id."
        ),
        "models": model_keys,
        "grids": grids,
        "candidate_counts": {key: len(value) for key, value in grids.items()},
        "selected_candidates": {
            key: {
                "candidate_id": value["candidate_id"],
                "params": value["params"],
                "mean_pr_auc": value["mean_pr_auc"],
                "mean_roc_auc": value["mean_roc_auc"],
                "mean_neg_brier": value["mean_neg_brier"],
            }
            for key, value in selected.items()
        },
        "cross_validation": {
            "n_splits": CV_POLICY["n_splits"],
            "splitter": CV_POLICY["splitter"],
            "seed": CV_POLICY["seed"],
            "weight_column": WEIGHT_COLUMN,
            "primary_scoring": CV_POLICY["primary_scoring"],
            "row_expansion": False,
        },
        "calibration": {
            "method": CALIBRATION_POLICY["method"],
            "ensemble": CALIBRATION_POLICY["ensemble"],
            "data": CALIBRATION_POLICY["data"],
        },
        "environment": environment_snapshot(),
        "encoders": {
            family: _artifact_identity(spec.artifact) for family, spec in ENCODERS.items()
        },
        "preprocessing_metadata": _artifact_identity(PREPROCESSING_METADATA_PATH),
        "feature_order": list(FEATURE_ORDER),
        "target": TARGET_COLUMN,
        "positive_class": POSITIVE_CLASS,
        "weight": WEIGHT_COLUMN,
        "threshold": DEFAULT_THRESHOLD,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "log": str(log_path),
    }
    path = LOGS_DIR / f"search_manifest_{run_id}.json"
    path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
        help="models to run (default: all four)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore cached candidate results and re-run the search",
    )
    args = parser.parse_args()

    set_global_seeds()
    _style()
    ensure_output_directories()
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + f"_four_model_seed{RANDOM_STATE}"
    log_path = LOGS_DIR / f"training_run_{run_id}.log"
    _log(f"run {run_id}: models={args.models} resume={not args.no_resume}", log_path)
    _log(f"environment={json.dumps(environment_snapshot(), default=_json_default)}", log_path)

    # Training data and folds are shared by all models; each model uses its own encoder.
    training_cache: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {}
    folds = None
    fold_diag_rows = []
    balance = None
    for key in args.models:
        Z, y, w = load_encoded_split(key, "train")
        y_arr = np.asarray(y, dtype=int)
        w_arr = np.asarray(w, dtype=float)
        training_cache[key] = (Z, y_arr, w_arr)
        if folds is None:
            folds = weighted_folds(y_arr, w_arr)
            diagnostics = fold_diagnostics(y_arr, w_arr, folds)
            balance = assess_fold_balance(diagnostics)
            fold_diag_rows = [
                {
                    "fold": diag.fold,
                    "rows": diag.rows,
                    "positive_rows": diag.positive_rows,
                    "weight_sum": diag.weight_sum,
                    "positive_weight": diag.positive_weight,
                    "negative_weight": diag.negative_weight,
                    "weighted_prevalence": diag.weighted_prevalence,
                    "weight_relative_deviation": diag.weight_relative_deviation,
                    "prevalence_relative_deviation": diag.prevalence_relative_deviation,
                }
                for diag in diagnostics
            ]
            _log(
                f"folds: {len(folds)} x {len(y_arr)} rows; "
                f"weight_sum={w_arr.sum():.0f}; material_imbalance={balance['material_imbalance']}",
                log_path,
            )
        else:
            reference = training_cache[args.models[0]]
            if len(y_arr) != len(reference[1]) or not np.array_equal(y_arr, reference[1]):
                raise RuntimeError("Training target differs between model encoders")

    pd.DataFrame(fold_diag_rows).to_csv(METRICS_DIR / "fold_diagnostics.csv", index=False)

    summaries_by_model: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, dict[str, Any]] = {}
    grids: dict[str, list[dict[str, Any]]] = {}

    for key in args.models:
        Z, y_arr, w_arr = training_cache[key]
        grids[key] = _candidate_grid(key)
        _log(
            f"{key}: {len(grids[key])} candidates x {len(folds)} folds; "
            f"encoder={MODEL_SPECS[key].encoder} width={Z.shape[1]}",
            log_path,
        )
        summaries = run_search(
            key,
            Z,
            y_arr,
            w_arr,
            folds,
            resume=not args.no_resume,
            log_path=log_path,
        )
        summaries_by_model[key] = summaries
        selected[key] = select_best(summaries)
        _write_search_tables(key, summaries)
        _log(
            f"{key}: selected {selected[key]['candidate_id']} "
            f"PR-AUC={selected[key]['mean_pr_auc']:.6f} params={_params_json(selected[key]['params'])}",
            log_path,
        )

    evaluations: dict[str, dict[str, Any]] = {}
    comparison_rows = []
    calibration_rows = []
    calibration_tables: dict[str, pd.DataFrame] = {}
    fit_records: dict[str, dict[str, Any]] = {}
    artifact_records: dict[str, dict[str, Any]] = {}
    verification_records: dict[str, dict[str, Any]] = {}

    for key in args.models:
        Z, y_arr, w_arr = training_cache[key]
        best = selected[key]
        model, fit_seconds, fit_warnings = fit_final_calibrated(
            key, best["params"], Z, y_arr, w_arr, log_path=log_path
        )
        artifact_path = MODELS_DIR / f"{key}_calibrated.joblib"
        save_model_artifact(model, artifact_path)
        _log(f"{key}: saved {artifact_path}", log_path)

        # Reload and verify the serialized artifact produces valid probabilities.
        reloaded = load_model_artifact(artifact_path)
        Z_val, y_val, w_val = load_encoded_split(key, "val")
        reloaded_probabilities = positive_class_probability(reloaded, Z_val)
        verification = {
            "artifact": _artifact_identity(artifact_path),
            "classes": [int(value) for value in np.asarray(reloaded.classes_)],
            "probabilities_finite": bool(np.isfinite(reloaded_probabilities).all()),
            "probabilities_in_unit_interval": bool(
                ((reloaded_probabilities >= 0) & (reloaded_probabilities <= 1)).all()
            ),
            "probability_column": "cancer=1",
            "prediction_matches_in_memory": bool(
                np.allclose(
                    reloaded_probabilities,
                    positive_class_probability(model, Z_val),
                    rtol=0,
                    atol=0,
                )
            ),
        }
        if not all(
            [
                verification["probabilities_finite"],
                verification["probabilities_in_unit_interval"],
                verification["prediction_matches_in_memory"],
            ]
        ):
            raise RuntimeError(f"{key}: artifact verification failed: {verification}")
        verification_records[key] = verification
        _log(f"{key}: artifact verification passed", log_path)

        metadata = {
            "model": key,
            "display_name": MODEL_SPECS[key].display_name,
            "estimator": MODEL_SPECS[key].estimator,
            "encoder": MODEL_SPECS[key].encoder,
            "encoder_width": int(Z.shape[1]),
            "scaler": MODEL_SPECS[key].scaler,
            "selected_candidate": best["candidate_id"],
            "selected_params": best["params"],
            "cv": {
                "mean_pr_auc": best["mean_pr_auc"],
                "std_pr_auc": best["std_pr_auc"],
                "mean_roc_auc": best["mean_roc_auc"],
                "std_roc_auc": best["std_roc_auc"],
                "mean_neg_brier": best["mean_neg_brier"],
                "std_neg_brier": best["std_neg_brier"],
                "n_splits": CV_POLICY["n_splits"],
                "splitter": CV_POLICY["splitter"],
                "seed": CV_POLICY["seed"],
            },
            "calibration": {
                "method": CALIBRATION_POLICY["method"],
                "ensemble": CALIBRATION_POLICY["ensemble"],
                "data": CALIBRATION_POLICY["data"],
            },
            "feature_order": list(FEATURE_ORDER),
            "target": TARGET_COLUMN,
            "positive_class": POSITIVE_CLASS,
            "weight": WEIGHT_COLUMN,
            "threshold": DEFAULT_THRESHOLD,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
            "fit_seconds": fit_seconds,
            "fit_warnings": sorted(set(fit_warnings)),
            "artifact": str(artifact_path),
            "artifact_sha256": verification["artifact"]["sha256"],
            "encoder_artifact": str(ENCODERS[MODEL_SPECS[key].encoder].artifact),
            "encoder_sha256": _sha256(ENCODERS[MODEL_SPECS[key].encoder].artifact),
        }
        metadata_path = MODELS_DIR / f"{key}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")

        evaluation = evaluate_on_validation(key, model, log_path=log_path)
        evaluations[key] = evaluation
        table, calibration_metrics = calibration_table(
            evaluation["y_true"], evaluation["probabilities"], evaluation["count"]
        )
        table.insert(0, "model", key)
        calibration_tables[key] = table
        calibration_rows.append(
            {
                "model": key,
                "display_name": MODEL_SPECS[key].display_name,
                **calibration_metrics,
            }
        )

        metrics = evaluation["metrics"]
        comparison_rows.append(
            {
                "model": key,
                "display_name": MODEL_SPECS[key].display_name,
                "encoder": MODEL_SPECS[key].encoder,
                "selected_candidate": best["candidate_id"],
                "selected_params": _params_json(best["params"]),
                "cv_mean_pr_auc": best["mean_pr_auc"],
                "cv_std_pr_auc": best["std_pr_auc"],
                "cv_mean_roc_auc": best["mean_roc_auc"],
                "cv_std_roc_auc": best["std_roc_auc"],
                "cv_mean_neg_brier": best["mean_neg_brier"],
                "cv_std_neg_brier": best["std_neg_brier"],
                "val_pr_auc": metrics["pr_auc"],
                "val_roc_auc": metrics["roc_auc"],
                "val_recall": metrics["recall"],
                "val_specificity": metrics["specificity"],
                "val_precision": metrics["precision"],
                "val_brier": metrics["brier"],
                "val_tp_w": metrics["tp_w"],
                "val_fp_w": metrics["fp_w"],
                "val_tn_w": metrics["tn_w"],
                "val_fn_w": metrics["fn_w"],
                "val_predicted_positive_weight": metrics["predicted_positive_weight"],
                "val_ece": calibration_metrics["ece"],
                "val_mce": calibration_metrics["mce"],
                "val_brier_skill_score": calibration_metrics["brier_skill_score"],
                "fit_seconds": fit_seconds,
                "search_seconds": sum(item["duration_seconds"] for item in summaries_by_model[key]),
                "inference_seconds": evaluation["inference_seconds"],
                "inference_seconds_per_row": evaluation["inference_seconds_per_row"],
                "inference_seconds_per_weighted_case": evaluation["inference_seconds_per_weighted_case"],
                "artifact_path": str(artifact_path),
                "artifact_sha256": verification["artifact"]["sha256"],
            }
        )
        fit_records[key] = {"fit_seconds": fit_seconds, "warnings": sorted(set(fit_warnings))}

    # ---- tables ---------------------------------------------------------
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    comparison.to_csv(METRICS_DIR / "validation_metrics.csv", index=False)

    validation_columns = [
        "model",
        "display_name",
        "rows",
        "weight_sum",
        "positive_weight",
        "negative_weight",
        "weighted_prevalence",
        "threshold",
        *METRIC_COLUMNS,
        "tp_w",
        "fp_w",
        "tn_w",
        "fn_w",
        "predicted_positive_weight",
    ]
    validation_table = pd.DataFrame(
        [
            {
                "model": key,
                "display_name": MODEL_SPECS[key].display_name,
                "rows": evaluation["rows"],
                "weight_sum": evaluation["weight_sum"],
                "positive_weight": evaluation["positive_weight"],
                "negative_weight": evaluation["negative_weight"],
                "weighted_prevalence": evaluation["weighted_prevalence"],
                **{name: evaluation["metrics"][name] for name in ("threshold", *METRIC_COLUMNS)},
                "tp_w": evaluation["metrics"]["tp_w"],
                "fp_w": evaluation["metrics"]["fp_w"],
                "tn_w": evaluation["metrics"]["tn_w"],
                "fn_w": evaluation["metrics"]["fn_w"],
                "predicted_positive_weight": evaluation["metrics"]["predicted_positive_weight"],
            }
            for key, evaluation in evaluations.items()
        ]
    )[validation_columns]
    validation_table.to_csv(METRICS_DIR / "validation_metrics_weighted.csv", index=False)

    row_equal_table = pd.DataFrame(
        [
            {
                "model": key,
                "note": "SECONDARY row-equal diagnostic; ignores count; never used for selection",
                **{
                    name: evaluation["row_equal"][name]
                    for name in ("threshold", *METRIC_COLUMNS)
                },
            }
            for key, evaluation in evaluations.items()
        ]
    )
    row_equal_table.to_csv(METRICS_DIR / "validation_metrics_row_equal_secondary.csv", index=False)

    pd.concat(calibration_tables.values(), ignore_index=True).to_csv(
        METRICS_DIR / "calibration_bins_val.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(METRICS_DIR / "calibration_metrics.csv", index=False)

    all_predictions = pd.concat(
        [
            pd.read_csv(evaluation["predictions_path"])
            for evaluation in evaluations.values()
        ],
        ignore_index=True,
    )
    all_predictions.to_csv(PREDICTIONS_DIR / "val_predictions_all.csv", index=False)

    all_fold_rows = []
    all_summary_rows = []
    for key, summaries in summaries_by_model.items():
        for summary in summaries:
            all_fold_rows.extend(dict(row) for row in summary["folds"])
            all_summary_rows.append(
                {name: value for name, value in summary.items() if name not in {"folds", "warnings"}}
            )
    pd.DataFrame(all_fold_rows).to_csv(METRICS_DIR / "cv_fold_results_all.csv", index=False)
    pd.DataFrame(all_summary_rows).to_csv(METRICS_DIR / "cv_candidate_summary_all.csv", index=False)

    # ---- plots ----------------------------------------------------------
    plot_paths = []
    if evaluations:
        plot_paths.append(str(plot_roc_curves(evaluations)))
        plot_paths.append(str(plot_pr_curves(evaluations)))
        plot_paths.append(str(plot_calibration_curves(evaluations, calibration_tables)))
        plot_paths.append(str(plot_probability_distributions(evaluations)))
        plot_paths.append(str(plot_metric_comparison(comparison)))
        plot_paths.append(str(plot_confusion_matrices(evaluations)))
        plot_paths.append(str(plot_cv_vs_validation(comparison)))
        plot_paths.append(str(plot_candidate_scores(summaries_by_model, selected)))

    # ---- manifest / final log ------------------------------------------
    registry_path = _append_registry(run_id, comparison, log_path)
    manifest_path = _write_manifest(run_id, args.models, grids, selected, log_path)
    summary = {
        "run_id": run_id,
        "models": args.models,
        "selected": {
            key: {
                "candidate_id": selected[key]["candidate_id"],
                "params": selected[key]["params"],
                "cv_mean_pr_auc": selected[key]["mean_pr_auc"],
                "cv_mean_roc_auc": selected[key]["mean_roc_auc"],
                "cv_mean_neg_brier": selected[key]["mean_neg_brier"],
            }
            for key in args.models
        },
        "validation": {
            row["model"]: {
                "pr_auc": row["val_pr_auc"],
                "roc_auc": row["val_roc_auc"],
                "recall": row["val_recall"],
                "specificity": row["val_specificity"],
                "precision": row["val_precision"],
                "brier": row["val_brier"],
                "ece": row["val_ece"],
                "mce": row["val_mce"],
            }
            for row in comparison_rows
        },
        "fit_records": fit_records,
        "artifact_verification": verification_records,
        "fold_balance": balance,
        "plots": plot_paths,
        "manifest": str(manifest_path),
        "registry": str(registry_path),
        "test_accessed": False,
    }
    summary_path = LOGS_DIR / f"training_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    _log(f"run {run_id}: wrote {summary_path}", log_path)
    _log("test_accessed=false", log_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - surface the full traceback in the console/log
        traceback.print_exc()
        raise
