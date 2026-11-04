"""Train, evaluate, and compare the Modified Barlow Architecture.

Trains two independent, unpenalized logistic regressions (premenopausal /
postmenopausal) on the frozen Train split, evaluates on Validation only,
combines subgroup predictions in original row order, and compares against
the frozen unified Logistic Regression model.

Modified Barlow Architecture — not a reproduction of Barlow et al. (2006).
``race`` and ``Hispanic`` are entirely excluded. ``menopaus`` routes rows
only. ``count`` is a frequency weight (sample_weight), never a feature.

Usage:
    python -m outputs.experiments.barlow_2006_modified.train_evaluate
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent.parent))

from barlow_common import (  # noqa: E402
    CATEGORY_ORDERS,
    MODEL_PARAMS,
    POST_FEATURES,
    PRE_FEATURES,
    REFERENCE_CATEGORIES,
    ROUTING_COLUMN,
    TARGET,
    WEIGHT,
    build_encoder,
    build_lr,
    compute_metrics,
    excluded_unknown,
    get_subgroup,
    load_split_frame,
    positive_probability,
    save_joblib,
    subgroup_descriptive,
)
from src.config import environment_snapshot, set_global_seeds  # noqa: E402

CONFIG_DIR = HERE / "config"
MODELS_DIR = HERE / "models"
PREPROCESSING_DIR = HERE / "preprocessing"
PREDICTIONS_DIR = HERE / "predictions"
REPORTS_DIR = HERE / "reports"
LOGS_DIR = HERE / "logs"

SUBGROUPS = [
    {"key": "premenopausal", "menopaus": 0, "features": PRE_FEATURES},
    {"key": "postmenopausal", "menopaus": 1, "features": POST_FEATURES},
]

UNIFIED_PREDICTIONS = (
    HERE.parent.parent.parent
    / "outputs"
    / "training"
    / "predictions"
    / "val_predictions_logistic_regression.csv"
)


def _log(lines: list[str], message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    lines.append(line)


def main() -> dict[str, Any]:
    set_global_seeds()
    for folder in (CONFIG_DIR, MODELS_DIR, PREPROCESSING_DIR, PREDICTIONS_DIR, REPORTS_DIR, LOGS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    log_path = LOGS_DIR / "training.log"

    _log(log_lines, "Modified Barlow Architecture — training started")
    _log(log_lines, "test_accessed=False (no test.csv opened or referenced)")

    train = load_split_frame("train")
    val = load_split_frame("val")

    excluded_train = excluded_unknown(train)
    excluded_val = excluded_unknown(val)
    _log(
        log_lines,
        f"excluded menopa==9 Train rows={len(excluded_train)} "
        f"weight={float(excluded_train[WEIGHT].sum()):.0f}",
    )
    _log(
        log_lines,
        f"excluded menopa==9 Val rows={len(excluded_val)} "
        f"weight={float(excluded_val[WEIGHT].sum()):.0f}",
    )

    # --- config snapshot -------------------------------------------------
    _log(log_lines, "writing config/experiment_config.json")
    config_payload: dict[str, Any] = {
        "experiment": "Modified Barlow Architecture (not a reproduction of Barlow et al. 2006)",
        "exclusions": {
            "race": "excluded entirely — not recovered, not used",
            "Hispanic": "excluded entirely — not recovered, not used",
            "menopaus": "routing variable only, never a feature",
            "invasive": "never a feature",
            "training": "never a feature",
            "menopaus_unknown_rule": "menopa == 9 excluded from both models and combined evaluation",
        },
        "target": TARGET,
        "positive_class": 1,
        "weight": WEIGHT,
        "weight_role": "frequency weight (sample_weight); never a feature",
        "splits": {"train": "data/04_splits/train.csv", "val": "data/04_splits/val.csv", "test": "never accessed"},
        "subgroups": [
            {
                "key": s["key"],
                "menopaus": s["menopaus"],
                "features": s["features"],
                "reference_categories": {f: REFERENCE_CATEGORIES[f] for f in s["features"]},
                "category_orders": {f: CATEGORY_ORDERS[f] for f in s["features"]},
            }
            for s in SUBGROUPS
        ],
        "model": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "params": {k: (str(v) if v is None else v) for k, v in MODEL_PARAMS.items()},
            "penalty": "None (unpenalized, as SAS PROC LOGISTIC)",
            "encoding": "OneHotEncoder drop='first' (SAS param=ref / ref=first equivalent)",
            "class_weight": None,
            "no_smote_no_oversampling": True,
        },
        "threshold": 0.5,
        "environment": environment_snapshot(),
    }
    (CONFIG_DIR / "experiment_config.json").write_text(
        json.dumps(config_payload, indent=2, default=str), encoding="utf-8"
    )
    feature_definitions = {
        "premenopausal": PRE_FEATURES,
        "postmenopausal": POST_FEATURES,
        "routing_column": ROUTING_COLUMN,
        "forbidden_features": ["race", "Hispanic", "menopaus", "count", "cancer", "training", "invasive"],
    }
    (CONFIG_DIR / "feature_definitions.json").write_text(
        json.dumps(feature_definitions, indent=2), encoding="utf-8"
    )
    category_metadata = {
        "category_orders": CATEGORY_ORDERS,
        "reference_categories": REFERENCE_CATEGORIES,
        "encoding": "BCSC codebook numeric ascending; drop first (reference)",
    }
    (CONFIG_DIR / "category_reference_metadata.json").write_text(
        json.dumps(category_metadata, indent=2), encoding="utf-8"
    )

    subgroup_rows: list[dict[str, Any]] = []
    combined_probs = np.full(len(val), np.nan)
    combined_row_ids = np.full(len(val), -1, dtype=int)
    val_routed_mask = val[ROUTING_COLUMN].isin([0, 1]).to_numpy()

    train_meta: dict[str, dict[str, Any]] = {}

    # --- train + evaluate each subgroup ---------------------------------
    for sub in SUBGROUPS:
        key = sub["key"]
        features = sub["features"]
        train_sub = get_subgroup(train, sub["menopaus"])
        val_sub = get_subgroup(val, sub["menopaus"])

        X_train = train_sub[features].to_numpy()
        y_train = train_sub[TARGET].to_numpy(dtype=int)
        w_train = train_sub[WEIGHT].to_numpy(dtype=float)
        X_val = val_sub[features].to_numpy()
        y_val = val_sub[TARGET].to_numpy(dtype=int)
        w_val = val_sub[WEIGHT].to_numpy(dtype=float)

        encoder = build_encoder(features)
        Z_train = encoder.fit_transform(X_train)
        Z_val = encoder.transform(X_val)

        model = build_lr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Z_train, y_train, sample_weight=w_train)

        save_joblib(model, MODELS_DIR / f"{key}_lr.joblib")
        save_joblib(encoder, PREPROCESSING_DIR / f"{key}_encoder.joblib")
        _log(log_lines, f"{key}: trained on {len(train_sub)} rows, weight={float(w_train.sum()):.0f}")

        train_pi = float(w_train[y_train == 1].sum() / w_train.sum())
        train_meta[key] = {"pi": train_pi, "desc": subgroup_descriptive(train_sub, y_train, w_train)}

        probs = positive_probability(model, Z_val)
        metrics = compute_metrics(y_val, probs, w_val, train_prevalence=train_pi)

        row = {"model": key, "menopaus": sub["menopaus"], "features": ";".join(features), **metrics}
        subgroup_rows.append(row)
        _log(
            log_lines,
            f"{key}: val ROC-AUC={metrics['roc_auc']:.6f} PR-AUC={metrics['pr_auc']:.6f} "
            f"Brier={metrics['brier']:.8f} logloss={metrics['log_loss']:.8f}",
        )

        # Reassemble combined predictions in validation row order.
        val_positions = np.flatnonzero(val["row_id"].to_numpy()[val_routed_mask] >= 0)
        # map val_sub row_ids back to val positions
        val_id_to_pos = {int(rid): i for i, rid in enumerate(val["row_id"].to_numpy())}
        for j, rid in enumerate(val_sub["row_id"].to_numpy(dtype=int)):
            pos = val_id_to_pos[int(rid)]
            combined_probs[pos] = probs[j]
            combined_row_ids[pos] = int(rid)

    # --- combined evaluation (descriptive, not a model average) -----------
    routed = val.loc[val_routed_mask].copy()
    routed_probs = combined_probs[val_routed_mask]
    y_comb = routed[TARGET].to_numpy(dtype=int)
    w_comb = routed[WEIGHT].to_numpy(dtype=float)
    # Training prevalence of the combined (routed) population.
    train_combined = train.loc[train[ROUTING_COLUMN].isin([0, 1])]
    pi_combined = float(
        train_combined.loc[train_combined[TARGET] == 1, WEIGHT].sum()
        / train_combined[WEIGHT].sum()
    )
    combined_metrics = compute_metrics(y_comb, routed_probs, w_comb, train_prevalence=pi_combined)
    _log(
        log_lines,
        f"combined: ROC-AUC={combined_metrics['roc_auc']:.6f} PR-AUC={combined_metrics['pr_auc']:.6f} "
        f"Brier={combined_metrics['brier']:.8f} logloss={combined_metrics['log_loss']:.8f}",
    )

    # --- combined predictions file (original row order) -------------------
    pred_df = pd.DataFrame(
        {
            "row_id": val["row_id"].to_numpy(dtype=int),
            "menopaus": val[ROUTING_COLUMN].to_numpy(dtype=int),
            "y_true": val[TARGET].to_numpy(dtype=int),
            "count": val[WEIGHT].to_numpy(dtype=float),
            "probability_combined": combined_probs,
            "routed": val_routed_mask,
        }
    )
    pred_df.to_csv(PREDICTIONS_DIR / "validation_predictions.csv", index=False)

    # --- unified comparison ------------------------------------------------
    comparison_rows: list[dict[str, Any]] = []
    from barlow_common import compute_metrics as _cm

    # Recompute unified metrics from frozen predictions (same val order) to
    # keep the comparison on the exact same rows.
    unified_metrics: dict[str, Any] | None = None
    if UNIFIED_PREDICTIONS.is_file():
        upred = pd.read_csv(UNIFIED_PREDICTIONS)
        _log(log_lines, f"unified predictions loaded: {UNIFIED_PREDICTIONS.name}, shape={upred.shape}")
        y_u = upred["y_true"].to_numpy(dtype=int)
        p_u = upred["probability"].to_numpy(dtype=float)
        w_u = upred["count"].to_numpy(dtype=float)
        y_a = val[TARGET].to_numpy(dtype=int)
        if not np.array_equal(y_u, y_a):
            raise ValueError("unified predictions y_true do not align with val.csv order")
        train_y = train[TARGET].to_numpy(dtype=int)
        train_w = train[WEIGHT].to_numpy(dtype=float)
        pi_unified = float(train_w[train_y == 1].sum() / train_w.sum())
        unified_metrics = _cm(y_u, p_u, w_u, train_prevalence=pi_unified)

    for row in subgroup_rows:
        comparison_rows.append(
            {
                "model": row["model"],
                "rows": row["rows"],
                "weight_sum": row["weight_sum"],
                "weighted_prevalence": row["weighted_prevalence"],
                "train_prevalence_baseline": row["train_prevalence_baseline"],
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
                "brier": row["brier"],
                "brier_baseline": row["brier_baseline"],
                "brier_skill_score": row["brier_skill_score"],
                "log_loss": row["log_loss"],
                "log_loss_baseline": row["log_loss_baseline"],
                "recall": row["recall"],
                "specificity": row["specificity"],
                "precision": row["precision"],
                "mean_predicted_probability": row["mean_predicted_probability"],
            }
        )
    comparison_rows.append(
        {
            "model": "combined_two_model",
            "rows": combined_metrics["rows"],
            "weight_sum": combined_metrics["weight_sum"],
            "weighted_prevalence": combined_metrics["weighted_prevalence"],
            "train_prevalence_baseline": combined_metrics["train_prevalence_baseline"],
            "roc_auc": combined_metrics["roc_auc"],
            "pr_auc": combined_metrics["pr_auc"],
            "brier": combined_metrics["brier"],
            "brier_baseline": combined_metrics["brier_baseline"],
            "brier_skill_score": combined_metrics["brier_skill_score"],
            "log_loss": combined_metrics["log_loss"],
            "log_loss_baseline": combined_metrics["log_loss_baseline"],
            "recall": combined_metrics["recall"],
            "specificity": combined_metrics["specificity"],
            "precision": combined_metrics["precision"],
            "mean_predicted_probability": combined_metrics["mean_predicted_probability"],
        }
    )
    if unified_metrics is not None:
        comparison_rows.append(
            {
                "model": "unified_logistic_regression",
                "rows": unified_metrics["rows"],
                "weight_sum": unified_metrics["weight_sum"],
                "weighted_prevalence": unified_metrics["weighted_prevalence"],
                "train_prevalence_baseline": unified_metrics["train_prevalence_baseline"],
                "roc_auc": unified_metrics["roc_auc"],
                "pr_auc": unified_metrics["pr_auc"],
                "brier": unified_metrics["brier"],
                "brier_baseline": unified_metrics["brier_baseline"],
                "brier_skill_score": unified_metrics["brier_skill_score"],
                "log_loss": unified_metrics["log_loss"],
                "log_loss_baseline": unified_metrics["log_loss_baseline"],
                "recall": unified_metrics["recall"],
                "specificity": unified_metrics["specificity"],
                "precision": unified_metrics["precision"],
                "mean_predicted_probability": unified_metrics["mean_predicted_probability"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    subgroup_df = pd.DataFrame(subgroup_rows)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    comparison_df.to_csv(REPORTS_DIR / "subgroup_metrics.csv", index=False)
    _log(log_lines, "wrote reports/model_comparison.csv and reports/subgroup_metrics.csv")

    # --- report ------------------------------------------------------------
    def fmt(v: Any, digits: int = 6) -> str:
        if v is None:
            return "N/A"
        return f"{float(v):.{digits}f}"

    pre = subgroup_rows[0]
    post = subgroup_rows[1]
    comb = combined_metrics

    lines: list[str] = []
    lines.append("# Modified Barlow Architecture — Experiment Report\n")
    lines.append(
        "**This is a Modified Barlow Architecture, not a reproduction of Barlow et al. "
        "(2006).** Subgroup feature lists follow the paper's architecture, but "
        "`race` and `Hispanic` are entirely excluded (not recovered or used), "
        "and the estimator differs from SAS in known, documented ways below."
    )
    lines.append("\n## 1. What was actually implemented\n")
    lines.append(
        "- Two independent, unpenalized logistic regressions:\n"
        "  - **Premenopausal** (`menopaus == 0`): `agegrp, brstproc, nrelbc, density`.\n"
        "  - **Postmenopausal** (`menopaus == 1`): `agegrp, bmi, agefirst, brstproc, "
        "nrelbc, hrt, surgmeno, lastmamm, density`.\n"
        "- `menopaus` used for routing only; `menopaus == 9` excluded from training, "
        "evaluation, and the combined evaluation.\n"
        "- `count` used exclusively as `sample_weight`. No class weighting, SMOTE, "
        "oversampling/undersampling, trees, boosting, or neural nets.\n"
        "- Encoders: one-hot, drop first level per feature (reference-category "
        "coding equivalent to SAS `param=ref` / `ref=first`), with BCSC codebook "
        "category order (numeric ascending), fitted on Train subgroup rows only.\n"
        "- Splits: frozen `train.csv` (fitting only) and `val.csv` (external "
        "evaluation only). `test.csv` was **never accessed**."
    )
    lines.append("\n## 2. Sample counts and weights\n")
    lines.append("| Split | Rows | `sum(count)` | Weighted prevalence |")
    lines.append("|---|---:|---:|---:|")
    for name, frame in (("Train", train), ("Validation", val)):
        y = frame[TARGET].to_numpy(dtype=int)
        w = frame[WEIGHT].to_numpy(dtype=float)
        pi = float(w[y == 1].sum() / w.sum())
        lines.append(
            f"| {name} | {len(frame):,} | {w.sum():,.0f} | {pi:.6%} |"
        )
    lines.append(
        f"| Excluded (`menopaus==9`) Train | {len(excluded_train):,} | "
        f"{float(excluded_train[WEIGHT].sum()):,.0f} | — |"
    )
    lines.append(
        f"| Excluded (`menopaus==9`) Validation | {len(excluded_val):,} | "
        f"{float(excluded_val[WEIGHT].sum()):,.0f} | — |"
    )
    lines.append("\n### Subgroup descriptive counts\n")
    lines.append(
        "| Subgroup | Train rows | Train `sum(count)` | Train prev. | "
        "Val rows | Val `sum(count)` | Val prev. |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in subgroup_rows:
        meta = train_meta[row["model"]]
        desc = meta["desc"]
        lines.append(
            f"| {row['model']} | {desc['rows']:,} | {desc['weight_sum']:,.0f} | "
            f"{row['train_prevalence_baseline']:.6%} | "
            f"{row['rows']:,} | {row['weight_sum']:,.0f} | "
            f"{row['weighted_prevalence']:.6%} |"
        )

    lines.append("\n## 3. Subgroup validation results\n")
    lines.append(
        "| Model | ROC-AUC | PR-AUC | Brier | Brier baseline | "
        "BSS | Log loss | Recall | Specificity | Precision |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in subgroup_rows:
        lines.append(
            f"| {row['model']} | {fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} | "
            f"{fmt(row['brier'], 8)} | {fmt(row['brier_baseline'], 8)} | "
            f"{fmt(row['brier_skill_score'])} | {fmt(row['log_loss'], 8)} | "
            f"{fmt(row['recall'])} | {fmt(row['specificity'])} | "
            f"{fmt(row['precision'])} |"
        )
    lines.append(
        "\nConfusion matrices (weighted, threshold 0.5):\n"
    )
    for row in subgroup_rows:
        lines.append(
            f"- **{row['model']}**: TP_w={row['tp_w']:,.0f} FP_w={row['fp_w']:,.0f} "
            f"TN_w={row['tn_w']:,.0f} FN_w={row['fn_w']:,.0f} "
            f"(predicted positive weight={row['predicted_positive_weight']:,.0f})"
        )

    lines.append("\n## 4. Combined validation results\n")
    lines.append(
        f"- Combined rows={comb['rows']:,}, weight={comb['weight_sum']:,.0f}, "
        f"weighted prevalence={comb['weighted_prevalence']:.6%}\n"
        f"- ROC-AUC={fmt(comb['roc_auc'])}, PR-AUC={fmt(comb['pr_auc'])}, "
        f"Brier={fmt(comb['brier'], 8)} (baseline {fmt(comb['brier_baseline'], 8)}, "
        f"BSS={fmt(comb['brier_skill_score'])}), Log loss={fmt(comb['log_loss'], 8)}\n"
        f"- Recall={fmt(comb['recall'])}, Specificity={fmt(comb['specificity'])}, "
        f"Precision={fmt(comb['precision'])}\n"
        "- Combined metrics were computed from row-level predictions concatenated "
        "in original validation order — not by averaging subgroup AUCs. "
        "`menopaus==9` validation rows were excluded."
    )

    lines.append("\n## 5. Comparison with unified Logistic Regression\n")
    lines.append(
        "| Model | ROC-AUC | PR-AUC | Brier | BSS | Log loss |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in comparison_df.iterrows():
        lines.append(
            f"| {r['model']} | {fmt(r['roc_auc'])} | {fmt(r['pr_auc'])} | "
            f"{fmt(r['brier'], 8)} | {fmt(r['brier_skill_score'])} | "
            f"{fmt(r['log_loss'], 8)} |"
        )
    if unified_metrics is not None:
        lines.append(
            "\nThe unified model was evaluated on the full Validation set "
            "(including `menopaus==9` rows) from the frozen "
            "`val_predictions_logistic_regression.csv` artifact. The Barlow "
            "combined model covers only rows with `menopaus ∈ {0,1}`. "
            "The comparison is therefore descriptive, not like-for-like."
        )

    lines.append("\n## 6. How this differs from Barlow et al. (2006)\n")
    lines.append(
        "1. **No `race`/`Hispanic`.** The postmenopausal model is a 9-feature "
        "modified version, not the paper's 11-feature model.\n"
        "2. **Estimators differ.** SAS `PROC LOGISTIC` (Fisher scoring, exact "
        "frequency-weighted likelihood, no penalty) vs scikit-learn "
        "unpenalized `LogisticRegression` (`penalty=None`, Newton-CG family "
        "solver differences, `tol`-based stopping). Coefficients, SEs, and "
        "CIs will not match SAS output exactly.\n"
        "3. **Splits differ.** This project uses a frozen 85/15 Train/Validation "
        "split derived from `train_full.csv`, not the paper's 75/25 split.\n"
        "4. **`menopaus == 9` handling.** Unknown-menopause rows exist in this "
        "dataset (Train: see §2) and are excluded here; the original paper "
        "reports models by `menopaus` subgroup with the same exclusion rule.\n"
        "5. **Metric conventions.** ROC-AUC (c-statistic) is un-binned pairwise "
        "concordance; SAS `NCONCORDBIN` bins pairs. PR-AUC, Brier, and Log loss "
        "are project additions, not in the paper."
    )

    lines.append("\n## 7. Interpretation\n")
    lines.append(
        "- A higher ROC-AUC alone does not establish clinical utility; PR-AUC, "
        "Brier skill vs the training-prevalence baseline, and log loss are "
        "reported alongside.\n"
        "- If either subgroup's Brier ≈ baseline or BSS ≈ 0, its probabilities "
        "add little over a constant-prevalence predictor.\n"
        "- At threshold 0.5 with prevalence ≈ 0.49%, recall will typically be "
        "near 0 and precision undefined — the decision threshold is not tuned."
    )

    lines.append("\n## 8. Reproducibility\n")
    lines.append(
        "```powershell\n"
        "python -m outputs.experiments.barlow_2006_modified.train_evaluate\n"
        "python -m unittest discover -s tests -v\n"
        "```\n"
        "Artifacts: `models/*_lr.joblib`, `preprocessing/*_encoder.joblib`, "
        "`predictions/validation_predictions.csv`, `reports/*.csv`, "
        "`logs/training.log`, `config/experiment_config.json`."
    )
    lines.append(
        "\ntest_accessed=False. No existing experiment, split file, or artifact "
        "was modified."
    )

    (REPORTS_DIR / "experiment_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    _log(log_lines, "wrote reports/experiment_report.md")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return {
        "subgroups": subgroup_rows,
        "combined": combined_metrics,
        "unified": unified_metrics,
        "comparison": comparison_rows,
    }


if __name__ == "__main__":
    main()
