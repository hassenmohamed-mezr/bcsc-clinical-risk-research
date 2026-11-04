"""One-time final Test evaluation of ALL frozen models.

Models (6):
- 4 unified models: logistic_regression, random_forest, gradient_boosting, mlp
  (calibrated artifacts in models/trained/)
- 2 Modified Barlow models: premenopausal_lr, postmenopausal_lr
  (+ their combined routing)

No model is refit. Frozen artifacts are only loaded and scored on test.csv.
User explicitly authorized this Test evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.cv import weighted_metrics  # noqa: E402

TEST_PATH = ROOT / "data" / "04_splits" / "test.csv"
TRAIN_PATH = ROOT / "data" / "04_splits" / "train.csv"
MODELS_DIR = ROOT / "models" / "trained"
ENC_DIR = ROOT / "models" / "preprocessing"
OUT = HERE / "test_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_ORDER = ["agegrp", "bmi", "agefirst", "menopaus", "surgmeno", "hrt",
                 "nrelbc", "density", "brstproc", "lastmamm"]
PRE = ["agegrp", "brstproc", "nrelbc", "density"]
POST = ["agegrp", "bmi", "agefirst", "brstproc", "nrelbc", "hrt",
        "surgmeno", "lastmamm", "density"]

UNIFIED = [
    ("logistic_regression", "onehot_pipeline.joblib"),
    ("random_forest", "ordinal_tree_pipeline.joblib"),
    ("gradient_boosting", "ordinal_tree_pipeline.joblib"),
    ("mlp", "onehot_pipeline.joblib"),
]


def wlogloss(y, p, w):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1], sample_weight=w))


def metrics(y, p, w, train_pi):
    y = np.asarray(y, int); p = np.asarray(p, float); w = np.asarray(w, float)
    m = weighted_metrics(y, p, w, threshold=0.5)
    brier_base = train_pi * (1 - train_pi)
    return {
        "rows": int(len(y)),
        "weight_sum": float(w.sum()),
        "weighted_prevalence": float(m["weighted_prevalence"]),
        "train_prevalence_baseline": float(train_pi),
        "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
        "brier": m["brier"], "brier_baseline": float(brier_base),
        "brier_skill_score": (1 - m["brier"] / brier_base) if brier_base else None,
        "log_loss": wlogloss(y, p, w),
        "log_loss_baseline": wlogloss(y, np.full(len(y), train_pi), w),
        "recall": m["recall"], "specificity": m["specificity"], "precision": m["precision"],
        "tp_w": m["tp_w"], "fp_w": m["fp_w"], "tn_w": m["tn_w"], "fn_w": m["fn_w"],
        "mean_predicted_probability": float(np.average(p, weights=w)),
    }


def prob(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def main():
    test = pd.read_csv(TEST_PATH)
    test.insert(0, "row_id", np.arange(len(test), dtype=int))
    train = pd.read_csv(TRAIN_PATH)
    tri = train["cancer"].to_numpy(int); trw = train["count"].to_numpy(float)
    pi_all = float(trw[tri == 1].sum() / trw.sum())

    results = {}
    preds = {"row_id": test["row_id"].to_numpy(), "menopaus": test["menopaus"].to_numpy(),
             "y_true": test["cancer"].to_numpy(), "count": test["count"].to_numpy()}

    # --- 4 unified frozen models on FULL test ---
    for key, enc_file in UNIFIED:
        model = joblib.load(MODELS_DIR / f"{key}_calibrated.joblib")
        enc = joblib.load(ENC_DIR / enc_file)
        p = prob(model, enc.transform(test[FEATURE_ORDER]))
        results[key] = metrics(test["cancer"].to_numpy(), p, test["count"].to_numpy(), pi_all)
        preds[f"prob_{key}"] = p
        print(f"{key}: ROC-AUC={results[key]['roc_auc']:.6f} PR-AUC={results[key]['pr_auc']:.6f} "
              f"Brier={results[key]['brier']:.8f}")

    # --- 2 Barlow models, routed ---
    combined = np.full(len(test), np.nan)
    for key, feats, meno in [("premenopausal", PRE, 0), ("postmenopausal", POST, 1)]:
        model = joblib.load(HERE / "models" / f"{key}_lr.joblib")
        enc = joblib.load(HERE / "preprocessing" / f"{key}_encoder.joblib")
        tr = train.loc[train["menopaus"] == meno]
        te = test.loc[test["menopaus"] == meno]
        pi = float(tr.loc[tr["cancer"] == 1, "count"].sum() / tr["count"].sum())
        p = prob(model, enc.transform(te[feats].to_numpy()))
        combined[te.index.to_numpy()] = p
        results[key] = {"menopaus": meno, **metrics(te["cancer"].to_numpy(), p, te["count"].to_numpy(), pi)}
        print(f"{key}: ROC-AUC={results[key]['roc_auc']:.6f} PR-AUC={results[key]['pr_auc']:.6f} "
              f"Brier={results[key]['brier']:.8f}")

    routed = test.loc[test["menopaus"].isin([0, 1])]
    tr_c = train.loc[train["menopaus"].isin([0, 1])]
    pi_c = float(tr_c.loc[tr_c["cancer"] == 1, "count"].sum() / tr_c["count"].sum())
    results["barlow_combined_two_model"] = metrics(
        routed["cancer"].to_numpy(), combined[routed.index.to_numpy()],
        routed["count"].to_numpy(), pi_c)
    preds["prob_barlow_combined"] = combined

    pd.DataFrame(preds).to_csv(OUT / "all_models_test_predictions.csv", index=False)
    pd.DataFrame([{"model": k, **v} for k, v in results.items()]).to_csv(
        OUT / "all_models_test_metrics.csv", index=False)
    (OUT / "all_models_test_metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    f = lambda v, d=6: "N/A" if v is None else f"{v:.{d}f}"
    order = ["logistic_regression", "random_forest", "gradient_boosting", "mlp",
             "premenopausal", "postmenopausal", "barlow_combined_two_model"]
    lines = ["# All-Models Final Test Evaluation (one-time, user-authorized)\n",
             f"Test rows={len(test):,}, sum(count)={test['count'].sum():,.0f}, "
             f"weighted prevalence={float(test.loc[test['cancer']==1,'count'].sum()/test['count'].sum()):.6%}.\n",
             "| Model | Rows | Weight | Prevalence | ROC-AUC | PR-AUC | Brier | Brier base | BSS | Log loss | Recall | Specificity | Precision | Mean prob |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in order:
        r = results[k]
        lines.append(f"| {k} | {r['rows']:,} | {r['weight_sum']:,.0f} | {r['weighted_prevalence']:.6%} | "
                     f"{f(r['roc_auc'])} | {f(r['pr_auc'])} | {f(r['brier'],8)} | {f(r['brier_baseline'],8)} | "
                     f"{f(r['brier_skill_score'])} | {f(r['log_loss'],8)} | {f(r['recall'])} | "
                     f"{f(r['specificity'])} | {f(r['precision'])} | {f(r['mean_predicted_probability'])} |")
    lines.append("\nUnified models scored on FULL Test (incl. menopa==9). "
                 "Barlow subgroups/combined cover only menopa in {0,1} (94,479 rows); "
                 "5,716 menopa==9 rows excluded from Barlow combined.")
    (OUT / "all_models_test_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote all_models_test_metrics.csv / .json / all_models_test_report.md")


if __name__ == "__main__":
    main()
