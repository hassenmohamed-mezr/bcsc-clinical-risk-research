"""One-time final Test evaluation for the frozen Modified Barlow models.

Explicit user-authorized exception to the usual no-Test rule. This script is
run ONCE. It loads data/04_splits/test.csv, evaluates the frozen Barlow
subgroup models (routed by menopa) and the frozen unified LR model, and writes
results under test_evaluation/. It does not refit anything.
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
OUT = HERE / "test_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

PRE = ["agegrp", "brstproc", "nrelbc", "density"]
POST = ["agegrp", "bmi", "agefirst", "brstproc", "nrelbc", "hrt", "surgmeno", "lastmamm", "density"]
UNIFIED_FEATURES = ["agegrp", "bmi", "agefirst", "menopaus", "surgmeno", "hrt", "nrelbc", "density", "brstproc", "lastmamm"]


def wlogloss(y, p, w):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    import numpy as _np
    return float(log_loss(y, _np.column_stack([1 - p, p]), labels=[0, 1], sample_weight=w))


def metrics(y, p, w, train_pi):
    m = weighted_metrics(np.asarray(y, int), np.asarray(p, float), np.asarray(w, float), threshold=0.5)
    p = np.asarray(p, float)
    w = np.asarray(w, float)
    y = np.asarray(y, int)
    brier_base = train_pi * (1 - train_pi)
    ll_base = wlogloss(y, np.full(len(y), train_pi), w)
    return {
        "rows": int(len(y)),
        "weight_sum": float(w.sum()),
        "weighted_prevalence": float(m["weighted_prevalence"]),
        "train_prevalence_baseline": float(train_pi),
        "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
        "brier": m["brier"], "brier_baseline": float(brier_base),
        "brier_skill_score": (1 - m["brier"] / brier_base) if brier_base else None,
        "log_loss": wlogloss(y, p, w), "log_loss_baseline": float(ll_base),
        "recall": m["recall"], "specificity": m["specificity"], "precision": m["precision"],
        "tp_w": m["tp_w"], "fp_w": m["fp_w"], "tn_w": m["tn_w"], "fn_w": m["fn_w"],
        "mean_predicted_probability": float(np.average(p, weights=w)),
    }


def prob(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def main():
    test = pd.read_csv(TEST_PATH)
    test.insert(0, "row_id", np.arange(len(test), dtype=int))
    train = pd.read_csv(ROOT / "data" / "04_splits" / "train.csv")

    results = {}
    combined = np.full(len(test), np.nan)

    for key, feats, meno in [("premenopausal", PRE, 0), ("postmenopausal", POST, 1)]:
        model = joblib.load(HERE / "models" / f"{key}_lr.joblib")
        enc = joblib.load(HERE / "preprocessing" / f"{key}_encoder.joblib")
        tr = train.loc[train["menopaus"] == meno]
        te = test.loc[test["menopaus"] == meno]
        pi = float(tr.loc[tr["cancer"] == 1, "count"].sum() / tr["count"].sum())
        p = prob(model, enc.transform(te[feats].to_numpy()))
        combined[te.index.to_numpy()] = p
        results[key] = {"menopaus": meno, "features": feats, **metrics(te["cancer"].to_numpy(), p, te["count"].to_numpy(), pi)}

    routed = test.loc[test["menopaus"].isin([0, 1])]
    tr_c = train.loc[train["menopaus"].isin([0, 1])]
    pi_c = float(tr_c.loc[tr_c["cancer"] == 1, "count"].sum() / tr_c["count"].sum())
    results["combined_two_model"] = metrics(
        routed["cancer"].to_numpy(), combined[routed.index.to_numpy()],
        routed["count"].to_numpy(), pi_c)

    # Unified frozen LR — full test set (it was trained on all train rows).
    umodel = joblib.load(ROOT / "models" / "trained" / "logistic_regression_calibrated.joblib")
    uenc = joblib.load(ROOT / "models" / "preprocessing" / "onehot_pipeline.joblib")
    tri = train["cancer"].to_numpy(int)
    trw = train["count"].to_numpy(float)
    pi_u = float(trw[tri == 1].sum() / trw.sum())
    pu = prob(umodel, uenc.transform(test[UNIFIED_FEATURES]))
    results["unified_logistic_regression_full_test"] = metrics(
        test["cancer"].to_numpy(), pu, test["count"].to_numpy(), pi_u)

    pred = pd.DataFrame({
        "row_id": test["row_id"], "menopaus": test["menopaus"],
        "y_true": test["cancer"], "count": test["count"],
        "probability_barlow_combined": combined, "probability_unified": pu,
    })
    pred.to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame([{"model": k, **v} for k, v in results.items()]).to_csv(OUT / "test_metrics.csv", index=False)
    (OUT / "test_metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    excl = test.loc[test["menopaus"] == 9]
    lines = [
        "# Final Test Evaluation (one-time, user-authorized)\n",
        "This is the single authorized Test evaluation. It deviates from the "
        "project's no-Test rule by explicit user decision. No model was refit; "
        "frozen artifacts were scored on test.csv.\n",
        f"Test rows={len(test):,}, sum(count)={test['count'].sum():,.0f}, "
        f"weighted prevalence={float(test.loc[test['cancer']==1,'count'].sum()/test['count'].sum()):.6%}.",
        f"Excluded from Barlow combined (menopaus==9): {len(excl):,} rows, "
        f"sum(count)={excl['count'].sum():,.0f}.",
        "\n| Model | Rows | ROC-AUC | PR-AUC | Brier | BSS | Log loss | Recall | Specificity | Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, r in results.items():
        f = lambda v, d=6: "N/A" if v is None else f"{v:.{d}f}"
        lines.append(f"| {k} | {r['rows']:,} | {f(r['roc_auc'])} | {f(r['pr_auc'])} | "
                     f"{f(r['brier'],8)} | {f(r['brier_skill_score'])} | {f(r['log_loss'],8)} | "
                     f"{f(r['recall'])} | {f(r['specificity'])} | {f(r['precision'])} |")
    lines.append("\nUnified model scored on the FULL test set (including menopa==9); "
                 "Barlow combined covers only menopa in {0,1}. Comparison is descriptive.")
    (OUT / "test_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
