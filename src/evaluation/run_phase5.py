"""Phase 5: ensemble proposal (no application)."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import SUPMT
import SUPMT_CONFIG

BOOT = ROOT / "outputs" / "evaluation" / "metrics" / "paired_bootstrap_ci.csv"
OUT = ROOT / "outputs" / "evaluation" / "reports" / "proposed_model_weights.md"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    paired = pd.read_csv(BOOT)
    # Determine if any candidate has CI entirely above 0 (for ROC or PR)
    above_zero = {}
    for cand in ["lr_gb", "lr_gb_mlp", "lr_only"]:
        sub = paired[(paired["candidate"] == cand)]
        roc = sub[sub["metric"] == "roc_auc"].iloc[0]
        pr = sub[sub["metric"] == "pr_auc"].iloc[0]
        above_zero[cand] = (roc["ci_lower"] > 0) or (pr["ci_lower"] > 0)

    # Point estimates from val predictions
    val = pd.read_csv(ROOT / "outputs" / "training" / "predictions" / "val_predictions_all.csv")
    pivot = val.pivot(index="row_id", columns="model", values="probability")
    meta = val.drop_duplicates("row_id").set_index("row_id")
    y = meta["y_true"].to_numpy(float)
    w = meta["count"].to_numpy(float)
    from sklearn.metrics import roc_auc_score, average_precision_score

    def pr_auc(prob):
        return average_precision_score(y, prob, sample_weight=w)

    candidates = {
        "all4": pivot[["logistic_regression", "gradient_boosting", "mlp", "random_forest"]].mean(axis=1).to_numpy(),
        "lr_gb": pivot[["logistic_regression", "gradient_boosting"]].mean(axis=1).to_numpy(),
        "lr_gb_mlp": pivot[["logistic_regression", "gradient_boosting", "mlp"]].mean(axis=1).to_numpy(),
        "lr_only": pivot["logistic_regression"].to_numpy(),
    }
    point_pr = {k: pr_auc(v) for k, v in candidates.items()}

    all_above = any(above_zero.values())
    if all_above:
        # recommend candidate with CI entirely above 0 (prefer highest PR)
        justified = [c for c in above_zero if above_zero[c]]
        rec = max(justified, key=lambda c: point_pr[c])
    else:
        # simplest candidate not worse on point estimates (fewer models)
        # point estimate vs all4
        not_worse = [c for c in ["lr_only", "lr_gb", "lr_gb_mlp"] if point_pr[c] >= point_pr["all4"] - 1e-12]
        # prefer fewer models
        model_count = {"lr_only": 1, "lr_gb": 2, "lr_gb_mlp": 3}
        if not_worse:
            rec = min(not_worse, key=lambda c: model_count[c])
        else:
            rec = "all4"

    weight_map = {
        "all4": {"logistic_regression": 1.0, "gradient_boosting": 1.0, "mlp": 1.0, "random_forest": 1.0},
        "lr_gb": {"logistic_regression": 1.0, "gradient_boosting": 1.0, "mlp": 0.0, "random_forest": 0.0},
        "lr_gb_mlp": {"logistic_regression": 1.0, "gradient_boosting": 1.0, "mlp": 1.0, "random_forest": 0.0},
        "lr_only": {"logistic_regression": 1.0, "gradient_boosting": 0.0, "mlp": 0.0, "random_forest": 0.0},
    }
    proposed = weight_map[rec]

    # Before/after on accepted test cases
    import json
    cases = json.load(open(ROOT / "BCSC_SUPMT_60_test_cases.json" if (ROOT / "BCSC_SUPMT_60_test_cases.json").exists() else Path(r"C:\Users\marwa\Downloads\BCSC_SUPMT_60_test_cases.json"), encoding="utf-8"))["cases"]
    accepted = [c for c in cases if c["expected"]["status"] == "accept"]

    before_rows = []
    after_rows = []
    for case in accepted:
        with contextlib.redirect_stdout(io.StringIO()):
            base = SUPMT.predict_patient(case["input"])
        # after: temporarily compute weighted average with proposed weights
        probs = [base[k] for k in ["logistic_regression", "gradient_boosting", "mlp", "random_forest"]]
        keys = ["logistic_regression", "gradient_boosting", "mlp", "random_forest"]
        num = sum(base[k] * proposed[k] for k in keys)
        den = sum(proposed[k] for k in keys)
        after_wa = num / den
        before_rows.append((case["id"], base["weighted_average"]))
        after_rows.append((case["id"], after_wa))

    lines = []
    lines.append("# Proposed Model Weights")
    lines.append("")
    lines.append(f"Recommendation: `{rec}`")
    lines.append("")
    lines.append("Paired bootstrap CI vs all4 includes 0 for all candidates:" + (" yes" if not all_above else " no"))
    lines.append("")
    lines.append("Candidate PR-AUC point estimates:")
    for k, v in point_pr.items():
        lines.append(f"- {k}: {v:.10f}")
    lines.append("")
    lines.append("Paired bootstrap CI table:")
    lines.append("")
    lines.append(paired.to_string(index=False))
    lines.append("")
    lines.append("Proposed SUPMT_CONFIG.MODEL_WEIGHTS diff:")
    lines.append("")
    lines.append("```python")
    lines.append("MODEL_WEIGHTS = {")
    for k in keys:
        lines.append(f'    "{k}": {proposed[k]},')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("Before/after on 40 accepted test cases:")
    lines.append("")
    lines.append("| Case | Before WA | After WA |")
    lines.append("|---|---:|---:|")
    for (cid, before), (_, after) in zip(before_rows, after_rows):
        lines.append(f"| {cid} | {before*100:.4f}% | {after*100:.4f}% |")
    lines.append("")
    lines.append("AWAITING APPROVAL")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("AWAITING APPROVAL")
    print("recommendation", rec)


if __name__ == "__main__":
    main()