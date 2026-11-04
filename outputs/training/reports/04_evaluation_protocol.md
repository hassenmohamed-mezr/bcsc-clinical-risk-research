# Unified Evaluation Protocol

Date: 2026-11-04. Scope: define one comparable evaluation protocol for Logistic Regression, Random Forest, Gradient Boosting, and MLP on the BCSC splits. This document defines metrics and reporting rules only. No model was trained or evaluated, no preprocessing artifact was changed, and `test.csv` was not opened.

> **Status: open items partly resolved by report 07.** The final experiment decisions are in
> `outputs/training/reports/07_final_experiment_decisions.md`. Weighted PR-AUC is confirmed as the
> primary CV selection metric, Gradient Boosting is fixed to scikit-learn, the MLP scaler is bundled
> in the model pipeline, and `surgmeno`/`hrt` code 9 stays one category. The metric definitions in
> this report remain authoritative.

## 1. Evaluation rules

- **Fit on Training, compare on Validation.** Models are fitted on `train` with `count` as `sample_weight`; model comparison uses `val` only. `test` is evaluated once, after the model, encoder, threshold rule, and protocol are frozen.
- **No test-driven decisions.** Test labels, test prevalence, and test metrics must not influence model choice, threshold choice, feature choice, or preprocessing.
- **Same rows for every model.** Every model is scored on every Validation row. Rows are never dropped per model; a model that cannot score all rows is non-comparable and must be fixed.
- **Probabilities are required.** Each model must expose the positive-class probability `p_i = P(cancer=1 | x_i)` through `predict_proba`, using `classes_` to select the `cancer=1` column. Raw `decision_function` scores are not accepted for probability metrics such as Brier, and are not used for the unified comparison.
- **Weights are frequency weights.** `count` is applied as `sample_weight` in fitting and in every metric. No row expansion is performed; weighted formulas are mathematically equivalent to expanding each row `count` times.
- **Transform-only preprocessing.** The saved encoders are applied with `transform`, never refit. The MLP `StandardScaler` is a step inside the MLP model pipeline: it is fitted on each CV training fold and on the full Training matrix for the final model, never on Validation or Test.
- **Primary selection metric.** Weighted PR-AUC (average precision with `count` as `sample_weight`) is the primary CV hyperparameter-selection metric. Weighted ROC-AUC and negative Brier are supporting scores. Unweighted/row-equal metrics are secondary diagnostics only and must never drive model selection.
- **No composite score.** All six metrics are reported. No single derived score replaces the table; the full weighted table is always reported, and no post-hoc metric may be introduced to favour a model.

## 2. Labels and threshold

| Item | Definition |
|---|---|
| Positive class | `cancer = 1` (BCSC cancer outcome recorded for the screening record) |
| Negative class | `cancer = 0` (no recorded cancer outcome) |
| Prediction | `p_i` = predicted probability of `cancer = 1`, required to be finite and in `[0, 1]` |
| Default threshold | `t = 0.5` |
| Hard classification | `ŷ_i = 1` if `p_i >= t`, otherwise `0` |
| Ties | A probability exactly equal to `t` is classified positive |
| Primary comparison | Threshold-independent metrics (ROC-AUC, PR-AUC, Brier) plus threshold-dependent metrics at `t = 0.5` |
| Threshold changes | Not tuned per model on Validation. Any non-default operating threshold must be pre-specified or selected on Training only, and reported separately with its source |

`t = 0.5` is a fixed statistical default for initial comparisons, not a clinically validated operating point. It must not be described as a clinical decision threshold.

## 3. Weighted quantities

For rows `i = 1..n`, target `y_i ∈ {0,1}`, predicted probability `p_i`, and frequency weight `w_i = count_i > 0`:

- Total weight: `W = Σ w_i`
- Positive weight: `W+ = Σ_{y_i=1} w_i`
- Negative weight: `W− = Σ_{y_i=0} w_i`
- Weighted prevalence: `π = W+ / W`

At threshold `t`, define the weighted confusion counts:

- `TP_w(t) = Σ_i w_i · 1[y_i = 1, p_i >= t]`
- `FP_w(t) = Σ_i w_i · 1[y_i = 0, p_i >= t]`
- `FN_w(t) = Σ_i w_i · 1[y_i = 1, p_i < t]`
- `TN_w(t) = Σ_i w_i · 1[y_i = 0, p_i < t]`

These satisfy `TP_w + FN_w = W+`, `TN_w + FP_w = W−`, and `TP_w + FP_w + TN_w + FN_w = W`.

All reported metrics are weighted. Raw row counts may be reported as a secondary diagnostic, clearly labelled, but must never be substituted for weighted results.

## 4. Metric definitions

### 4.1 ROC-AUC

**What it measures.** Discrimination across all possible thresholds: the probability that a randomly selected positive-weight case receives a higher predicted probability than a randomly selected negative-weight case. It measures ranking, not calibration.

**Calculation.**

- At each threshold: `TPR(t) = TP_w(t) / W+` and `FPR(t) = FP_w(t) / W−`.
- `ROC-AUC = ∫ TPR d(FPR)` over the weighted ROC curve, with tied scores handled by the standard rank/trapezoid convention.
- Reference implementation: `roc_auc_score(y, p, sample_weight=w)`.

**Threshold / probability requirement.** Threshold-independent; requires continuous probabilities (or scores) for all rows. It does not use a fixed `t`.

**Weighting.** Both TPR and FPR use weighted counts, so each aggregated row contributes in proportion to `count`. Report weighted prevalence alongside it.

**Range and direction.** `[0, 1]`; higher is better; `0.5` is chance-level for a balanced ranking interpretation. ROC-AUC can look optimistic when positives are rare.

### 4.2 PR-AUC (Average Precision)

**What it measures.** The trade-off between precision and recall across thresholds, summarised as average precision. It focuses on the positive class and is more informative than ROC-AUC when positives are rare.

**Calculation.**

- At each threshold: `P(t) = TP_w(t) / (TP_w(t) + FP_w(t))` and `R(t) = TP_w(t) / W+`.
- `PR-AUC = AP = Σ_n (R_n − R_{n−1}) · P_n`, with thresholds ordered by decreasing score and `R_0 = 0`.
- This is step-wise average precision, not trapezoidal interpolation under the PR curve.
- Reference implementation: `average_precision_score(y, p, sample_weight=w)`.

**Threshold / probability requirement.** Threshold-independent; requires probabilities.

**Weighting.** Precision, recall, and the increments in recall are computed from weighted counts.

**Range and direction.** `[0, 1]`; higher is better. The no-skill baseline is the weighted prevalence `π`, not `0.5`. PR-AUC must always be reported with `π` because it is prevalence-sensitive and not comparable across populations with different cancer rates.

### 4.3 Recall / Sensitivity

**What it measures.** The fraction of weighted positive cases that are correctly classified as positive at the chosen threshold.

**Calculation.** `Recall(t) = TP_w(t) / (TP_w(t) + FN_w(t)) = TP_w(t) / W+`.

**Threshold / probability requirement.** Threshold-dependent. It is computed from `p_i` and the fixed `t` (equivalently, from hard predictions produced at that `t`).

**Weighting.** Weighted true positives divided by total positive weight. A missed row contributes `count` missed cases, not one.

**Range and direction.** `[0, 1]`; higher is better. Report `t` with every value.

### 4.4 Specificity

**What it measures.** The fraction of weighted negative cases that are correctly classified as negative at the chosen threshold.

**Calculation.** `Specificity(t) = TN_w(t) / (TN_w(t) + FP_w(t)) = TN_w(t) / W−`.

**Threshold / probability requirement.** Threshold-dependent; computed from `p_i` and `t`.

**Weighting.** Weighted true negatives divided by total negative weight.

**Range and direction.** `[0, 1]`; higher is better. Report `t`. Specificity is not the same as negative predictive value; NPV is not part of this protocol.

### 4.5 Precision

**What it measures.** Of the weighted records classified positive at the chosen threshold, the fraction that are truly positive. Also called positive predictive value (PPV).

**Calculation.** `Precision(t) = TP_w(t) / (TP_w(t) + FP_w(t))`.

**Threshold / probability requirement.** Threshold-dependent; computed from `p_i` and `t`.

**Weighting.** Weighted true positives divided by total weight of predicted positives.

**Range and direction.** `[0, 1]`; higher is better. Precision is undefined when no weight is predicted positive at `t`; see Section 6.

### 4.6 Brier Score

**What it measures.** Mean squared error of the predicted probabilities. It captures both discrimination and calibration; it rewards probabilities that are close to the observed 0/1 outcomes, not just correctly ranked.

**Calculation.** `Brier = (1/W) · Σ_i w_i · (p_i − y_i)^2`.

**Threshold / probability requirement.** Threshold-independent; requires probabilities in `[0, 1]`. It cannot be computed from hard class labels alone.

**Weighting.** Weighted mean squared error, with `W` as the denominator.

**Range and direction.** `[0, 1]`; lower is better. For context, a constant predictor `p_i = π` gives `Brier_ref = π(1 − π)`.

**Optional supplementary skill score.** `Brier Skill Score = 1 − Brier / Brier_ref`. Positive values beat the constant-prevalence baseline; it is `N/A` if `π` is 0 or 1. This score is supplementary and does not replace the Brier score.

### 4.7 Metric summary

| Metric | Uses predicted probabilities | Uses threshold `t` | Weighted denominator | Better |
|---|---|---|---|---|
| ROC-AUC | yes (ranking) | no | `W+`, `W−` across thresholds | higher |
| PR-AUC (AP) | yes (ranking) | no | `W+`; precision uses predicted-positive weight | higher |
| Recall / Sensitivity | yes, or hard labels at `t` | yes | `W+` | higher |
| Specificity | yes, or hard labels at `t` | yes | `W−` | higher |
| Precision | yes, or hard labels at `t` | yes | predicted-positive weight | higher |
| Brier | yes | no | `W` | lower |

## 5. Reporting format

### 5.1 Primary Validation table (one row per model)

| Model | Encoder | ROC-AUC | PR-AUC | Validation prevalence `π` | Brier |
|---|---|---|---|---|---|

### 5.2 Threshold-dependent table at `t = 0.5` (one row per model)

| Model | Threshold | Recall | Specificity | Precision | `TP_w` | `FP_w` | `TN_w` | `FN_w` | Predicted-positive weight |
|---|---|---|---|---|---|---|---|---|---|

- Report `t` explicitly in every threshold-dependent table and figure.
- Report the weighted confusion counts so every rate can be reconstructed.
- Report `π` with PR-AUC, and report weighted `W+`, `W−`, and `W`.
- Raw row counts and an unweighted confusion matrix may be included as a clearly labelled diagnostic appendix only.
- Round rates to four decimal places and Brier to six decimal places; keep full precision in any machine-readable output.
- Initial comparisons use point estimates only. Confidence intervals or significance tests are optional and, if added, must be pre-specified with a fixed resampling seed and a stated resampling unit (aggregated rows with their `count` weights). They must not be introduced selectively for one model.
- Threshold sweeps may be shown as supplementary ROC/PR curves or a fixed grid (for example `t = 0.05, 0.10, …, 0.95`). They must not be used to pick the best threshold per model for the primary table.

### 5.3 Required metadata per result

Record the model name and version, encoder family, hyperparameters, random seed, scikit-learn version, split name, evaluation row count, weight sum, `π`, and threshold. A metric without this metadata is not comparable.

## 6. Undefined metrics and missing classes

Undefined values are reported as `N/A` with a reason. They are never silently replaced by `0`, `0.5`, or `1`, and rows are never dropped to make a metric computable.

| Situation | Handling |
|---|---|
| `W+ = 0` (no weighted positives) | ROC-AUC, PR-AUC, and recall are `N/A`; precision is computed only if some weight is predicted positive, otherwise `N/A`; specificity and Brier remain defined |
| `W− = 0` (no weighted negatives) | ROC-AUC, PR-AUC, and specificity are `N/A`; recall, precision, and Brier may still be defined but the result is flagged as degenerate |
| Either class absent | The comparison is invalid; report the split as non-evaluable rather than ranking models |
| No predicted positives at `t` | Precision is `N/A` (zero denominator); recall, specificity, and Brier remain defined |
| All rows predicted positive at `t` | Specificity is `0`; precision equals `π`; recall is `1`; these are defined but flagged |
| No predicted negatives at `t` | Specificity is `0` by the formula; precision and recall remain defined |
| Constant probabilities | ROC-AUC is `0.5` when both classes exist; PR-AUC equals `π`; flag the model as degenerate rather than treating the values as informative |
| `Brier_ref = 0` (`π` is 0 or 1) | Brier Skill Score is `N/A`; the Brier score itself may still be defined |
| `NaN`, infinite, or out-of-range probabilities | Fail the evaluation; do not clip, impute, or drop rows silently |
| Missing target or missing/non-positive weight | Fail the evaluation; the loader already enforces binary target and strictly positive weights |
| Model exposes no `cancer = 1` probability column | Fail the evaluation; do not substitute hard labels for probability metrics |
| Empty evaluation slice | All metrics are `N/A`; do not report a value |

A model with any failed validity check is excluded from comparison until fixed; it is not assigned a fallback score.

## 7. Comparison principles

- Threshold-independent metrics answer whether a model ranks and calibrates risk well; threshold-dependent metrics answer what happens at the fixed `t = 0.5` operating point. Both are required.
- Because weighted prevalence is approximately `0.49%`, ROC-AUC can appear high even when positive-class performance is modest. PR-AUC must be interpreted against `π ≈ 0.0049`, and recall/precision must be read together with specificity.
- No model is declared better on a single metric. The pre-specified selection hierarchy is: weighted PR-AUC as the primary criterion, with weighted ROC-AUC and weighted Brier as supporting metrics and the full weighted table always reported. The hierarchy was confirmed in report 07 and must not be changed after seeing results.
- Small differences must not be presented as meaningful without a pre-specified uncertainty method.

## 8. Open decisions

1. **Resolved: PR-AUC is the primary selection metric.** Weighted PR-AUC is primary; weighted ROC-AUC and Brier are supporting; unweighted metrics are secondary diagnostics only. Confirmed in report 07.
2. Decide whether confidence intervals or repeated weighted bootstrap will be used; if yes, fix the seed and resampling unit before evaluation.
3. Define the clinical operating threshold if one is needed; `t = 0.5` is only the initial statistical default.
4. Decide whether the optional Brier Skill Score and threshold sweeps will be included in the main report or an appendix.
5. Unweighted diagnostics are optional and secondary only; weighted results remain authoritative. Any unweighted table must be labelled as treating each aggregated row equally.
6. **Resolved: Gradient Boosting and MLP packaging.** scikit-learn `GradientBoostingClassifier` is fixed; the MLP uses One-Hot plus an in-pipeline `StandardScaler` serialized with the classifier. See report 07.
7. **Resolved: `surgmeno`/`hrt` code 9.** Kept as one category; the source combines Unknown and Not Applicable; raw data and mapping are unchanged. See report 07.

## 9. Reference calculation pattern

The following pattern is illustrative. It assumes aligned arrays `y`, `p`, `w` and both classes present. The shared weighted implementation now lives in `src/training/cv.py` (`weighted_metrics`, `weighted_scores`, `cross_validate_weighted`); use it rather than re-deriving the formulas per model.

```python
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

t = 0.5
pred = (p >= t).astype(int)

tp_w = w[(y == 1) & (pred == 1)].sum()
fp_w = w[(y == 0) & (pred == 1)].sum()
fn_w = w[(y == 1) & (pred == 0)].sum()
tn_w = w[(y == 0) & (pred == 0)].sum()

recall = tp_w / (tp_w + fn_w)
specificity = tn_w / (tn_w + fp_w)
precision = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else np.nan
brier = np.average((p - y) ** 2, weights=w)
roc_auc = roc_auc_score(y, p, sample_weight=w)
pr_auc = average_precision_score(y, p, sample_weight=w)
```

This pattern is a definition aid. The shared implementation in `src/training/cv.py` is the executable reference and is covered by `tests/test_training_cv.py`; no project model has been fitted.

Last updated: 2026-11-04
