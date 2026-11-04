# Modified Barlow Architecture — Experiment Report

**This is a Modified Barlow Architecture, not a reproduction of Barlow et al. (2006).** Subgroup feature lists follow the paper's architecture, but `race` and `Hispanic` are entirely excluded (not recovered or used), and the estimator differs from SAS in known, documented ways below.

## 1. What was actually implemented

- Two independent, unpenalized logistic regressions:
  - **Premenopausal** (`menopaus == 0`): `agegrp, brstproc, nrelbc, density`.
  - **Postmenopausal** (`menopaus == 1`): `agegrp, bmi, agefirst, brstproc, nrelbc, hrt, surgmeno, lastmamm, density`.
- `menopaus` used for routing only; `menopaus == 9` excluded from training, evaluation, and the combined evaluation.
- `count` used exclusively as `sample_weight`. No class weighting, SMOTE, oversampling/undersampling, trees, boosting, or neural nets.
- Encoders: one-hot, drop first level per feature (reference-category coding equivalent to SAS `param=ref` / `ref=first`), with BCSC codebook category order (numeric ascending), fitted on Train subgroup rows only.
- Splits: frozen `train.csv` (fitting only) and `val.csv` (external evaluation only). `test.csv` was **never accessed**.

## 2. Sample counts and weights

| Split | Rows | `sum(count)` | Weighted prevalence |
|---|---:|---:|---:|
| Train | 152,149 | 1,525,868 | 0.488378% |
| Validation | 28,316 | 269,271 | 0.488356% |
| Excluded (`menopaus==9`) Train | 8,434 | 119,308 | — |
| Excluded (`menopaus==9`) Validation | 1,565 | 17,138 | — |

### Subgroup descriptive counts

| Subgroup | Train rows | Train `sum(count)` | Train prev. | Val rows | Val `sum(count)` | Val prev. |
|---|---:|---:|---:|---:|---:|---:|
| premenopausal | 20,841 | 363,363 | 0.307131% | 3,834 | 62,861 | 0.283164% |
| postmenopausal | 122,874 | 1,043,197 | 0.570937% | 22,917 | 189,272 | 0.550002% |

## 3. Subgroup validation results

| Model | ROC-AUC | PR-AUC | Brier | Brier baseline | BSS | Log loss | Recall | Specificity | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| premenopausal | 0.558866 | 0.003766 | 0.00282464 | 0.00282368 | -0.000337 | 0.01951842 | 0.000000 | 1.000000 | N/A |
| postmenopausal | 0.607458 | 0.008135 | 0.00546559 | 0.00546981 | 0.000772 | 0.03369872 | 0.000000 | 1.000000 | N/A |

Confusion matrices (weighted, threshold 0.5):

- **premenopausal**: TP_w=0 FP_w=0 TN_w=62,683 FN_w=178 (predicted positive weight=0)
- **postmenopausal**: TP_w=0 FP_w=0 TN_w=188,231 FN_w=1,041 (predicted positive weight=0)

## 4. Combined validation results

- Combined rows=26,751, weight=252,133, weighted prevalence=0.483475%
- ROC-AUC=0.623213, PR-AUC=0.007664, Brier=0.00480716 (baseline 0.00481141, BSS=0.000885), Log loss=0.03016333
- Recall=0.000000, Specificity=1.000000, Precision=N/A
- Combined metrics were computed from row-level predictions concatenated in original validation order — not by averaging subgroup AUCs. `menopaus==9` validation rows were excluded.

## 5. Comparison with unified Logistic Regression

| Model | ROC-AUC | PR-AUC | Brier | BSS | Log loss |
|---|---:|---:|---:|---:|---:|
| premenopausal | 0.558866 | 0.003766 | 0.00282464 | -0.000337 | 0.01951842 |
| postmenopausal | 0.607458 | 0.008135 | 0.00546559 | 0.000772 | 0.03369872 |
| combined_two_model | 0.623213 | 0.007664 | 0.00480716 | 0.000885 | 0.03016333 |
| unified_logistic_regression | 0.624226 | 0.007885 | 0.00485498 | 0.000972 | 0.03038698 |

The unified model was evaluated on the full Validation set (including `menopaus==9` rows) from the frozen `val_predictions_logistic_regression.csv` artifact. The Barlow combined model covers only rows with `menopaus ∈ {0,1}`. The comparison is therefore descriptive, not like-for-like.

## 6. How this differs from Barlow et al. (2006)

1. **No `race`/`Hispanic`.** The postmenopausal model is a 9-feature modified version, not the paper's 11-feature model.
2. **Estimators differ.** SAS `PROC LOGISTIC` (Fisher scoring, exact frequency-weighted likelihood, no penalty) vs scikit-learn unpenalized `LogisticRegression` (`penalty=None`, Newton-CG family solver differences, `tol`-based stopping). Coefficients, SEs, and CIs will not match SAS output exactly.
3. **Splits differ.** This project uses a frozen 85/15 Train/Validation split derived from `train_full.csv`, not the paper's 75/25 split.
4. **`menopaus == 9` handling.** Unknown-menopause rows exist in this dataset (Train: see §2) and are excluded here; the original paper reports models by `menopaus` subgroup with the same exclusion rule.
5. **Metric conventions.** ROC-AUC (c-statistic) is un-binned pairwise concordance; SAS `NCONCORDBIN` bins pairs. PR-AUC, Brier, and Log loss are project additions, not in the paper.

## 7. Interpretation

- A higher ROC-AUC alone does not establish clinical utility; PR-AUC, Brier skill vs the training-prevalence baseline, and log loss are reported alongside.
- If either subgroup's Brier ≈ baseline or BSS ≈ 0, its probabilities add little over a constant-prevalence predictor.
- At threshold 0.5 with prevalence ≈ 0.49%, recall will typically be near 0 and precision undefined — the decision threshold is not tuned.

## 8. Reproducibility

```powershell
python -m outputs.experiments.barlow_2006_modified.train_evaluate
python -m unittest discover -s tests -v
```
Artifacts: `models/*_lr.joblib`, `preprocessing/*_encoder.joblib`, `predictions/validation_predictions.csv`, `reports/*.csv`, `logs/training.log`, `config/experiment_config.json`.

test_accessed=False. No existing experiment, split file, or artifact was modified.