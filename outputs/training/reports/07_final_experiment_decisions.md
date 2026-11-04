# Final Training Experiment Decisions

Date: 2026-11-04. Scope: align the training configuration, model factories, weighted CV/scoring utilities, and related reports with the eight confirmed experiment decisions. No model was trained, tuned, or evaluated. No encoder or fitted preprocessing artifact was refitted. Validation was not loaded for fitting or selection, and `test.csv` was not accessed.

## 1. Verdict

All eight decisions are now reflected in code and/or an explicit, testable policy. The main implementation changes are:

- MLP: `StandardScaler` is now a step inside the MLP model pipeline (`SampleWeightPipeline`), not a separate object. The pipeline is fitted fresh on each CV training fold, so the scaler never sees that fold's validation rows, and the scaler + MLP serialize together.
- MLP: `early_stopping=False`; early-stopping-only parameters were removed; `max_iter` is an initial setting and a Training-only CV tuning dimension.
- Gradient Boosting: scikit-learn `GradientBoostingClassifier` is fixed for this phase; no external boosting library is imported or installed.
- Weighted CV: a deterministic, non-expanding weighted class-stratified fold builder, weighted metrics, weighted scorers, fold diagnostics, and a fold fitting/scoring helper now exist. `count` is passed to both `fit()` and scoring.
- Calibration: required, sigmoid (Platt), `ensemble=False`, Training-only weighted CV; the factory cannot select Validation or Test as the calibration split.
- Code 9: `surgmeno` and `hrt` keep one category in both encoder families; no raw-data or mapping change.

The configuration self-check passes 14 checks without live data and 15 with live Train/Validation data. The test suite passes 38/38. `test_accessed=false` and `models_trained=false` in both configuration runs.

## 2. Decision status

| # | Confirmed decision | Status | Implementation evidence | Verification |
|---:|---|---|---|---|
| 1 | Use `sklearn.ensemble.GradientBoostingClassifier`; no XGBoost/other boosting library | Implemented | `MODEL_SPECS["gradient_boosting"]` uses `sklearn.ensemble.GradientBoostingClassifier` with `ordinal_tree`; no external boosting import exists | `test_estimators_construct_with_central_seed`, `test_model_artifacts_are_unfitted_and_accept_sample_weight` |
| 2 | MLP uses existing One-Hot pipeline and `StandardScaler` inside the model pipeline; scaler fit per CV training fold; scaler + model saved together | Implemented | `build_model_pipeline("mlp")` returns `SampleWeightPipeline([("scaler", StandardScaler()), ("classifier", MLPClassifier(...))])`; `cross_validate_weighted` builds a fresh pipeline per fold and fits it on `X[train_index]`; `save_model_artifact`/`load_model_artifact` | `test_mlp_pipeline_bundles_scaler_and_disables_early_stopping`, `test_mlp_pipeline_serializes_scaler_and_model_together`, `test_scaler_is_fitted_on_training_fold_only` |
| 3 | Disable MLP early stopping; tune `max_iter` with Training-only CV; never use external Validation for early stopping or selection | Implemented | MLP `early_stopping=False`; `validation_fraction` and `n_iter_no_change` removed from `MODEL_SPECS`; `max_iter=200` initial and `max_iter` in `TUNING_SPACES["mlp"]`; `CV_POLICY["data"]="Training only"`; `CV_POLICY["validation_use"]` reserves Validation for evaluation | `test_mlp_pipeline_bundles_scaler_and_disables_early_stopping`, `test_full_configuration_validates` |
| 4 | Logistic Regression uses One-Hot without an additional scaler | Implemented | `MODEL_SPECS["logistic_regression"]` has `encoder="onehot"`, `scaler=None`; `build_model_pipeline` returns the estimator, not a scaled pipeline | `test_mlp_uses_training_only_standard_scaler`, `test_model_artifacts_are_unfitted_and_accept_sample_weight` |
| 5 | Use `count` as `sample_weight` for fitting and scoring; no row expansion; weighted PR-AUC primary; unweighted only secondary | Implemented | `weighted_metrics` passes `sample_weight` to ROC-AUC, average precision, Brier, and weighted confusion counts; `cross_validate_weighted` passes the training-fold weight to `fit` and the validation-fold weight to scoring; `CV_POLICY["primary_scoring"]="pr_auc"`; `row_equal_metrics` is explicitly labelled secondary | `test_weighted_metrics_match_sklearn_and_weighted_confusion`, `test_weighted_scores_use_count_and_pr_auc_is_primary`, `test_weights_reach_both_fit_and_weighted_scoring`, `test_unweighted_diagnostic_is_clearly_secondary` |
| 6 | Prefer weighted cancer-stratified folds without expansion; report rows/`sum(count)`/weighted prevalence; check material imbalance | Implemented | `WeightedStratifiedKFold` / `weighted_folds` perform deterministic greedy weighted class-stratified assignment; `fold_diagnostics` reports rows, positive rows, weight sum, positive/negative weight, weighted prevalence; `assess_fold_balance` checks pre-declared tolerances | `test_folds_cover_every_row_once_without_expansion`, `test_fold_diagnostics_report_weights_and_weighted_prevalence`, `test_material_imbalance_is_flagged`, `test_sklearn_compatible_splitter` |
| 7 | Calibration required; Training-only CV; weighted; no external Validation/Test | Implemented as configuration and factory; not fitted | `CALIBRATION_POLICY` (sigmoid, `ensemble=False`, Training-only, weighted); `build_training_calibrated_model` accepts only `split="train"` and wires `WeightedStratifiedKFold`; `CalibratedClassifierCV.fit(sample_weight=...)` is available and the MLP pipeline exposes `sample_weight` | `test_calibration_is_training_only_weighted_and_sigmoid`, `test_training_calibration_helper_uses_weighted_training_folds`, `test_full_configuration_validates` |
| 8 | Keep `surgmeno`/`hrt` code 9 as one category; document combined meaning; do not change raw data or mapping | Implemented | `CODE_9_POLICY` states single category, no split, no raw/mapping change; both encoder metadata families emit exactly one `surgmeno_9` and one `hrt_9`; saved one-hot categories contain one code 9 per feature | `test_code_9_remains_one_category_for_surgmeno_and_hrt` |

## 3. Implementation details

### 3.1 Model-to-input mapping preserved

| Model | Estimator | Encoder | Model artifact |
|---|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | `onehot` (43 columns) | estimator only; no scaler |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `ordinal_tree` (26 columns) | estimator only; no scaler |
| Gradient Boosting | `sklearn.ensemble.GradientBoostingClassifier` | `ordinal_tree` (26 columns) | estimator only; no scaler |
| MLP | `sklearn.neural_network.MLPClassifier` | `onehot` (43 columns) | `SampleWeightPipeline(StandardScaler, MLPClassifier)` |

The saved One-Hot encoder remains the upstream preprocessing artifact and is applied with `transform` only by `load_encoded_split`. The MLP model pipeline therefore starts from the existing 43-column One-Hot matrix and bundles the required scaler with the classifier. The encoder is not refit inside the model pipeline, so the fitted preprocessing artifact remains the single source of category mappings.

### 3.2 MLP scaling per fold

`build_model_pipeline("mlp")` constructs an unfitted pipeline. `cross_validate_weighted` calls the factory once per fold and fits the returned pipeline on that fold's training rows only:

```python
model = model_factory()
model.fit(X[train_index], y[train_index], sample_weight=count[train_index])
scores = model.predict_proba(X[validation_index])[:, positive_column]
```

Because the scaler is the first pipeline step, its `mean_`/`var_` are estimated from the fold's training rows. The validation rows are only transformed by the already-fitted scaler. The unit test `test_scaler_is_fitted_on_training_fold_only` checks the fitted scaler mean and variance against `X[train_index]` and against `X[validation_index]` using a synthetic matrix and a non-training spy classifier.

For the final model, the same pipeline is fitted once on the full Training matrix; the scaler then reflects Training only. `save_model_artifact` writes the pipeline as one joblib object, and `load_model_artifact` requires the probability interface. The serialization test round-trips the unfitted scaler + MLP pipeline and confirms the steps and early-stopping setting survive.

`SampleWeightPipeline` exposes `fit(X, y, sample_weight=None, ...)` explicitly because scikit-learn's base `Pipeline` only accepts step-prefixed weights when metadata routing is disabled. The subclass forwards the weight to the final estimator as `classifier__sample_weight`. MLP scaling and weighted fitting therefore work together without a caller dropping either one.

### 3.3 MLP early stopping and `max_iter`

- `early_stopping=False`.
- `validation_fraction` and `n_iter_no_change` are not stored in `MODEL_SPECS["mlp"]`; they are early-stopping-only settings and would be misleading while early stopping is disabled.
- Initial `max_iter=200`.
- `TUNING_SPACES["mlp"]` includes `max_iter ∈ {200, 400, 600}`. The resulting grid is 3 hidden-layer configurations × 2 alpha values × 3 `max_iter` values = 18 candidates, within the configured maximum of 24.
- `TUNING_POLICY["validation_use"]` and `CV_POLICY["validation_use"]` reserve the external Validation split for final evaluation. The MLP is not given a `validation_fraction` and cannot use Validation as an internal early-stopping set.

### 3.4 Weighted cross-validation and scoring

`src/training/cv.py` adds:

| Function / class | Purpose |
|---|---|
| `weighted_folds` | Deterministic weighted, cancer-stratified fold assignment without row expansion |
| `WeightedStratifiedKFold` | scikit-learn-compatible splitter carrying the `count` vector |
| `fold_diagnostics` | Per-fold rows, positive rows, `sum(count)`, positive/negative weight, weighted prevalence, deviations |
| `assess_fold_balance` | Pre-declared 5% represented-weight and 10% weighted-prevalence tolerances |
| `weighted_metrics` | Weighted ROC-AUC, PR-AUC, recall, specificity, precision, Brier, and weighted confusion counts |
| `weighted_scores` | CV scorer values: `pr_auc` (primary), `roc_auc`, `neg_brier` |
| `row_equal_metrics` | Explicitly secondary row-equal diagnostic only |
| `positive_class_probability` | Selects the `cancer=1` column from `predict_proba` |
| `cross_validate_weighted` | Fresh model per fold; weights to `fit` and to weighted scoring |

Fold construction is deterministic: within each class, rows are ordered by descending `count` with a seeded permutation for ties, then assigned one at a time to the fold with the smallest accumulated class weight. No row is duplicated, dropped, expanded, or resampled. This is a greedy approximation to weighted stratification, not an exact optimal partition; the diagnostics below show the achieved balance.

`count` remains a forbidden model feature (`FORBIDDEN_MODEL_COLUMNS`). The encoded matrices still contain only encoded feature columns, and the existing encoded-matrix test checks all four widths.

### 3.5 Fold diagnostics on the Training split

Command: `python -m src.training.cv --with-train`. This loads Training only, builds folds, and reports diagnostics. It does not fit a model and does not load Validation or Test.

Training reference: 152,149 rows; `sum(count)=1,525,868`; positive weight `7,452`; weighted prevalence `0.488378%`.

| Fold | Rows | Positive rows | `sum(count)` | Positive weight | Weighted prevalence | Weight deviation | Prevalence deviation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30,430 | 1,150 | 305,174 | 1,490 | 0.488246% | 0.000131% | 0.026969% |
| 1 | 30,429 | 1,150 | 305,174 | 1,491 | 0.488574% | 0.000131% | 0.040127% |
| 2 | 30,430 | 1,150 | 305,174 | 1,491 | 0.488574% | 0.000131% | 0.040127% |
| 3 | 30,430 | 1,150 | 305,173 | 1,490 | 0.488248% | 0.000197% | 0.026642% |
| 4 | 30,430 | 1,150 | 305,173 | 1,490 | 0.488248% | 0.000197% | 0.026642% |

All folds are far inside the configured material-imbalance tolerances (5% represented weight, 10% weighted prevalence), so `material_imbalance=false`. The represented-weight spread is 1 unit and the weighted-prevalence spread is about 0.0004 percentage points. The external Validation split was not involved.

### 3.6 Probability calibration

Calibration is required. The selected policy is:

| Item | Decision |
|---|---|
| Method | `sigmoid` (Platt scaling) |
| Why not isotonic | Isotonic regression is more flexible but less stable with a rare positive class (~0.49% weighted prevalence) |
| Ensemble | `ensemble=False`; one base pipeline is fitted on the full Training split and one calibrator is fitted on weighted out-of-fold probabilities |
| CV | `WeightedStratifiedKFold` over Training only; same 5 folds as the tuning policy |
| Weights | `count` is passed to every fold fit and to the calibrator; weights are never silently dropped |
| Final fit | Base pipeline fitted on the full Training split with `count`; calibrator fitted on weighted out-of-fold probabilities from Training-only CV |
| Validation | Evaluation only; never used to fit, tune, or select the calibrator |
| Test | Forbidden |
| Artifact | One joblib object: `CalibratedClassifierCV` containing the fitted model pipeline and calibrator |

`build_calibrated_model(model_key, cv)` requires an explicit CV argument and rejects `cv=None`, so it cannot silently fall back to an integer splitter. `build_training_calibrated_model(model_key, y, count)` additionally accepts only `split="train"` and constructs the weighted splitter over the supplied Training target/weight vector; `split="val"` and `split="test"` raise `ValueError`.

Weighted calibration compatibility was checked against scikit-learn 1.9.1 source and signatures:

- `CalibratedClassifierCV.fit(X, y, sample_weight=None, **fit_params)` accepts `sample_weight`.
- With `ensemble=False`, scikit-learn obtains out-of-fold predictions with the configured CV, fits the base estimator on the full input, and fits the calibrator with the same `sample_weight`.
- With `ensemble=True`, each fold fit receives the fold's weights and the calibrator receives the test-fold weights.
- The MLP `SampleWeightPipeline.fit` signature contains `sample_weight`, so `CalibratedClassifierCV`'s support check does not warn and drop base-estimator weights.

No calibrator was fitted in this task, so no calibration-quality claim is made.

### 3.7 `count` weighting and the primary metric

- `count` is never a feature: it is in `FORBIDDEN_MODEL_COLUMNS`; encoded matrices are checked against the saved metadata.
- Rows are never expanded for folds or fitting. The aggregated row remains the unit of storage and splitting; `count` supplies its represented frequency.
- Weighted PR-AUC (`average_precision_score(..., sample_weight=count)`) is the primary CV selection metric. ROC-AUC and negative Brier are supporting scores.
- Unweighted/row-equal metrics are available only through `row_equal_metrics`, whose name and docstring state that each aggregated row is treated equally; they are not used for selection.
- Validation evaluation must use the same weighted metric functions, not unweighted substitutes.

### 3.8 `surgmeno` and `hrt` code 9

The BCSC source combines Unknown and Not Applicable in code 9 for these two features. The project keeps code 9 as a single category:

- `CODE_9_POLICY["treatment"] = "single category"`.
- `split_into_unknown_and_not_applicable = False`.
- No raw-data change and no mapping change.
- Both `onehot` and `ordinal_tree` metadata contain exactly one `surgmeno_9` and one `hrt_9` column; no invented `*_unknown` columns appear for these features.
- The saved one-hot encoder's categories contain one code 9 per feature.

The ambiguity remains a documented data limitation: a single code-9 coefficient/indicator cannot distinguish Unknown from Not Applicable.

## 4. Preserved constraints

- Target `cancer`; positive class `1`; negative class `0`.
- Approved 10 features in the fixed `FEATURES` order.
- `count` is the frequency weight, never a feature.
- `invasive`, `race`, and `Hispanic` remain excluded.
- Frozen Train/Validation/Test files are unchanged; only Train and Validation are referenced.
- Test is absent from `SPLIT_REFERENCES`; `load_encoded_split(..., "test")` raises before reading data.
- Validation remains outside CV fitting, tuning, early stopping, and calibration; it is the final comparison split.
- Saved encoders and preprocessing metadata were not refit or modified.
- `RANDOM_STATE=42`, `N_JOBS=1`, `DEFAULT_THRESHOLD=0.5` are unchanged.
- MLP, scaler, and calibration artifacts are serializable with joblib.

## 5. Verification

### 5.1 Configuration checks

```text
python -m src.training.config
```

Result: **14 checks passed** — features/target/weight, split references, encoder metadata/artifacts, each of the four model entries, model specs, code-9 policy, weighted CV policy, calibration policy, evaluation policy, tuning/seed policy, and output directories. `test_accessed=false`, `models_trained=false`.

```text
python -m src.training.config --with-splits
```

Result: **15 checks passed** — the same checks plus live Train/Validation data. The live check confirmed the frozen row counts, weight sums, positive weights, and weighted rates. `test_accessed=false`, `models_trained=false`.

### 5.2 Read-only fold diagnostics

```text
python -m src.training.cv --with-train
```

Result: five weighted Training folds as shown in Section 3.5, all within tolerance, `material_imbalance=false`. No model was fitted; Validation and Test were not loaded.

### 5.3 Unit tests

```text
python -m unittest discover -s tests -v
```

Result: **38/38 tests passed**.

- 16 preprocessing tests (unchanged).
- 14 training-configuration tests: feature exclusion, model keys and encoder widths, central seed, unfitted/weighted model artifacts, MLP scaler scope, MLP scaler+model pipeline, early-stopping disabled and `max_iter` tuning, MLP artifact serialization, calibration policy/factory, Training-only calibration helper, code 9, Test rejection, encoded matrix widths, and full validation.
- 8 weighted-CV tests: row coverage without expansion, fold diagnostics, material-imbalance detection, sklearn-compatible splitter, weighted metrics against scikit-learn, weighted PR-AUC primary, secondary row-equal diagnostics, weights reaching both `fit` and scoring, and per-fold scaler fitting. The CV tests use synthetic data and a non-training spy; no project model is fitted.

## 6. Limitations and unresolved items

1. **Weighted stratification is a greedy approximation.** The splitter balances per-class represented weight deterministically, but it is not an optimal constrained partition and has no formal balance guarantee for future data. The achieved Training balance is reported above; the fallback is documented in `CV_POLICY["fallback"]`.
2. **Calibration is configured but not fitted.** No calibration curve, reliability diagram, or calibrated metric has been produced. Sigmoid was chosen for stability with rare positives; whether it is adequate must be checked during the evaluation phase on Validation only.
3. **Calibration weighting depends on scikit-learn behavior.** Public `CalibratedClassifierCV.fit(sample_weight=...)` support is verified for 1.9.1. The internal calibrator forwarding was inspected in this version; re-verify if scikit-learn is upgraded. The policy says to stop rather than drop weights if support changes.
4. **MLP artifact boundary.** The saved MLP artifact contains the scaler and MLP, not the One-Hot encoder. The encoder remains the existing transform-only preprocessing artifact. If a single raw-input artifact is later required, wrap the frozen encoder separately; refitting the encoder inside CV folds is not the current policy.
5. **`SampleWeightPipeline` assumes metadata routing is disabled.** It forwards `classifier__sample_weight`, the default scikit-learn path. If metadata routing is enabled globally, adapt the fit call; do not silently lose weights.
6. **No training or tuning has run.** Initial hyperparameters and the `max_iter` grid are baselines; no performance, ranking, or calibration claim is made.
7. **Dependencies are unpinned.** Joblib byte identity and exact floating-point repetition across environments are not guaranteed; record the environment snapshot and artifact hashes per run.
8. **Code 9 remains semantically ambiguous.** The single category is honest about the source but cannot distinguish Unknown from Not Applicable; resolving it needs reliable source documentation, not a data invention.
9. **`cross_validate_weighted` is implemented but not executed on BCSC data.** Future tuning code must use it (or an equivalent that passes weights to scoring) and record every fold and candidate score.

## 7. Files changed

| File | Change |
|---|---|
| `src/training/config.py` | MLP early stopping disabled; `max_iter` added to the tuning grid; MLP scaler+model pipeline factory; calibration and weighted-CV policies; code-9 policy; model pipeline/calibration factories; expanded validation and summary |
| `src/training/pipeline.py` | New `SampleWeightPipeline` and joblib model artifact save/load helpers |
| `src/training/cv.py` | New Training-only weighted fold construction, fold diagnostics, weighted metrics/scorers, and cross-validation helper |
| `tests/test_training_config.py` | Updated and expanded configuration, MLP pipeline, serialization, calibration, and code-9 tests |
| `tests/test_training_cv.py` | New weighted-fold, fold-diagnostic, weighted-scoring, and per-fold scaler tests |
| `outputs/training/reports/07_final_experiment_decisions.md` | This report |
| `outputs/training/reports/06_training_configuration.md` | Aligned with the final MLP, CV, and calibration decisions |
| `outputs/training/reports/05_reproducibility_policy.md` | Aligned MLP/GB/CV settings with the final decisions |
| `outputs/training/reports/04_evaluation_protocol.md` | Resolved open decisions that are now fixed |
| `outputs/training/reports/03_model_encoder_selection.md` | Resolved encoder/scaler open decisions |
| `outputs/training/reports/02_train_validation_compatibility.md` | Added a status note pointing to the fixed encoder/scaler choice |
| `outputs/training/reports/01_preprocessing_review.md` | Added a status note resolving the code-9 policy divergence |
| `PROJECT_MAP.md` | Added the new modules, report, entry point, conventions, and status |

Last updated: 2026-11-04
