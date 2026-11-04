# Unified Training Configuration

Date: 2026-11-04. Scope: one centralized training configuration for Logistic Regression, Random Forest, Gradient Boosting, and MLP. No model was trained or evaluated, no dataset or fitted preprocessing artifact was modified or refitted, and `test.csv` was not accessed.

> **Status: superseded in part by report 07.** The final experiment decisions are recorded in
> `outputs/training/reports/07_final_experiment_decisions.md`. Where this report and report 07
> disagree, report 07 is authoritative. The sections below have been updated to match the final
> decisions, so the remaining text is the current configuration reference, not the earlier draft.

## Verdict

Implemented `src/training/config.py` as the single training-time configuration. It imports the approved features, target, weight, split paths, and reproducibility settings from their existing owners instead of redefining them. All four model specifications resolve to the correct saved encoder, the correct encoded width, an unfitted estimator with the central seed, and the shared weighted evaluation/tuning rules. The MLP model artifact bundles `StandardScaler` + `MLPClassifier` in a `SampleWeightPipeline`, and weighted CV utilities live in `src/training/cv.py`. The read-only self-check passes 14 configuration checks, or 15 when the live Train/Validation files are included, and the full test suite passes 38/38.

## 1. Design principles

1. **Import shared values; do not duplicate them.** `FEATURES`, `TARGET`, `WEIGHT`, `PROJECT_ROOT`, and `SPLIT_PATHS` come from `src.data.load_splits`. `RANDOM_STATE`, `N_JOBS`, and `DEFAULT_THRESHOLD` come from `src.config`. Encoded widths and names come from `models/preprocessing/preprocessing_metadata.json`, not from hard-coded copies.
2. **Separate model-specific choices from shared rules.** Each model entry declares only its estimator, encoder, optional scaler, and model-specific hyperparameters. Evaluation, tuning, seed, split, and output rules are defined once and shared.
3. **Keep the saved encoders in charge.** The configuration references the joblib artifacts by family and path. It never rebuilds, refits, or replaces them. `load_encoded_split()` applies `transform` only.
4. **Never expose the Test split.** Only `train` and `val` are present in `SPLIT_REFERENCES`; `load_encoded_split()` rejects any other split before reading data.
5. **Keep `count` out of the model matrix.** `count` is declared as the frequency weight and is listed in `FORBIDDEN_MODEL_COLUMNS` together with `cancer`, `training`, `invasive`, `race`, and `Hispanic`.

## 2. Single sources of truth

| Configuration item | Imported from | Used in |
|---|---|---|
| Feature names and order | `src.data.load_splits.FEATURES` | `FEATURE_ORDER`, all model inputs |
| Target | `src.data.load_splits.TARGET` | `TARGET_COLUMN` |
| Frequency weight | `src.data.load_splits.WEIGHT` | `WEIGHT_COLUMN`, evaluation and tuning policy |
| Split file paths | `src.data.load_splits.SPLIT_PATHS` | `SPLIT_REFERENCES` |
| Encoder family names | `src.preprocessing.pipeline.FAMILIES` | `ENCODERS` keys |
| Project root | `src.data.load_splits.PROJECT_ROOT` | artifact and output paths |
| Central seed | `src.config.RANDOM_STATE` | every estimator, CV, and experiment record |
| Thread count | `src.config.N_JOBS` | every estimator that supports `n_jobs` |
| Default threshold | `src.config.DEFAULT_THRESHOLD` | evaluation policy |
| Encoder output widths/names | `models/preprocessing/preprocessing_metadata.json` | model-to-matrix validation |

## 3. Configuration contents

### 3.1 Features, target, and weight

| Item | Value |
|---|---|
| Feature order | `agegrp, bmi, agefirst, menopaus, surgmeno, hrt, nrelbc, density, brstproc, lastmamm` |
| Target column | `cancer` |
| Positive class | `1` |
| Negative class | `0` |
| Frequency weight | `count` |
| Forbidden model columns | `cancer`, `count`, `training`, `invasive`, `race`, `Hispanic` |

`FEATURE_ORDER` is a tuple built directly from `FEATURES`, so the order cannot drift from the loader or the saved pipelines.

### 3.2 Training and Validation references

Only the frozen Training and Validation splits are referenced. The Test path is not present in the configuration.

| Split | Path | Role | Rows | Weight sum | Positive weight | Weighted cancer rate |
|---|---|---|---:|---:|---:|---:|
| `train` | `data/04_splits/train.csv` | Fit models; fit the MLP scaler; internal CV | 152,149 | 1,525,868 | 7,452 | 0.488378% |
| `val` | `data/04_splits/val.csv` | Model comparison and reporting | 28,316 | 269,271 | 1,315 | 0.488356% |

These values are frozen guards copied from report 02. `validate_config(check_split_data=True)` reloads Train and Validation and fails if any row count, weight sum, positive weight, or weighted rate changes. It never loads Test. File SHA-256 fingerprints are not embedded in the module; report 05 requires recording them in each run's `splits.json`.

### 3.3 Encoder references

| Family | Artifact | Inputs | Outputs | Assigned models | Transformation |
|---|---|---:|---:|---|---|
| `onehot` | `models/preprocessing/onehot_pipeline.joblib` | 10 | 43 | Logistic Regression, MLP | One-hot for every feature; Unknown (9) is its own category |
| `ordinal_tree` | `models/preprocessing/ordinal_tree_pipeline.joblib` | 10 | 26 | Random Forest, Gradient Boosting | Ordinal ranks for ordered features plus Unknown flags; one-hot for nominal features |

The family names are imported from `src.preprocessing.pipeline.FAMILIES`; the configuration does not maintain a separate list. The widths are read from `preprocessing_metadata.json`; the configuration does not hard-code 43 or 26. The self-check also loads both joblib artifacts and verifies `n_features_in_`, the stored feature order, the output names, and the metadata width.

### 3.4 Post-encoder scaling (MLP only, bundled with the model)

| Scaler | Estimator | Parameters | Fit scope |
|---|---|---|---|
| `standard` | `sklearn.preprocessing.StandardScaler` | `with_mean=True`, `with_std=True` | Fitted only on the training portion of each CV fold; applied unchanged to that fold's validation rows. For the final model, fitted on the full Training encoded matrix only. |

Only the MLP references this scaler. Logistic Regression, Random Forest, and Gradient Boosting use `scaler=None`. For the MLP, `build_model_pipeline("mlp")` returns `SampleWeightPipeline([("scaler", StandardScaler()), ("classifier", MLPClassifier(...))])`, so the scaler is fitted per fold by the pipeline itself and is serialized together with the classifier as one joblib artifact. The scaler does not exist as a separate artifact, and the saved encoders are not modified.

### 3.5 Model specifications

| Key | Model | Estimator | Encoder | Matrix width | Model artifact |
|---|---|---|---|---:|---|
| `logistic_regression` | Logistic Regression | `sklearn.linear_model.LogisticRegression` | `onehot` | 43 | estimator only; no scaler |
| `random_forest` | Random Forest | `sklearn.ensemble.RandomForestClassifier` | `ordinal_tree` | 26 | estimator only; no scaler |
| `gradient_boosting` | Gradient Boosting | `sklearn.ensemble.GradientBoostingClassifier` | `ordinal_tree` | 26 | estimator only; no scaler |
| `mlp` | MLP Neural Network | `sklearn.neural_network.MLPClassifier` | `onehot` | 43 | `SampleWeightPipeline(StandardScaler, MLPClassifier)` |

The Gradient Boosting estimator is fixed to scikit-learn `GradientBoostingClassifier` for this phase; no XGBoost or other boosting library is used.

Initial model-specific parameters:

| Key | Initial parameters |
|---|---|
| `logistic_regression` | `penalty="l2"`, `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, `class_weight=None` |
| `random_forest` | `n_estimators=300`, `criterion="gini"`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"`, `bootstrap=True`, `class_weight=None` |
| `gradient_boosting` | `loss="log_loss"`, `learning_rate=0.05`, `n_estimators=200`, `subsample=0.8`, `criterion="friedman_mse"`, `min_samples_split=2`, `min_samples_leaf=1`, `max_depth=3` |
| `mlp` | `hidden_layer_sizes=(64, 32)`, `activation="relu"`, `solver="adam"`, `alpha=0.0001`, `batch_size=1024`, `learning_rate="constant"`, `learning_rate_init=0.001`, `max_iter=200`, `shuffle=True`, `early_stopping=False` |

The MLP has no `validation_fraction` or `n_iter_no_change` in `MODEL_SPECS`: those parameters only apply to early stopping, which is disabled in this phase. `max_iter` is an initial setting and a Training-only CV tuning dimension.

The common values are injected once by `_resolved_model_params()`:

- `random_state=RANDOM_STATE` for every model.
- `n_jobs=N_JOBS` for models whose estimator supports it (Logistic Regression and Random Forest). `MLPClassifier` has no `n_jobs`. The calibration factory also passes `n_jobs=N_JOBS` to `CalibratedClassifierCV`.
- No model entry stores a local `random_state` or `n_jobs`; the validator rejects such duplication.

`build_estimator(key)` constructs an **unfitted** estimator. `build_scaler(key)` constructs an **unfitted** scaler or returns `None`. `build_model_pipeline(key)` returns the unfitted estimator, or the scaler + estimator `SampleWeightPipeline` for the MLP. `build_calibrated_model(key, cv)` and `build_training_calibrated_model(key, y, count)` return an unfitted `CalibratedClassifierCV`.

### 3.6 Evaluation policy

The policy mirrors `outputs/training/reports/04_evaluation_protocol.md`.

| Metric | Function | Threshold-dependent | Direction | Weighted denominator |
|---|---|---|---|---|
| ROC-AUC | `sklearn.metrics.roc_auc_score` | no | higher | weighted TPR/FPR across thresholds |
| PR-AUC (average precision) | `sklearn.metrics.average_precision_score` | no | higher | weighted positive weight; precision uses predicted-positive weight |
| Recall / Sensitivity | weighted confusion counts | yes | higher | `W+` |
| Specificity | weighted confusion counts | yes | higher | `W-` |
| Precision / PPV | weighted confusion counts | yes | higher | predicted-positive weight |
| Brier | `sklearn.metrics.brier_score_loss` | no | lower | `W` |

Shared rules:

- Positive class `cancer = 1`; negative class `0`.
- Default threshold `t = 0.5`; prediction rule `p >= t`.
- Probabilities are required from `predict_proba`, selecting the column whose `classes_` entry equals `1`.
- `count` is passed as `sample_weight` to fitting and every metric; no row expansion or resampling.
- Comparison uses Validation only; Test is evaluated once after freezing.
- Every Validation row must be scored by every model.
- Undefined metrics are `N/A` with a reason; they are never replaced by `0`, `0.5`, or `1`.
- Weighted confusion counts `TP_w`, `FP_w`, `TN_w`, `FN_w` and weighted prevalence are reported with the metrics.

### 3.7 Hyperparameter tuning policy

| Rule | Value |
|---|---|
| Search | Exhaustive search over a pre-declared grid |
| Data | Training only |
| Cross-validation | `WeightedStratifiedKFold(n_splits=5, random_state=42)` over Training only, carrying the `count` vector |
| Weighting | `count` as `sample_weight` in every fold fit **and** in every weighted scoring call |
| Primary scoring | `average_precision` (weighted PR-AUC) |
| Supporting scoring | weighted `roc_auc`, weighted `neg_brier_score` |
| Unweighted metrics | secondary diagnostics only; never used for selection |
| Maximum candidates per model | 24 |
| Refit | Selected candidate refit on the full Training split with `count` |
| Validation use | Compare final candidates once; never use for search, early stopping, or calibration |
| Test use | Forbidden until the model, encoder, threshold, and protocol are frozen |
| Recording | Record the grid, folds, seed, every candidate score, and the selected parameters |

Pre-declared initial grids:

| Model | Grid |
|---|---|
| `logistic_regression` | `C ∈ {0.01, 0.1, 1.0, 10.0}`, `solver="lbfgs"` |
| `random_forest` | `n_estimators ∈ {300, 600}`, `max_depth ∈ {None, 16}`, `min_samples_leaf ∈ {1, 10}`, `max_features="sqrt"` |
| `gradient_boosting` | `learning_rate ∈ {0.03, 0.05, 0.1}`, `n_estimators ∈ {100, 200}`, `max_depth ∈ {2, 3}`, `subsample ∈ {0.8, 1.0}` |
| `mlp` | `hidden_layer_sizes ∈ {(32,), (64, 32), (128, 64)}`, `alpha ∈ {0.0001, 0.001}`, `max_iter ∈ {200, 400, 600}` |

The MLP grid has 3 × 2 × 3 = 18 candidates, within the 24 candidate cap. The weighted scoring path is implemented in `src/training/cv.py` (`weighted_metrics`, `weighted_scores`, `cross_validate_weighted`), so tuning must call it rather than plain `GridSearchCV` scoring; a plain scorer would receive no `sample_weight` and would silently optimize the row-equal objective.

### 3.8 Seed policy

The configuration imports the central values from `src.config`:

| Setting | Value |
|---|---|
| `RANDOM_STATE` | `42` |
| `N_JOBS` | `1` |
| `DEFAULT_THRESHOLD` | `0.5` |

`SEED_POLICY` records that every estimator receives `random_state=42`, the tuning folds use seed 42, and no training code may define a local seed. Launch-time deterministic settings (`PYTHONHASHSEED`, BLAS/OpenMP thread limits) remain in report 05.

### 3.9 Output directories

| Purpose | Path |
|---|---|
| Trained models | `models/trained/` |
| Experiment records | `outputs/training/experiments/` |
| Predictions | `outputs/training/predictions/` |
| Metrics | `outputs/training/metrics/` |
| Logs | `outputs/training/logs/` |
| Reports | `outputs/training/reports/` |

The append-only registry is `outputs/training/experiments/registry.csv`. Run identifiers follow:

```text
YYYYMMDDTHHMMSS_<model>_<encoder>_seed42
```

Each run directory is expected to contain the files in `RUN_ARTIFACTS`: `config.json`, `environment.json`, `splits.json`, `metrics.json`, `predictions_val.csv`, `model.joblib`, `log.txt`, and `config_notes.md`.

The directories are defined, not created on import. `ensure_output_directories()` creates them on demand and is never called by importing the module.

## 4. Verification

### 4.1 Configuration self-check

```text
python -m src.training.config
```

Result: **14 checks passed** — features/target/weight, split references, encoder metadata/artifacts, each of the four model entries, model specs, code-9 policy, weighted CV policy, calibration policy, evaluation policy, tuning/seed policy, and output directories.

```text
python -m src.training.config --with-splits
```

Result: **15 checks passed** — the same checks plus live Train/Validation data. The live check confirmed the frozen row counts, weight sums, positive weights, and weighted cancer rates. Both runs report:

```text
test_accessed = false
models_trained = false
```

### 4.2 All four models consume the configuration

A read-only smoke check loaded the Validation split and applied each model's saved encoder with `transform` only:

| Model | Encoded matrix | dtype | Finite | Estimator constructed | Scaler |
|---|---|---|---|---|---|
| `logistic_regression` | `(28316, 43)` | `float32` | yes | `LogisticRegression` | none |
| `random_forest` | `(28316, 26)` | `float32` | yes | `RandomForestClassifier` | none |
| `gradient_boosting` | `(28316, 26)` | `float32` | yes | `GradientBoostingClassifier` | none |
| `mlp` | `(28316, 43)` | `float32` | yes | `MLPClassifier` | `StandardScaler` |

The validator also confirmed for every model that:

- The estimator imports and constructs with `random_state=42`.
- `n_jobs=1` is present where the estimator supports it.
- `fit()` accepts `sample_weight`, including the MLP's `SampleWeightPipeline.fit`.
- `predict_proba` is available.
- The encoder artifact's input order equals `FEATURE_ORDER` and its output names equal the metadata.
- No encoded name is `count`, `cancer`, `training`, `invasive`, `race`, or `Hispanic`.

### 4.3 Test suite

```text
python -m unittest discover -s tests -v
```

Result: **38/38 tests passed** — the 16 existing preprocessing tests, 14 training-configuration tests (feature exclusion, required model keys, encoder widths, central seed injection, MLP scaler scope, MLP scaler+model pipeline, early-stopping disabled and `max_iter` tuning, MLP artifact serialization, calibration policy/factory, Training-only calibration helper, code 9, Test-split rejection, encoded matrix shape, and full configuration validation), and 8 weighted-CV tests (fold coverage without expansion, fold diagnostics, material-imbalance detection, sklearn-compatible splitter, weighted metrics against scikit-learn, weighted PR-AUC primary, secondary row-equal diagnostics, weights reaching both `fit` and scoring, and per-fold scaler fitting).

### 4.4 Read-only fold diagnostics

```text
python -m src.training.cv --with-train
```

Result: five weighted Training folds; each has 30,429–30,430 rows, `sum(count)` 305,173–305,174, positive weight 1,490–1,491, and weighted prevalence 0.488246–0.488574%. `material_imbalance=false`. Validation and Test were not loaded and no model was fitted.

## 5. Usage

```python
from src.config import set_global_seeds
from src.training.config import (
    build_model_pipeline,
    load_encoded_split,
)
from src.training.cv import cross_validate_weighted, weighted_folds

set_global_seeds()

# Encoded Training matrix for one model; transform only, never refit the encoder.
Z_train, y_train, w_train = load_encoded_split("random_forest", "train")
model = build_model_pipeline("random_forest")   # constructed, not fitted

# MLP: the scaler is the first step of the model pipeline and is fitted per fold.
Z_train_mlp, y_train_mlp, w_train_mlp = load_encoded_split("mlp", "train")
mlp = build_model_pipeline("mlp")               # SampleWeightPipeline(StandardScaler, MLPClassifier)

folds = weighted_folds(y_train_mlp, w_train_mlp, n_splits=5, random_state=42)
results = cross_validate_weighted(
    lambda: build_model_pipeline("mlp"), Z_train_mlp, y_train_mlp, w_train_mlp, folds=folds
)
```

`load_encoded_split()` rejects `test` and any split not listed in `SPLIT_REFERENCES`. Actual tuning runs should select candidates from `weighted_scores` (weighted PR-AUC primary), not from a plain unweighted scorer.

## 6. Guarantees and limitations

Guarantees:

- `count` is never a model feature and is always the frequency weight.
- Every model is mapped to one saved encoder and the correct matrix width.
- Encoders are referenced, not rebuilt; all learned preprocessing remains Training-fitted.
- The central seed and thread count are injected once, not repeated per model.
- Training and Validation references are explicit and guarded; Test is absent.
- The four models share one metric, tuning, and recording policy.

Limitations and open decisions:

1. **Gradient Boosting implementation is fixed for this phase.** The entry uses `sklearn.ensemble.GradientBoostingClassifier`; `HistGradientBoostingClassifier` or an external library is a separate future experiment that must be documented if adopted.
2. **MLP scaler packaging is resolved.** The scaler is the first step of `SampleWeightPipeline`, fitted per CV training fold and on full Training for the final model, and serialized with the classifier as one joblib artifact. See report 07.
3. **MLP early stopping is disabled.** `early_stopping=False`; early-stopping-only parameters are absent. `max_iter` is tuned with Training-only CV. The external Validation split is never used for early stopping or selection.
4. **Logistic Regression deliberately has no scaler.** The one-hot matrix is predominantly binary indicators; adding a scaler would be a separate documented experiment.
5. **Weighted tuning scoring is implemented but not yet run on BCSC data.** `cross_validate_weighted` passes `count` to both `fit` and weighted scoring; any future tuning entry point must use it or an equivalent that does the same.
6. **Primary selection metric is final: weighted PR-AUC.** Unweighted metrics are secondary diagnostics only. See report 07.
7. **`surgmeno`/`hrt` code 9 stays one category.** The source combines Unknown and Not Applicable; the ambiguity is documented and raw data/mapping are unchanged.
8. **Weighted stratification is a deterministic greedy approximation.** Folds are balanced on represented weight and weighted prevalence within tolerance on Training; there is no formal optimality guarantee. See `CV_POLICY["fallback"]` and report 07.
9. **Initial configurations are baselines, not tuned results.** No model has been fitted, so no performance claim is made.
10. **Dependencies remain unpinned.** Exact bitwise reproducibility is not guaranteed; see report 05.
11. **Output directories are not created yet.** Call `ensure_output_directories()` in the training entry point.

## 7. Files added or changed

| File | Change |
|---|---|
| `src/training/__init__.py` | New package marker |
| `src/training/config.py` | Unified training configuration, model/calibration factories, policies, and read-only validator |
| `src/training/pipeline.py` | `SampleWeightPipeline` and joblib model artifact save/load helpers |
| `src/training/cv.py` | Training-only weighted folds, fold diagnostics, weighted metrics/scorers, CV helper |
| `tests/test_training_config.py` | Read-only tests for the configuration, MLP pipeline, serialization, calibration, and code 9 |
| `tests/test_training_cv.py` | Tests for weighted folds, diagnostics, weighted scoring, and per-fold scaler fitting |
| `outputs/training/reports/06_training_configuration.md` | This report |
| `outputs/training/reports/07_final_experiment_decisions.md` | Authoritative final decision report |
| `PROJECT_MAP.md` | Updated with the modules, reports, tests, and status |

Last updated: 2026-11-04
