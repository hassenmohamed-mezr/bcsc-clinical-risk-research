# PROJECT_MAP

## 1. Summary
Research repository for the BCSC Risk Estimation dataset (aggregated screening-mammography records, ~280,660 rows). Work covers TXT→CSV conversion, EDA, feature-cancer association, PCA/t-SNE, weighted train/validation/test splitting, a reusable split loader, a leakage-safe preprocessing pipeline (raw-input mapping + model encoders), a unified training configuration for four models, Training-only weighted CV/scoring utilities, and final experiment decisions (MLP scaling and early stopping, Gradient Boosting library, `count` weighting, fold construction, calibration, and code-9 handling); no model training or evaluation exists yet.

## 2. Stack
- Python (script + Jupyter notebooks); no pinned versions (`requirements.txt` is empty by design).
- Observed environment (2026-11-04): Python 3.15.0, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0; PyTorch, XGBoost, LightGBM, CatBoost, and TensorFlow/Keras are not installed.
- Libraries used in code: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, joblib.
- No DB, API, web server, Docker, or external services. Data shipped as local CSV/TXT.
- License: Apache 2.0 (LICENSE). No packaging/test config found.

## 3. Folder Map
- `data/01_raw/`: source `risk.txt` plus CSV copies (git-ignored); `02_interim/`: filtered CSVs; `03_processed/`: empty; `04_splits/`: `train_full.csv`, `train.csv`, `val.csv`, `test.csv`.
- `notebooks/02_eda/`: EDA, feature-vs-cancer, PCA/t-SNE notebooks; `03_preprocessing/`: column-drop notebook; `01_data_discovery/`, `04_modeling/`, `05_final_evaluation/`: empty.
- `src/data/`: `risk_to_csv.py` (TXT→CSV) and `load_splits.py` (split loader); `src/preprocessing/split_original.py`, `split_validation.py`, `raw_input_mapping.py` (interface→BCSC mapper), `pipeline.py` (encoder families), `fit_preprocessing.py` (train-only artifact fitting); `src/config.py` (central reproducibility settings); `src/training/config.py` (unified model/experiment configuration), `pipeline.py` (`SampleWeightPipeline` + joblib artifact helpers), `cv.py` (weighted folds, diagnostics, metrics, CV helper); `src/modeling|evaluation/`: empty.
- `outputs/eda/{tables,figures}`: 14 CSVs + 18 images; `outputs/preprocessing/reports/`: reviews 01–14 (splits, leakage audit, missing-value policy, encoding analysis, preprocessing pipeline); `outputs/training/reports/`: reviews 01–07 (preprocessing implementation review, train/validation compatibility check, model-to-encoder selection, unified evaluation protocol, reproducibility policy, unified training configuration, final experiment decisions); other `outputs/*`: empty.
- `models/preprocessing/`: fitted mapping JSON, `onehot`/`ordinal_tree` joblib pipelines, metadata; `models/trained/`: configured but not created; `tests/`: `test_preprocessing.py`, `test_training_config.py`, `test_training_cv.py`; `docs/{preprocessing,modeling,final_evaluation}`: empty.
- `docs/`: state matrix, dataset notes, EDA notes; `references/`: placeholder notes only.

## 4. Entry Points and Run
- `python src/data/risk_to_csv.py [src] [dst]` (defaults `risk.txt` → `risk.csv`; validated with asserts).
- `python src/preprocessing/split_original.py` → `data/04_splits/{train_full,test}.csv`.
- `python src/preprocessing/split_validation.py` → `data/04_splits/{train,val}.csv` (weighted 85/15, seed 42).
- `python -m src.data.load_splits` runs the loader self-check (train + val only).
- `python -m src.config` prints the observed Python/library/thread/seed snapshot used in experiment records.
- `python -m src.training.config [--with-splits]` validates the unified training configuration without training; `--with-splits` also reloads Train/Validation and checks the frozen row/weight/rate references.
- `python -m src.training.cv --with-train` loads Training only, builds the weighted 5-fold split, and prints per-fold rows, `sum(count)`, positive weight, weighted prevalence, and balance diagnostics; it fits no model and does not load Validation or Test.
- `python -m src.preprocessing.fit_preprocessing` fits both encoder families on `train` only and writes `models/preprocessing/`.
- `python -m unittest discover -s tests -v` runs the preprocessing, training-configuration, and weighted-CV test suites (pytest is not installed).
- Notebooks run manually with working directory = data folder; they read/write bare filenames (`risk.csv`, `risk_filtered.csv`) — (unverified) intended run location.
- No CLI, server, or orchestrator; README states no verified install steps.

## 5. Main Flow
`risk.txt` → `risk_to_csv.py` (16 fixed columns, sanity checks) → `risk.csv` → `drop.ipynb` (drops `race`, `Hispanic`, `invasive`) → `risk_filtered.csv` → `split_original.py` (original `training` flag) → `train_full.csv`/`test.csv` → `split_validation.py` (weighted stratified 15% validation) → `train.csv`/`val.csv` → `load_splits.load_split()` → `(X, y, w)` with `count` as `sample_weight` → `fit_preprocessing.py` (fit on train) / `pipeline.transform_features` (transform val/test) → encoded model matrix → `src/training/config.py` (model/encoder mapping, initial parameters, `build_model_pipeline`/calibration factories, shared weighted evaluation/tuning rules) → `src/training/cv.py` (weighted folds, diagnostics, weighted metrics, `cross_validate_weighted`) → calibration via Training-only weighted CV → training experiment (not started). EDA notebooks separately use `count` weights for tables/figures.

## 6. Key Files
- `src/data/risk_to_csv.py`: column list, allowed-value validation, conversion.
- `src/data/load_splits.py`: single source of truth — `FEATURES` (10 fixed, ordered), `TARGET="cancer"`, `WEIGHT="count"`; `load_split("train"|"val"|"test") -> (X, y, w)` validates columns/binary target/positive weights, applies no preprocessing, and returns `w` for `sample_weight`.
- `src/config.py`: central reproducibility settings — `RANDOM_STATE=42`, `N_JOBS=1`, `DEFAULT_THRESHOLD=0.5`, global Python/NumPy seeding, optional PyTorch seeding, and `environment_snapshot()` for experiment records.
- `src/training/config.py`: unified training configuration — imports `FEATURES`/`TARGET`/`WEIGHT`/`SPLIT_PATHS` and the central seed/thread/threshold; declares per-model encoder, optional scaler, and initial hyperparameters; provides `build_estimator`, `build_scaler`, `build_model_pipeline`, `build_calibrated_model`, and `build_training_calibrated_model`; defines `CODE_9_POLICY`, `CV_POLICY`, `CALIBRATION_POLICY`, weighted metrics, tuning policy, output layout, and a read-only validator (`python -m src.training.config`).
- `src/training/pipeline.py`: `SampleWeightPipeline` (forwards `sample_weight` to the final estimator, used for MLP `StandardScaler` + `MLPClassifier`) and `save_model_artifact`/`load_model_artifact` joblib helpers.
- `src/training/cv.py`: Training-only weighted fold builder (`weighted_folds`, `WeightedStratifiedKFold`), per-fold diagnostics and balance assessment, weighted metrics/scorers, and `cross_validate_weighted`; CLI `python -m src.training.cv --with-train`.
- `src/preprocessing/split_original.py`: preserves the original `training` split.
- `src/preprocessing/split_validation.py`: weighted stratified 85/15 train/validation split (seed 42).
- `src/preprocessing/raw_input_mapping.py`: stage-A interface→BCSC mapping (`0` = Unknown → 9; agegrp has no Unknown), strict validation, JSON save/load.
- `src/preprocessing/pipeline.py`: stage-B `ColumnTransformer` families `onehot` (43 cols) and `ordinal_tree` (26 cols), fixed `FEATURES` order, `handle_unknown="ignore"` for nominal codes, ordinal rank+Unknown flags, joblib save/load, `encode_user_inputs` inference path.
- `src/preprocessing/fit_preprocessing.py`: fits both encoder families on `train` only and saves reusable artifacts + metadata.
- `tests/test_preprocessing.py`: 16 stdlib `unittest` tests covering mapping, feature order/dimensions, train-only fitting, Unknown/unseen handling, serialization, and inference parity.
- `tests/test_training_config.py`: 14 tests covering configuration, MLP scaler+model pipeline and serialization, calibration factories, code 9, and Test rejection.
- `tests/test_training_cv.py`: 8 tests covering weighted folds, diagnostics, weighted scoring, and per-fold scaler fitting.
- `notebooks/03_preprocessing/drop.ipynb`: controlled column exclusion + assert.
- `notebooks/02_eda/EDA.ipynb`: weighted summaries, Wilson CI, unknown-rate table, training split.
- `notebooks/02_eda/Full Feature vs Cancer.ipynb`: weighted contingency, chi-square, Cramér's V / bias-corrected V, outputs to `BCSC_Full_Feature_Analysis`.
- `notebooks/02_eda/Visualization_2d&3d.ipynb`: PCA/t-SNE figures.
- `docs/PROJECT_STATE_MATRIX.md`: authoritative inventory/status.
- `outputs/preprocessing/reports/05_leakage_audit.md`: leakage/integrity audit of splits, preprocessing, and EDA (verdict: PASS WITH WARNINGS).
- `outputs/preprocessing/reports/06_missing_inventory.md`: null counts and observed special codes for the 10 features on train+val (completed; no nulls, code 9 unverified).
- `outputs/preprocessing/reports/07_weighted_missing_rates.md`: row-based and frequency-weighted missing rates for the 10 features (completed; all 0.00%, special codes excluded).
- `outputs/preprocessing/reports/08_missingness_patterns.md`: target-group special-code rates, co-occurrence, and limitations (completed; descriptive only, no tests).
- `outputs/preprocessing/reports/09_missing_value_semantics.md`: official BCSC meanings/classification of special codes for the 10 features (completed; code 9 = Unknown except surgmeno/hrt mixed, agegrp 10 valid).
- `outputs/preprocessing/reports/10_missing_treatment_options.md`: option comparison (retain unknown, mode, indicator, model-native, recode NA) without selecting a strategy (completed).
- `outputs/preprocessing/reports/11_missing_treatment_decisions.md`: per-feature treatment decisions (retain Unknown; conditional NA recode for surgmeno/hrt) (completed).
- `outputs/preprocessing/reports/12_missing_value_policy.md`: authoritative missing-value policy for implementation (completed with two pending items).
- `outputs/preprocessing/reports/13_encoding_strategy_analysis.md`: feature-type/encoding comparison with per-feature recommendations and unresolved items (completed; no encoding implemented).
- `outputs/preprocessing/reports/14_preprocessing_pipeline.md`: two-stage preprocessing implementation, train-only fitting, encoder rationale, Unknown handling, tests, and unresolved items (completed; artifacts fitted on train only).
- `outputs/training/reports/01_preprocessing_review.md`: independent review of the same implementation and saved artifacts (completed; artifacts load and transform validation correctly; policy divergence and domain-validation gaps flagged).
- `outputs/training/reports/02_train_validation_compatibility.md`: read-only train/val compatibility check against both saved pipelines (completed; PASS, no blocking mismatch; raw-order and semantic caveats noted).
- `outputs/training/reports/03_model_encoder_selection.md`: per-model-family encoder recommendation, scaling/transformation needs, and unresolved decisions (completed; no model training and no encoder changes).
- `outputs/training/reports/04_evaluation_protocol.md`: unified weighted evaluation protocol — ROC-AUC, PR-AUC, recall, specificity, precision, Brier, label/threshold conventions, reporting format, and undefined-metric handling (completed; no model evaluated and test untouched).
- `outputs/training/reports/05_reproducibility_policy.md`: central seed, Python/NumPy/scikit-learn/PyTorch/Gradient-Boosting settings, deterministic runtime controls, data/split rules, experiment-record layout, and reproducibility limitations (completed; no model trained and test untouched).
- `outputs/training/reports/06_training_configuration.md`: unified configuration structure, per-model encoder/scaler/initial parameters, shared weighted evaluation and tuning policy, output directories, and read-only verification that all four models resolve to the correct encoded matrix (completed; updated to the final decisions; no model trained and test untouched).
- `outputs/training/reports/07_final_experiment_decisions.md`: authoritative final experiment decisions — Gradient Boosting library, MLP One-Hot + in-pipeline scaler and early-stopping policy, Logistic Regression scaling, `count` weighting for fit and scoring, weighted fold construction and diagnostics, Training-only weighted calibration, and code-9 handling; decision-status table, verification results, limitations, and files changed (completed; no model trained and test untouched).

## 7. Config/Env (names only)
No `.env` or `.env.example`; configuration is code-based. Central reproducibility settings live in `src/config.py` (`RANDOM_STATE`, `N_JOBS`, `DEFAULT_THRESHOLD`, seeding helpers, `environment_snapshot`). Training-time configuration lives in `src/training/config.py`, which imports those shared values and `load_splits` identifiers instead of redefining them: `FEATURE_ORDER`, `SPLIT_REFERENCES`, `ENCODERS`, `SCALERS`, `MODEL_SPECS`, `METRICS`, `EVALUATION_POLICY`, `TUNING_POLICY`, `TUNING_SPACES`, `SEED_POLICY`, `CODE_9_POLICY`, `CV_POLICY`, `CALIBRATION_POLICY`, `OUTPUT_DIRECTORIES`, `RUN_ARTIFACTS`. Other constants live in code: notebook paths (`INPUT_FILE`, `OUTPUT_FILE`, `DATA_PATH`, `OUTPUT_DIR`), split scripts (`RANDOM_STATE`, `VAL_FRACTION`), loader (`FEATURES`, `TARGET`, `WEIGHT`, `SPLIT_PATHS`).

## 8. Conventions
- snake_case Python; flat procedural scripts/notebooks; stage-numbered folders (`01_`…`05_`).
- Validation by `assert`/`raise ValueError`; prints for reporting; no logging framework.
- Target `cancer`; `count` = frequency weight, never a feature; 10 fixed features in `load_splits.FEATURES`; excluded columns (`race`, `Hispanic`, `invasive`) handled explicitly.
- Splits are fixed files under `data/04_splits/`; never re-split or reshuffle inside training code.
- Preprocessing is two-stage: numeric interface choices → documented BCSC codes (`raw_input_mapping`) → model-specific encoder (`pipeline`). Learned encoders are fitted on `train` only and applied unchanged to val/test; inference reuses the same fitted pipeline.
- Model comparison follows `outputs/training/reports/04_evaluation_protocol.md`: `count`-weighted metrics, Validation only, positive class `cancer=1`, and `t=0.5` for initial threshold-dependent results; Test is evaluated once after freezing.
- All training code imports `RANDOM_STATE`/`N_JOBS` from `src.config`; official runs use seed 42, set `PYTHONHASHSEED` and BLAS/OpenMP threads at launch, and record configuration, environment, split fingerprints, metrics, and predictions under `outputs/training/experiments/`.
- Model-to-encoder mapping is fixed in `src/training/config.py`: Logistic Regression → `onehot` with no scaler, Random Forest → `ordinal_tree`, Gradient Boosting → `ordinal_tree` with scikit-learn `GradientBoostingClassifier`, MLP → `onehot` plus a `StandardScaler` inside the model pipeline. Initial parameters are baselines; tuning is Training-only 5-fold weighted CV with `count` passed to both `fit` and weighted scoring, and weighted PR-AUC as the primary search score.
- Use `load_encoded_split(model_key, "train"|"val")` for a transform-only encoded matrix; it rejects `test`, checks the matrix width against the saved metadata, and never refits an encoder. Use `build_model_pipeline`/`build_calibrated_model` for unfitted artifacts; the MLP artifact is a `SampleWeightPipeline(StandardScaler, MLPClassifier)` serialized as one joblib object.
- Weighted folds come from `src/training/cv.py` (`weighted_folds`/`WeightedStratifiedKFold`), never row expansion. Per-fold diagnostics must report rows, `sum(count)`, and weighted prevalence; material imbalance must be documented. Validation is never part of a fold.
- Calibration is required and uses Training-only weighted CV, sigmoid (Platt), `ensemble=False`, and one serialized artifact; the external Validation and Test splits are never used to fit or select the calibrator.
- `surgmeno` and `hrt` code 9 remain one category because the source combines Unknown and Not Applicable; do not split it or modify raw data/mapping.
- Raw split CSV column order differs from `FEATURES`; always use `load_split()` (or reorder) before transformation. The pipeline rejects reordered frames by design.
- Documentation is evidence-based, separating verified vs pending items.

## 9. Tests
Preprocessing suite: `tests/test_preprocessing.py`, 16 stdlib `unittest` tests, all passing (`python -m unittest discover -s tests -v`; pytest is not installed). Covers interface→BCSC mapping and rejection rules, feature count/order, stable encoded dimensions (43/26), train-only category fitting/no refit during validation transform, Unknown and unseen-code handling, joblib/JSON save-reload consistency, and mapper→pipeline inference parity. Inline checks also remain: asserts in scripts and the `load_splits` self-check (`python -m src.data.load_splits`).

Training configuration suite: `tests/test_training_config.py`, 14 stdlib `unittest` tests. Covers feature exclusion of `count`/`cancer`, required model keys, encoder widths, central seed/`n_jobs` injection, MLP scaler scope, MLP scaler+model pipeline and early-stopping setting, MLP artifact serialization, calibration policy/factory and Training-only helper, code-9 handling, rejection of the Test split, encoded-matrix shape for all four models, and full configuration validation.

Weighted-CV suite: `tests/test_training_cv.py`, 8 stdlib `unittest` tests. Covers non-expanding fold coverage, per-fold weight/prevalence diagnostics, material-imbalance detection, the sklearn-compatible splitter, weighted metrics against scikit-learn, weighted PR-AUC as primary, secondary row-equal diagnostics, weights reaching both `fit` and scoring, and per-fold scaler fitting with a non-training spy model.

Combined suite: **38/38 passing**. `python -m src.training.config [--with-splits]` is the standalone read-only configuration check; `python -m src.training.cv --with-train` prints fold diagnostics.

## 10. Unknowns
- Category-code semantics and dataset provenance/redistribution terms (unverified); `references/` placeholders.
- Notebook-relative paths and whether raw CSVs match regenerated output (unverified).
- The 10-feature list is now fixed in code but not model-validated; `training` split semantics remain (unverified) per reports 01–02; images of unknown purpose.
- EDA/association notebooks and PCA/t-SNE used the full dataset including test rows (no persisted transformers); influence of EDA statistics on the feature-exclusion decision is (unverified).
- Exact bitwise reproducibility is not guaranteed: dependencies are unpinned, external MLP/Gradient-Boosting libraries are not installed (and are not used in this phase), and BLAS/OpenMP/HW differences remain; see report 05 for mitigations and per-run recording requirements.
- `surgmeno`/`hrt` code 9 combines Unknown and Not Applicable in the source; the single-category policy is documented but the row-level distinction remains unverified.

## 11. Progress / Status
- Completed: conversion, EDA, association analysis, PCA/t-SNE, original split preservation, weighted 85/15 validation split, `load_splits` module with self-check, the read-only leakage audit, the missing-value policy (reports 06–12), the feature-type/encoding analysis (report 13), and the two-stage preprocessing pipeline with tests and saved artifacts (report 14).
- Missing-value phase: complete with pending items — the source combines Unknown and Not Applicable in `surgmeno`/`hrt` code 9, and the final decision keeps code 9 as one category (report 07); define the handling contract if true nulls ever appear.
- Encoding/preprocessing phase: implemented with both `onehot` (43 cols) and `ordinal_tree` (26 cols) families, fitted on train only and saved under `models/preprocessing/`; the model-family/encoder mapping is now fixed (report 07).
- Preprocessing tests: 16/16 passing; validation is transform-only and test remains unopened.
- Preprocessing review (2026-11-04): saved artifacts load and reproduce the expected 43/26 columns; feature order, train-only categories, validation transform-only behaviour, and exclusions verified. Open findings: the `surgmeno`/`hrt` conditional Not Applicable recode was not implemented, and the final decision keeps code 9 as one category; direct BCSC-coded transforms do not strictly reject every undocumented category.
- Train/validation compatibility check (2026-11-04): PASS for both `onehot` (43) and `ordinal_tree` (26); dtypes, domains, target, `count`, missingness, category coverage, and encoded dimensions all match. No unseen validation categories or all-zero encoded columns. Use `load_split()` because raw CSV order is not canonical.
- Model-to-encoder selection (2026-11-04): Logistic Regression → `onehot` (no scaler), Random Forest → `ordinal_tree`, Gradient Boosting → `ordinal_tree` with scikit-learn `GradientBoostingClassifier`, MLP → `onehot` plus a `StandardScaler` inside the model pipeline. No model has been trained.
- Evaluation protocol (2026-11-04): unified `count`-weighted ROC-AUC, PR-AUC, recall, specificity, precision, and Brier; positive class `cancer=1`, default threshold `0.5`, Validation-only comparison, and Test evaluated once after freezing. Weighted PR-AUC is the confirmed primary selection metric. Undefined metrics are reported as `N/A` with a reason. No model has been evaluated.
- Reproducibility policy (2026-11-04): implemented `src/config.py` with `RANDOM_STATE=42` (matching the frozen split), `N_JOBS=1`, global Python/NumPy seeding, optional PyTorch seeding, and environment snapshots. Experiment-record layout and deterministic launch settings documented in report 05; Gradient Boosting fixed to scikit-learn and MLP early stopping disabled. No model trained.
- Unified training configuration (2026-11-04): implemented `src/training/config.py` and report 06. All four models have a declared encoder, optional scaler, and initial parameter set; shared weighted metric, tuning, seed, split, and output policies are defined once. Read-only validation passes 14 checks (15 with live Train/Validation data); all four models resolve to the expected 43/26-column matrices. No model trained and Test untouched.
- Final experiment decisions (2026-11-04): implemented `src/training/pipeline.py` (`SampleWeightPipeline`, joblib model artifact helpers) and `src/training/cv.py` (weighted folds, diagnostics, weighted metrics/scorers, `cross_validate_weighted`); added report 07 and expanded the tests. Training fold diagnostics: 5 folds of 30,429–30,430 rows with `sum(count)` 305,173–305,174 and weighted prevalence 0.488246–0.488574% (`material_imbalance=false`). Configuration checks pass 14/15 and the test suite passes 38/38. No model trained and Test untouched.
- Open before modeling: run a real tuning search through `cross_validate_weighted` (weighted PR-AUC primary); fit and validate calibration on Training-only CV; produce the Validation comparison once models are frozen. Weighted stratification is a deterministic greedy approximation and should be monitored per run.
- Self-check: train 152,149 rows / 1,525,868 weight / 0.488378% weighted cancer rate; val 28,316 rows / 269,271 weight / 0.488356%; test not loaded.
- Not started: model training, validation, selection, final evaluation.

## 12. Leakage Audit (completed)
- Scope: split files vs filtered input, original-split fidelity, validation origin, feature/record overlap, weight conservation, weighted feature distributions, `src/` and notebook fit/transform usage, and full-dataset EDA decisions.
- Verdict: **PASS WITH WARNINGS** — conservation, original split, and validation-origin checks all passed; warnings for expected train/val duplicate profiles, feature-distribution gaps (up to 3.87pp), and full-dataset EDA/one-hot/PCA/t-SNE fit on data including test (exploratory only, no persisted preprocessing objects).
- Details and modeling rules: `outputs/preprocessing/reports/05_leakage_audit.md`.

Last updated: 2026-11-04 | git HEAD: 40e7c4f
