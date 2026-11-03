# PROJECT_MAP

## 1. Summary
Research repository for the BCSC Risk Estimation dataset (aggregated screening-mammography records, ~280,660 rows). Work covers TXT→CSV conversion, EDA, feature-cancer association, PCA/t-SNE, weighted train/validation/test splitting, a reusable split loader, a leakage-safe preprocessing pipeline (raw-input mapping + model encoders), and a unified training configuration for four models; no model training or evaluation exists yet.

## 2. Stack
- Python (script + Jupyter notebooks); no pinned versions (`requirements.txt` is empty by design).
- Observed environment (2026-11-04): Python 3.15.0, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0; PyTorch, XGBoost, LightGBM, CatBoost, and TensorFlow/Keras are not installed.
- Libraries used in code: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, joblib.
- No DB, API, web server, Docker, or external services. Data shipped as local CSV/TXT.
- License: Apache 2.0 (LICENSE). No packaging/test config found.

## 3. Folder Map
- `data/01_raw/`: source `risk.txt` plus CSV copies (git-ignored); `02_interim/`: filtered CSVs; `03_processed/`: empty; `04_splits/`: `train_full.csv`, `train.csv`, `val.csv`, `test.csv`.
- `notebooks/02_eda/`: EDA, feature-vs-cancer, PCA/t-SNE notebooks; `03_preprocessing/`: column-drop notebook; `01_data_discovery/`, `04_modeling/`, `05_final_evaluation/`: empty.
- `src/data/`: `risk_to_csv.py` (TXT→CSV) and `load_splits.py` (split loader); `src/preprocessing/split_original.py`, `split_validation.py`, `raw_input_mapping.py` (interface→BCSC mapper), `pipeline.py` (encoder families), `fit_preprocessing.py` (train-only artifact fitting); `src/config.py` (central reproducibility settings); `src/training/config.py` (unified model/experiment configuration); `src/modeling|evaluation/`: empty.
- `outputs/eda/{tables,figures}`: 14 CSVs + 18 images; `outputs/preprocessing/reports/`: reviews 01–14 (splits, leakage audit, missing-value policy, encoding analysis, preprocessing pipeline); `outputs/training/reports/`: reviews 01–06 (preprocessing implementation review, train/validation compatibility check, model-to-encoder selection, unified evaluation protocol, reproducibility policy, unified training configuration); other `outputs/*`: empty.
- `models/preprocessing/`: fitted mapping JSON, `onehot`/`ordinal_tree` joblib pipelines, metadata; `models/trained/`: configured but not created; `tests/`: `test_preprocessing.py`, `test_training_config.py`; `docs/{preprocessing,modeling,final_evaluation}`: empty.
- `docs/`: state matrix, dataset notes, EDA notes; `references/`: placeholder notes only.

## 4. Entry Points and Run
- `python src/data/risk_to_csv.py [src] [dst]` (defaults `risk.txt` → `risk.csv`; validated with asserts).
- `python src/preprocessing/split_original.py` → `data/04_splits/{train_full,test}.csv`.
- `python src/preprocessing/split_validation.py` → `data/04_splits/{train,val}.csv` (weighted 85/15, seed 42).
- `python -m src.data.load_splits` runs the loader self-check (train + val only).
- `python -m src.config` prints the observed Python/library/thread/seed snapshot used in experiment records.
- `python -m src.training.config [--with-splits]` validates the unified training configuration without training; `--with-splits` also reloads Train/Validation and checks the frozen row/weight/rate references.
- `python -m src.preprocessing.fit_preprocessing` fits both encoder families on `train` only and writes `models/preprocessing/`.
- `python -m unittest discover -s tests -v` runs the preprocessing and training-configuration test suites (pytest is not installed).
- Notebooks run manually with working directory = data folder; they read/write bare filenames (`risk.csv`, `risk_filtered.csv`) — (unverified) intended run location.
- No CLI, server, or orchestrator; README states no verified install steps.

## 5. Main Flow
`risk.txt` → `risk_to_csv.py` (16 fixed columns, sanity checks) → `risk.csv` → `drop.ipynb` (drops `race`, `Hispanic`, `invasive`) → `risk_filtered.csv` → `split_original.py` (original `training` flag) → `train_full.csv`/`test.csv` → `split_validation.py` (weighted stratified 15% validation) → `train.csv`/`val.csv` → `load_splits.load_split()` → `(X, y, w)` with `count` as `sample_weight` → `fit_preprocessing.py` (fit on train) / `pipeline.transform_features` (transform val/test) → encoded model matrix → `src/training/config.py` (model/encoder mapping, initial parameters, shared weighted evaluation/tuning rules) → training experiment (not started). EDA notebooks separately use `count` weights for tables/figures.

## 6. Key Files
- `src/data/risk_to_csv.py`: column list, allowed-value validation, conversion.
- `src/data/load_splits.py`: single source of truth — `FEATURES` (10 fixed, ordered), `TARGET="cancer"`, `WEIGHT="count"`; `load_split("train"|"val"|"test") -> (X, y, w)` validates columns/binary target/positive weights, applies no preprocessing, and returns `w` for `sample_weight`.
- `src/config.py`: central reproducibility settings — `RANDOM_STATE=42`, `N_JOBS=1`, `DEFAULT_THRESHOLD=0.5`, global Python/NumPy seeding, optional PyTorch seeding, and `environment_snapshot()` for experiment records.
- `src/training/config.py`: unified training configuration — imports `FEATURES`/`TARGET`/`WEIGHT`/`SPLIT_PATHS` and the central seed/thread/threshold; declares per-model encoder, optional scaler, and initial hyperparameters; defines shared weighted metrics, tuning policy, output layout, and a read-only validator (`python -m src.training.config`).
- `src/preprocessing/split_original.py`: preserves the original `training` split.
- `src/preprocessing/split_validation.py`: weighted stratified 85/15 train/validation split (seed 42).
- `src/preprocessing/raw_input_mapping.py`: stage-A interface→BCSC mapping (`0` = Unknown → 9; agegrp has no Unknown), strict validation, JSON save/load.
- `src/preprocessing/pipeline.py`: stage-B `ColumnTransformer` families `onehot` (43 cols) and `ordinal_tree` (26 cols), fixed `FEATURES` order, `handle_unknown="ignore"` for nominal codes, ordinal rank+Unknown flags, joblib save/load, `encode_user_inputs` inference path.
- `src/preprocessing/fit_preprocessing.py`: fits both encoder families on `train` only and saves reusable artifacts + metadata.
- `tests/test_preprocessing.py`: 16 stdlib `unittest` tests covering mapping, feature order/dimensions, train-only fitting, Unknown/unseen handling, serialization, and inference parity.
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
- `outputs/training/reports/06_training_configuration.md`: unified configuration structure, per-model encoder/scaler/initial parameters, shared weighted evaluation and tuning policy, output directories, and read-only verification that all four models resolve to the correct encoded matrix (completed; no model trained and test untouched).

## 7. Config/Env (names only)
No `.env` or `.env.example`; configuration is code-based. Central reproducibility settings live in `src/config.py` (`RANDOM_STATE`, `N_JOBS`, `DEFAULT_THRESHOLD`, seeding helpers, `environment_snapshot`). Training-time configuration lives in `src/training/config.py`, which imports those shared values and `load_splits` identifiers instead of redefining them: `FEATURE_ORDER`, `SPLIT_REFERENCES`, `ENCODERS`, `SCALERS`, `MODEL_SPECS`, `METRICS`, `EVALUATION_POLICY`, `TUNING_POLICY`, `TUNING_SPACES`, `SEED_POLICY`, `OUTPUT_DIRECTORIES`, `RUN_ARTIFACTS`. Other constants live in code: notebook paths (`INPUT_FILE`, `OUTPUT_FILE`, `DATA_PATH`, `OUTPUT_DIR`), split scripts (`RANDOM_STATE`, `VAL_FRACTION`), loader (`FEATURES`, `TARGET`, `WEIGHT`, `SPLIT_PATHS`).

## 8. Conventions
- snake_case Python; flat procedural scripts/notebooks; stage-numbered folders (`01_`…`05_`).
- Validation by `assert`/`raise ValueError`; prints for reporting; no logging framework.
- Target `cancer`; `count` = frequency weight, never a feature; 10 fixed features in `load_splits.FEATURES`; excluded columns (`race`, `Hispanic`, `invasive`) handled explicitly.
- Splits are fixed files under `data/04_splits/`; never re-split or reshuffle inside training code.
- Preprocessing is two-stage: numeric interface choices → documented BCSC codes (`raw_input_mapping`) → model-specific encoder (`pipeline`). Learned encoders are fitted on `train` only and applied unchanged to val/test; inference reuses the same fitted pipeline.
- Model comparison follows `outputs/training/reports/04_evaluation_protocol.md`: `count`-weighted metrics, Validation only, positive class `cancer=1`, and `t=0.5` for initial threshold-dependent results; Test is evaluated once after freezing.
- All training code imports `RANDOM_STATE`/`N_JOBS` from `src.config`; official runs use seed 42, set `PYTHONHASHSEED` and BLAS/OpenMP threads at launch, and record configuration, environment, split fingerprints, metrics, and predictions under `outputs/training/experiments/`.
- Model-to-encoder mapping is fixed in `src/training/config.py`: Logistic Regression → `onehot`, Random Forest → `ordinal_tree`, Gradient Boosting → `ordinal_tree` (scikit-learn numeric matrix), MLP → `onehot` plus a Training-fitted `StandardScaler`. Initial parameters are baselines; tuning is Training-only 5-fold stratified CV with `count` weights and PR-AUC as the primary search score.
- Use `load_encoded_split(model_key, "train"|"val")` for a transform-only encoded matrix; it rejects `test`, checks the matrix width against the saved metadata, and never refits an encoder. Use `build_estimator`/`build_scaler` for unfitted objects.
- Raw split CSV column order differs from `FEATURES`; always use `load_split()` (or reorder) before transformation. The pipeline rejects reordered frames by design.
- Documentation is evidence-based, separating verified vs pending items.

## 9. Tests
Preprocessing suite: `tests/test_preprocessing.py`, 16 stdlib `unittest` tests, all passing (`python -m unittest discover -s tests -v`; pytest is not installed). Covers interface→BCSC mapping and rejection rules, feature count/order, stable encoded dimensions (43/26), train-only category fitting/no refit during validation transform, Unknown and unseen-code handling, joblib/JSON save-reload consistency, and mapper→pipeline inference parity. Inline checks also remain: asserts in scripts and the `load_splits` self-check (`python -m src.data.load_splits`).

Training configuration suite: `tests/test_training_config.py`, 7 stdlib `unittest` tests. Covers feature exclusion of `count`/`cancer`, required model keys, encoder widths, central seed/`n_jobs` injection, MLP scaler scope, rejection of the Test split, encoded-matrix shape for all four models, and full configuration validation. Combined suite: **23/23 passing**. `python -m src.training.config [--with-splits]` is the standalone read-only configuration check.

## 10. Unknowns
- Category-code semantics and dataset provenance/redistribution terms (unverified); `references/` placeholders.
- Notebook-relative paths and whether raw CSVs match regenerated output (unverified).
- The 10-feature list is now fixed in code but not model-validated; `training` split semantics remain (unverified) per reports 01–02; images of unknown purpose.
- EDA/association notebooks and PCA/t-SNE used the full dataset including test rows (no persisted transformers); influence of EDA statistics on the feature-exclusion decision is (unverified).
- Exact bitwise reproducibility is not guaranteed: dependencies are unpinned, external MLP/Gradient-Boosting libraries are not installed, and BLAS/OpenMP/HW differences remain; see report 05 for mitigations and per-run recording requirements.

## 11. Progress / Status
- Completed: conversion, EDA, association analysis, PCA/t-SNE, original split preservation, weighted 85/15 validation split, `load_splits` module with self-check, the read-only leakage audit, the missing-value policy (reports 06–12), the feature-type/encoding analysis (report 13), and the two-stage preprocessing pipeline with tests and saved artifacts (report 14).
- Missing-value phase: complete with pending items — confirm `surgmeno`/`hrt` code 9 with `menopaus=1` means Unknown, and define the handling contract if true nulls ever appear.
- Encoding/preprocessing phase: implemented with both `onehot` (43 cols) and `ordinal_tree` (26 cols) families, fitted on train only and saved under `models/preprocessing/`; final model-family/encoder selection remains provisional.
- Preprocessing tests: 16/16 passing; validation is transform-only and test remains unopened.
- Preprocessing review (2026-11-04): saved artifacts load and reproduce the expected 43/26 columns; feature order, train-only categories, validation transform-only behaviour, and exclusions verified. Open findings: `surgmeno`/`hrt` conditional Not Applicable recode not implemented, and direct BCSC-coded transforms do not strictly reject every undocumented category.
- Train/validation compatibility check (2026-11-04): PASS for both `onehot` (43) and `ordinal_tree` (26); dtypes, domains, target, `count`, missingness, category coverage, and encoded dimensions all match. No unseen validation categories or all-zero encoded columns. Use `load_split()` because raw CSV order is not canonical.
- Model-to-encoder selection (2026-11-04): recommended Logistic Regression → `onehot` (scaling optional), Random Forest → `ordinal_tree`, Gradient Boosting → `ordinal_tree` for numeric-matrix implementations, MLP → `onehot` plus a Training-fitted `StandardScaler`. No scaler artifact exists yet and no model has been trained.
- Evaluation protocol (2026-11-04): unified `count`-weighted ROC-AUC, PR-AUC, recall, specificity, precision, and Brier; positive class `cancer=1`, default threshold `0.5`, Validation-only comparison, and Test evaluated once after freezing. Undefined metrics are reported as `N/A` with a reason. No model has been evaluated.
- Reproducibility policy (2026-11-04): implemented `src/config.py` with `RANDOM_STATE=42` (matching the frozen split), `N_JOBS=1`, global Python/NumPy seeding, optional PyTorch seeding, and environment snapshots. Experiment-record layout and deterministic launch settings documented in report 05; no model trained.
- Unified training configuration (2026-11-04): implemented `src/training/config.py` and report 06. All four models have a declared encoder, optional scaler, and initial parameter set; shared weighted metric, tuning, seed, split, and output policies are defined once. Read-only validation passes 11 checks (12 with live Train/Validation data); all four models resolve to the expected 43/26-column matrices and 7 new tests pass (23/23 total). No model trained and Test untouched.
- Open before modeling: choose the Gradient Boosting implementation and whether to use native categorical handling; decide how the MLP scaler is packaged/serialised; decide whether Logistic Regression uses a scaler; implement and validate weighted tuning score routing; resolve the `surgmeno`/`hrt` recode.
- Self-check: train 152,149 rows / 1,525,868 weight / 0.488378% weighted cancer rate; val 28,316 rows / 269,271 weight / 0.488356%; test not loaded.
- Not started: model training, validation, selection, final evaluation.

## 12. Leakage Audit (completed)
- Scope: split files vs filtered input, original-split fidelity, validation origin, feature/record overlap, weight conservation, weighted feature distributions, `src/` and notebook fit/transform usage, and full-dataset EDA decisions.
- Verdict: **PASS WITH WARNINGS** — conservation, original split, and validation-origin checks all passed; warnings for expected train/val duplicate profiles, feature-distribution gaps (up to 3.87pp), and full-dataset EDA/one-hot/PCA/t-SNE fit on data including test (exploratory only, no persisted preprocessing objects).
- Details and modeling rules: `outputs/preprocessing/reports/05_leakage_audit.md`.

Last updated: 2026-11-04 | git HEAD: 40e7c4f
