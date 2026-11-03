# PROJECT_MAP

## 1. Summary
Research repository for the BCSC Risk Estimation dataset (aggregated screening-mammography records, ~280,660 rows). Work covers TXT→CSV conversion, EDA, feature-cancer association, PCA/t-SNE, weighted train/validation/test splitting, a reusable split loader, and a leakage-safe preprocessing pipeline (raw-input mapping + model encoders); no model training or evaluation exists yet.

## 2. Stack
- Python (script + Jupyter notebooks); no pinned versions (`requirements.txt` is empty by design).
- Libraries used in code: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn.
- No DB, API, web server, Docker, or external services. Data shipped as local CSV/TXT.
- License: Apache 2.0 (LICENSE). No packaging/test config found.

## 3. Folder Map
- `data/01_raw/`: source `risk.txt` plus CSV copies (git-ignored); `02_interim/`: filtered CSVs; `03_processed/`: empty; `04_splits/`: `train_full.csv`, `train.csv`, `val.csv`, `test.csv`.
- `notebooks/02_eda/`: EDA, feature-vs-cancer, PCA/t-SNE notebooks; `03_preprocessing/`: column-drop notebook; `01_data_discovery/`, `04_modeling/`, `05_final_evaluation/`: empty.
- `src/data/`: `risk_to_csv.py` (TXT→CSV) and `load_splits.py` (split loader); `src/preprocessing/split_original.py`, `split_validation.py`, `raw_input_mapping.py` (interface→BCSC mapper), `pipeline.py` (encoder families), `fit_preprocessing.py` (train-only artifact fitting); `src/modeling|evaluation/`: empty.
- `outputs/eda/{tables,figures}`: 14 CSVs + 18 images; `outputs/preprocessing/reports/`: reviews 01–14 (splits, leakage audit, missing-value policy, encoding analysis, preprocessing pipeline); other `outputs/*`: empty.
- `models/preprocessing/`: fitted mapping JSON, `onehot`/`ordinal_tree` joblib pipelines, metadata; `tests/`: `test_preprocessing.py`; `docs/{preprocessing,modeling,final_evaluation}`: empty.
- `docs/`: state matrix, dataset notes, EDA notes; `references/`: placeholder notes only.

## 4. Entry Points and Run
- `python src/data/risk_to_csv.py [src] [dst]` (defaults `risk.txt` → `risk.csv`; validated with asserts).
- `python src/preprocessing/split_original.py` → `data/04_splits/{train_full,test}.csv`.
- `python src/preprocessing/split_validation.py` → `data/04_splits/{train,val}.csv` (weighted 85/15, seed 42).
- `python -m src.data.load_splits` runs the loader self-check (train + val only).
- `python -m src.preprocessing.fit_preprocessing` fits both encoder families on `train` only and writes `models/preprocessing/`.
- `python -m unittest discover -s tests -v` runs the preprocessing test suite (pytest is not installed).
- Notebooks run manually with working directory = data folder; they read/write bare filenames (`risk.csv`, `risk_filtered.csv`) — (unverified) intended run location.
- No CLI, server, or orchestrator; README states no verified install steps.

## 5. Main Flow
`risk.txt` → `risk_to_csv.py` (16 fixed columns, sanity checks) → `risk.csv` → `drop.ipynb` (drops `race`, `Hispanic`, `invasive`) → `risk_filtered.csv` → `split_original.py` (original `training` flag) → `train_full.csv`/`test.csv` → `split_validation.py` (weighted stratified 15% validation) → `train.csv`/`val.csv` → `load_splits.load_split()` → `(X, y, w)` with `count` as `sample_weight` → `fit_preprocessing.py` (fit on train) / `pipeline.transform_features` (transform val/test) → encoded model matrix. EDA notebooks separately use `count` weights for tables/figures.

## 6. Key Files
- `src/data/risk_to_csv.py`: column list, allowed-value validation, conversion.
- `src/data/load_splits.py`: single source of truth — `FEATURES` (10 fixed, ordered), `TARGET="cancer"`, `WEIGHT="count"`; `load_split("train"|"val"|"test") -> (X, y, w)` validates columns/binary target/positive weights, applies no preprocessing, and returns `w` for `sample_weight`.
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

## 7. Config/Env (names only)
None found: no `.env`, `.env.example`, or config module. Constants live in code: notebook paths (`INPUT_FILE`, `OUTPUT_FILE`, `DATA_PATH`, `OUTPUT_DIR`), split scripts (`RANDOM_STATE`, `VAL_FRACTION`), loader (`FEATURES`, `TARGET`, `WEIGHT`, `SPLIT_PATHS`).

## 8. Conventions
- snake_case Python; flat procedural scripts/notebooks; stage-numbered folders (`01_`…`05_`).
- Validation by `assert`/`raise ValueError`; prints for reporting; no logging framework.
- Target `cancer`; `count` = frequency weight, never a feature; 10 fixed features in `load_splits.FEATURES`; excluded columns (`race`, `Hispanic`, `invasive`) handled explicitly.
- Splits are fixed files under `data/04_splits/`; never re-split or reshuffle inside training code.
- Preprocessing is two-stage: numeric interface choices → documented BCSC codes (`raw_input_mapping`) → model-specific encoder (`pipeline`). Learned encoders are fitted on `train` only and applied unchanged to val/test; inference reuses the same fitted pipeline.
- Documentation is evidence-based, separating verified vs pending items.

## 9. Tests
Preprocessing suite: `tests/test_preprocessing.py`, 16 stdlib `unittest` tests, all passing (`python -m unittest discover -s tests -v`; pytest is not installed). Covers interface→BCSC mapping and rejection rules, feature count/order, stable encoded dimensions (43/26), train-only category fitting/no refit during validation transform, Unknown and unseen-code handling, joblib/JSON save-reload consistency, and mapper→pipeline inference parity. Inline checks also remain: asserts in scripts and the `load_splits` self-check (`python -m src.data.load_splits`).

## 10. Unknowns
- Category-code semantics and dataset provenance/redistribution terms (unverified); `references/` placeholders.
- Notebook-relative paths and whether raw CSVs match regenerated output (unverified).
- The 10-feature list is now fixed in code but not model-validated; `training` split semantics remain (unverified) per reports 01–02; images of unknown purpose.
- EDA/association notebooks and PCA/t-SNE used the full dataset including test rows (no persisted transformers); influence of EDA statistics on the feature-exclusion decision is (unverified).

## 11. Progress / Status
- Completed: conversion, EDA, association analysis, PCA/t-SNE, original split preservation, weighted 85/15 validation split, `load_splits` module with self-check, the read-only leakage audit, the missing-value policy (reports 06–12), the feature-type/encoding analysis (report 13), and the two-stage preprocessing pipeline with tests and saved artifacts (report 14).
- Missing-value phase: complete with pending items — confirm `surgmeno`/`hrt` code 9 with `menopaus=1` means Unknown, and define the handling contract if true nulls ever appear.
- Encoding/preprocessing phase: implemented with both `onehot` (43 cols) and `ordinal_tree` (26 cols) families, fitted on train only and saved under `models/preprocessing/`; final model-family/encoder selection remains provisional.
- Preprocessing tests: 16/16 passing; validation is transform-only and test remains unopened.
- Self-check: train 152,149 rows / 1,525,868 weight / 0.488378% weighted cancer rate; val 28,316 rows / 269,271 weight / 0.488356%; test not loaded.
- Not started: model training, validation, selection, final evaluation.

## 12. Leakage Audit (completed)
- Scope: split files vs filtered input, original-split fidelity, validation origin, feature/record overlap, weight conservation, weighted feature distributions, `src/` and notebook fit/transform usage, and full-dataset EDA decisions.
- Verdict: **PASS WITH WARNINGS** — conservation, original split, and validation-origin checks all passed; warnings for expected train/val duplicate profiles, feature-distribution gaps (up to 3.87pp), and full-dataset EDA/one-hot/PCA/t-SNE fit on data including test (exploratory only, no persisted preprocessing objects).
- Details and modeling rules: `outputs/preprocessing/reports/05_leakage_audit.md`.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
