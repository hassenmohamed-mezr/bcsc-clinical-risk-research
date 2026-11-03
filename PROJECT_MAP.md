# PROJECT_MAP

## 1. Summary
Research repository for the BCSC Risk Estimation dataset (aggregated screening-mammography records, ~280,660 rows). Work so far is data conversion, EDA, feature-cancer association, and PCA/t-SNE visualization; no model training or evaluation exists.

## 2. Stack
- Python (script + Jupyter notebooks); no pinned versions (`requirements.txt` is empty by design).
- Libraries used in code: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn.
- No DB, API, web server, Docker, or external services. Data shipped as local CSV/TXT.
- License: Apache 2.0 (LICENSE). No packaging/test config found.

## 3. Folder Map
- `data/01_raw/`: source `risk.txt` plus CSV copies (git-ignored); `02_interim/`: filtered CSVs; `03_processed/`, `04_splits/`: empty.
- `notebooks/02_eda/`: EDA, feature-vs-cancer, PCA/t-SNE notebooks; `03_preprocessing/`: column-drop notebook; `01_data_discovery/`, `04_modeling/`, `05_final_evaluation/`: empty.
- `src/data/`: TXT→CSV conversion script; `src/preprocessing|modeling|evaluation/`: empty.
- `outputs/eda/{tables,figures}`: 14 CSVs + 18 images; other `outputs/*`: empty.
- `models/*`, `tests/`, `docs/{preprocessing,modeling,final_evaluation}`: empty.
- `docs/`: state matrix, dataset notes, EDA notes; `references/`: placeholder notes only.

## 4. Entry Points and Run
- `python src/data/risk_to_csv.py [src] [dst]` (defaults `risk.txt` → `risk.csv`; validated with asserts).
- Notebooks run manually with working directory = data folder; they read/write bare filenames (`risk.csv`, `risk_filtered.csv`) — (unverified) intended run location.
- No CLI, server, or orchestrator; README states no verified install steps.

## 5. Main Flow
`risk.txt` → `risk_to_csv.py` (16 fixed columns, sanity checks) → `risk.csv` → `drop.ipynb` (drops `race`, `Hispanic`, `invasive`) → `risk_filtered.csv` → `Visualization_2d&3d.ipynb` (one-hot, PCA 2D/3D, t-SNE) and `EDA.ipynb` / `Full Feature vs Cancer.ipynb` (weighted via `count`) → CSVs/PNGs in `outputs/eda/`.

## 6. Key Files
- `src/data/risk_to_csv.py`: column list, allowed-value validation, conversion.
- `notebooks/03_preprocessing/drop.ipynb`: controlled column exclusion + assert.
- `notebooks/02_eda/EDA.ipynb`: weighted summaries, Wilson CI, unknown-rate table, training split.
- `notebooks/02_eda/Full Feature vs Cancer.ipynb`: weighted contingency, chi-square, Cramér's V / bias-corrected V, outputs to `BCSC_Full_Feature_Analysis`.
- `notebooks/02_eda/Visualization_2d&3d.ipynb`: PCA/t-SNE figures.
- `docs/PROJECT_STATE_MATRIX.md`: authoritative inventory/status.

## 7. Config/Env (names only)
None found: no `.env`, `.env.example`, or config module. Only in-notebook constants (`INPUT_FILE`, `OUTPUT_FILE`, `COLUMNS_TO_DROP`, `DATA_PATH`, `EXCLUDED_COLUMNS`, `OUTPUT_DIR`).

## 8. Conventions
- snake_case Python; flat procedural scripts/notebooks; stage-numbered folders (`01_`…`05_`).
- Validation by `assert`/`raise ValueError`; prints for reporting; no logging framework.
- Target `cancer`; `count` = frequency weight, never a feature; excluded columns handled explicitly.
- Documentation is evidence-based, separating verified vs pending items.

## 9. Tests
No test suite or framework; `tests/` is empty. Only inline sanity checks inside the script and notebooks.

## 10. Unknowns
- Category-code semantics and dataset provenance/redistribution terms (unverified); `references/` placeholders.
- Notebook-relative paths and whether raw CSVs match regenerated output (unverified).
- Feature set not finalized; `training` column role unverified; images of unknown purpose.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
