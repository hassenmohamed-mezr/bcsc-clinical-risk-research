# Data Leakage Audit

**Overall verdict: PASS WITH WARNINGS**

## Results
1. **Feature overlap — WARNING.** 16,538 train/val shared combinations (allowed by design), 37,126 train/test, 15,838 val/test. Exact duplicate records train↔val: 8,775 distinct, explained by 59,931 duplicated rows already in the filtered input; train/val↔test share 0.
2. **Conservation — PASS.** train+val equals train_full and train_full+test equals filtered (full-column multiset comparison); weights 1,525,868+269,271=1,795,139 and +597,859=2,392,998.
3. **Original split — PASS.** test equals filtered[`training`==0]; train_full equals filtered[`training`==1].
4. **Validation origin — PASS.** All val rows are contained in train_full; none originate from test.
5. **Distribution — WARNING.** Weighted cancer rates 0.488378% (train) vs 0.488356% (val). Largest gaps: `brstproc` 3.87pp (code 0), `surgmeno` 2.64pp (0), `density` 2.13pp (9), `agegrp` 1.99pp (7). Test (0.480214%) is descriptive only.
6. **Code — WARNING.** Full-data `OneHotEncoder.fit_transform` (`Visualization_2d&3d.ipynb` cell 0 L78), weighted PCA (L≈100), `TSNE.fit_transform` (L217) on data including test; figures/tables only, no persisted transformer. `load_splits.py` L16 `.fit` is a docstring example; `src/` has no scalers, imputers, or selection.
7. **EDA — WARNING.** `EDA.ipynb` and `Full Feature vs Cancer.ipynb` use full `risk.csv` including test rows. `drop.ipynb` removes `race`, `Hispanic`, `invasive`; EDA influence is (unverified).

**Limitations:** aggregated rows lack patient IDs, so repeat profiles cannot be distinguished from repeated examinations.

## Rules for modeling
- Fit every preprocessing object on training data only; apply unchanged to validation and test.
- Tune hyperparameters and select models using validation data only.
- Keep test isolated; use it once for final evaluation.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
