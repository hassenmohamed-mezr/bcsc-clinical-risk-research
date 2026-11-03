# Original Split (training column)

Input: `data/02_interim/risk_filtered.csv` — 280,660 rows, 13 columns (no `race`, `Hispanic`, `invasive`).
Split by the original `training` column only; no shuffling, re-splitting, or transformation. Row order and all columns (including `count`) preserved.

| File | training | rows | sum(count) | weighted cancer cases | weighted cancer rate |
|---|---|---|---|---|---|
| `data/04_splits/train_full.csv` | 1 | 180,465 | 1,795,139 | 8,767 | 0.488374% |
| `data/04_splits/test.csv` | 0 | 100,195 | 597,859 | 2,871 | 0.480214% |
| total | — | 280,660 | 2,392,998 | 11,638 | 0.486335% |

Mapping used: **1 = train**, **0 = test** (conventional; still unverified by repo docs).

Asserts passed: row totals, weighted totals, identical column names/order, no nulls introduced, both splits non-empty.

Script: `src/preprocessing/split_original.py` (deterministic, idempotent, paths overridable via argparse).

Last updated: 2026-11-03 | git HEAD: 40e7c4f
