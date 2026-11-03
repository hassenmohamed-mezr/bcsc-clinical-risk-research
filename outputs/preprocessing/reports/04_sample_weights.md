# Sample Weights

**Module:** `src/data/load_splits.py` — `load_split("train" | "val" | "test") -> (X, y, w)`.

**Features (fixed order, n=10):** `agegrp`, `bmi`, `agefirst`, `menopaus`, `surgmeno`, `hrt`, `nrelbc`, `density`, `brstproc`, `lastmamm`.

**Target:** `cancer` (binary 0/1).

**Weight:** `count`, returned separately as `w`. It is a frequency weight — each row aggregates `count` examinations — and must never be used as a feature or physically expanded.

**Convention:** pass `w` to fit and metric calls:
`model.fit(X, y, sample_weight=w)`;
`roc_auc_score(y, scores, sample_weight=w)`.

**Warning:** metrics computed without `sample_weight` treat rows as single examinations and misrepresent the frequency-weighted dataset.

Self-check (rows / weight / weighted cancer rate): train 152,149 / 1,525,868 / 0.488378%; val 28,316 / 269,271 / 0.488356%. Test not loaded.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
