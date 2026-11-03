# Encoding Strategy Analysis

Source: BCSC documentation (reports 09, 12). No encoder fitted; test untouched.

| Feature | Type / order | Special code | Encoding | Rationale | Status |
|---|---|---|---|---|---|
| agegrp | ordinal 1–10 (5-year bands) | none | Ordinal ranks | Order and spacing meaningful | Evidence-based |
| bmi | ordinal bands 1–4 | 9 Unknown | Ordinal + Unknown dummy | Ordered bands; 9 separate | Evidence-based |
| density | ordinal 1–4 | 9 Unknown/different system | Ordinal + Unknown dummy | Ordered density | Evidence-based |
| nrelbc | ordinal 0, 1, 2+ | 9 Unknown | Ordinal + Unknown dummy | Count order, top-coded | Evidence-based |
| menopaus | nominal binary | 9 Unknown | One-hot | No numeric distance | Evidence-based |
| brstproc, lastmamm | nominal binary | 9 Unknown | One-hot | Unordered states | Evidence-based |
| agefirst | nominal 0/1/2 | 9 Unknown | One-hot | Nulliparous breaks order | Evidence-based |
| surgmeno, hrt | nominal binary | 9 Unknown/NA unresolved | One-hot (combined 9) | No unsupported recode | Provisional |

**Comparison:** One-hot avoids false distances and handles Unknown at inference; dimensionality is low (≤10 levels). Ordinal is compact and monotonic but implies equal spacing and needs an Unknown dummy. Native categorical fits only CatBoost/LightGBM.

**Unresolved:** surgmeno/hrt Unknown-vs-NA split; model family (linear models may prefer one-hot for ordered features). No encoding implemented.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
