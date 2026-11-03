# Missing-Value Policy (authoritative)

All 10 features are categorical integer codes; no nulls exist in train/val. Raw data stays unchanged; `count` is the frequency weight, never a feature.

| Feature(s) | Type | Actual null | Unknown / NA | Treatment | Indicator | Reason | Pending |
|---|---|---|---|---|---|---|---|
| agegrp | categorical 1–10 | none | none (9 = 75–79, 10 = 80–84 valid) | Passthrough | No | Documented valid categories | — |
| menopaus, bmi, agefirst, nrelbc, density, brstproc, lastmamm | categorical | none | 9 = Unknown | Retain 9 as explicit Unknown level; no imputation | No | Preserves information and semantics | — |
| surgmeno, hrt | categorical | none | 9 = Unknown or not menopausal | `menopaus=0` → Not applicable; otherwise 9 → Unknown | No | Documented conditional meaning | Confirm 9 with menopaus=1 is Unknown (provisional) |

## Global rules
- Fit every learned preprocessing object on train only; apply unchanged to validation and test.
- Never use `cancer` or test data for imputation, category definitions, or thresholds.
- No mode/mean imputation, no row or feature deletion; keep raw splits intact.
- If actual nulls appear later, flag and treat as Unknown — never silently impute.
- Encode Unknown/NA as explicit categories, fitted on train only.

**Status:** complete with pending items — surgmeno/hrt unknown-vs-NA confirmation; handling contract if true nulls ever appear.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
