# Training Column Review

Source: `data/01_raw/risk.csv` (280,660 rows, 16 columns). Read-only review; weighted metrics use `count` (sum = 2,392,998).

## Values
- `training` is `int64`, no nulls, exactly two values: **0** and **1**.
- No local document defines the meaning. `docs/PROJECT_STATE_MATRIX.md` only notes the column "exists in the source header and is summarized in EDA"; `README` mentions model training as absent. Semantics are therefore external to the repo.

## Sizes (rows and weighted)
| training | rows | % rows | sum(count) | % weighted |
|---|---|---|---|---|
| 0 | 100,195 | 35.7% | 597,859 | 25.0% |
| 1 | 180,465 | 64.3% | 1,795,139 | 75.0% |

## Cancer Rate per group
- `training = 0`: 0.4802% (weighted cancer cases ≈ 2,871).
- `training = 1`: 0.4884% (weighted cancer cases ≈ 8,769).
- Overall ≈ 0.4863%, consistent with `docs/exploratory_analysis/EDA_METHODOLOGY_AND_FINDINGS.md`.

## Overlap Check
- 265,901 distinct full-feature combinations; **14,759 combinations appear in both groups (yes, overlap exists)**.
- Those combinations cover 29,518 rows (10.5%) and 40,598 weighted examinations (1.7%).
- So the split is by aggregated row, not by unique patient/feature profile; the same profile can be labelled 0 and 1.

## Conclusion
- Value **1 = training** and **0 = test/validation** is the conventional reading and is consistent with the 75/25 weighted split — but no repository document states it: **(unverified)**.
- Split is row-level with detectable profile leakage across groups; do not assume disjoint populations.
- Cancer prevalence is nearly identical in both groups, so the split is reasonably balanced.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
