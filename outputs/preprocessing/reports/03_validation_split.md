# Validation Split Report

- **Input:** `data/04_splits/train_full.csv` (original test set never opened)
- **Algorithm:** Weighted stratified random split by target class + deterministic single-row local optimization
- **Random state:** 42

## Counts
| Set | Rows | Frequency weight (`count`) |
|---|---|---|
| Original | 180,465 | 1,795,139 |
| Training | 152,149 | 1,525,868 |
| Validation | 28,316 | 269,271 |

- Achieved validation weight share: **15.000008%** (target 15%; tolerance 14–16% met).
- Per class: class 0 val share 15.0000% (residual 0.200), class 1 val share 14.9994% (residual 0.050).

## Weighted cancer rate
| Set | Rate | Relative difference |
|---|---|---|
| Original | 0.488374% | — |
| Training | 0.488378% | 0.0007% |
| Validation | 0.488356% | 0.0039% |

Both well within the 10% relative tolerance.

## Checks
All required integrity checks passed: row totals, exact one-time row preservation, identical columns/order, unchanged `count` values, weighted totals, validation share, and stratification.

Shared feature combinations (diagnostic only, expected and allowed): **16,538**.

No deviations from the target or constraints. No helper columns added; no rows expanded, merged, or modified.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
