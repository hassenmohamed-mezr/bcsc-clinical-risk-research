# Missingness Patterns

Train+val only: 180,465 rows, weight 1,795,139 (cancer=0: 1,786,372; cancer=1: 8,767).

**Actual nulls:** zero in all 10 features for both target groups (weighted rate 0.00%), so no null-based target pattern.

**Encoded special values** (unverified; not treated as missing), weighted %:

| Feature | Code | Cancer=0 | Cancer=1 | Δ pp |
|---|---|---|---|---|
| hrt | 9 | 41.02 | 32.14 | 8.88 |
| surgmeno | 9 | 52.20 | 43.97 | 8.23 |
| density | 9 | 27.27 | 29.98 | 2.70 |
| menopaus | 9 | 7.61 | 5.43 | 2.18 |
| agegrp | 10 | 2.91 | 4.85 | 1.94 |

Other five features differ ≤1.7pp.

**Co-occurrence:** 89.6% of weight has ≥1 code-9 feature; top pairs bmi+agefirst 43.5%, surgmeno+hrt 37.8%.

**Interpretation:** `hrt` and `surgmeno` show the largest descriptive gaps, but codes are unverified and cannot be read causally; they are informative only if code 9 is confirmed as unknown. No tests were run: aggregated rows repeat profiles, so weighted tests would be pseudo-replicated.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
