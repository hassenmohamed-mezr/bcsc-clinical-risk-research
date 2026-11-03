# Missing-Value Treatment Decisions

Evidence: official BCSC coding (report 09); nulls = 0; test data not used.

| Feature | Decision | Representation | Basis |
|---|---|---|---|
| agegrp | No treatment | 9 = 75–79, 10 = 80–84 | Documented valid codes; evidence-based |
| menopaus, bmi, agefirst, nrelbc, density, brstproc, lastmamm | Retain Unknown level | code 9 → "Unknown" category, not NaN | Preserves information; evidence-based |
| surgmeno, hrt | Conditional recode | menopaus=0 → "Not applicable"; other 9 → "Unknown" | Documented conditional meaning; provisional |

**Missing indicator:** not justified — Unknown is a semantic category, so an indicator would duplicate it; revisit only if a model requires numeric NaN input.

**Rejected:** mode imputation (fabricates values, distorts weighted estimates, hides informative missingness at 41–56% shares); plain 9→NaN (erases the distinction from actual missingness); model-native NA alone (helps trees only, still needs encoding).

**Confidence:** agegrp, menopaus, bmi, agefirst, nrelbc, density, brstproc, lastmamm are evidence-based. The surgmeno/hrt recode is provisional: it assumes 9 with menopaus=1 is Unknown, pending codebook confirmation. No cancer-based rules, and no validation-performance tuning.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
