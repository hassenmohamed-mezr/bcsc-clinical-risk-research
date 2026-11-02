# Feature Association Analysis

## Objective

The preserved `notebooks/02_eda/Full Feature vs Cancer.ipynb` investigates the
association between available features and the `cancer` target.

## Existing methodology

The notebook uses weighted contingency tables based on `count`, category-level
cancer rates with weighted Wilson intervals, chi-square tests, standard
Cramér's V, and bias-corrected Cramér's V. It writes category tables and plots
to the preserved `outputs/eda/` directories.

## Existing outputs

- `outputs/eda/tables/ALL_FEATURES_ASSOCIATION_SUMMARY.csv`
- Feature-specific category-rate CSV tables
- Feature-specific cancer-rate plots

The summary ranks corrected Cramér's V. The observed ranking includes
`agegrp` as the largest recorded association, followed by `menopaus`,
`density`, `brstproc`, `nrelbc`, `lastmamm`, `surgmeno`, `Hispanic`, `race`,
`hrt`, `bmi`, and `agefirst`.

## Interpretation and limitations

These are exploratory associations, not model-performance measurements and not
proof of causation. Statistical association may reflect confounding, coding,
aggregation, missing/unknown categories, or the large weighted sample size.
`count` is a frequency weight and is not a predictive feature.
