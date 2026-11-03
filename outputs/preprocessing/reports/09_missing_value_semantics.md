# Missing-Value Semantics

Source: BCSC official documentation (`bcsc-research.org/datasets/rfdataset/dataset`); local references hold no codebook. No documented `Missing` (not recorded) codes exist; actual nulls are zero.

| Feature | Code | Official meaning | Classification | Evidence |
|---|---|---|---|---|
| menopaus | 9 | unknown | Unknown | Doc; high |
| agegrp | 10 | 80–84 years | Valid | Doc; high |
| agegrp | 9 | 75–79 years | Valid | Doc; high |
| bmi | 9 | unknown | Unknown | Doc; high |
| agefirst | 9 | unknown | Unknown | Doc; high |
| nrelbc | 9 | unknown | Unknown | Doc; high |
| brstproc | 9 | unknown | Unknown | Doc; high |
| lastmamm | 9 | unknown | Unknown | Doc; high |
| density | 9 | unknown / different measurement system | Unknown | Doc; high |
| surgmeno | 9 | unknown or not menopausal | Unknown/NA | Doc; UNVERIFIED |
| hrt | 9 | unknown or not menopausal | Unknown/NA | Doc; UNVERIFIED |

`agefirst`/`nrelbc` 0/1/2 are valid categories, not missing.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
