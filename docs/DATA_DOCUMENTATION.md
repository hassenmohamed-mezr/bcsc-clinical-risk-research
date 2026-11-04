# Data Documentation

## Data Sources

- Source file: `data/01_raw/risk.txt`
- Converted file: `data/01_raw/risk.csv`
- Conversion script: `src/data/risk_to_csv.py`
- Filtered file: `data/02_interim/risk_filtered.csv`
- Splits: `data/04_splits/train_full.csv`, `train.csv`, `val.csv`, `test.csv`

Evidence: `src/data/risk_to_csv.py`, `src/preprocessing/split_original.py`, `src/preprocessing/split_validation.py`, `src/data/load_splits.py`.

## Feature Set

The canonical feature order is defined in `src/data/load_splits.py`:

1. `agegrp`
2. `bmi`
3. `agefirst`
4. `menopaus`
5. `surgmeno`
6. `hrt`
7. `nrelbc`
8. `density`
9. `brstproc`
10. `lastmamm`

## Target and Weight

- Target: `cancer` (binary 0/1)
- Frequency weight: `count` (positive, used as `sample_weight`)
- `count` is never a feature.

Evidence: `src/data/load_splits.py`, `src/training/config.py`.

## Excluded Columns

The following are not model features:

- `training` (metadata/split indicator)
- `invasive`
- `race`
- `Hispanic`

Evidence: `src/data/risk_to_csv.py`, `src/data/load_splits.py`, `PROJECT_MAP.md`.

## Category Codes

| Feature | Codes | Notes |
|---|---|---|
| `agegrp` | 1-10 | 5-year age bands; 9 and 10 are valid ages, not Unknown |
| `bmi` | 1-4, 9 | 9 = Unknown |
| `agefirst` | 0, 1, 2, 9 | 0=<30, 1=>=30, 2=nulliparous, 9=Unknown |
| `menopaus` | 0, 1, 9 | 0=premenopausal, 1=postmenopausal/age>=55, 9=Unknown |
| `surgmeno` | 0, 1, 9 | 0=natural, 1=surgical, 9=Unknown or not menopausal |
| `hrt` | 0, 1, 9 | 0=no, 1=yes, 9=Unknown or not menopausal |
| `nrelbc` | 0, 1, 2, 9 | 0=none, 1=one, 2=two or more, 9=Unknown |
| `density` | 1-4, 9 | 1=almost entirely fat, 2=scattered, 3=heterogeneously dense, 4=extremely dense, 9=Unknown/different system |
| `brstproc` | 0, 1, 9 | 0=no, 1=yes, 9=Unknown |
| `lastmamm` | 0, 1, 9 | 0=negative, 1=false positive, 9=Unknown |

Evidence: `src/preprocessing/raw_input_mapping.py`, `outputs/preprocessing/reports/09_missing_value_semantics.md`, `src/data/risk_to_csv.py`.

## Missing Value Policy

- All 10 features are categorical integer codes.
- No actual nulls exist in train/val according to reports.
- Unknown is encoded as explicit category 9 for most features.
- `agegrp` has no Unknown code.
- `surgmeno` and `hrt` code 9 combines Unknown and Not Applicable.
- No mode/mean imputation; raw data stays unchanged.
- Future true nulls are not fully contracted; current validators reject them.

Evidence: `outputs/preprocessing/reports/12_missing_value_policy.md`, `src/data/load_splits.py`.

## Splitting Strategy

- Original split preserved from `training` flag (`split_original.py`).
- Validation split created from training only (`split_validation.py`), weighted stratified 85/15, seed 42.
- Test split remains frozen and was used only for one authorized final evaluation.

Evidence: `outputs/preprocessing/reports/02_original_split.md`, `03_validation_split.md`, `PROJECT_MAP.md`.

## Data Validation

`load_split()` validates:

- required columns
- no duplicated columns
- binary target
- positive weights
- no missing feature values
- exact feature order

Evidence: `src/data/load_splits.py`.

## Data Limitations

- Aggregated data; rows represent groups, not individual examinations.
- `count` is a frequency weight.
- Category semantics rely on BCSC documentation; local codebook is incomplete.
- `surgmeno`/`hrt` code 9 is unresolved at row level.