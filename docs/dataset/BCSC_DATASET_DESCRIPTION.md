# BCSC Dataset Description

## Verified source assets

- `risk.txt` is preserved under `data/01_raw/`.
- CSV copies of the source data are preserved under `data/01_raw/`.
- The CSV header contains clinical and mammography variables, `cancer`,
  `training`, and `count`.

## Data characteristics

The dataset is aggregated. Its rows represent groups of examinations sharing
the same recorded characteristics and outcome. `count` represents the number of
examinations represented by each row and is treated as a frequency weight in
the existing analyses.

The observed target in the existing notebooks is `cancer`. The existing
association and exploratory analyses use weighted counts rather than treating
each aggregate row as one independent examination.

## Limitations

The inspected assets do not provide a complete external codebook or provenance
record for every category code. Category meanings should therefore be verified
against the original dataset source before clinical interpretation.
