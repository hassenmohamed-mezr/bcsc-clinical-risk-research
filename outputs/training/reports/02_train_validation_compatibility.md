# Train / Validation Compatibility Check

Date: 2026-11-04. Scope: Training and Validation splits against the saved preprocessing artifacts. Only `transform_features` was used; nothing was fitted or refit. `test.csv` was not opened. No datasets or artifacts were modified.

## Verdict

**PASS — no blocking incompatibility.** Train and Validation are compatible with both saved encoders. All 10 features are present in the required order after loading, all categorical values are documented, target and weights are valid, and both pipelines produce stable, finite matrices. Four caveats are noted below; none is caused by the current data.

## Checks and results

### 1. Feature names and ordering

- `load_split()` returns exactly the canonical order: `agegrp, bmi, agefirst, menopaus, surgmeno, hrt, nrelbc, density, brstproc, lastmamm`.
- Raw CSV headers use a different order (`menopaus, agegrp, density, bmi, …`), but the loader selects `FEATURES` explicitly. Both `X_train` and `X_val` are therefore correctly ordered.
- Both artifacts store `feature_names_in_` in the canonical order.

### 2. Data types and categorical code validity

| Feature | Type | Allowed codes | Train observed | Val observed | Unexpected |
|---|---|---|---|---|---|
| agegrp | int64 | 1–10 | 1–10 | 1–10 | none |
| bmi | int64 | 1–4, 9 | same | same | none |
| agefirst | int64 | 0, 1, 2, 9 | same | same | none |
| menopaus | int64 | 0, 1, 9 | same | same | none |
| surgmeno | int64 | 0, 1, 9 | same | same | none |
| hrt | int64 | 0, 1, 9 | same | same | none |
| nrelbc | int64 | 0, 1, 2, 9 | same | same | none |
| density | int64 | 1–4, 9 | same | same | none |
| brstproc | int64 | 0, 1, 9 | same | same | none |
| lastmamm | int64 | 0, 1, 9 | same | same | none |

- No feature nulls in either split. No undocumented codes.
- Validation contains no category absent from Training (`val − train = ∅` for every feature).

### 3. Target availability and class representation

| Split | Target dtype | Values | Rows 0 / 1 | Weight 0 / 1 | Weighted positive rate |
|---|---|---|---|---|---|
| train | int64 | 0, 1 | 146,399 / 5,750 | 1,518,416 / 7,452 | 0.488378% |
| val | int64 | 0, 1 | 27,297 / 1,019 | 267,956 / 1,315 | 0.488356% |

- `cancer` is present, non-null, binary, and both classes have positive row and weight counts in both splits.
- Class imbalance remains substantial (weighted positive rate ≈ 0.488%); this is a modeling consideration, not a compatibility failure.

### 4. `count` availability and validity

| Split | dtype | Min | Max | Sum | Non-positive | Non-finite | Integer |
|---|---|---|---|---|---|---|---|
| train | int64 | 1 | 7,295 | 1,525,868 | 0 | 0 | yes |
| val | int64 | 1 | 3,064 | 269,271 | 0 | 0 | yes |

- `count` is present, positive, finite, and integer-valued; it is a valid sklearn `sample_weight` and is not part of `X`.

### 5. Missing and unexpected values

- Feature nulls: train 0, validation 0.
- Target nulls: 0; weight nulls: 0.
- Unexpected categorical values: none in either split.
- Raw splits contain `training` as metadata and `cancer`/`count` as target/weight; the loader excludes them from `X`. `race`, `Hispanic`, and `invasive` are absent from the split headers.

### 6. Compatibility with both pipelines

Both artifacts loaded successfully and accepted both splits:

| Pipeline | Train output | Val output | Metadata | Names match |
|---|---|---|---|---|
| onehot | (152,149, 43) | (28,316, 43) | 43 | yes |
| ordinal_tree | (152,149, 26) | (28,316, 26) | 26 | yes |

- Output dtype is `float32`; no NaN or infinite values.
- `sklearn.utils.validation.check_X_y` passed for both pipelines and both splits.
- One-hot: every feature block sums to 1 per row; values are only 0/1.
- Ordinal tree: ordinal ranks and Unknown flags match the documented mapping exactly; nominal one-hot blocks sum to 1 per row.
- No encoded column is all-zero in either split.
- Every artifact category is observed in Training; no Validation category is outside the fitted domain.

### 7. Consistency of encoded dimensions

- Train and Validation have identical output widths for each family.
- Widths and feature names match `preprocessing_metadata.json` exactly.
- No row-count mismatch: encoded rows equal target rows and weight rows for both splits.

### 8. Model-training error risk

No current feature, category, shape, target, or weight mismatch would cause a training error. Remaining risks are operational/semantic:

1. **Raw column order.** Reading `train.csv`/`val.csv` directly and passing the DataFrame to the pipeline will raise because raw order differs from `FEATURES`. Use `load_split()` or reorder columns; do not bypass the loader.
2. **`surgmeno`/`hrt` policy divergence.** Both remain a combined code-9 category; the conditional Not Applicable recode from report 12 is not implemented. This does not break training but leaves a pending semantic decision.
3. **Latent invalid-code handling.** The one-hot family accepts undocumented codes as all-zero blocks, and nominal encoders in both families ignore unseen codes. No such values exist in train/val, so this is not triggering today; strict domain validation would make future failures explicit.
4. **Encoder choice remains open.** Both artifacts are compatible; one must be selected with the final model family.

Artifact SHA-256 values were identical before and after this check; no refitting occurred.

Last updated: 2026-11-04
