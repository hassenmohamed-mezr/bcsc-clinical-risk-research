# Preprocessing Review

Date: 2026-11-04. Scope: approved features, raw-input mapper, both encoder families, train-only fitting, validation transform, column exclusions, and saved artifacts. No implementation was modified and no pipeline was refit. `test.csv` was not read.

## Verdict

Verified: the saved artifacts load, expose the approved 10 features in the fixed order, and transform validation data correctly. Two findings need follow-up: a policy/implementation divergence for `surgmeno`/`hrt`, and incomplete domain validation when transforming BCSC-coded data. Neither affects the validated user-input path today.

## Verified facts

### 1. Features and ordering

- `src/data/load_splits.py` defines exactly: `agegrp, bmi, agefirst, menopaus, surgmeno, hrt, nrelbc, density, brstproc, lastmamm`.
- Both saved pipelines store `feature_names_in_` in exactly that order; the metadata and mapper JSON repeat it.
- `count`, `cancer`, `training`, `invasive`, `race`, and `Hispanic` appear in none of the feature lists or encoded names. Transforming a frame with any of them added raises `ValueError`; reordered columns also raise.
- Train is 152,149 rows / 1,525,868 weight; validation is 28,316 rows / 269,271 weight.

### 2. Raw-input mapping

- `models/preprocessing/raw_input_mapping.json` loads and its choices equal the documented `CHOICES` exactly.
- Interface `0` maps to BCSC `9` for the nine features with Unknown; `agegrp` rejects `0` with `InputMappingError`. All interface values are strict integer choices; invalid, missing, extra, boolean, and non-integer inputs are rejected.
- Mapping follows documented order: `agefirst 1/2/3 → 0/1/2`, `nrelbc 1/2/3 → 0/1/2`, binary features `1/2 → 0/1`, and `agegrp 1–10 → 1–10`.
- Every validation BCSC code is mappable. For both families, `encode_user_inputs(mapper, pipeline, interface)` equals direct `transform_features(pipeline, codes)` exactly.

### 3. Encoder assignment

| Feature | One-hot family | Ordinal-tree family |
|---|---|---|
| agegrp | one-hot, 10 columns | ordinal rank 1–10, no Unknown flag |
| bmi | one-hot, 5 columns | rank 1–4 + Unknown flag |
| nrelbc | one-hot, 4 columns | rank 1–3 + Unknown flag |
| density | one-hot, 5 columns | rank 1–4 + Unknown flag |
| agefirst | one-hot, 4 columns | one-hot, 4 columns |
| menopaus, surgmeno, hrt, brstproc, lastmamm | one-hot, 3 columns each | one-hot, 3 columns each |

- One-hot output: **43 columns**. Ordinal-tree output: **26 columns** (7 ordinal + 19 nominal).
- Artifact names and dimensions match `preprocessing_metadata.json` exactly.
- `agegrp` code 9 and 10 are valid age categories, not Unknown.

### 4. Train-only fitting

- `fit_preprocessing.py` loads only `load_split("train")` and passes only that frame to `fit_pipeline`.
- `transform_features` calls only `pipeline.transform`; no fit occurs during validation transformation.
- Metadata records `fit_split="train"`, `test_used=false`, 152,149 fitted rows, and weight 1,525,868, matching the train split.
- `count` is not passed to either encoder; it is loaded only to record the fitted weight sum. No weighted category derivation occurs.
- One-hot categories equal the train per-feature unique code sets exactly: `[10, 5, 4, 3, 3, 3, 4, 5, 3, 3]`.
- Transforming validation leaves every fitted category list unchanged.
- Limitation: provenance is inferred from source code and metadata; the artifacts contain no train-file hash or row fingerprint.

### 5. Validation transform and Unknown handling

- Loaded artifacts transformed validation to `(28316, 43)` and `(28316, 26)` with zero NaN/inf values.
- One-hot blocks sum to 1 per feature per row; nominal one-hot blocks in the ordinal-tree output also sum to 1. Ordinal ranks and Unknown flags match the documented codebook for every validation row.
- Unknown code 9 is an explicit one-hot category or an ordinal rank-0 + flag. `agegrp` has no Unknown flag.
- `surgmeno`/`hrt` code 9 is encoded as one combined category.

### 6. Saved artifacts

| Artifact | Loaded | SHA-256 |
|---|---|---|
| `raw_input_mapping.json` | yes | `1e03383950ca52f544982f5911cab58e114bd3a0c21726699b9f6d0827c3fbfd` |
| `onehot_pipeline.joblib` | yes | `cb4625af62f04c5f3980a6b9734ef90f71201063d45959b957f7d40bbe6458f4` |
| `ordinal_tree_pipeline.joblib` | yes | `a95345c4932be9623d6b6c29f7d3e5b906ded00485488b96dd63ca218962139d` |
| `preprocessing_metadata.json` | yes | `8029e005e0b72ddb984849843191f49b087c941f7a01e7eeaa22d253e1bfbba6` |

Both joblib objects are fitted `ColumnTransformer` instances with the expected transformers and output names.

## Potential issues

1. **`surgmeno`/`hrt` policy divergence.** Report 12 selects `menopaus=0 → Not applicable`, otherwise `9 → Unknown` (provisional). The implementation maps interface `0` to combined BCSC `9` and encodes it as one category; no conditional split exists. Report 14 documents this provisional choice, but the authoritative policy is not fully implemented.
2. **Weak validation of BCSC-coded transform inputs.** Nominal one-hot encoders use `handle_unknown="ignore"`; an undocumented nominal code such as `agefirst=7` becomes an all-zero block instead of raising. The one-hot family also accepts an undocumented `agegrp=11` as all-zero. The ordinal-tree family rejects invalid ordinal codes but still ignores invalid nominal codes. The raw-input mapper protects the user path; direct source-data transforms are less strict.
3. **Ordinal spacing assumptions.** Ranks for `bmi`, `density`, and `nrelbc` imply equal spacing, although the bands are not equally wide and `nrelbc` is top-coded at 2+. The family is documented as tree-oriented; it is not appropriate for a linear model without review.
4. **No training-data fingerprint.** Artifacts record row/weight counts but not a hash of `train.csv`; stale artifacts would not be detectable from the files alone.
5. **Portability.** Unpickling the ordinal pipeline requires `src.preprocessing.pipeline` to be importable; loading was verified from the repository root only.
6. **No rerun of the test suite in this review.** Running it would fit new in-memory pipelines; the saved artifacts were exercised directly instead. Report 14 records 16/16 tests passing at implementation time.

## Unresolved decisions

- Confirm whether `surgmeno`/`hrt` code 9 with `menopaus=1` is Unknown, then decide whether to implement the conditional Not Applicable recode (requires refitting).
- Select the final model family, and therefore `onehot` or `ordinal_tree`.
- Define behaviour for future true nulls; current validators reject them.
- Decide whether to add strict documented-domain validation and a train-data fingerprint to the artifacts.
- Execute test-set transformation only after preprocessing is frozen; it remains unexecuted.

Last updated: 2026-11-04
