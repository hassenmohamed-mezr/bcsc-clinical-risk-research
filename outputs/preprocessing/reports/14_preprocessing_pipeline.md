# Preprocessing Pipeline

Status: implemented, fitted on train only; no model training. Test split never opened.

## 1. Two-stage architecture and data flow

**Stage A — raw input mapping** (`src/preprocessing/raw_input_mapping.py`): converts numeric interface choices into original BCSC codes. Interface `0` = explicit Unknown (or Unknown/NA where BCSC uses one combined code), mapped to BCSC `9`. Interface `1..n` follows the documented BCSC category order:

| Feature | Interface mapping (BCSC code) |
|---|---|
| agegrp | 1–10 → 1–10 (no Unknown; 9 = 75–79, 10 = 80–84) |
| bmi, density | 1–4 → 1–4, 0 → 9 |
| agefirst | 1→0 (<30), 2→1 (≥30), 3→2 (nulliparous), 0→9 |
| nrelbc | 1→0, 2→1, 3→2 (2+), 0→9 |
| menopaus, surgmeno, hrt, brstproc, lastmamm | 1→0, 2→1, 0→9 |

Invalid, missing, extra, non-integer, boolean, or null choices raise `InputMappingError`; nothing is silently converted to Unknown. The mapper is serializable (JSON) and reusable.

**Stage B — model encoding** (`src/preprocessing/pipeline.py`): a fitted `ColumnTransformer` takes BCSC-code frames in the fixed `FEATURES` order and emits the model matrix.

Inference path (single, shared with training): `user row → RawInputMapper.transform_frame → BCSC codes → fitted pipeline.transform → encoded matrix`. `encode_user_inputs` is the public convenience for this path. There is no separate manual inference preprocessing.

## 2. Artifacts

Fitted by `python -m src.preprocessing.fit_preprocessing` (train only):

| Artifact | Contents |
|---|---|
| `models/preprocessing/raw_input_mapping.json` | interface→BCSC mapping, version 1 |
| `models/preprocessing/onehot_pipeline.joblib` | 43-column encoder |
| `models/preprocessing/ordinal_tree_pipeline.joblib` | 26-column encoder |
| `models/preprocessing/preprocessing_metadata.json` | fit split, 152,149 rows / 1,525,868 weight, features, dimensions |

## 3. Encoders and rationale

| Family | Design | Output | Intended use |
|---|---|---|---|
| `onehot` | one-hot for all 10 features, Unknown is its own category | 43 | Safe default, linear models |
| `ordinal_tree` | documented rank for agegrp/bmi/nrelbc/density; Unknown rank 0 + flag for bmi/nrelbc/density; one-hot for nominal features | 26 | Tree models; compact, monotonic |

Ordinal ranks come from the fixed BCSC codebook, not learned statistics. One-hot categories are learned from train only. `agegrp` gets no Unknown flag because its code 9 is a valid age band. `surgmeno`/`hrt` keep the dataset's combined `9` = Unknown/NA as one category until the pending distinction is confirmed. Family choice is provisional: the final model family is not chosen, so both artifacts exist; one is selected later, never both.

## 4. Train-only fitting enforcement

- `fit_preprocessing` loads only `train`; `fit_pipeline` receives one frame and fits the encoders on it.
- `transform_features` only calls `transform`; it never fits. Validation is used for transform checks; test is untouched.
- Feature order and membership are enforced: reordered, missing, extra, `count`, `cancer`, or `training` columns are rejected.
- No imputation, row expansion, or row/feature dropping. `count` remains a frequency weight for later model fitting; category derivation does not use it.

A test removes `lastmamm=9` from the fitting frame: the encoder has no such category, validation rows with 9 transform to an all-zero `lastmamm` block, and the fitted category set is unchanged.

## 5. Unknown and special-code handling

- Nominal Unknown `9` is an explicit category when present in train; unseen nominal codes are ignored (`handle_unknown="ignore"`) and produce an all-zero block.
- Ordinal Unknown `9` → rank 0 with flag 1. Undocumented ordinal codes (e.g., agegrp 11) raise an error rather than being coerced to Unknown.
- `agegrp` 9/10 are valid ordered categories; no missing/NA code exists.
- No actual nulls exist in the splits; a future-null contract remains undefined.

## 6. Tests and results

Command: `python -m unittest discover -s tests -v` (pytest is not installed). **16 tests, all passing**:

- mapping choices, full-record mapping, Unknown→9, invalid input rejection, exact-field checks, frame mapping, JSON round-trip;
- feature count/order and forbidden columns, fixed dimensions (43/26), stable dimensions across splits, ordinal ranks/Unknown flags;
- train-only categories and no refit during validation transform, Unknown/unseen handling;
- joblib save/reload equality, mapper→pipeline inference parity.

## 7. Unresolved decisions and limitations

1. `surgmeno`/`hrt` Unknown-vs-Not-Applicable split still needs codebook confirmation; combined `9` is used provisionally.
2. Final model family is undecided; encoder family selection follows from it.
3. No contract for future true nulls; current validators reject them.
4. Encoded artifacts must be regenerated if the training split or feature list changes.
5. Tests use samples for speed (20k train / 10k validation); artifacts are fitted on the full train split. Test-set transform remains unexecuted by design.

Last updated: 2026-11-03 | git HEAD: 40e7c4f
