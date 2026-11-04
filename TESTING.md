# Testing

## Test Files

- `tests/test_preprocessing.py`
- `tests/test_training_config.py`
- `tests/test_training_cv.py`
- `tests/test_training_models.py`
- `tests/test_barlow_modified.py`
- `test_SUPMT_inference.py`

## Framework

- Standard library `unittest`
- `pytest` is not installed

## Commands

```bash
python -m unittest discover -s tests -v
python test_SUPMT_inference.py
```

## Test Categories

### Preprocessing (`tests/test_preprocessing.py`)

- interface-to-BCSC mapping
- invalid input rejection
- exact feature fields
- frame mapping
- JSON round-trip
- feature count/order
- train-only fitting
- Unknown/unseen code handling
- save/reload consistency
- human input mapping and conversion
- inference parity

### Training Configuration (`tests/test_training_config.py`)

- feature exclusion
- model keys
- encoder widths
- seed/n_jobs
- MLP scaler scope
- calibration policy
- Test split rejection

### Weighted CV (`tests/test_training_cv.py`)

- non-expanding folds
- diagnostics
- weighted scoring
- scaler fitting

### Training Models (`tests/test_training_models.py`)

- candidate grids
- parameter namespaces
- PR-AUC selection
- calibration tables

### Modified Barlow (`tests/test_barlow_modified.py`)

- feature lists
- routing
- forbidden features
- reference-first encoding
- unpenalized LR
- row-order preservation

### Inference Runtime (`test_SUPMT_inference.py`)

- 60 cases from `BCSC_SUPMT_60_test_cases.json`
- 40 accepted, 20 rejected
- finite probabilities
- weighted average recomputation
- reference range deviation reporting

## Known Test Limitations

- Tests use samples for speed in some cases; artifacts are fitted on full train.
- Test split is not used in preprocessing/training tests.
- Inference reference ranges are approximate and not ground truth.
- No clinical accuracy validation.