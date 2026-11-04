# Development Guide

## Environment Setup

- Python 3.15.0 (observed)
- scikit-learn 1.9.1
- NumPy 2.5.4
- pandas 3.1.0
- joblib 1.6.0

`requirements.txt` is intentionally empty; versions are not pinned.

## Running the Project

```bash
python -m src.data.load_splits
python -m src.preprocessing.fit_preprocessing
python -m src.training.config --with-splits
python -m src.training.cv --with-train
python SUPMT.py
python test_SUPMT_inference.py
```

## Running Tests

```bash
python -m unittest discover -s tests -v
python test_SUPMT_inference.py
```

## Understanding the Codebase

- Start with `README.md` and `PROJECT_MAP.md`.
- Data loading: `src/data/load_splits.py`
- Preprocessing: `src/preprocessing/`
- Training: `src/training/`
- Inference: `SUPMT.py`, `SUPMT_CONFIG.py`

## Adding a New Model

1. Add model spec in `src/training/config.py`.
2. Add encoder mapping.
3. Train/tune/calibrate using `src/training/train_models.py`.
4. Save artifact under `models/trained/`.
5. Update `SUPMT.py` mapping if inference should include it.
6. Update documentation.

## Adding or Changing Features

- Update `FEATURES` in `src/data/load_splits.py` (requires retraining).
- Update preprocessing mapping and encoders.
- Regenerate artifacts.
- Update docs and tests.

## Updating Documentation

- Keep `PROJECT_MAP.md` synchronized with repository tree.
- Add evidence paths for technical claims.
- Do not document planned work as implemented.

## Reproducibility

- Use random state 42.
- Keep `count` as frequency weight.
- Do not fit on validation/test.
- Use saved artifacts for inference.