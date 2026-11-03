"""Reusable loader for the BCSC train / validation / test splits.

Single source of truth for the feature list, the binary target, and the
frequency weight. Returns data only: no encoding, scaling, imputation, or other
preprocessing is applied here.

Usage:
    from src.data.load_splits import FEATURES, load_split

    X_train, y_train, w_train = load_split("train")
    X_val, y_val, w_val = load_split("val")

`count` is a frequency weight, never a feature. Pass it to model fitting and to
metrics that support sample weights:

    model.fit(X_train, y_train, sample_weight=w_train)
    roc_auc_score(y_val, scores, sample_weight=w_val)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---- single source of truth ---------------------------------------------
TARGET = "cancer"
WEIGHT = "count"

FEATURES = [
    "agegrp",
    "bmi",
    "agefirst",
    "menopaus",
    "surgmeno",
    "hrt",
    "nrelbc",
    "density",
    "brstproc",
    "lastmamm",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATHS = {
    "train": PROJECT_ROOT / "data" / "04_splits" / "train.csv",
    "val": PROJECT_ROOT / "data" / "04_splits" / "val.csv",
    "test": PROJECT_ROOT / "data" / "04_splits" / "test.csv",
}


def _validate_split(df: pd.DataFrame, name: str) -> None:
    required = FEATURES + [TARGET, WEIGHT]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"split '{name}' is missing required columns: {missing}")

    if df.columns.duplicated().any():
        duplicated = sorted(set(df.columns[df.columns.duplicated()]))
        raise ValueError(f"split '{name}' has duplicated column names: {duplicated}")

    if df[TARGET].isna().any():
        raise ValueError(f"split '{name}': missing values in '{TARGET}'")
    invalid_targets = set(df[TARGET].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(f"split '{name}': '{TARGET}' must be binary 0/1, found {sorted(invalid_targets)}")

    if df[WEIGHT].isna().any():
        raise ValueError(f"split '{name}': missing values in '{WEIGHT}'")
    if not pd.api.types.is_numeric_dtype(df[WEIGHT]):
        raise ValueError(f"split '{name}': '{WEIGHT}' must be numeric")
    if not (df[WEIGHT] > 0).all():
        raise ValueError(f"split '{name}': '{WEIGHT}' must be strictly positive")

    if df[FEATURES].isna().any().any():
        bad = df[FEATURES].isna().any()
        raise ValueError(f"split '{name}': missing feature values in {list(bad[bad].index)}")


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load one split and return ``(X, y, w)``.

    Args:
        name: one of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        X: feature frame with exactly ``FEATURES`` (in that order).
        y: binary target ``cancer``.
        w: frequency weight ``count`` (pass as ``sample_weight``).
    """
    if name not in SPLIT_PATHS:
        raise ValueError(f"unknown split '{name}'; allowed values: {sorted(SPLIT_PATHS)}")

    path = SPLIT_PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(f"split file not found for '{name}': {path}")

    df = pd.read_csv(path)
    _validate_split(df, name)

    X = df[FEATURES]
    if list(X.columns) != FEATURES:
        raise ValueError(f"split '{name}': feature order does not match FEATURES")

    y = df[TARGET]
    w = df[WEIGHT]

    if not (len(X) == len(y) == len(w)):
        raise ValueError(f"split '{name}': X/y/w lengths differ")
    if len(X) == 0:
        raise ValueError(f"split '{name}' is empty")

    return X, y, w


def _self_check(names: tuple[str, ...] = ("train", "val")) -> None:
    order = None
    for name in names:
        X, y, w = load_split(name)
        assert list(X.columns) == FEATURES, "unexpected features or order"
        assert not {"count", "cancer", "training"} & set(X.columns), "leakage column in X"
        assert len(X) == len(y) == len(w), "length mismatch"
        assert (w > 0).all(), "non-positive weights"
        assert set(y.unique()) <= {0, 1}, "non-binary target"
        if order is None:
            order = list(X.columns)
        assert order == list(X.columns), "feature order differs between splits"

        total = float(w.sum())
        rate = float(w[y == 1].sum()) / total
        print(f"{name:<5} rows={len(X):>7,} weight_sum={int(total):>10,} cancer_rate={rate:.6%}")


if __name__ == "__main__":
    _self_check()
