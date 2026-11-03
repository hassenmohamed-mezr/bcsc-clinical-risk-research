"""Model preprocessing (stage 2) for the BCSC risk features.

Takes data already expressed in original BCSC codes (source splits or the
output of :mod:`src.preprocessing.raw_input_mapping`) and produces the encoded
model matrix. Two encoder families are provided because the final model family
is not yet chosen:

* ``onehot`` - one-hot for every feature. Safe default for linear models:
  no false numeric ordering; Unknown (code 9) becomes its own category.
* ``ordinal_tree`` - ordinal ranks for the ordered features (agegrp, bmi,
  density, nrelbc), with an explicit Unknown flag for bmi/density/nrelbc, and
  one-hot for nominal features. Compact and monotonic, intended for trees.

Fitting contract: the one-hot categories (and any future learned parameters)
are fitted on the Training split only. Ordinal ranks come from the documented
codebook, so they are fixed domain mappings rather than data-derived
statistics. Validation/Test are only ever transformed.

Usage
-----
    from src.data.load_splits import load_split
    from src.preprocessing.pipeline import fit_pipeline, transform_features

    X_train, y_train, w_train = load_split("train")
    pipe = fit_pipeline(X_train, family="onehot")          # fit on train only
    X_val, _, _ = load_split("val")
    Z_val = transform_features(pipe, X_val)                 # transform only
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.data.load_splits import FEATURES, TARGET, WEIGHT
from src.preprocessing.raw_input_mapping import RawInputMapper

UNKNOWN_CODE = 9
FAMILIES = ("onehot", "ordinal_tree")

# Documented category order (BCSC codebook; reports 09/12). Excludes code 9.
ORDINAL_CODES: dict[str, list[int]] = {
    "agegrp": list(range(1, 11)),   # 1 = 35-39 ... 10 = 80-84
    "bmi": [1, 2, 3, 4],
    "density": [1, 2, 3, 4],
    "nrelbc": [0, 1, 2],            # 2 = "2 or more"
}
ORDINAL_FEATURES = [f for f in FEATURES if f in ORDINAL_CODES]
NOMINAL_FEATURES = [f for f in FEATURES if f not in ORDINAL_CODES]
FORBIDDEN_INPUTS = {TARGET, WEIGHT, "training"}


class OrdinalWithUnknown(BaseEstimator, TransformerMixin):
    """Map documented ordered codes to ranks and optionally flag Unknown.

    Ranks are 1..k in codebook order. When ``unknown_code`` is set, Unknown is
    rank 0 with the companion flag set to 1; when it is ``None`` (agegrp, whose
    code 9 is a valid age band) only the rank column is emitted. Codes outside
    the documented set are rejected: they are data errors, not silently coerced
    to Unknown.
    """

    def __init__(
        self,
        feature: str,
        valid_codes: Iterable[int],
        unknown_code: int | None = UNKNOWN_CODE,
    ):
        # sklearn clone() requires __init__ to store parameters unmodified.
        self.feature = feature
        self.valid_codes = valid_codes
        self.unknown_code = unknown_code

    def _valid_codes(self) -> list[int]:
        return list(self.valid_codes)

    def _as_1d(self, X) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim == 2:
            if arr.shape[1] != 1:
                raise ValueError(f"{self.feature}: expected a single column, got shape {arr.shape}")
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(f"{self.feature}: expected 1-D input, got shape {arr.shape}")
        return arr

    def _validate(self, arr: np.ndarray) -> np.ndarray:
        codes = arr.astype("int64")
        if not np.all(codes == arr):
            raise ValueError(f"{self.feature}: non-integer codes found")
        allowed = set(self._valid_codes())
        if self.unknown_code is not None:
            allowed.add(self.unknown_code)
        bad = sorted(set(codes.tolist()) - allowed)
        if bad:
            raise ValueError(f"{self.feature}: undocumented code(s) {bad}; allowed {sorted(allowed)}")
        return codes

    def fit(self, X, y=None):
        self._validate(self._as_1d(X))
        self.n_features_in_ = 1
        return self

    def transform(self, X) -> np.ndarray:
        codes = self._validate(self._as_1d(X))
        rank = np.zeros(len(codes), dtype=np.float32)
        lookup = {code: i + 1 for i, code in enumerate(self._valid_codes())}
        if self.unknown_code is None:
            rank[:] = [lookup[c] for c in codes]
            return rank.reshape(-1, 1)
        unknown = (codes == self.unknown_code).astype(np.float32)
        valid = codes != self.unknown_code
        rank[valid] = [lookup[c] for c in codes[valid]]
        return np.column_stack([rank, unknown])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = [f"{self.feature}_ordinal"]
        if self.unknown_code is not None:
            names.append(f"{self.feature}_unknown")
        return np.array(names, dtype=object)


def _check_design(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("expected a pandas DataFrame of BCSC codes")
    missing = [f for f in FEATURES if f not in X.columns]
    extra = [c for c in X.columns if c not in FEATURES]
    if missing or extra:
        raise ValueError(f"input columns must be exactly FEATURES; missing={missing}, extra={extra}")
    if list(X.columns) != FEATURES:
        raise ValueError("input columns must follow the fixed FEATURES order")
    forbidden = FORBIDDEN_INPUTS & set(X.columns)
    if forbidden:
        raise ValueError(f"forbidden columns in model input: {sorted(forbidden)}")


def build_pipeline(family: str = "onehot") -> ColumnTransformer:
    if family not in FAMILIES:
        raise ValueError(f"unknown encoder family {family!r}; allowed {FAMILIES}")

    if family == "onehot":
        return ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
              list(FEATURES))],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    ordinal_steps = [
        (
            f"ord_{name}",
            OrdinalWithUnknown(
                name,
                ORDINAL_CODES[name],
                unknown_code=None if name == "agegrp" else UNKNOWN_CODE,
            ),
            [name],
        )
        for name in ORDINAL_FEATURES
    ]
    nominal_step = (
        "nominal",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
        list(NOMINAL_FEATURES),
    )
    return ColumnTransformer(
        ordinal_steps + [nominal_step],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_pipeline(X_train: pd.DataFrame, family: str = "onehot") -> ColumnTransformer:
    """Fit the encoder strictly on Training data (original BCSC codes)."""
    _check_design(X_train)
    pipeline = build_pipeline(family)
    pipeline.fit(X_train)
    return pipeline


def transform_features(pipeline: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Transform codes with an already-fitted pipeline; never fits here."""
    _check_design(X)
    return pipeline.transform(X)


def feature_names(pipeline: ColumnTransformer) -> list[str]:
    return [str(name) for name in pipeline.get_feature_names_out()]


def fit_from_user_inputs(mapper: RawInputMapper, X_interface: pd.DataFrame, family: str = "onehot") -> ColumnTransformer:
    """Convenience for tests/dev: map interface values, then fit on the codes."""
    codes = mapper.transform_frame(X_interface)
    return fit_pipeline(codes, family=family)


def encode_user_inputs(mapper: RawInputMapper, pipeline: ColumnTransformer, X_interface: pd.DataFrame) -> np.ndarray:
    """Single inference path: interface values -> BCSC codes -> encoded matrix."""
    return transform_features(pipeline, mapper.transform_frame(X_interface))


def save_pipeline(pipeline: ColumnTransformer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str | Path) -> ColumnTransformer:
    pipeline = joblib.load(path)
    if not hasattr(pipeline, "transform"):
        raise TypeError(f"object at {path} is not a fitted preprocessing pipeline")
    return pipeline
