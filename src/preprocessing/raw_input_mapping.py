"""Raw user-input mapping for the BCSC preprocessing stage.

Stage 1 of the two-stage preprocessing design: converts numeric *interface*
choices (what a future UI/inference client submits) into the original BCSC
category codes. Interface codes are NOT BCSC codes; they are a project-local
re-indexing defined explicitly below from the official BCSC codebook
(reports 09/12), so no undocumented meaning is invented.

Convention
----------
* Interface code ``0`` = explicit "Unknown" (or "Unknown / not applicable"
  where the BCSC dataset itself uses one code for both, i.e. surgmeno/hrt),
  mapped to BCSC code ``9``.
* Interface codes ``1..n`` follow the documented BCSC category order and map
  to the corresponding valid BCSC code.
* ``agegrp`` has no unknown code; interface ``0`` is rejected.

Inputs are validated strictly: unsupported values raise ``InputMappingError``
and are never silently turned into Unknown.

Usage
-----
    mapper = RawInputMapper()
    codes = mapper.transform({"agegrp": 4, "bmi": 0, ...})   # -> dict of BCSC codes
    frame = mapper.transform_frame(user_df)                 # -> DataFrame, BCSC codes
    mapper.save("models/preprocessing/raw_input_mapping.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

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

UNKNOWN_INPUT = 0
UNKNOWN_BCSC_CODE = 9

# interface code -> BCSC code (documented order; see BCSC codebook)
CHOICES: dict[str, dict[int, int]] = {
    "agegrp": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
    "bmi": {1: 1, 2: 2, 3: 3, 4: 4, UNKNOWN_INPUT: 9},
    "agefirst": {1: 0, 2: 1, 3: 2, UNKNOWN_INPUT: 9},
    "menopaus": {1: 0, 2: 1, UNKNOWN_INPUT: 9},
    "surgmeno": {1: 0, 2: 1, UNKNOWN_INPUT: 9},
    "hrt": {1: 0, 2: 1, UNKNOWN_INPUT: 9},
    "nrelbc": {1: 0, 2: 1, 3: 2, UNKNOWN_INPUT: 9},
    "density": {1: 1, 2: 2, 3: 3, 4: 4, UNKNOWN_INPUT: 9},
    "brstproc": {1: 0, 2: 1, UNKNOWN_INPUT: 9},
    "lastmamm": {1: 0, 2: 1, UNKNOWN_INPUT: 9},
}

# Human-readable interface labels (documentation only; not used by encoders).
LABELS: dict[str, dict[int, str]] = {
    "agegrp": {i: label for i, label in enumerate(
        ["35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84"], 1)},
    "bmi": {1: "10-24.99", 2: "25-29.99", 3: "30-34.99", 4: "35+", 0: "Unknown"},
    "agefirst": {1: "age <30", 2: "age >=30", 3: "nulliparous", 0: "Unknown"},
    "menopaus": {1: "premenopausal", 2: "postmenopausal or age>=55", 0: "Unknown"},
    "surgmeno": {1: "natural", 2: "surgical", 0: "Unknown / not applicable"},
    "hrt": {1: "no", 2: "yes", 0: "Unknown / not applicable"},
    "nrelbc": {1: "zero", 2: "one", 3: "two or more", 0: "Unknown"},
    "density": {1: "almost entirely fat", 2: "scattered fibroglandular", 3: "heterogeneously dense",
                4: "extremely dense", 0: "Unknown / different system"},
    "brstproc": {1: "no", 2: "yes", 0: "Unknown"},
    "lastmamm": {1: "negative", 2: "false positive", 0: "Unknown"},
}


class InputMappingError(ValueError):
    """Raised when a raw user input cannot be mapped to a BCSC code."""


def _coerce_code(feature: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise InputMappingError(f"{feature}: expected an integer interface code, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InputMappingError(f"{feature}: expected an integer interface code, got {value!r}") from None
    if not np.isfinite(number) or not number.is_integer():
        raise InputMappingError(f"{feature}: expected an integer interface code, got {value!r}")
    return int(number)


class RawInputMapper:
    """Reusable, serializable interface-code -> BCSC-code mapper."""

    features: tuple[str, ...] = tuple(FEATURES)
    choices: dict[str, dict[int, int]] = {k: dict(v) for k, v in CHOICES.items()}

    def map_value(self, feature: str, value: Any) -> int:
        if feature not in self.choices:
            raise InputMappingError(f"unknown feature: {feature!r}")
        code = _coerce_code(feature, value)
        mapping = self.choices[feature]
        if code not in mapping:
            allowed = sorted(mapping)
            raise InputMappingError(
                f"{feature}: unsupported interface code {code}; allowed: {allowed}"
            )
        return mapping[code]

    def transform(self, record: Mapping[str, Any]) -> dict[str, int]:
        missing = [f for f in self.features if f not in record]
        extra = [k for k in record if k not in self.features]
        if missing or extra:
            raise InputMappingError(f"record must contain exactly {list(self.features)}; missing={missing}, extra={extra}")
        return {f: self.map_value(f, record[f]) for f in self.features}

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(df)
        out: dict[str, pd.Series] = {}
        for feature in self.features:
            mapping = self.choices[feature]
            series = df[feature]
            mapped = []
            for value in series:
                code = _coerce_code(feature, value)
                if code not in mapping:
                    raise InputMappingError(
                        f"{feature}: unsupported interface code {code}; allowed: {sorted(mapping)}"
                    )
                mapped.append(mapping[code])
            out[feature] = pd.Series(mapped, index=df.index, dtype="int64")
        return pd.DataFrame(out, columns=list(self.features))

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [f for f in self.features if f not in df.columns]
        extra = [c for c in df.columns if c not in self.features]
        if missing or extra:
            raise InputMappingError(f"input must contain exactly {list(self.features)}; missing={missing}, extra={extra}")

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "features": list(self.features),
            "unknown_input": UNKNOWN_INPUT,
            "choices": {f: {str(k): v for k, v in sorted(m.items())} for f, m in self.choices.items()},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RawInputMapper":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if list(data["features"]) != list(FEATURES):
            raise InputMappingError("saved mapping feature order does not match FEATURES")
        mapper = cls()
        mapper.choices = {
            f: {int(k): int(v) for k, v in data["choices"][f].items()} for f in FEATURES
        }
        if mapper.choices != CHOICES:
            raise InputMappingError("saved mapping does not match the documented codebook mapping")
        return mapper
