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


# Human-friendly input fields. These are NOT BCSC columns; they are converted to
# the ten interface-coded FEATURES by the helpers below.
HUMAN_FIELDS: tuple[str, ...] = (
    "age",
    "bmi",
    "has_given_birth",
    "age_at_first_birth",
    "is_postmenopausal",
    "menopause_type",
    "currently_on_hrt",
    "first_degree_relatives_with_breast_cancer",
    "breast_density",
    "prior_breast_procedure",
    "last_mammogram_result",
)

_LABEL_LOOKUP: dict[str, dict[str, int]] = {
    feature: {str(label).strip().lower(): code for code, label in LABELS[feature].items()}
    for feature in LABELS
}
# Accepted human wording differs slightly from the codebook label for density.
_LABEL_LOOKUP["density"].setdefault("almost entirely fatty", 1)


def _as_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or isinstance(value, str):
        raise InputMappingError(f"{field}: expected a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InputMappingError(f"{field}: expected a number, got {value!r}") from None
    if not np.isfinite(number):
        raise InputMappingError(f"{field}: expected a finite number, got {value!r}")
    return number


def _require_exact_human_fields(patient: Mapping[str, Any]) -> None:
    if not isinstance(patient, Mapping):
        raise InputMappingError(f"patient must be a mapping, got {type(patient).__name__}")
    missing = [field for field in HUMAN_FIELDS if field not in patient]
    extra = [key for key in patient if key not in HUMAN_FIELDS]
    if missing or extra:
        raise InputMappingError(
            f"patient must contain exactly {list(HUMAN_FIELDS)}; missing={missing}, extra={extra}"
        )


def human_to_interface_record(patient: Mapping[str, Any]) -> dict[str, int]:
    """Convert one human-readable patient record to interface codes.

    The returned dictionary has exactly the ten ``FEATURES`` keys and contains
    interface codes (``0..n``), not BCSC codes.  ``RawInputMapper`` performs the
    interface -> BCSC conversion afterwards.
    """
    _require_exact_human_fields(patient)

    # age -> agegrp (35-39=1 ... 80-84=10). agegrp has no Unknown code.
    age = _as_number(patient["age"], "age")
    if not float(age).is_integer():
        raise InputMappingError(f"age: must be an integer number of years, got {age!r}")
    if age < 35 or age > 84:
        raise InputMappingError(f"age: must be between 35 and 84, got {age!r}")
    agegrp = int((age - 35) // 5) + 1

    # bmi -> bmi interface code (None is the only Unknown form accepted here).
    bmi_raw = patient["bmi"]
    if bmi_raw is None:
        bmi_code = UNKNOWN_INPUT
    else:
        bmi = _as_number(bmi_raw, "bmi")
        if bmi < 10:
            raise InputMappingError(f"bmi: must be >= 10, got {bmi!r}")
        if bmi < 25:
            bmi_code = 1
        elif bmi < 30:
            bmi_code = 2
        elif bmi < 35:
            bmi_code = 3
        else:
            bmi_code = 4

    # has_given_birth + age_at_first_birth -> agefirst.
    has_given_birth = patient["has_given_birth"]
    if has_given_birth is not None and not isinstance(has_given_birth, bool):
        raise InputMappingError(
            f"has_given_birth: expected True/False/None, got {has_given_birth!r}"
        )
    age_at_first_birth = patient["age_at_first_birth"]
    if has_given_birth is False:
        if age_at_first_birth is not None:
            raise InputMappingError(
                "age_at_first_birth must be None when has_given_birth is False"
            )
        agefirst_code = 3  # nulliparous
    elif has_given_birth is True:
        if age_at_first_birth is None:
            agefirst_code = UNKNOWN_INPUT  # unknown age at first birth
        else:
            first_birth_age = _as_number(age_at_first_birth, "age_at_first_birth")
            if first_birth_age <= 0:
                raise InputMappingError(
                    f"age_at_first_birth: must be > 0, got {first_birth_age!r}"
                )
            if first_birth_age > age:
                raise InputMappingError(
                    f"age_at_first_birth: cannot exceed current age, got {first_birth_age!r}"
                )
            agefirst_code = 1 if first_birth_age < 30 else 2
    else:  # None: unknown parity; do not infer from a supplied birth age.
        if age_at_first_birth is not None:
            raise InputMappingError(
                "age_at_first_birth requires has_given_birth to be True"
            )
        agefirst_code = UNKNOWN_INPUT

    # is_postmenopausal -> menopaus.
    is_postmenopausal = patient["is_postmenopausal"]
    if is_postmenopausal is None:
        menopaus_code = UNKNOWN_INPUT
    elif isinstance(is_postmenopausal, bool):
        menopaus_code = 2 if is_postmenopausal else 1
    else:
        raise InputMappingError(
            f"is_postmenopausal: expected True/False/None, got {is_postmenopausal!r}"
        )

    # menopause_type -> surgmeno (conditional on postmenopausal).
    menopause_type = patient["menopause_type"]
    if menopause_type is None:
        surgmeno_code = UNKNOWN_INPUT
    else:
        if not isinstance(menopause_type, str):
            raise InputMappingError(
                f"menopause_type: expected a string, got {menopause_type!r}"
            )
        if is_postmenopausal is not True:
            raise InputMappingError(
                "menopause_type is only allowed when is_postmenopausal is True"
            )
        code = _LABEL_LOOKUP["surgmeno"].get(menopause_type.strip().lower())
        if code is None or code == UNKNOWN_INPUT:
            raise InputMappingError(
                f"menopause_type: unsupported value {menopause_type!r}"
            )
        surgmeno_code = code

    # currently_on_hrt -> hrt (conditional on postmenopausal).
    currently_on_hrt = patient["currently_on_hrt"]
    if currently_on_hrt is None:
        hrt_code = UNKNOWN_INPUT
    else:
        if not isinstance(currently_on_hrt, bool):
            raise InputMappingError(
                f"currently_on_hrt: expected True/False, got {currently_on_hrt!r}"
            )
        if is_postmenopausal is not True:
            raise InputMappingError(
                "currently_on_hrt is only allowed when is_postmenopausal is True"
            )
        hrt_code = 2 if currently_on_hrt else 1

    # first_degree_relatives_with_breast_cancer -> nrelbc.
    relatives = patient["first_degree_relatives_with_breast_cancer"]
    if relatives is None:
        nrelbc_code = UNKNOWN_INPUT
    elif isinstance(relatives, str):
        value = relatives.strip()
        if value == "0":
            nrelbc_code = 1
        elif value == "1":
            nrelbc_code = 2
        elif value == "2+":
            nrelbc_code = 3
        else:
            raise InputMappingError(
                f"first_degree_relatives_with_breast_cancer: unsupported value {relatives!r}"
            )
    else:
        if isinstance(relatives, bool) or not isinstance(relatives, (int, float, np.integer, np.floating)):
            raise InputMappingError(
                f"first_degree_relatives_with_breast_cancer: expected 0/1/'2+' or None, got {relatives!r}"
            )
        number = float(relatives)
        if not np.isfinite(number):
            raise InputMappingError(
                f"first_degree_relatives_with_breast_cancer: expected a finite value, got {relatives!r}"
            )
        if number == 0:
            nrelbc_code = 1
        elif number == 1:
            nrelbc_code = 2
        elif number >= 2:
            nrelbc_code = 3
        else:
            raise InputMappingError(
                f"first_degree_relatives_with_breast_cancer: unsupported value {relatives!r}"
            )

    # breast_density -> density.
    breast_density = patient["breast_density"]
    if breast_density is None:
        density_code = UNKNOWN_INPUT
    else:
        if not isinstance(breast_density, str):
            raise InputMappingError(
                f"breast_density: expected a string or None, got {breast_density!r}"
            )
        code = _LABEL_LOOKUP["density"].get(breast_density.strip().lower())
        if code is None or code == UNKNOWN_INPUT:
            raise InputMappingError(
                f"breast_density: unsupported value {breast_density!r}"
            )
        density_code = code

    # prior_breast_procedure -> brstproc.
    prior = patient["prior_breast_procedure"]
    if prior is None:
        brstproc_code = UNKNOWN_INPUT
    elif isinstance(prior, bool):
        brstproc_code = 2 if prior else 1
    else:
        raise InputMappingError(
            f"prior_breast_procedure: expected True/False/None, got {prior!r}"
        )

    # last_mammogram_result -> lastmamm.
    last_result = patient["last_mammogram_result"]
    if last_result is None:
        lastmamm_code = UNKNOWN_INPUT
    else:
        if not isinstance(last_result, str):
            raise InputMappingError(
                f"last_mammogram_result: expected a string or None, got {last_result!r}"
            )
        code = _LABEL_LOOKUP["lastmamm"].get(last_result.strip().lower())
        if code is None or code == UNKNOWN_INPUT:
            raise InputMappingError(
                f"last_mammogram_result: unsupported value {last_result!r}"
            )
        lastmamm_code = code

    record = {
        "agegrp": agegrp,
        "bmi": bmi_code,
        "agefirst": agefirst_code,
        "menopaus": menopaus_code,
        "surgmeno": surgmeno_code,
        "hrt": hrt_code,
        "nrelbc": nrelbc_code,
        "density": density_code,
        "brstproc": brstproc_code,
        "lastmamm": lastmamm_code,
    }

    # Reuse CHOICES as the final authority on valid interface codes.
    for feature, code in record.items():
        if code not in CHOICES[feature]:
            raise InputMappingError(
                f"{feature}: generated interface code {code} is not supported"
            )
    return record


def human_to_interface_frame(patients: Any) -> pd.DataFrame:
    """Convert one patient or a list of patients to an interface-coded frame."""
    if isinstance(patients, Mapping):
        records = [patients]
    elif isinstance(patients, (list, tuple)):
        records = list(patients)
    else:
        raise InputMappingError(
            f"patients must be a dict or a list of dicts, got {type(patients).__name__}"
        )
    if not records:
        raise InputMappingError("patients must not be empty")
    rows = [human_to_interface_record(patient) for patient in records]
    return pd.DataFrame(rows, columns=list(FEATURES))


def encode_human_inputs(patients: Any, mapper: RawInputMapper, pipeline: Any) -> np.ndarray:
    """Optional convenience: human input -> interface -> BCSC -> encoded matrix."""
    from src.preprocessing.pipeline import encode_user_inputs

    interface = human_to_interface_frame(patients)
    return encode_user_inputs(mapper, pipeline, interface)
