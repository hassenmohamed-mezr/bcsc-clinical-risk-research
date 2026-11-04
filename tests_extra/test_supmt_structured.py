"""Tests for SUPMT_structured (read-only)."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import SUPMT
import SUPMT_structured

CASES_PATH = ROOT / "BCSC_SUPMT_60_test_cases.json"


def load_accepted():
    with CASES_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return [c for c in payload["cases"] if c["expected"]["status"] == "accept"]


def test_structured_equals_predict_patient():
    import unittest
    if not CASES_PATH.is_file():
        raise unittest.SkipTest("BCSC_SUPMT_60_test_cases.json not found in repository")
    for case in load_accepted():
        with contextlib.redirect_stdout(io.StringIO()):
            base = SUPMT.predict_patient(case["input"])
        structured = SUPMT_structured.predict_patient_structured(case["input"])
        for key in ("logistic_regression", "gradient_boosting", "mlp", "random_forest"):
            assert abs(structured["probabilities"][key] - base[key]) <= 1e-12
        assert abs(structured["weighted_average"] - base["weighted_average"]) <= 1e-12


if __name__ == "__main__":
    test_structured_equals_predict_patient()
    print("structured parity test passed")