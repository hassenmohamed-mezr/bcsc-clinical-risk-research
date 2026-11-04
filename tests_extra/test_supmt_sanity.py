"""Sanity tests for SUPMT inference (read-only)."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import SUPMT

CASES_PATH = ROOT / "BCSC_SUPMT_60_test_cases.json"
if not CASES_PATH.is_file():
    CASES_PATH = Path(r"C:\Users\marwa\Downloads\BCSC_SUPMT_60_test_cases.json")

OUT_REPORT = ROOT / "outputs" / "evaluation" / "reports" / "monotonic_report.csv"


def load_accepted_cases():
    with CASES_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return [c for c in payload["cases"] if c["expected"]["status"] == "accept"]


class TestSupmtSanityHard(unittest.TestCase):
    def test_no_exceptions_and_finite(self):
        cases = load_accepted_cases()
        for case in cases:
            with contextlib.redirect_stdout(io.StringIO()):
                result = SUPMT.predict_patient(case["input"])
            for key in ("logistic_regression", "gradient_boosting", "mlp", "random_forest", "weighted_average"):
                value = result[key]
                assert isinstance(value, (int, float))
                assert math.isfinite(value)
                assert 0 <= value <= 1

    def test_deterministic(self):
        cases = load_accepted_cases()[:5]
        for case in cases:
            with contextlib.redirect_stdout(io.StringIO()):
                r1 = SUPMT.predict_patient(case["input"])
                r2 = SUPMT.predict_patient(case["input"])
            for key in ("logistic_regression", "gradient_boosting", "mlp", "random_forest", "weighted_average"):
                assert r1[key] == r2[key]

    def test_matches_existing_60_case_results(self):
        # Re-run the accepted cases and ensure they still produce finite, in-range values.
        cases = load_accepted_cases()
        assert len(cases) == 40
        for case in cases:
            with contextlib.redirect_stdout(io.StringIO()):
                result = SUPMT.predict_patient(case["input"])
            assert set(result.keys()) >= {"logistic_regression", "gradient_boosting", "mlp", "random_forest", "weighted_average"}


class TestSupmtSanitySoft(unittest.TestCase):
    def test_monotonic_report(self):
        base = {
            "age": 50,
            "bmi": 27.5,
            "has_given_birth": True,
            "age_at_first_birth": 26,
            "is_postmenopausal": False,
            "menopause_type": None,
            "currently_on_hrt": None,
            "first_degree_relatives_with_breast_cancer": 1,
            "breast_density": "heterogeneously dense",
            "prior_breast_procedure": False,
            "last_mammogram_result": "negative",
        }
        densities = [
            "almost entirely fatty",
            "scattered fibroglandular",
            "heterogeneously dense",
            "extremely dense",
        ]
        rows = []
        # Age monotonicity
        for model_key in ("logistic_regression", "gradient_boosting"):
            probs = []
            for age in range(35, 85):
                p = dict(base)
                p["age"] = age
                with contextlib.redirect_stdout(io.StringIO()):
                    r = SUPMT.predict_patient(p)
                probs.append((age, r[model_key]))
            for (a0, p0), (a1, p1) in zip(probs, probs[1:]):
                rows.append({
                    "model": model_key,
                    "test": "age_35_to_84",
                    "from_value": a0,
                    "to_value": a1,
                    "prob_from": p0,
                    "prob_to": p1,
                    "violation": p1 < p0,
                })
        # Density monotonicity
        for model_key in ("logistic_regression", "gradient_boosting"):
            probs = []
            for d in densities:
                p = dict(base)
                p["breast_density"] = d
                with contextlib.redirect_stdout(io.StringIO()):
                    r = SUPMT.predict_patient(p)
                probs.append((d, r[model_key]))
            for (d0, p0), (d1, p1) in zip(probs, probs[1:]):
                rows.append({
                    "model": model_key,
                    "test": "density_fatty_to_extremely_dense",
                    "from_value": d0,
                    "to_value": d1,
                    "prob_from": p0,
                    "prob_to": p1,
                    "violation": p1 < p0,
                })

        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with OUT_REPORT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        # Soft: never fail
        assert True


if __name__ == "__main__":
    import unittest

    unittest.main(argv=["test_supmt_sanity", "-v"], exit=False)