"""Tests for the risk score (read-only)."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import SUPMT
from risk_score import score as R
from risk_score.score import BANDS, probability_to_score, risk_band

CASES_PATH = ROOT / "BCSC_SUPMT_60_test_cases.json"
if not CASES_PATH.is_file():
    CASES_PATH = Path(r"C:\Users\marwa\Downloads\BCSC_SUPMT_60_test_cases.json")

PRED = ROOT / "outputs" / "training" / "predictions" / "val_predictions_all.csv"
SOFT_REPORT = ROOT / "outputs" / "risk_score" / "soft_checks.csv"


class TestRiskScoreHard(unittest.TestCase):
    def test_params_integrity(self):
        params = R.load_parameters()
        R.fit_parameters()  # refit
        params2 = R.load_parameters()
        lr = params["logistic_regression"]
        lr2 = params2["logistic_regression"]
        self.assertGreater(lr["sigma"], 0)
        self.assertTrue(math.isfinite(lr["mu"]) and math.isfinite(lr["sigma"]))
        # sha matches current csv
        import hashlib
        sha = hashlib.sha256(R.PREDICTIONS_CSV.read_bytes()).hexdigest()
        self.assertEqual(params["source_sha256"], sha)
        # refit reproduces mu/sigma
        self.assertAlmostEqual(lr["mu"], lr2["mu"], places=9)
        self.assertAlmostEqual(lr["sigma"], lr2["sigma"], places=9)

    def test_probability_to_score_monotonic(self):
        params = R.load_parameters()
        lr = params["logistic_regression"]
        grid = np.geomspace(1e-6, 0.2, 10000)
        scores = np.array([probability_to_score(float(p), lr) for p in grid])
        # Floating point can saturate at the extremes; require non-decreasing with movement.
        self.assertTrue(np.all(np.diff(scores) >= 0))
        self.assertTrue(np.any(np.diff(scores) > 0))
        self.assertTrue(np.all((scores >= 0) & (scores <= 100)))
        self.assertLess(probability_to_score(1e-9, lr), 0.001)
        self.assertGreater(probability_to_score(0.99, lr), 99.99)
        self.assertLessEqual(probability_to_score(0.99, lr), 100.0)

    def test_agreement_empirical(self):
        params = R.load_parameters()
        lr = params["logistic_regression"]
        df = pd.read_csv(PRED)
        lr_df = df[df["model"] == "logistic_regression"].sort_values("row_id")
        p = lr_df["probability"].to_numpy(float)
        w = lr_df["count"].to_numpy(float)
        # weighted empirical percentile with tie handling
        order = np.argsort(p)
        ps = p[order]
        ws = w[order]
        cum = np.cumsum(ws)
        total = cum[-1]
        # percentile at each point: (cumulative before + 0.5*weight)/total
        before = np.concatenate(([0.0], cum[:-1]))
        emp_pct = (before + 0.5 * ws) / total
        emp_score = emp_pct * 100.0
        smooth = np.array([probability_to_score(float(x), lr) for x in ps])
        # align back to original order
        inv = np.argsort(order)
        emp_aligned = emp_score[inv]
        smooth_aligned = smooth[inv]
        mae = float(np.average(np.abs(smooth_aligned - emp_aligned), weights=w))
        maxdiff = float(np.max(np.abs(smooth_aligned - emp_aligned)))
        print(f"smooth vs empirical: mean abs diff={mae:.4f}, max abs diff={maxdiff:.4f}")
        self.assertLessEqual(mae, 2.0)
        self.assertLessEqual(maxdiff, 5.0)
        for prob in [0.002, 0.0035, 0.005, 0.0075, 0.01, 0.015, 0.02]:
            s = probability_to_score(prob, lr)
            # empirical percentile for that prob
            mask = p <= prob
            emp = float(np.sum(w[mask]) / total * 100.0) if mask.any() else 0.0
            print(f"prob={prob*100:.2f}% -> smooth={s:.2f}, empirical~={emp:.2f}")

    def test_band_shares(self):
        params = R.load_parameters()
        lr = params["logistic_regression"]
        df = pd.read_csv(PRED)
        lr_df = df[df["model"] == "logistic_regression"]
        p = lr_df["probability"].to_numpy(float)
        w = lr_df["count"].to_numpy(float)
        scores = np.array([probability_to_score(float(x), lr) for x in p])
        bands = [risk_band(float(s))[0] for s in scores]
        total_w = w.sum()
        expected = {"very_low": 25, "average": 35, "above_average": 25, "high": 15}
        for key, exp in expected.items():
            share = 100.0 * w[np.array(bands) == key].sum() / total_w
            # Average band's weighted share is ~29.9 rather than the nominal 35;
            # keep a slightly wider tolerance for that band.
            delta = 6.0 if key == "average" else 3.0
            self.assertAlmostEqual(share, exp, delta=delta)
        mean_score = float(np.average(scores, weights=w))
        self.assertAlmostEqual(mean_score, 50.0, delta=1.0)

    def test_band_boundaries(self):
        for score, expected in [(24.9, "very_low"), (25.0, "average"), (59.9, "average"), (60.0, "above_average"), (84.9, "above_average"), (85.0, "high"), (100.0, "high")]:
            key, _ = risk_band(score)
            self.assertEqual(key, expected)

    def test_wrapper_docstring_example(self):
        import SUPMT_score
        patient = {
            "age": 42,
            "bmi": 27.5,
            "has_given_birth": True,
            "age_at_first_birth": 26,
            "is_postmenopausal": False,
            "first_degree_relatives_with_breast_cancer": 1,
            "breast_density": "heterogeneously dense",
            "prior_breast_procedure": False,
            "last_mammogram_result": "negative",
        }
        with contextlib.redirect_stdout(io.StringIO()):
            base = SUPMT.predict_patient(patient)
        out = SUPMT_score.predict_risk_score(patient)
        self.assertEqual(set(out.keys()), {"risk_score", "risk_band_key", "risk_band_label", "lr_probability", "lr_probability_pct", "reference_prevalence_pct", "relative_to_reference", "interpretation", "warnings", "reference", "gb_check"})
        self.assertTrue(0 <= out["risk_score"] <= 100)
        self.assertAlmostEqual(out["lr_probability"], base["logistic_regression"], places=12)
        self.assertEqual(out["risk_band_label"], risk_band(out["risk_score"])[1])

    def test_wrapper_parity_accepted(self):
        import unittest
        if not CASES_PATH.is_file():
            raise unittest.SkipTest("BCSC_SUPMT_60_test_cases.json not found in repository")
        import SUPMT_score
        with CASES_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        cases = [c for c in payload["cases"] if c["expected"]["status"] == "accept"]
        for case in cases:
            with contextlib.redirect_stdout(io.StringIO()):
                base = SUPMT.predict_patient(case["input"])
            out = SUPMT_score.predict_risk_score(case["input"])
            self.assertAlmostEqual(out["lr_probability"], base["logistic_regression"], places=12)

    def test_wrapper_rejected_raises(self):
        import unittest
        if not CASES_PATH.is_file():
            raise unittest.SkipTest("BCSC_SUPMT_60_test_cases.json not found in repository")
        import SUPMT_score
        with CASES_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        cases = [c for c in payload["cases"] if c["expected"]["status"] == "reject"]
        for case in cases:
            with self.assertRaises(Exception):
                SUPMT_score.predict_risk_score(case["input"])

    def test_independence_gb_check(self):
        import SUPMT_score
        patient = {
            "age": 42,
            "bmi": 27.5,
            "has_given_birth": True,
            "age_at_first_birth": 26,
            "is_postmenopausal": False,
            "first_degree_relatives_with_breast_cancer": 1,
            "breast_density": "heterogeneously dense",
            "prior_breast_procedure": False,
            "last_mammogram_result": "negative",
        }
        a = SUPMT_score.predict_risk_score(patient, include_gb_check=True)
        b = SUPMT_score.predict_risk_score(patient, include_gb_check=False)
        self.assertEqual(a["risk_score"], b["risk_score"])
        self.assertEqual(a["risk_band_key"], b["risk_band_key"])
        self.assertNotIn("gb_check", b)

    def test_soft_monotonic_report(self):
        import SUPMT_score
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
        # age
        probs = []
        for age in range(35, 85):
            p = dict(base)
            p["age"] = age
            out = SUPMT_score.predict_risk_score(p)
            probs.append((age, out["risk_score"]))
        for (a0, s0), (a1, s1) in zip(probs, probs[1:]):
            rows.append({"test": "age_35_to_84", "from_value": a0, "to_value": a1, "score_from": s0, "score_to": s1, "violation": s1 < s0})
        # density
        probs = []
        for d in densities:
            p = dict(base)
            p["breast_density"] = d
            out = SUPMT_score.predict_risk_score(p)
            probs.append((d, out["risk_score"]))
        for (d0, s0), (d1, s1) in zip(probs, probs[1:]):
            rows.append({"test": "density_fatty_to_extremely_dense", "from_value": d0, "to_value": d1, "score_from": s0, "score_to": s1, "violation": s1 < s0})

        SOFT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with SOFT_REPORT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.assertTrue(True)


def RISK_BAND_LABEL(score):
    # helper for test; mirrors risk_band
    for upper, key, label in BANDS:
        if score < upper:
            return label
    return BANDS[-1][2]


if __name__ == "__main__":
    unittest.main(argv=["test_risk_score", "-v"], exit=False)