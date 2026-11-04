"""Seeded scan of 5000 valid human inputs through SUPMT (read-only)."""

from __future__ import annotations

import csv
import io
import contextlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import SUPMT

OUT = ROOT / "outputs" / "evaluation" / "reports" / "input_scan.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

DENSITIES = [
    "almost entirely fatty",
    "scattered fibroglandular",
    "heterogeneously dense",
    "extremely dense",
]
RELATIVES = [None, 0, 1, "2+"]
BMI_VALUES = [None, 18.0, 24.99, 25.0, 29.99, 30.0, 34.99, 35.0, 40.0]


def random_patient(rng: random.Random) -> dict:
    age = rng.randint(35, 84)
    post = rng.choice([None, True, False])
    has_birth = rng.choice([True, False])
    if has_birth:
        first_birth = rng.choice([None, rng.randint(15, min(45, age))]) if age >= 15 else None
    else:
        first_birth = None
    if post is True:
        menopause_type = rng.choice([None, "natural", "surgical"])
        hrt = rng.choice([None, True, False])
    else:
        menopause_type = None
        hrt = None
    return {
        "age": age,
        "bmi": rng.choice(BMI_VALUES),
        "has_given_birth": has_birth,
        "age_at_first_birth": first_birth,
        "is_postmenopausal": post,
        "menopause_type": menopause_type,
        "currently_on_hrt": hrt,
        "first_degree_relatives_with_breast_cancer": rng.choice(RELATIVES),
        "breast_density": rng.choice([None] + DENSITIES),
        "prior_breast_procedure": rng.choice([None, True, False]),
        "last_mammogram_result": rng.choice([None, "negative", "false positive"]),
    }


def main():
    rng = random.Random(42)
    rows = []
    model_keys = ["logistic_regression", "gradient_boosting", "mlp", "random_forest"]
    for i in range(5000):
        patient = random_patient(rng)
        with contextlib.redirect_stdout(io.StringIO()):
            result = SUPMT.predict_patient(patient)
        row = {"case_index": i}
        for key in model_keys:
            row[key] = result[key]
        row["weighted_average"] = result["weighted_average"]
        row["above_10pct"] = any(result[k] > 0.10 for k in model_keys)
        rows.append(row)

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import statistics
    for key in model_keys + ["weighted_average"]:
        vals = [r[key] for r in rows]
        print(
            key,
            "min", min(vals),
            "p50", statistics.median(vals),
            "p99", sorted(vals)[int(0.99 * len(vals))],
            "max", max(vals),
        )
    print("above_10pct cases:", sum(r["above_10pct"] for r in rows))
    # RF constant-like check
    rf_vals = [r["random_forest"] for r in rows]
    print("RF std:", statistics.pstdev(rf_vals))
    print("saved", OUT)


if __name__ == "__main__":
    main()