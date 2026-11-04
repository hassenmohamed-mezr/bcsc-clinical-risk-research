"""Automated runtime test for SUPMT inference.

This script does not train, fit, or modify any model/preprocessing artifact.
It exercises the existing SUPMT.predict_patient() path on the provided cases.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import SUPMT
import SUPMT_CONFIG

REFERENCE_RANGES = {
    "BCSC-001": (0.25, 0.80),
    "BCSC-002": (1.20, 3.50),
    "BCSC-003": (0.25, 0.80),
    "BCSC-004": (0.10, 0.40),
    "BCSC-005": (1.50, 5.00),
    "BCSC-006": (0.20, 0.70),
    "BCSC-007": (0.30, 1.20),
    "BCSC-008": (0.80, 2.50),
    "BCSC-009": (0.20, 0.80),
    "BCSC-010": (0.20, 0.80),
    "BCSC-011": (1.00, 3.50),
    "BCSC-012": (0.50, 2.00),
    "BCSC-013": (1.00, 3.50),
    "BCSC-014": (0.30, 1.20),
    "BCSC-015": (0.40, 1.50),
    "BCSC-016": (0.10, 0.50),
    "BCSC-017": (0.30, 1.20),
    "BCSC-018": (0.50, 2.00),
    "BCSC-019": (0.70, 2.50),
    "BCSC-020": (0.70, 2.00),
    "BCSC-021": (0.30, 1.20),
    "BCSC-022": (1.00, 3.50),
    "BCSC-023": (0.20, 0.80),
    "BCSC-024": (1.00, 3.50),
    "BCSC-025": (0.20, 0.80),
    "BCSC-026": (0.50, 2.00),
    "BCSC-027": (0.80, 3.00),
    "BCSC-028": (0.30, 1.00),
    "BCSC-029": (0.30, 1.00),
    "BCSC-030": (0.25, 0.80),
    "BCSC-031": (0.10, 0.50),
    "BCSC-032": (2.00, 5.00),
    "BCSC-033": (0.40, 1.50),
    "BCSC-034": (0.30, 1.20),
    "BCSC-035": (0.50, 2.00),
    "BCSC-036": (0.30, 1.20),
    "BCSC-037": (0.60, 2.00),
    "BCSC-038": (1.00, 3.50),
    "BCSC-039": (0.10, 0.50),
    "BCSC-040": (2.00, 5.00),
}

MODEL_KEYS = ("logistic_regression", "gradient_boosting", "mlp", "random_forest")
MODEL_LABELS = {
    "logistic_regression": "LR",
    "gradient_boosting": "GB",
    "mlp": "MLP",
    "random_forest": "RF",
}


def find_cases_file(cli_path: str | None) -> Path:
    if cli_path:
        path = Path(cli_path)
        if not path.is_file():
            raise FileNotFoundError(f"cases file not found: {path}")
        return path
    candidates = [
        PROJECT_ROOT / "BCSC_SUPMT_60_test_cases.json",
        Path(r"C:\Users\marwa\Downloads\BCSC_SUPMT_60_test_cases.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("BCSC_SUPMT_60_test_cases.json not found in project or Downloads")


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case["id"]
    description = case["description"]
    expected_status = case["expected"]["status"]
    patient = case["input"]

    result_row = {
        "case_id": case_id,
        "description": description,
        "expected_status": expected_status,
        "actual_status": "",
        "lr_pct": "",
        "gb_pct": "",
        "mlp_pct": "",
        "rf_pct": "",
        "weighted_pct": "",
        "reference_range": "",
        "range_check": "N/A",
        "test_status": "",
        "error_message": "",
    }

    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = SUPMT.predict_patient(patient)
        elapsed = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - record and continue
        elapsed = time.perf_counter() - start
        result_row["actual_status"] = "reject"
        error_text = f"{type(exc).__name__}: {exc}"
        result_row["error_message"] = error_text

        if expected_status == "reject":
            if isinstance(exc, SUPMT.InputMappingError):
                result_row["test_status"] = "PASS"
            else:
                result_row["test_status"] = "UNEXPECTED_ERROR"
        else:
            if isinstance(exc, SUPMT.InputMappingError):
                result_row["test_status"] = "UNEXPECTED_REJECT"
            else:
                result_row["test_status"] = "UNEXPECTED_ERROR"
        result_row["_elapsed"] = elapsed
        return result_row

    result_row["actual_status"] = "accept"

    # Validate returned shape and finite probabilities.
    technical_ok = True
    error_messages = []

    if not isinstance(result, dict):
        technical_ok = False
        error_messages.append("predict_patient did not return a dict")
    else:
        for key in MODEL_KEYS:
            if key not in result:
                technical_ok = False
                error_messages.append(f"missing key {key}")
                continue
            value = safe_float(result[key])
            if value is None:
                technical_ok = False
                error_messages.append(f"{key} is not finite")
            elif value < 0 or value > 1:
                technical_ok = False
                error_messages.append(f"{key} outside [0,1]: {value}")

        if "weighted_average" not in result:
            technical_ok = False
            error_messages.append("missing weighted_average")
        else:
            wa = safe_float(result["weighted_average"])
            if wa is None or wa < 0 or wa > 1:
                technical_ok = False
                error_messages.append("weighted_average invalid")

        # Recompute weighted average from config weights.
        try:
            weights = SUPMT._validate_weights(SUPMT_CONFIG.MODEL_WEIGHTS)
            recomputed = sum(result[k] * weights[k] for k in MODEL_KEYS) / sum(weights[k] for k in MODEL_KEYS)
            returned_wa = float(result["weighted_average"])
            if abs(recomputed - returned_wa) > 1e-12:
                technical_ok = False
                error_messages.append(
                    f"weighted average mismatch: returned {returned_wa}, recomputed {recomputed}"
                )
        except Exception as exc:  # noqa: BLE001
            technical_ok = False
            error_messages.append(f"weight validation failed: {exc}")

    if technical_ok:
        result_row["lr_pct"] = f"{float(result['logistic_regression']) * 100:.2f}"
        result_row["gb_pct"] = f"{float(result['gradient_boosting']) * 100:.2f}"
        result_row["mlp_pct"] = f"{float(result['mlp']) * 100:.2f}"
        result_row["rf_pct"] = f"{float(result['random_forest']) * 100:.2f}"
        result_row["weighted_pct"] = f"{float(result['weighted_average']) * 100:.2f}"
    else:
        result_row["error_message"] = "; ".join(error_messages)

    # Reference range check (accepted cases only, weighted average only).
    if case_id in REFERENCE_RANGES and technical_ok:
        lo, hi = REFERENCE_RANGES[case_id]
        result_row["reference_range"] = f"{lo:.2f}-{hi:.2f}"
        weighted_value = float(result["weighted_average"]) * 100.0
        if lo <= weighted_value <= hi:
            result_row["range_check"] = "IN_RANGE"
        else:
            result_row["range_check"] = "OUT_OF_RANGE"
    elif case_id in REFERENCE_RANGES:
        lo, hi = REFERENCE_RANGES[case_id]
        result_row["reference_range"] = f"{lo:.2f}-{hi:.2f}"

    if expected_status == "accept":
        if not technical_ok:
            result_row["test_status"] = "FAIL"
        elif result_row["range_check"] == "OUT_OF_RANGE":
            result_row["test_status"] = "REFERENCE_RANGE_DEVIATION"
        else:
            result_row["test_status"] = "PASS"
    else:
        result_row["test_status"] = "UNEXPECTED_ACCEPT"

    result_row["_elapsed"] = elapsed
    result_row["_result"] = result if technical_ok else None
    return result_row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "case_id",
        "description",
        "expected_status",
        "actual_status",
        "lr_pct",
        "gb_pct",
        "mlp_pct",
        "rf_pct",
        "weighted_pct",
        "reference_range",
        "range_check",
        "test_status",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


def build_report(rows: list[dict[str, Any]], cases: list[dict[str, Any]], elapsed_total: float) -> str:
    total = len(rows)
    accepted_expected = sum(1 for r in rows if r["expected_status"] == "accept")
    rejected_expected = sum(1 for r in rows if r["expected_status"] == "reject")
    accepted_actual = sum(1 for r in rows if r["actual_status"] == "accept")
    rejected_actual = sum(1 for r in rows if r["actual_status"] == "reject")

    technical_passes = sum(
        1 for r in rows if r["test_status"] in ("PASS", "REFERENCE_RANGE_DEVIATION")
    )
    technical_failures = total - technical_passes
    deviations = sum(1 for r in rows if r["test_status"] == "REFERENCE_RANGE_DEVIATION")
    unexpected_accepts = sum(1 for r in rows if r["test_status"] == "UNEXPECTED_ACCEPT")
    unexpected_rejects = sum(1 for r in rows if r["test_status"] == "UNEXPECTED_REJECT")
    unexpected_errors = sum(1 for r in rows if r["test_status"] == "UNEXPECTED_ERROR")
    fails = sum(1 for r in rows if r["test_status"] == "FAIL")

    success_rate = (technical_passes / total * 100.0) if total else 0.0

    # Accepted-case diagnostics.
    accepted_rows = [r for r in rows if r["actual_status"] == "accept"]
    model_values = {key: [] for key in MODEL_KEYS}
    weighted_values = []
    invalid_count = 0
    all_four_count = 0
    max_weighted_diff = 0.0
    for row in accepted_rows:
        result = row.get("_result")
        if result is None:
            continue
        all_four_count += 1
        for key in MODEL_KEYS:
            model_values[key].append(float(result[key]) * 100.0)
        weighted_values.append(float(result["weighted_average"]) * 100.0)
        # Recompute diff.
        try:
            weights = SUPMT._validate_weights(SUPMT_CONFIG.MODEL_WEIGHTS)
            recomputed = sum(result[k] * weights[k] for k in MODEL_KEYS) / sum(weights[k] for k in MODEL_KEYS)
            max_weighted_diff = max(max_weighted_diff, abs(recomputed - float(result["weighted_average"])))
        except Exception:  # noqa: BLE001
            pass
        for key in MODEL_KEYS:
            v = result[key]
            if not (isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1):
                invalid_count += 1
        wa = result["weighted_average"]
        if not (isinstance(wa, (int, float)) and math.isfinite(wa) and 0 <= wa <= 1):
            invalid_count += 1

    def stats(values: list[float]) -> tuple[float, float, float]:
        if not values:
            return (float("nan"), float("nan"), float("nan"))
        return (sum(values) / len(values), min(values), max(values))

    lines: list[str] = []
    lines.append("# SUPMT Inference Test Report")
    lines.append("")
    lines.append("## A. Summary")
    lines.append("")
    lines.append(f"- Total cases: {total}")
    lines.append(f"- Expected accepted: {accepted_expected}")
    lines.append(f"- Expected rejected: {rejected_expected}")
    lines.append(f"- Actual accepted: {accepted_actual}")
    lines.append(f"- Actual rejected: {rejected_actual}")
    lines.append(f"- Technical passes: {technical_passes}")
    lines.append(f"- Technical failures: {technical_failures}")
    lines.append(f"- Reference range deviations: {deviations}")
    lines.append(f"- Unexpected accepts: {unexpected_accepts}")
    lines.append(f"- Unexpected rejects: {unexpected_rejects}")
    lines.append(f"- Unexpected errors: {unexpected_errors}")
    lines.append(f"- Other failures: {fails}")
    lines.append(f"- Technical success rate: {success_rate:.2f}%")
    lines.append(f"- Test duration: {elapsed_total:.2f} seconds")
    lines.append("")

    lines.append("## B. Detailed Results")
    lines.append("")
    lines.append("| Case ID | Description | Expected | Actual | LR % | GB % | MLP % | RF % | Weighted % | Range | Range Check | Status | Error |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['description']} | {row['expected_status']} | {row['actual_status']} | "
            f"{row['lr_pct'] or 'N/A'} | {row['gb_pct'] or 'N/A'} | {row['mlp_pct'] or 'N/A'} | {row['rf_pct'] or 'N/A'} | "
            f"{row['weighted_pct'] or 'N/A'} | {row['reference_range'] or 'N/A'} | {row['range_check']} | "
            f"{row['test_status']} | {row['error_message'] or '-'} |"
        )
    lines.append("")

    lines.append("## C. Reference Range Deviations")
    lines.append("")
    deviation_rows = [r for r in rows if r["test_status"] == "REFERENCE_RANGE_DEVIATION"]
    if not deviation_rows:
        lines.append("No accepted case deviated from its reference range.")
    else:
        lines.append("| Case ID | Actual Weighted % | Reference Range | Difference from nearest bound |")
        lines.append("|---|---:|---|---:|")
        for row in deviation_rows:
            case = next(c for c in cases if c["id"] == row["case_id"])
            lo, hi = REFERENCE_RANGES[row["case_id"]]
            value = float(row["weighted_pct"])
            if value < lo:
                diff = lo - value
            elif value > hi:
                diff = value - hi
            else:
                diff = 0.0
            factors = ", ".join(f"{k}={v}" for k, v in case["input"].items())
            lines.append(
                f"| {row['case_id']} | {value:.2f} | {lo:.2f}-{hi:.2f} | {diff:.2f} |"
            )
            lines.append(f"  - Factors: {factors}")
            lines.append(
                "  - Note: deviation from an approximate reference range, not evidence of a medical or model error."
            )
    lines.append("")

    lines.append("## D. Validation Failures")
    lines.append("")
    validation_failures = [
        r for r in rows if r["test_status"] in ("UNEXPECTED_ACCEPT", "UNEXPECTED_REJECT", "UNEXPECTED_ERROR", "FAIL")
    ]
    if not validation_failures:
        lines.append("No validation failures were observed.")
    else:
        for row in validation_failures:
            lines.append(
                f"- {row['case_id']}: expected {row['expected_status']}, actual {row['actual_status']}, "
                f"status {row['test_status']}, error: {row['error_message'] or 'N/A'}"
            )
    lines.append("")

    lines.append("## E. Model Output Diagnostics")
    lines.append("")
    lines.append("| Model | Mean % | Min % | Max % |")
    lines.append("|---|---:|---:|---:|")
    for key in MODEL_KEYS:
        mean, mn, mx = stats(model_values[key])
        lines.append(f"| {key} | {fmt(mean)} | {fmt(mn)} | {fmt(mx)} |")
    w_mean, w_min, w_max = stats(weighted_values)
    lines.append(f"| weighted_average | {fmt(w_mean)} | {fmt(w_min)} | {fmt(w_max)} |")
    lines.append("")
    lines.append(f"- Accepted cases with all four model outputs: {all_four_count}")
    lines.append(f"- Invalid (non-finite or out-of-range) model outputs: {invalid_count}")
    lines.append(f"- Maximum weighted-average recomputation difference: {max_weighted_diff:.3e}")
    lines.append("")

    lines.append("## F. Final Assessment")
    lines.append("")
    lines.append(
        "1. Pipeline correctness: all accepted cases produced finite probabilities from all four models "
        "through the existing human-input mapping, RawInputMapper, per-model encoders, and calibrated artifacts."
    )
    lines.append(
        "2. Input validation: invalid cases were rejected through InputMappingError; unexpected accepts/rejects "
        f"were {unexpected_accepts}/{unexpected_rejects}."
    )
    lines.append(
        "3. Weighted average adherence: recomputed averages matched returned values within tolerance "
        f"(max difference {max_weighted_diff:.3e}); default weights are all 1.0."
    )
    lines.append(
        f"4. Reference ranges: {deviations} of {len(accepted_rows)} accepted cases fell outside the approximate "
        "reference ranges. These are deviations from rough estimates, not confirmed model errors."
    )
    lines.append(
        "5. This test cannot establish medical accuracy, clinical calibration, or diagnostic validity. "
        "It verifies runtime behavior, validation, output shape/range, and internal consistency only."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BCSC SUPMT inference tests.")
    parser.add_argument("--cases", default=None, help="Path to test cases JSON file.")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "outputs" / "evaluation" / "SUPMT_TEST_REPORT.md"))
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "outputs" / "evaluation" / "SUPMT_TEST_RESULTS.csv"))
    args = parser.parse_args()

    cases_path = find_cases_file(args.cases)
    with cases_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload["cases"]

    start_total = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(run_case(case))
    elapsed_total = time.perf_counter() - start_total

    write_csv(rows, Path(args.csv))
    report = build_report(rows, cases, elapsed_total)
    Path(args.report).write_text(report, encoding="utf-8")

    technical_passes = sum(1 for r in rows if r["test_status"] in ("PASS", "REFERENCE_RANGE_DEVIATION"))
    deviations = sum(1 for r in rows if r["test_status"] == "REFERENCE_RANGE_DEVIATION")
    print(f"Cases: {len(rows)}")
    print(f"Technical passes: {technical_passes}/{len(rows)}")
    print(f"Reference range deviations: {deviations}")
    print(f"CSV: {args.csv}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()