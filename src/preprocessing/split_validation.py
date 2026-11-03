"""Weighted stratified random split of the training data into train/validation.

Splits `data/04_splits/train_full.csv` into ~85% / ~15% of total frequency
weight (`count`), stratified by the binary `cancer` target. The original test
set (`data/04_splits/test.csv`) is never read or touched.

Algorithm
---------
1. Load and validate the input (count, cancer).
2. Split rows into class_0 / class_1.
3. Per class: shuffle row indices with RANDOM_STATE, greedily assign each row to
   validation when that brings the validation weight closer to 15% of the class
   weight, otherwise to training (deterministic tie-break: keep in training).
4. Per class: deterministic local optimization moving single rows between the
   two sides while it strictly reduces |val_weight - target|.

No row is split, merged, expanded, or modified; only whole rows move.

Usage:
    python src/preprocessing/split_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---- configuration -------------------------------------------------------
RANDOM_STATE = 42
VAL_FRACTION = 0.15
WEIGHT_COL = "count"
TARGET_COL = "cancer"

INPUT_FILE = "data/04_splits/train_full.csv"
TRAIN_FILE = "data/04_splits/train.csv"
VAL_FILE = "data/04_splits/val.csv"

# ---- validation tolerances required by the task --------------------------
SHARE_MIN, SHARE_MAX = 0.14, 0.16
RATE_REL_TOL = 0.10


def _validate(df: pd.DataFrame) -> None:
    assert WEIGHT_COL in df.columns, f"missing column: {WEIGHT_COL}"
    assert TARGET_COL in df.columns, f"missing column: {TARGET_COL}"
    assert not df[WEIGHT_COL].isna().any(), f"nulls in {WEIGHT_COL}"
    assert not df[TARGET_COL].isna().any(), f"nulls in {TARGET_COL}"
    assert np.issubdtype(df[WEIGHT_COL].dtype, np.integer), "count must be integer"
    assert (df[WEIGHT_COL] > 0).all(), "count must be positive"
    assert set(df[TARGET_COL].unique()) <= {0, 1}, "cancer must be binary 0/1"


def _greedy(weights: np.ndarray, order: np.ndarray, target: float) -> np.ndarray:
    """Return a boolean mask: True = validation. Ties keep the row in training."""
    in_val = np.zeros(len(weights), dtype=bool)
    running = 0.0
    for i in order:
        candidate = running + weights[i]
        if abs(candidate - target) < abs(running - target):
            in_val[i] = True
            running = candidate
    return in_val


def _optimize(weights: np.ndarray, in_val: np.ndarray, target: float) -> np.ndarray:
    """Deterministic single-row local search minimizing |val_weight - target|."""
    in_val = in_val.copy()
    val_w = float(weights[in_val].sum())
    for _ in range(1_000_000):  # safety ceiling, expected to converge far sooner
        residual = val_w - target
        best_gain = 0.0
        best_move = None

        # candidate removals from validation (reduces val weight by w)
        val_idx = np.flatnonzero(in_val)
        if in_val.sum() > 1:
            gain = np.abs(residual) - np.abs(residual - weights[val_idx])
            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain, best_move = float(gain[k]), ("out", int(val_idx[k]))

        # candidate additions to validation (increases val weight by w)
        trn_idx = np.flatnonzero(~in_val)
        if trn_idx.size:
            gain = np.abs(residual) - np.abs(residual + weights[trn_idx])
            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain, best_move = float(gain[k]), ("in", int(trn_idx[k]))

        if best_move is None:
            break
        direction, idx = best_move
        if direction == "out":
            in_val[idx] = False
            val_w -= float(weights[idx])
        else:
            in_val[idx] = True
            val_w += float(weights[idx])
    else:  # pragma: no cover - defensive
        raise RuntimeError("local optimization did not converge")
    return in_val


def _class_allocation(df: pd.DataFrame, cls: int, rng: np.random.Generator) -> pd.DataFrame:
    part = df[df[TARGET_COL] == cls]
    weights = part[WEIGHT_COL].to_numpy(dtype=float)
    class_total = float(weights.sum())
    target = VAL_FRACTION * class_total

    order = rng.permutation(len(part))
    in_val = _greedy(weights, order, target)

    # guarantee both sides non-empty when the class allows it
    if not in_val.any() and len(part) > 1:
        in_val[int(np.argmin(np.abs(weights - target)))] = True
    if in_val.all() and len(part) > 1:
        in_val[int(np.argmin(np.abs(weights - target)))] = False

    in_val = _optimize(weights, in_val, target)

    val_w = float(weights[in_val].sum())
    achieved = val_w / class_total if class_total else 0.0
    print(
        f"class {cls}: rows={len(part):>7,} weight={class_total:>11,.0f} "
        f"val_rows={int(in_val.sum()):>7,} val_weight={val_w:>9,.0f} "
        f"share={achieved:.4%} residual={abs(val_w - target):,.3f}"
    )
    return part.assign(_split=in_val)


def split(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_path)
    _validate(df)
    rng = np.random.default_rng(RANDOM_STATE)

    parts = [_class_allocation(df, cls, rng) for cls in (0, 1)]
    tagged = pd.concat(parts).sort_index()

    assert len(tagged) == len(df), "rows lost during allocation"
    assert tagged["_split"].isin([False, True]).all(), "bad allocation flags"
    assert tagged["_split"].any() and (~tagged["_split"]).any(), "empty split"

    val = tagged[tagged["_split"]].drop(columns="_split")
    train = tagged[~tagged["_split"]].drop(columns="_split")

    # ---- structural integrity ----
    assert len(train) + len(val) == len(df), "row totals do not add up"
    assert list(train.columns) == list(val.columns) == list(df.columns), "columns differ"
    assert WEIGHT_COL in train.columns and WEIGHT_COL in val.columns, "count missing"

    combined = pd.concat([train, val]).sort_values(
        by=list(df.columns), kind="mergesort"
    ).reset_index(drop=True)
    reference = df.sort_values(by=list(df.columns), kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(combined, reference, check_dtype=True)

    # ---- weighted integrity ----
    total_w = int(df[WEIGHT_COL].sum())
    train_w = int(train[WEIGHT_COL].sum())
    val_w = int(val[WEIGHT_COL].sum())
    assert train_w + val_w == total_w, "weighted totals do not add up"
    share = val_w / total_w
    assert SHARE_MIN <= share <= SHARE_MAX, f"validation share {share:.4%} outside 14-16%"

    # ---- stratification integrity ----
    orig_rate = float((df[TARGET_COL] * df[WEIGHT_COL]).sum() / total_w)
    for name, part in (("train", train), ("val", val)):
        w = int(part[WEIGHT_COL].sum())
        rate = float((part[TARGET_COL] * part[WEIGHT_COL]).sum() / w)
        rel = abs(rate - orig_rate) / orig_rate
        assert rel <= RATE_REL_TOL, f"{name} weighted cancer rate off by {rel:.2%}"
        print(f"{name:<7} rows={len(part):>7,} weight={w:>11,} cancer_rate={rate:.6%} rel_diff={rel:.4%}")

    # ---- diagnostic only: shared feature combinations ----
    features = [c for c in df.columns if c not in (WEIGHT_COL, TARGET_COL)]
    shared = len(set(map(tuple, train[features].to_numpy())) & set(map(tuple, val[features].to_numpy())))
    print(f"total_w={total_w:,} train_w={train_w:,} val_w={val_w:,} share={share:.6%}")
    print(f"orig_cancer_rate={orig_rate:.6%} shared_feature_combos={shared:,}")

    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--train", default=TRAIN_FILE)
    parser.add_argument("--val", default=VAL_FILE)
    args = parser.parse_args()

    train, val = split(Path(args.input))

    for path, frame in ((Path(args.train), train), (Path(args.val), val)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print(f"written {args.train} ({len(train):,} rows), {args.val} ({len(val):,} rows)")


if __name__ == "__main__":
    main()
