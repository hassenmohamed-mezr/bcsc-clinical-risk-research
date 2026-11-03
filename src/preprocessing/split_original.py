"""Split the interim dataset using the ORIGINAL `training` column.

No shuffling, re-splitting, scaling, encoding, or imputation is performed.
Rows keep their original order and `training` is retained in both outputs.

Usage:
    python src/preprocessing/split_original.py
    python src/preprocessing/split_original.py --input <csv> --train <csv> --test <csv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---- configuration -------------------------------------------------------
INPUT_FILE = "data/02_interim/risk_filtered.csv"
TRAIN_FILE = "data/04_splits/train_full.csv"
TEST_FILE = "data/04_splits/test.csv"
TRAINING_COL = "training"
TARGET_COL = "cancer"
WEIGHT_COL = "count"

# Original split mapping: 1 = training, 0 = test (unverified in repo docs).
TRAIN_VALUE = 1
TEST_VALUE = 0


def split(input_path: Path, train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_path)

    assert TRAINING_COL in df.columns, f"missing column: {TRAINING_COL}"
    assert WEIGHT_COL in df.columns, f"missing column: {WEIGHT_COL}"
    assert TARGET_COL in df.columns, f"missing column: {TARGET_COL}"

    values = set(df[TRAINING_COL].unique().tolist())
    assert values == {TRAIN_VALUE, TEST_VALUE}, f"unexpected {TRAINING_COL} values: {values}"
    assert not df[TRAINING_COL].isna().any(), f"nulls in {TRAINING_COL}"

    train = df[df[TRAINING_COL] == TRAIN_VALUE].copy()
    test = df[df[TRAINING_COL] == TEST_VALUE].copy()

    # ---- integrity checks ----
    assert len(train) + len(test) == len(df), "row counts do not add up"
    assert abs(train[WEIGHT_COL].sum() + test[WEIGHT_COL].sum() - df[WEIGHT_COL].sum()) < 1e-9, (
        "weighted counts do not add up"
    )
    assert list(train.columns) == list(test.columns) == list(df.columns), "column order changed"
    assert not train.isna().any().any(), "nulls introduced in train"
    assert not test.isna().any().any(), "nulls introduced in test"
    assert len(train) > 0 and len(test) > 0, "empty split"

    for path in (train_path, test_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    return train, test


def _report(name: str, part: pd.DataFrame) -> None:
    weighted = float(part[WEIGHT_COL].sum())
    cases = float((part[TARGET_COL] * part[WEIGHT_COL]).sum())
    rate = cases / weighted if weighted else 0.0
    print(
        f"{name:<12} rows={len(part):>7,}  weighted={weighted:>12,.0f}  "
        f"cases={cases:>8,.0f}  cancer_rate={rate:.6%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--train", default=TRAIN_FILE)
    parser.add_argument("--test", default=TEST_FILE)
    args = parser.parse_args()

    train, test = split(Path(args.input), Path(args.train), Path(args.test))

    print(f"input        rows={len(train) + len(test):>7,}  columns={len(train.columns)}")
    _report("train (1)", train)
    _report("test (0)", test)
    print(f"written      {args.train} | {args.test}")


if __name__ == "__main__":
    main()
