import sys
import pandas as pd

# Column names, in the same order as the dataset documentation (variables 1-16)
COLUMNS = [
    "menopaus", "agegrp", "density", "race", "Hispanic", "bmi",
    "agefirst", "nrelbc", "brstproc", "lastmamm", "surgmeno", "hrt",
    "invasive", "cancer", "training", "count",
]

# Allowed values per column, from the dataset documentation
ALLOWED = {
    "menopaus": {0, 1, 9},
    "agegrp": set(range(1, 11)),
    "density": {1, 2, 3, 4, 9},
    "race": {1, 2, 3, 4, 5, 9},
    "Hispanic": {0, 1, 9},
    "bmi": {1, 2, 3, 4, 9},
    "agefirst": {0, 1, 2, 9},
    "nrelbc": {0, 1, 2, 9},
    "brstproc": {0, 1, 9},
    "lastmamm": {0, 1, 9},
    "surgmeno": {0, 1, 9},
    "hrt": {0, 1, 9},
    "invasive": {0, 1},
    "cancer": {0, 1},
    "training": {0, 1},
}


def convert(src: str = "risk.txt", dst: str = "risk.csv") -> pd.DataFrame:
    # Columns are separated by one or more spaces (the file also has Windows \r\n endings)
    df = pd.read_csv(src, sep=r"\s+", header=None, names=COLUMNS, engine="python")

    # ---- sanity checks ----
    assert df.shape[1] == 16, f"expected 16 columns, got {df.shape[1]}"
    assert not df.isna().any().any(), "file contains missing/malformed values"
    for col, allowed in ALLOWED.items():
        bad = set(df[col].unique()) - allowed
        assert not bad, f"column '{col}' has unexpected values: {bad}"
    assert (df["invasive"] <= df["cancer"]).all(), "found invasive=1 with cancer=0"

    df.to_csv(dst, index=False)

    # ---- quick report ----
    print(f"rows            : {len(df):,}")
    print(f"sum of count    : {df['count'].sum():,}")
    print(f"cancer cases    : {(df['cancer'] * df['count']).sum():,}")
    print(f"saved to        : {dst}")
    return df


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "risk.txt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "risk.csv"
    convert(src, dst)