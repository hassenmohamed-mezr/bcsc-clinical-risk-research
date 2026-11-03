"""Fit and persist the reusable BCSC preprocessing artifacts.

Fits every learned preprocessing object on the Training split only and writes:

    models/preprocessing/raw_input_mapping.json
    models/preprocessing/onehot_pipeline.joblib
    models/preprocessing/ordinal_tree_pipeline.joblib
    models/preprocessing/preprocessing_metadata.json

Validation/Test data are never read here; ``count`` is not used by these
encoders (it is a frequency weight for model fitting, not a preprocessing
input). No rows are expanded or dropped.

Usage:
    python -m src.preprocessing.fit_preprocessing
"""

from __future__ import annotations

import json
from pathlib import Path

import sklearn
from sklearn.compose import ColumnTransformer

from src.data.load_splits import FEATURES, TARGET, WEIGHT, load_split
from src.preprocessing.pipeline import (
    FAMILIES,
    feature_names,
    fit_pipeline,
    save_pipeline,
)
from src.preprocessing.raw_input_mapping import RawInputMapper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "models" / "preprocessing"


def main() -> dict:
    X_train, _, w_train = load_split("train")

    mapper = RawInputMapper()
    mapper_path = ARTIFACT_DIR / "raw_input_mapping.json"
    mapper.save(mapper_path)

    metadata = {
        "version": 1,
        "fit_split": "train",
        "test_used": False,
        "rows_fitted": int(len(X_train)),
        "weight_sum_fitted": int(w_train.sum()),
        "features": list(FEATURES),
        "target": TARGET,
        "frequency_weight": WEIGHT,
        "sklearn_version": sklearn.__version__,
        "unknown_code": 9,
        "interface_unknown": 0,
        "families": {},
    }

    for family in FAMILIES:
        pipeline: ColumnTransformer = fit_pipeline(X_train, family=family)
        path = ARTIFACT_DIR / f"{family}_pipeline.joblib"
        save_pipeline(pipeline, path)
        names = feature_names(pipeline)
        metadata["families"][family] = {
            "artifact": path.name,
            "n_features_in": len(FEATURES),
            "n_features_out": len(names),
            "feature_names": names,
        }
        print(f"{family:<13} fit on {len(X_train):,} rows -> {len(names):>2} encoded features -> {path}")

    metadata_path = ARTIFACT_DIR / "preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"mapping        -> {mapper_path}")
    print(f"metadata       -> {metadata_path}")
    return metadata


if __name__ == "__main__":
    main()
