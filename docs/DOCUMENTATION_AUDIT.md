# Documentation Audit

| Documentation Claim | Evidence File | Evidence Location | Verification Status |
|---|---|---|---|
| Four trained models exist | `models/trained/logistic_regression_metadata.json` | model keys | Verified |
| Logistic Regression uses one-hot | `src/training/config.py` | `MODEL_SPECS["logistic_regression"].encoder` | Verified |
| Gradient Boosting uses ordinal_tree | `src/training/config.py` | `MODEL_SPECS["gradient_boosting"].encoder` | Verified |
| MLP uses one-hot + StandardScaler | `src/training/config.py` | `MODEL_SPECS["mlp"].encoder`, `.scaler` | Verified |
| Random Forest uses ordinal_tree | `src/training/config.py` | `MODEL_SPECS["random_forest"].encoder` | Verified |
| Encoders fitted on train only | `src/preprocessing/fit_preprocessing.py` | `main()` loads only train | Verified |
| Inference does not refit | `SUPMT.py` | `_load_system()` calls `load_pipeline` and `load_model_artifact` | Verified |
| Feature order is fixed | `src/data/load_splits.py` | `FEATURES` | Verified |
| count is frequency weight | `src/data/load_splits.py` | `WEIGHT`, docstring | Verified |
| surgmeno/hrt code 9 combined | `outputs/preprocessing/reports/12_missing_value_policy.md` | table row | Verified |
| Test split one-time evaluation | `PROJECT_MAP.md` | section 11 | Verified |
| 60 inference test cases | `test_SUPMT_inference.py` | run output | Verified |
| No clinical validation | Repository scope | Not found as implemented | Not Verified / Not Implemented |
| Future Egyptian data collection | Project context | Planned only | Planned / Not Implemented |