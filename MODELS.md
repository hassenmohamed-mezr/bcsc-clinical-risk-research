# Models

## Model Inventory

Four trained and calibrated models are stored under `models/trained/`.

| Model key | Display name | Algorithm | Encoder | Scaler | Artifact |
|---|---|---|---|---|---|
| `logistic_regression` | Logistic Regression | `sklearn.linear_model.LogisticRegression` | `onehot` | None | `models/trained/logistic_regression_calibrated.joblib` |
| `random_forest` | Random Forest | `sklearn.ensemble.RandomForestClassifier` | `ordinal_tree` | None | `models/trained/random_forest_calibrated.joblib` |
| `gradient_boosting` | Gradient Boosting | `sklearn.ensemble.GradientBoostingClassifier` | `ordinal_tree` | None | `models/trained/gradient_boosting_calibrated.joblib` |
| `mlp` | MLP Neural Network | `sklearn.neural_network.MLPClassifier` | `onehot` | StandardScaler (bundled in MLP artifact) | `models/trained/mlp_calibrated.joblib` |

Evidence: `src/training/config.py` (`MODEL_SPECS`), model metadata JSON files.

## Calibration

All four artifacts are `CalibratedClassifierCV` objects with:

- method: sigmoid
- ensemble: false
- data: Training only

Evidence: model metadata JSON files, `src/training/train_models.py`.

## Selected Hyperparameters

| Model | Selected candidate | Selected params |
|---|---|---|
| Logistic Regression | `logistic_regression_001` | `C=0.1`, `solver=lbfgs` |
| Random Forest | `random_forest_003` | `n_estimators=300`, `max_depth=16`, `min_samples_leaf=10`, `max_features=sqrt` |
| Gradient Boosting | `gradient_boosting_021` | `learning_rate=0.1`, `n_estimators=200`, `max_depth=2`, `subsample=1.0` |
| MLP | `mlp_003` | `hidden_layer_sizes=[32]`, `alpha=0.001`, `max_iter=200` |

Evidence: `models/trained/logistic_regression_metadata.json` (and the corresponding metadata files for the other three models).

## Input and Encoding

- Feature order for all models: `agegrp, bmi, agefirst, menopaus, surgmeno, hrt, nrelbc, density, brstproc, lastmamm`
- One-Hot encoder output width: 43
- Ordinal-Tree encoder output width: 26
- MLP includes `StandardScaler` inside its `SampleWeightPipeline`

Evidence: `models/preprocessing/preprocessing_metadata.json`, `src/training/config.py`.

## Evaluation Results

### Training CV (from `model_comparison.csv`)

| Model | CV PR-AUC (mean ± std) | CV ROC-AUC (mean ± std) | CV Brier (mean ± std, from -neg Brier) |
|---|---:|---:|---:|
| logistic_regression | 0.0082903217 ± 0.0003717784 | 0.6308427726 ± 0.0060869493 | 0.0048546007 ± 0.0000023470 |
| gradient_boosting | 0.0082913360 ± 0.0003353842 | 0.6319280055 ± 0.0039187848 | 0.0048545909 ± 0.0000021604 |
| mlp | 0.0068142444 ± 0.0004433911 | 0.5921166497 ± 0.0151021729 | 0.0048625204 ± 0.0000032341 |
| random_forest | 0.0054117301 ± 0.0001446913 | 0.5129516181 ± 0.0095022749 | 0.0048726834 ± 0.0000020911 |
### Validation (from `model_comparison.csv`)

| Model | Validation PR-AUC | Validation ROC-AUC | Validation Brier | Validation BSS |
|---|---:|---:|---:|---:|
| logistic_regression | 0.0078848623 | 0.6242263499 | 0.0048549828 | 0.0009720912 |
| gradient_boosting | 0.0078455035 | 0.6269441674 | 0.0048549638 | 0.0009760016 |
| mlp | 0.0069350720 | 0.5944424307 | 0.0048609369 | -0.0002531039 |
| random_forest | 0.0052653633 | 0.5077782931 | 0.0048596865 | 0.0000041929 |
Note: weighted prevalence is about 0.49%, so PR-AUC must be read against that baseline. Recall at threshold 0.5 is 0 by construction at this prevalence and is not a model defect.

Evidence: `models/trained/logistic_regression_metadata.json`, `outputs/training/metrics/model_comparison.csv`.

## Prediction Interface

- Models expose `predict_proba`.
- Positive class is `cancer=1`.
- Inference uses `positive_class_probability()` from `src/training/cv.py`.

Evidence: `SUPMT.py`, `src/training/cv.py`.

## Limitations

- Very low prevalence (~0.49%) makes PR-AUC interpretation important.
- Recall at threshold 0.5 is 0 for all models in validation.
- `surgmeno`/`hrt` code 9 is unresolved Unknown/NA.
- No clinical validation or deployment certification is present.