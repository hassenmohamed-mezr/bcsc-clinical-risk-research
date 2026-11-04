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

Reported validation metrics (from saved metadata/comparison artifacts):

| Model | Validation PR-AUC | Validation ROC-AUC | Validation Brier |
|---|---:|---:|---:|
| Logistic Regression | 0.00829032168635534 | 0.6308427726120697 | 0.00485460074139491 |
| Gradient Boosting | 0.008291336041341874 | 0.631928005505103 | 0.004854590866248082 |
| MLP | 0.0068142443535213985 | 0.5921166497226285 | 0.004862520383702512 |
| Random Forest | 0.005411730104276897 | 0.5129516180738664 | 0.004872683445716941 |

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