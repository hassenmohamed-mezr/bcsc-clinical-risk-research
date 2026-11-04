# Experiments and Results

## Evaluation Protocol

- Training-only 5-fold weighted CV
- Weighted PR-AUC primary selection metric
- Sigmoid calibration, `ensemble=False`, Training data only
- Validation used for final comparison
- Test evaluated once after freezing, by explicit authorization

Evidence: `outputs/training/reports/04_evaluation_protocol.md`, `07_final_experiment_decisions.md`.

## Model Comparison (Validation)

| Model | Encoder | Validation PR-AUC | Validation ROC-AUC | Validation Brier |
|---|---|---:|---:|---:|
| Logistic Regression | onehot | 0.00829032168635534 | 0.6308427726120697 | 0.00485460074139491 |
| Gradient Boosting | ordinal_tree | 0.008291336041341874 | 0.631928005505103 | 0.004854590866248082 |
| MLP | onehot | 0.0068142443535213985 | 0.5921166497226285 | 0.004862520383702512 |
| Random Forest | ordinal_tree | 0.005411730104276897 | 0.5129516180738664 | 0.004872683445716941 |

Evidence: `outputs/training/metrics/model_comparison.csv`, `models/trained/logistic_regression_metadata.json`.

## Modified Barlow Architecture

A separate experiment trains two unpenalized logistic regressions by menopausal status
(premenopausal and postmenopausal), excludes `race`/`Hispanic`, and excludes
`menopaus==9` rows. Results are documented in project reports.

Evidence: `outputs/experiments/barlow_2006_modified/reports/experiment_report.md`.

## Inference Runtime Test

`test_SUPMT_inference.py` ran 60 cases:

- 40 accepted
- 20 rejected
- 60 technical passes
- 9 reference range deviations

Evidence: `SUPMT_TEST_REPORT.md`, `SUPMT_TEST_RESULTS.csv`.

## Limitations of Results

- Low prevalence (~0.49%) requires careful metric interpretation.
- Recall at threshold 0.5 is 0 for all models in validation.
- Test results are runtime behavior checks, not clinical accuracy proof.
- Reference ranges in inference tests are approximate.