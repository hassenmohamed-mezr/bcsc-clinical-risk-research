# Experiments and Results

## Evaluation Protocol

- Training-only 5-fold weighted CV
- Weighted PR-AUC primary selection metric
- Sigmoid calibration, `ensemble=False`, Training data only
- Validation used for final comparison
- Test evaluated once after freezing, by explicit authorization

Evidence: `outputs/training/reports/04_evaluation_protocol.md`, `07_final_experiment_decisions.md`.

## Model Comparison

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