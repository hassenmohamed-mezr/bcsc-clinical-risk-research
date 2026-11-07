# TEST_SET_THRESHOLD_ANALYSIS

## Source
- Original held-out Test Set: data/04_splits/test.csv
- Rows: 100,195
- Ground truth: cancer column (2,536 positive)
- count ignored

## Probability sources
- logistic_regression, gradient_boosting, mlp, random_forest: model.predict_proba on per-model encoded test features
- weighted_average: current MODEL_WEIGHTS (all 1.0)

## logistic_regression

| Threshold | Total Predicted Positive | Actual Cancer | Actual Non-Cancer | Cancer Precision (%) | Below Threshold |
|---|---:|---:|---:|---:|---:|
| 100% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 90% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 80% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 70% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 60% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 50% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 40% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 30% | 0 | 0 | 0 | N/A (no positives) | 100195 |

## gradient_boosting

| Threshold | Total Predicted Positive | Actual Cancer | Actual Non-Cancer | Cancer Precision (%) | Below Threshold |
|---|---:|---:|---:|---:|---:|
| 100% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 90% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 80% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 70% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 60% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 50% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 40% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 30% | 0 | 0 | 0 | N/A (no positives) | 100195 |

## mlp

| Threshold | Total Predicted Positive | Actual Cancer | Actual Non-Cancer | Cancer Precision (%) | Below Threshold |
|---|---:|---:|---:|---:|---:|
| 100% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 90% | 1 | 0 | 1 | 0.00 | 100194 |
| 80% | 2 | 0 | 2 | 0.00 | 100193 |
| 70% | 2 | 0 | 2 | 0.00 | 100193 |
| 60% | 3 | 0 | 3 | 0.00 | 100192 |
| 50% | 5 | 0 | 5 | 0.00 | 100190 |
| 40% | 7 | 0 | 7 | 0.00 | 100188 |
| 30% | 10 | 0 | 10 | 0.00 | 100185 |

## random_forest

| Threshold | Total Predicted Positive | Actual Cancer | Actual Non-Cancer | Cancer Precision (%) | Below Threshold |
|---|---:|---:|---:|---:|---:|
| 100% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 90% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 80% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 70% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 60% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 50% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 40% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 30% | 0 | 0 | 0 | N/A (no positives) | 100195 |

## weighted_average

| Threshold | Total Predicted Positive | Actual Cancer | Actual Non-Cancer | Cancer Precision (%) | Below Threshold |
|---|---:|---:|---:|---:|---:|
| 100% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 90% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 80% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 70% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 60% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 50% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 40% | 0 | 0 | 0 | N/A (no positives) | 100195 |
| 30% | 0 | 0 | 0 | N/A (no positives) | 100195 |

## Direct Answers

- At 100%:
  - logistic_regression: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - gradient_boosting: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - mlp: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - random_forest: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - weighted_average: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)

- At 90%:
  - logistic_regression: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - gradient_boosting: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - mlp: No: 1 predicted high-risk were Non-Cancer. (Total=1, Actual Cancer=0)
  - random_forest: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - weighted_average: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)

- At 80%:
  - logistic_regression: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - gradient_boosting: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - mlp: No: 2 predicted high-risk were Non-Cancer. (Total=2, Actual Cancer=0)
  - random_forest: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - weighted_average: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)

- At 70%:
  - logistic_regression: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - gradient_boosting: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - mlp: No: 2 predicted high-risk were Non-Cancer. (Total=2, Actual Cancer=0)
  - random_forest: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - weighted_average: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)

- At 60%:
  - logistic_regression: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - gradient_boosting: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - mlp: No: 3 predicted high-risk were Non-Cancer. (Total=3, Actual Cancer=0)
  - random_forest: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - weighted_average: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)

- At 50%:
  - logistic_regression: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - gradient_boosting: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - mlp: No: 5 predicted high-risk were Non-Cancer. (Total=5, Actual Cancer=0)
  - random_forest: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - weighted_average: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)

- At 40%:
  - logistic_regression: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - gradient_boosting: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - mlp: No: 7 predicted high-risk were Non-Cancer. (Total=7, Actual Cancer=0)
  - random_forest: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - weighted_average: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)

- At 30%:
  - logistic_regression: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - gradient_boosting: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - mlp: No: 10 predicted high-risk were Non-Cancer. (Total=10, Actual Cancer=0)
  - random_forest: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)
  - weighted_average: No predicted positives; precision N/A. (Total=0, Actual Cancer=0)

## Notes
- Thresholds >=30% produce very few or zero positives because predicted probabilities are ~0.5%.
- Precision of 100% occurs only when Total Predicted Positive > 0 and all are cancer.
- No model retraining or weight changes were performed.
