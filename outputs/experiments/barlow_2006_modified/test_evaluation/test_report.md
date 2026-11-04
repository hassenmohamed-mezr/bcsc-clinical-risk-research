# Final Test Evaluation (one-time, user-authorized)

This is the single authorized Test evaluation. It deviates from the project's no-Test rule by explicit user decision. No model was refit; frozen artifacts were scored on test.csv.

Test rows=100,195, sum(count)=597,859, weighted prevalence=0.480214%.
Excluded from Barlow combined (menopaus==9): 5,716 rows, sum(count)=45,513.

| Model | Rows | ROC-AUC | PR-AUC | Brier | BSS | Log loss | Recall | Specificity | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| premenopausal | 14,846 | 0.621831 | 0.004995 | 0.00303130 | 0.009986 | 0.02038619 | 0.000000 | 1.000000 | N/A |
| postmenopausal | 79,633 | 0.617000 | 0.008693 | 0.00557522 | 0.017890 | 0.03421224 | 0.000000 | 1.000000 | N/A |
| combined_two_model | 94,479 | 0.635679 | 0.008260 | 0.00492126 | 0.016258 | 0.03065799 | 0.000000 | 1.000000 | N/A |
| unified_logistic_regression_full_test | 100,195 | 0.639755 | 0.008215 | 0.00477317 | 0.017851 | 0.02984482 | 0.000000 | 1.000000 | N/A |

Unified model scored on the FULL test set (including menopa==9); Barlow combined covers only menopa in {0,1}. Comparison is descriptive.