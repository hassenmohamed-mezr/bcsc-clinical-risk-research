# Proposed Model Weights

Recommendation: `lr_gb`

Paired bootstrap CI vs all4 includes 0 for all candidates: no

Candidate PR-AUC point estimates:
- all4: 0.0077945834
- lr_gb: 0.0079183227
- lr_gb_mlp: 0.0079082770
- lr_only: 0.0078848623

Paired bootstrap CI table:

candidate   vs  metric  median_diff  ci_lower  ci_upper
    lr_gb all4 roc_auc     0.005226  0.001970  0.008160
    lr_gb all4  pr_auc     0.000136 -0.000702  0.000561
lr_gb_mlp all4 roc_auc     0.003539  0.002806  0.004330
lr_gb_mlp all4  pr_auc     0.000113  0.000078  0.000154
  lr_only all4 roc_auc     0.002718 -0.002324  0.007684
  lr_only all4  pr_auc     0.000103 -0.000852  0.000530

Proposed SUPMT_CONFIG.MODEL_WEIGHTS diff:

```python
MODEL_WEIGHTS = {
    "logistic_regression": 1.0,
    "gradient_boosting": 1.0,
    "mlp": 0.0,
    "random_forest": 0.0,
}
```

Before/after on 40 accepted test cases:

| Case | Before WA | After WA |
|---|---:|---:|
| BCSC-001 | 0.4126% | 0.3491% |
| BCSC-002 | 2.1573% | 2.5259% |
| BCSC-003 | 0.4132% | 0.3620% |
| BCSC-004 | 0.2698% | 0.1296% |
| BCSC-005 | 3.7038% | 4.9904% |
| BCSC-006 | 0.3387% | 0.2420% |
| BCSC-007 | 0.4976% | 0.5046% |
| BCSC-008 | 1.0493% | 1.5168% |
| BCSC-009 | 0.3120% | 0.2072% |
| BCSC-010 | 0.3554% | 0.2938% |
| BCSC-011 | 0.8039% | 0.9975% |
| BCSC-012 | 0.5155% | 0.5475% |
| BCSC-013 | 1.1644% | 1.5465% |
| BCSC-014 | 0.4690% | 0.4336% |
| BCSC-015 | 0.4919% | 0.5491% |
| BCSC-016 | 0.2612% | 0.1173% |
| BCSC-017 | 0.5121% | 0.5563% |
| BCSC-018 | 0.5139% | 0.4994% |
| BCSC-019 | 0.7693% | 0.9579% |
| BCSC-020 | 0.5250% | 0.5753% |
| BCSC-021 | 0.4465% | 0.4166% |
| BCSC-022 | 0.5533% | 0.5946% |
| BCSC-023 | 0.2956% | 0.1764% |
| BCSC-024 | 1.0051% | 1.2875% |
| BCSC-025 | 0.2801% | 0.1391% |
| BCSC-026 | 0.4753% | 0.5072% |
| BCSC-027 | 0.7264% | 0.9372% |
| BCSC-028 | 0.4041% | 0.3750% |
| BCSC-029 | 0.4041% | 0.3750% |
| BCSC-030 | 0.4132% | 0.3620% |
| BCSC-031 | 0.2612% | 0.1173% |
| BCSC-032 | 5.6619% | 5.2626% |
| BCSC-033 | 0.5032% | 0.4805% |
| BCSC-034 | 0.6513% | 0.7923% |
| BCSC-035 | 0.4244% | 0.4132% |
| BCSC-036 | 0.6401% | 0.7430% |
| BCSC-037 | 0.3831% | 0.3354% |
| BCSC-038 | 2.2555% | 2.9775% |
| BCSC-039 | 0.3077% | 0.2016% |
| BCSC-040 | 8.4227% | 3.5231% |

AWAITING APPROVAL