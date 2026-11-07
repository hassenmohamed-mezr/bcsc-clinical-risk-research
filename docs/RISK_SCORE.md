# Risk Score

## Formula

The risk score is a rank-based relative-risk indicator, not a probability:

```text
score = 100 * Phi((ln(p) - mu) / sigma)
```

where:
- `p` = Logistic Regression probability (`p_lr`)
- `mu = -5.4080596412098485`
- `sigma = 0.4751694327986758`
- `Phi` = standard normal CDF

Fitted parameters:
- Weighted mean of `ln(probability)` over validation LR predictions, weights = `count`
- Weighted standard deviation (population, ddof=0)
- Weighted median probability: `0.44495941817186624` (~0.445%)

## Fitting and Refitting

Parameters are fitted by `risk_score/score.py::fit_parameters()` from:
`outputs/training/predictions/val_predictions_all.csv`

Reproduction checks:
- LR weighted ROC-AUC = 0.6242 ± 0.0001
- Weighted prevalence = 0.004884 ± 0.000001
- mu = -5.408 ± 0.005
- sigma = 0.475 ± 0.005
- Weighted median probability = 0.448% ± 0.005 percentage points

Parameters must be refit if the LR model or validation distribution changes.

## Band Table (Validation, Smooth Score)

| Band | Weighted share | Weighted observed rate | Ratio to prevalence | 95% CI (observed rate) |
|---|---:|---:|---:|---|
| very_low | 27.49% | 0.2526% | 0.5173 | 0.1932% - 0.3276% |
| average | 29.87% | 0.4227% | 0.8655 | 0.3413% - 0.5246% |
| above_average | 26.59% | 0.5782% | 1.1840 | 0.4921% - 0.6777% |
| high | 16.05% | 0.8656% | 1.7725 | 0.7377% - 1.0093% |

Band boundaries:
- 24.9 -> very_low
- 25.0 -> average
- 59.9 -> average
- 60.0 -> above_average
- 84.9 -> above_average
- 85.0 -> high
- 100 -> high

## Mapping Examples

| LR probability | Smooth score | Band | Empirical percentile |
|---:|---:|---|---:|
| 0.20% | 4.48 | very_low | 3.81% |
| 0.35% | 30.16 | average | 32.47% |
| 0.50% | 59.13 | average | 56.76% |
| 0.75% | 86.09 | high | 85.59% |
| 1.00% | 95.45 | high | 96.28% |
| 1.50% | 99.45 | high | 99.69% |
| 2.00% | 99.92 | high | 99.96% |
| 3.00% | 99.997 | high | 99.999% |

Smooth vs empirical percentile: mean absolute difference 1.4187 points, max 2.9880 points.

## Output semantics

The inference wrapper exposes two different values:

- `weighted_average_pct`: estimated model probability from the configured
  weighted average of all active models. This is the value shown as
  `Estimated Risk` in the desktop interface.
- `risk_score`: relative percentile score derived from the Logistic Regression
  probability. It is shown separately as `Relative Risk Score` and is not a
  probability.

The configured models and their weighted average are implemented in
`SUPMT.py`; the score calibration remains tied to the Logistic Regression
validation distribution.

## Usage

```python
from SUPMT_score import predict_risk_score

result = predict_risk_score(patient)
print(result["risk_score"], result["risk_band_label"])
```

## Limitations
- The score is a relative rank indicator, not an infection probability.
- Scores above about 99.5 are saturated; very high-risk inputs are not distinguishable in display (e.g. 2% and 3% both near 100).
- The top band corresponds to roughly 1% observed annual risk, about twice the reference average.
- The function is a smooth approximation of the empirical percentile.
- Parameters are tied to the current LR and validation distribution.
- No clinical validation.
- Bands are product choices.
- Scores change only at age-group boundaries because age is binned.
- Known non-monotonic spots for LR: age 39->40, 79->80, heterogeneously dense -> extremely dense.