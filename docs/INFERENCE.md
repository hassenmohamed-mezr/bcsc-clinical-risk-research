# Inference

## Entry Point

- `SUPMT.py`
- Configuration: `SUPMT_CONFIG.py`

## Artifact Loading

`SUPMT.py` loads once and caches:

- `RawInputMapper` from `models/preprocessing/raw_input_mapping.json`
- per-model preprocessing pipelines from metadata `encoder_artifact`
- calibrated model artifacts from `models/trained/logistic_regression_calibrated.joblib` (and corresponding artifacts for the other models)

It validates feature order, artifact hashes, and encoder family compatibility.

Evidence: `SUPMT.py::_load_system`.

## Input Schema

Human input fields:

- `age`
- `bmi`
- `has_given_birth`
- `age_at_first_birth`
- `is_postmenopausal`
- `menopause_type`
- `currently_on_hrt`
- `first_degree_relatives_with_breast_cancer`
- `breast_density`
- `prior_breast_procedure`
- `last_mammogram_result`

Missing fields are normalized to `None`.

Evidence: `SUPMT.py::_normalize_patient`, `src/preprocessing/raw_input_mapping.py`.

## Validation

Validation is implemented in `human_to_interface_record()` and rejects:

- out-of-range age
- non-integer age
- BMI below 10
- unsupported strings
- contradictory birth fields
- invalid conditional menopause fields
- invalid boolean fields

Evidence: `src/preprocessing/raw_input_mapping.py`.

## Mapping and Preprocessing

1. Human fields → interface codes
2. `RawInputMapper.transform_frame()` → BCSC codes
3. Per-model `ColumnTransformer.transform()` → encoded matrix

Encoders are loaded, not refitted.

## Model Predictions

Each calibrated model returns `predict_proba`; the positive class (`cancer=1`) probability is extracted with `positive_class_probability()`.

## Aggregation

Default weights are all `1.0` in `SUPMT_CONFIG.MODEL_WEIGHTS`.

```python
weighted_average = sum(predictions[name] * MODEL_WEIGHTS[name]) / sum(MODEL_WEIGHTS.values())
```

Weights are validated for finiteness, non-negativity, and positive sum.

## Final Output

```text
Logistic Regression: XX.XX%
Gradient Boosting: XX.XX%
MLP: XX.XX%
Random Forest: XX.XX%

Weighted Average: XX.XX%
```

No diagnosis, risk category, or treatment recommendation is produced.

## Error Handling

- Missing/invalid artifacts raise `SUPMTError`.
- Invalid patient input raises `InputMappingError`.
- Preprocessing/prediction failures are wrapped in `SUPMTError`.
- No silent fallback to dummy predictions.

## Runtime Requirements

- Python 3.15.0 (observed)
- scikit-learn 1.9.1
- NumPy 2.5.4
- pandas 3.1.0
- joblib 1.6.0

## Known Limitations

- Single-patient inference is the intended use case.
- No service/API wrapper.
- Artifact loading assumes repository-relative paths.
- `surgmeno`/`hrt` code 9 remains combined Unknown/NA.