# SUPMT Inference Test Report

## A. Summary

- Total cases: 60
- Expected accepted: 40
- Expected rejected: 20
- Actual accepted: 40
- Actual rejected: 20
- Technical passes: 60
- Technical failures: 0
- Reference range deviations: 9
- Unexpected accepts: 0
- Unexpected rejects: 0
- Unexpected errors: 0
- Other failures: 0
- Technical success rate: 100.00%
- Test duration: 1.18 seconds

## B. Detailed Results

| Case ID | Description | Expected | Actual | LR % | GB % | MLP % | RF % | Weighted % | Range | Range Check | Status | Error |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| BCSC-001 | Premenopausal baseline | accept | accept | 0.34 | 0.35 | 0.49 | 0.47 | 0.41 | 0.25-0.80 | IN_RANGE | PASS | - |
| BCSC-002 | Postmenopausal baseline | accept | accept | 2.75 | 2.30 | 3.00 | 0.57 | 2.16 | 1.20-3.50 | IN_RANGE | PASS | - |
| BCSC-003 | Unknown menopause, partial fields | accept | accept | 0.34 | 0.38 | 0.46 | 0.47 | 0.41 | 0.25-0.80 | IN_RANGE | PASS | - |
| BCSC-004 | Minimum age | accept | accept | 0.18 | 0.08 | 0.36 | 0.46 | 0.27 | 0.10-0.40 | IN_RANGE | PASS | - |
| BCSC-005 | Maximum age | accept | accept | 3.77 | 6.21 | 4.21 | 0.63 | 3.70 | 1.50-5.00 | IN_RANGE | PASS | - |
| BCSC-006 | BMI category boundary 25 | accept | accept | 0.24 | 0.24 | 0.41 | 0.46 | 0.34 | 0.20-0.70 | IN_RANGE | PASS | - |
| BCSC-007 | BMI category boundary 30 | accept | accept | 0.48 | 0.53 | 0.50 | 0.48 | 0.50 | 0.30-1.20 | IN_RANGE | PASS | - |
| BCSC-008 | BMI category boundary 35 | accept | accept | 1.70 | 1.33 | 0.63 | 0.53 | 1.05 | 0.80-2.50 | IN_RANGE | PASS | - |
| BCSC-009 | BMI high | accept | accept | 0.22 | 0.19 | 0.37 | 0.46 | 0.31 | 0.20-0.80 | IN_RANGE | PASS | - |
| BCSC-010 | BMI fractional low | accept | accept | 0.38 | 0.21 | 0.37 | 0.46 | 0.36 | 0.20-0.80 | IN_RANGE | PASS | - |
| BCSC-011 | Natural menopause, HRT on | accept | accept | 1.09 | 0.90 | 0.71 | 0.51 | 0.80 | 1.00-3.50 | OUT_OF_RANGE | REFERENCE_RANGE_DEVIATION | - |
| BCSC-012 | Surgical menopause, HRT on | accept | accept | 0.57 | 0.52 | 0.48 | 0.49 | 0.52 | 0.50-2.00 | IN_RANGE | PASS | - |
| BCSC-013 | Surgical menopause, HRT off | accept | accept | 1.69 | 1.40 | 1.00 | 0.56 | 1.16 | 1.00-3.50 | IN_RANGE | PASS | - |
| BCSC-014 | Family history 2+ | accept | accept | 0.39 | 0.48 | 0.50 | 0.51 | 0.47 | 0.30-1.20 | IN_RANGE | PASS | - |
| BCSC-015 | Family history integer 2 | accept | accept | 0.50 | 0.60 | 0.40 | 0.47 | 0.49 | 0.40-1.50 | IN_RANGE | PASS | - |
| BCSC-016 | Density fatty | accept | accept | 0.16 | 0.07 | 0.36 | 0.45 | 0.26 | 0.10-0.50 | IN_RANGE | PASS | - |
| BCSC-017 | Density scattered | accept | accept | 0.63 | 0.48 | 0.44 | 0.50 | 0.51 | 0.30-1.20 | IN_RANGE | PASS | - |
| BCSC-018 | Density heterogeneously dense | accept | accept | 0.53 | 0.47 | 0.58 | 0.47 | 0.51 | 0.50-2.00 | IN_RANGE | PASS | - |
| BCSC-019 | Density extremely dense | accept | accept | 0.91 | 1.01 | 0.64 | 0.52 | 0.77 | 0.70-2.50 | IN_RANGE | PASS | - |
| BCSC-020 | Nulliparous postmenopausal | accept | accept | 0.58 | 0.57 | 0.48 | 0.47 | 0.53 | 0.70-2.00 | OUT_OF_RANGE | REFERENCE_RANGE_DEVIATION | - |
| BCSC-021 | Unknown BMI | accept | accept | 0.40 | 0.43 | 0.49 | 0.46 | 0.45 | 0.30-1.20 | IN_RANGE | PASS | - |
| BCSC-022 | Unknown family history | accept | accept | 0.54 | 0.65 | 0.52 | 0.50 | 0.55 | 1.00-3.50 | OUT_OF_RANGE | REFERENCE_RANGE_DEVIATION | - |
| BCSC-023 | Unknown prior procedure | accept | accept | 0.19 | 0.17 | 0.38 | 0.45 | 0.30 | 0.20-0.80 | IN_RANGE | PASS | - |
| BCSC-024 | Unknown mammogram | accept | accept | 1.38 | 1.20 | 0.95 | 0.49 | 1.01 | 1.00-3.50 | IN_RANGE | PASS | - |
| BCSC-025 | Unknown age at first birth | accept | accept | 0.13 | 0.15 | 0.38 | 0.46 | 0.28 | 0.20-0.80 | IN_RANGE | PASS | - |
| BCSC-026 | Unknown menopause type | accept | accept | 0.54 | 0.47 | 0.42 | 0.47 | 0.48 | 0.50-2.00 | OUT_OF_RANGE | REFERENCE_RANGE_DEVIATION | - |
| BCSC-027 | Unknown HRT | accept | accept | 0.88 | 0.99 | 0.53 | 0.50 | 0.73 | 0.80-3.00 | OUT_OF_RANGE | REFERENCE_RANGE_DEVIATION | - |
| BCSC-028 | All optional fields omitted | accept | accept | 0.40 | 0.35 | 0.40 | 0.47 | 0.40 | 0.30-1.00 | IN_RANGE | PASS | - |
| BCSC-029 | All fields null except age | accept | accept | 0.40 | 0.35 | 0.40 | 0.47 | 0.40 | 0.30-1.00 | IN_RANGE | PASS | - |
| BCSC-030 | Menopause unknown age 53 | accept | accept | 0.34 | 0.38 | 0.46 | 0.47 | 0.41 | 0.25-0.80 | IN_RANGE | PASS | - |
| BCSC-031 | Age 35 boundary | accept | accept | 0.16 | 0.07 | 0.36 | 0.45 | 0.26 | 0.10-0.50 | IN_RANGE | PASS | - |
| BCSC-032 | Age 84 boundary | accept | accept | 3.86 | 6.66 | 11.48 | 0.64 | 5.66 | 2.00-5.00 | OUT_OF_RANGE | REFERENCE_RANGE_DEVIATION | - |
| BCSC-033 | Age 55 unknown menopause | accept | accept | 0.48 | 0.49 | 0.56 | 0.49 | 0.50 | 0.40-1.50 | IN_RANGE | PASS | - |
| BCSC-034 | False positive mammogram | accept | accept | 0.88 | 0.70 | 0.47 | 0.55 | 0.65 | 0.30-1.20 | IN_RANGE | PASS | - |
| BCSC-035 | No prior procedure | accept | accept | 0.42 | 0.41 | 0.40 | 0.47 | 0.42 | 0.50-2.00 | OUT_OF_RANGE | REFERENCE_RANGE_DEVIATION | - |
| BCSC-036 | Prior procedure true | accept | accept | 0.55 | 0.94 | 0.58 | 0.50 | 0.64 | 0.30-1.20 | IN_RANGE | PASS | - |
| BCSC-037 | HRT false natural | accept | accept | 0.36 | 0.31 | 0.38 | 0.48 | 0.38 | 0.60-2.00 | OUT_OF_RANGE | REFERENCE_RANGE_DEVIATION | - |
| BCSC-038 | HRT true natural | accept | accept | 3.28 | 2.68 | 2.47 | 0.59 | 2.26 | 1.00-3.50 | IN_RANGE | PASS | - |
| BCSC-039 | All known fields, low BMI category | accept | accept | 0.27 | 0.13 | 0.37 | 0.45 | 0.31 | 0.10-0.50 | IN_RANGE | PASS | - |
| BCSC-040 | All known fields, older age | accept | accept | 2.79 | 4.26 | 26.06 | 0.59 | 8.42 | 2.00-5.00 | OUT_OF_RANGE | REFERENCE_RANGE_DEVIATION | - |
| BCSC-041 | Age below range | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age: must be between 35 and 84, got 30.0 |
| BCSC-042 | Age above range | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age: must be between 35 and 84, got 90.0 |
| BCSC-043 | Age below minimum by one | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age: must be between 35 and 84, got 34.0 |
| BCSC-044 | Age above maximum by one | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age: must be between 35 and 84, got 85.0 |
| BCSC-045 | Age missing | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age: expected a number, got None |
| BCSC-046 | Age string | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age: expected a number, got '42' |
| BCSC-047 | Age non-integer | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age: must be an integer number of years, got 42.5 |
| BCSC-048 | BMI below minimum | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: bmi: must be >= 10, got 9.9 |
| BCSC-049 | BMI negative | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: bmi: must be >= 10, got -1.0 |
| BCSC-050 | BMI nonnumeric | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: bmi: expected a number, got '27.5' |
| BCSC-051 | Birth status invalid string | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: has_given_birth: expected True/False/None, got 'yes' |
| BCSC-052 | Birth status integer not boolean | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: has_given_birth: expected True/False/None, got 1 |
| BCSC-053 | Contradictory nulliparity and age at first birth | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age_at_first_birth must be None when has_given_birth is False |
| BCSC-054 | Age at first birth without birth status | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age_at_first_birth requires has_given_birth to be True |
| BCSC-055 | Age at first birth zero | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age_at_first_birth: must be > 0, got 0.0 |
| BCSC-056 | Age at first birth exceeds current age | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age_at_first_birth: cannot exceed current age, got 45.0 |
| BCSC-057 | Age at first birth string | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: age_at_first_birth: expected a number, got '26' |
| BCSC-058 | Invalid menopause type | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: menopause_type: unsupported value 'medical' |
| BCSC-059 | Menopause type supplied while premenopausal | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: menopause_type is only allowed when is_postmenopausal is True |
| BCSC-060 | HRT supplied while premenopausal | reject | reject | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | InputMappingError: invalid patient input: currently_on_hrt is only allowed when is_postmenopausal is True |

## C. Reference Range Deviations

| Case ID | Actual Weighted % | Reference Range | Difference from nearest bound |
|---|---:|---|---:|
| BCSC-011 | 0.80 | 1.00-3.50 | 0.20 |
  - Factors: age=65, bmi=28.4, has_given_birth=True, age_at_first_birth=27, is_postmenopausal=True, menopause_type=natural, currently_on_hrt=True, first_degree_relatives_with_breast_cancer=1, breast_density=heterogeneously dense, prior_breast_procedure=False, last_mammogram_result=negative
  - Note: deviation from an approximate reference range, not evidence of a medical or model error.
| BCSC-020 | 0.53 | 0.70-2.00 | 0.17 |
  - Factors: age=70, bmi=27.1, has_given_birth=False, age_at_first_birth=None, is_postmenopausal=True, menopause_type=natural, currently_on_hrt=False, first_degree_relatives_with_breast_cancer=0, breast_density=scattered fibroglandular, prior_breast_procedure=False, last_mammogram_result=negative
  - Note: deviation from an approximate reference range, not evidence of a medical or model error.
| BCSC-022 | 0.55 | 1.00-3.50 | 0.45 |
  - Factors: age=59, bmi=30.5, has_given_birth=True, age_at_first_birth=30, is_postmenopausal=True, menopause_type=natural, currently_on_hrt=False, first_degree_relatives_with_breast_cancer=None, breast_density=extremely dense, prior_breast_procedure=False, last_mammogram_result=negative
  - Note: deviation from an approximate reference range, not evidence of a medical or model error.
| BCSC-026 | 0.48 | 0.50-2.00 | 0.02 |
  - Factors: age=60, bmi=28.8, has_given_birth=True, age_at_first_birth=27, is_postmenopausal=True, menopause_type=None, currently_on_hrt=False, first_degree_relatives_with_breast_cancer=1, breast_density=scattered fibroglandular, prior_breast_procedure=False, last_mammogram_result=negative
  - Note: deviation from an approximate reference range, not evidence of a medical or model error.
| BCSC-027 | 0.73 | 0.80-3.00 | 0.07 |
  - Factors: age=67, bmi=31.0, has_given_birth=True, age_at_first_birth=33, is_postmenopausal=True, menopause_type=natural, currently_on_hrt=None, first_degree_relatives_with_breast_cancer=0, breast_density=heterogeneously dense, prior_breast_procedure=False, last_mammogram_result=negative
  - Note: deviation from an approximate reference range, not evidence of a medical or model error.
| BCSC-032 | 5.66 | 2.00-5.00 | 0.66 |
  - Factors: age=84, bmi=39.9, has_given_birth=True, age_at_first_birth=40, is_postmenopausal=True, menopause_type=natural, currently_on_hrt=False, first_degree_relatives_with_breast_cancer=2, breast_density=extremely dense, prior_breast_procedure=True, last_mammogram_result=false positive
  - Note: deviation from an approximate reference range, not evidence of a medical or model error.
| BCSC-035 | 0.42 | 0.50-2.00 | 0.08 |
  - Factors: age=57, bmi=32.0, has_given_birth=False, age_at_first_birth=None, is_postmenopausal=True, menopause_type=natural, currently_on_hrt=False, first_degree_relatives_with_breast_cancer=0, breast_density=scattered fibroglandular, prior_breast_procedure=False, last_mammogram_result=negative
  - Note: deviation from an approximate reference range, not evidence of a medical or model error.
| BCSC-037 | 0.38 | 0.60-2.00 | 0.22 |
  - Factors: age=69, bmi=24.5, has_given_birth=True, age_at_first_birth=24, is_postmenopausal=True, menopause_type=natural, currently_on_hrt=False, first_degree_relatives_with_breast_cancer=1, breast_density=almost entirely fatty, prior_breast_procedure=False, last_mammogram_result=negative
  - Note: deviation from an approximate reference range, not evidence of a medical or model error.
| BCSC-040 | 8.42 | 2.00-5.00 | 3.42 |
  - Factors: age=79, bmi=26.4, has_given_birth=True, age_at_first_birth=25, is_postmenopausal=True, menopause_type=surgical, currently_on_hrt=False, first_degree_relatives_with_breast_cancer=2+, breast_density=extremely dense, prior_breast_procedure=True, last_mammogram_result=false positive
  - Note: deviation from an approximate reference range, not evidence of a medical or model error.

## D. Validation Failures

No validation failures were observed.

## E. Model Output Diagnostics

| Model | Mean % | Min % | Max % |
|---|---:|---:|---:|
| logistic_regression | 0.8930 | 0.1313 | 3.8635 |
| gradient_boosting | 1.0030 | 0.0748 | 6.6617 |
| mlp | 1.6113 | 0.3593 | 26.0556 |
| random_forest | 0.4979 | 0.4497 | 0.6382 |
| weighted_average | 1.0013 | 0.2612 | 8.4227 |

- Accepted cases with all four model outputs: 40
- Invalid (non-finite or out-of-range) model outputs: 0
- Maximum weighted-average recomputation difference: 0.000e+00

## F. Final Assessment

1. Pipeline correctness: all accepted cases produced finite probabilities from all four models through the existing human-input mapping, RawInputMapper, per-model encoders, and calibrated artifacts.
2. Input validation: invalid cases were rejected through InputMappingError; unexpected accepts/rejects were 0/0.
3. Weighted average adherence: recomputed averages matched returned values within tolerance (max difference 0.000e+00); default weights are all 1.0.
4. Reference ranges: 9 of 40 accepted cases fell outside the approximate reference ranges. These are deviations from rough estimates, not confirmed model errors.
5. This test cannot establish medical accuracy, clinical calibration, or diagnostic validity. It verifies runtime behavior, validation, output shape/range, and internal consistency only.
