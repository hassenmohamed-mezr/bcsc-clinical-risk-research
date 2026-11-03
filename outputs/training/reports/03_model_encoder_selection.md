# Model Encoder Selection

Date: 2026-11-04. Scope: choose between the existing `onehot_pipeline.joblib` (43 columns) and `ordinal_tree_pipeline.joblib` (26 columns) for four planned model families. No model was trained, no encoder was modified or refit, and no test data was used.

## Recommendation

| Model family | Existing pipeline | Scaling | Additional transformation | Rationale |
|---|---|---|---|---|
| Logistic Regression | `onehot` | Not required; optional train-fitted scaler | None | Avoids false linear effects from ordinal codes; all 43 inputs are 0/1 and comparable. |
| Random Forest | `ordinal_tree` | Not required | None | Trees use category order, not numeric distance; compact 26-column representation. |
| Gradient Boosting | `ordinal_tree` (numeric-matrix implementations) | Not required | None for the existing pipeline | Trees split ordinal ranks without assuming equal spacing; implementation caveat below. |
| MLP Neural Network | `onehot` | Required | `StandardScaler` fitted on Training output only | Avoids artificial ordinal distances; scaling stabilises optimisation. |

## Evaluation

### Categorical representation

- `onehot` represents every category explicitly, including Unknown (9), with no numeric ordering. It is the safest representation for models that use numeric distances or linear coefficients.
- `ordinal_tree` uses documented ranks for `agegrp`, `bmi`, `nrelbc`, and `density`, and one-hot for the six nominal features. Ranks preserve order but not spacing.
- For trees, equal-spacing assumptions are largely irrelevant: splits depend on order and thresholds, not the size of the gap between codes.
- For Logistic Regression and MLP, ordinal ranks would imply linear or distance-based effects that the data does not establish. This is why `onehot` is preferred for both.

### Model-by-model suitability

**Logistic Regression — `onehot`.** A linear model assigns one coefficient per encoded column. Ordinal ranks would force a constant change per category step and could create false effects for `bmi`, `density`, and `nrelbc`. One-hot avoids this. Scaling is not required because all columns are binary and therefore already on a common scale; it is optional if a solver or regularisation experiment needs standardised inputs.

**Random Forest — `ordinal_tree`.** Random forests are invariant to monotonic feature transformations. The ordinal ranks preserve the documented order of the four ordered features, the Unknown flags isolate missingness, and nominal features remain one-hot. The 26-column matrix is more compact than 43 columns. No scaling is needed. `onehot` is a valid fallback but offers no advantage for this model.

**Gradient Boosting — `ordinal_tree` for numeric-matrix implementations.** The same tree argument applies to sklearn `GradientBoostingClassifier` and `HistGradientBoostingClassifier`, XGBoost, and LightGBM when they receive a numeric matrix. No scaling is needed.
Implementation caveat: CatBoost, LightGBM, and `HistGradientBoostingClassifier` can optionally treat raw categorical codes natively. That would require a different preprocessing route; raw BCSC codes must not be passed as ordinary numeric features without declaring them categorical. The existing `ordinal_tree` is the common, implementation-neutral choice; `onehot` is the fallback when a specific implementation performs better with explicit indicator columns.

**MLP Neural Network — `onehot` plus scaling.** The MLP can learn nonlinear relationships, but feeding ordinal ranks still imposes an arbitrary numeric geometry, especially for `bmi`, `density`, and `nrelbc`. One-hot gives the network explicit category inputs. A `StandardScaler` is required for stable optimisation; it must be fitted on the encoded Training matrix only and applied unchanged to Validation and Test. The existing artifacts contain no scaler, so this is an additional downstream step, not a change to either encoder.

### Scaling and transformations

- **Logistic Regression:** no mandatory scaling for the 43 binary columns. If scaling is added for solver or regularisation reasons, fit it on Training only.
- **Random Forest / Gradient Boosting:** no scaling; tree splits are invariant to monotonic scaling.
- **MLP:** `StandardScaler` required; fit on `onehot` Training output only. Do not fit on Validation or Test.
- **All models:** no PCA, feature selection, or imputation is required at this stage; the existing encoders already preserve Unknown as explicit information.
- All four sklearn candidates accept `sample_weight`, so `count` can be supplied during training without row expansion.

## Unresolved decisions

1. Select the final Gradient Boosting implementation and decide whether to use native categorical handling or the existing numeric `ordinal_tree` route.
2. Decide whether the MLP scaler should be a separate artifact or wrapped with the encoder in a new serialised composite pipeline. Either way, it must be fitted on Training only.
3. Decide whether Logistic Regression should use the optional scaler; it is not required for the current binary one-hot matrix.
4. Resolve the pending `surgmeno`/`hrt` Unknown vs Not Applicable recode before final model comparison.
5. Confirm class-imbalance handling separately; it is a model-training decision, not an encoder choice.

Last updated: 2026-11-04
