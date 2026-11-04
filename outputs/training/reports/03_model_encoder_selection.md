# Model Encoder Selection

Date: 2026-11-04. Scope: choose between the existing `onehot_pipeline.joblib` (43 columns) and `ordinal_tree_pipeline.joblib` (26 columns) for four planned model families. No model was trained, no encoder was modified or refit, and no test data was used.

> **Status: open items resolved by report 07.** The final experiment decisions are in
> `outputs/training/reports/07_final_experiment_decisions.md`. Gradient Boosting is fixed to
> scikit-learn `GradientBoostingClassifier`, MLP scaling is a step inside the MLP model pipeline,
> Logistic Regression has no scaler, and `surgmeno`/`hrt` code 9 stays one category.

## Recommendation

| Model family | Existing pipeline | Scaling | Additional transformation | Rationale |
|---|---|---|---|---|
| Logistic Regression | `onehot` | None | None | Avoids false linear effects from ordinal codes; all 43 inputs are 0/1 and comparable, so no scaler is needed. |
| Random Forest | `ordinal_tree` | Not required | None | Trees use category order, not numeric distance; compact 26-column representation. |
| Gradient Boosting | `ordinal_tree` | Not required | None for the existing pipeline | Trees split ordinal ranks without assuming equal spacing; scikit-learn `GradientBoostingClassifier` is fixed for this phase. |
| MLP Neural Network | `onehot` | `StandardScaler` inside the model pipeline | `SampleWeightPipeline([StandardScaler, MLPClassifier])` | Avoids artificial ordinal distances; the scaler is fitted per CV training fold and serialized with the MLP. |

## Evaluation

### Categorical representation

- `onehot` represents every category explicitly, including Unknown (9), with no numeric ordering. It is the safest representation for models that use numeric distances or linear coefficients.
- `ordinal_tree` uses documented ranks for `agegrp`, `bmi`, `nrelbc`, and `density`, and one-hot for the six nominal features. Ranks preserve order but not spacing.
- For trees, equal-spacing assumptions are largely irrelevant: splits depend on order and thresholds, not the size of the gap between codes.
- For Logistic Regression and MLP, ordinal ranks would imply linear or distance-based effects that the data does not establish. This is why `onehot` is preferred for both.

### Model-by-model suitability

**Logistic Regression — `onehot`.** A linear model assigns one coefficient per encoded column. Ordinal ranks would force a constant change per category step and could create false effects for `bmi`, `density`, and `nrelbc`. One-hot avoids this. No scaler is used: all columns are binary indicators and therefore already on a common scale.

**Random Forest — `ordinal_tree`.** Random forests are invariant to monotonic feature transformations. The ordinal ranks preserve the documented order of the four ordered features, the Unknown flags isolate missingness, and nominal features remain one-hot. The 26-column matrix is more compact than 43 columns. No scaling is needed. `onehot` is a valid fallback but offers no advantage for this model.

**Gradient Boosting — `ordinal_tree` with scikit-learn `GradientBoostingClassifier`.** The same tree argument applies to scikit-learn `GradientBoostingClassifier` when it receives a numeric matrix. No scaling is needed. This phase does not use XGBoost, LightGBM, CatBoost, or `HistGradientBoostingClassifier`; native categorical handling is a future, separately documented experiment that would require a different preprocessing route. Raw BCSC codes must not be passed as ordinary numeric features without declaring them categorical.

**MLP Neural Network — `onehot` plus in-pipeline scaling.** The MLP can learn nonlinear relationships, but feeding ordinal ranks still imposes an arbitrary numeric geometry, especially for `bmi`, `density`, and `nrelbc`. One-hot gives the network explicit category inputs. A `StandardScaler` is required for stable optimisation and is the first step of the MLP model pipeline:

```python
SampleWeightPipeline([("scaler", StandardScaler()), ("classifier", MLPClassifier(...))])
```

Fitting the pipeline on a CV training fold fits the scaler only on that fold's training rows; the fitted scaler then transforms that fold's validation rows unchanged. The final model fits the same pipeline on the full Training matrix. The scaler and MLP are serialized together as one joblib artifact; the existing encoder artifacts contain no scaler and are not modified.

### Scaling and transformations

- **Logistic Regression:** no scaler. The 43 binary indicator columns are already on a common scale; adding a scaler would be a separate documented experiment.
- **Random Forest / Gradient Boosting:** no scaling; tree splits are invariant to monotonic scaling.
- **MLP:** `StandardScaler` inside the model pipeline, fitted per CV training fold and on full Training for the final model. Never fitted on Validation or Test.
- **All models:** no PCA, feature selection, or imputation is required at this stage; the existing encoders already preserve Unknown as explicit information.
- All four sklearn candidates accept `sample_weight`, so `count` can be supplied during training without row expansion.

## Unresolved decisions

1. **Resolved: Gradient Boosting implementation.** Use scikit-learn `GradientBoostingClassifier` with the existing `ordinal_tree` route. Native categorical handling and external boosting libraries are deferred.
2. **Resolved: MLP scaler packaging.** The scaler is a step inside `SampleWeightPipeline` and is serialized with the MLP as one artifact. It is fitted per CV training fold.
3. **Resolved: Logistic Regression scaling.** No scaler; the one-hot matrix is predominantly binary indicators.
4. **Resolved: `surgmeno`/`hrt` code 9.** Keep code 9 as one category; the source combines Unknown and Not Applicable, and raw data/mapping are unchanged. See report 07.
5. Confirm class-imbalance handling separately; it is a model-training decision, not an encoder choice.

Last updated: 2026-11-04
