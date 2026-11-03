# Reproducibility Policy

Date: 2026-11-04. Scope: reproducibility settings for all upcoming Logistic Regression, Random Forest, Gradient Boosting, and MLP experiments. No model was trained, the frozen Train/Validation/Test split was not modified or regenerated, and `test.csv` was not accessed.

## 1. Observed environment

Inspected on 2026-11-04 from the active Python environment:

| Component | Version / status |
|---|---|
| Python | 3.15.0 |
| NumPy | 2.5.4 |
| pandas | 3.1.0 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| PyTorch | not installed |
| XGBoost | not installed |
| LightGBM | not installed |
| CatBoost | not installed |
| TensorFlow / Keras | not installed |
| Platform | Windows-11-10.0.26300-SP0 |
| Dependency pinning | `requirements.txt` is empty |

Consequences:

- The only installed MLP implementation is `sklearn.neural_network.MLPClassifier`. PyTorch is not part of the current environment; its settings are defined conditionally in Section 5.
- The only installed Gradient Boosting implementations are `sklearn.ensemble.GradientBoostingClassifier` and `sklearn.ensemble.HistGradientBoostingClassifier`. The final Gradient Boosting library remains unresolved in report 03; Section 6 records the settings required if an external library is selected later.
- Versions above are observed, not pinned. Exact environment recreation is therefore not currently guaranteed; each experiment must record its own environment snapshot.

## 2. Central configuration module

Implemented `src/config.py` as the single source for upcoming training experiments:

| Setting | Value / purpose |
|---|---|
| `RANDOM_STATE` | `42` |
| `N_JOBS` | `1` |
| `DEFAULT_THRESHOLD` | `0.5` |
| `set_global_seeds(seed=42)` | Seeds Python `random` and NumPy's global RNG; returns a dedicated `numpy.random.Generator` |
| `seed_torch(seed=42, deterministic=True)` | Optional PyTorch seeding; raises a clear error while PyTorch is not installed |
| `environment_snapshot()` | Returns Python, OS, library versions, seed settings, and thread environment for the experiment record |
| `THREAD_ENV_VARS` | Names of launch-time environment variables that must be set before Python starts |

`RANDOM_STATE = 42` matches the seed used to create the frozen split (`src/preprocessing/split_validation.py`). The split script retains its local `RANDOM_STATE = 42` because the split is frozen and must not be regenerated; the value is identical to the central setting. All new training, tuning, and evaluation code must import `RANDOM_STATE` from `src.config` rather than defining another local seed.

Smoke checks performed: `python -m src.config` prints the environment snapshot; repeated `set_global_seeds()` calls reproduce the same Python/NumPy random sequence; boolean seeds are rejected; `seed_torch()` raises `ImportError` in the current environment. No model was fitted.

## 3. One central seed and per-model seeds

### 3.1 Default experiment seed

- Official model comparisons use `RANDOM_STATE = 42`.
- Models are not given different default seeds; a different seed is a new experiment, not a different model configuration.
- Multi-seed robustness studies are allowed only with a pre-declared seed list, for example `[42, 43, 44, 45, 46]`, and every seed is recorded and reported separately. Do not report only the best seed.
- Do not change `RANDOM_STATE` to improve a result. If the central seed ever changes, all models must be re-run and the change documented; the existing split remains frozen.

### 3.2 Python and NumPy

Call `set_global_seeds()` once at the start of every training/evaluation process, before constructing models or drawing any random sample:

```python
from src.config import RANDOM_STATE, N_JOBS, set_global_seeds

rng = set_global_seeds()
```

- Python's `random` module is seeded with `RANDOM_STATE`.
- NumPy's legacy global RNG is seeded with `RANDOM_STATE`.
- The returned `numpy.random.Generator` is independently seeded and should be used for new code that needs an explicit generator (for example, bootstrap resampling or internal folds).
- Do not mix the legacy global RNG and the returned generator for the same random decision.
- Set `PYTHONHASHSEED=42` at process launch. It cannot be changed reliably after Python starts.

For independent parallel processes, derive child seeds deterministically from one root, for example `np.random.SeedSequence(RANDOM_STATE).spawn(n)`, and record the mapping. Official comparisons should use `N_JOBS = 1` and avoid process-level randomness.

## 4. scikit-learn settings

Every estimator must be constructed with explicit `random_state=RANDOM_STATE`, even where the current solver is deterministic, so a later solver change cannot silently introduce randomness. Pass `n_jobs=N_JOBS` where the estimator supports it.

| Estimator | Required seed / determinism settings | Notes |
|---|---|---|
| `LogisticRegression` | `random_state=RANDOM_STATE` | `lbfgs` is deterministic; `liblinear`, `sag`, and `saga` use the seed. Keep solver and `max_iter` in the experiment configuration. |
| `RandomForestClassifier` | `random_state=RANDOM_STATE`, `n_jobs=N_JOBS` | Bootstrap sampling uses the seed. `n_jobs=1` is the reproducible default. |
| `GradientBoostingClassifier` | `random_state=RANDOM_STATE` | `subsample < 1.0` uses the seed; record `subsample`. No `n_jobs`. |
| `HistGradientBoostingClassifier` | `random_state=RANDOM_STATE` | `early_stopping` uses a seeded internal validation split; record `early_stopping`, `validation_fraction`, and `n_iter_no_change`. OpenMP threading is controlled at launch (Section 8). |
| `MLPClassifier` | `random_state=RANDOM_STATE`, `shuffle=True` | Weight initialisation and shuffling use the seed; `early_stopping` uses a seeded validation split. Keep solver, architecture, `max_iter`, and early-stopping settings in the configuration. |

Rules:

- No estimator may rely on its default `random_state=None`.
- Do not introduce `train_test_split` on the frozen data. If internal cross-validation is required, use `StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)` on Training only, with the same folds for every compared model; pass `count` as `sample_weight` inside each fold.
- Any random resampling, bootstrap, or hyperparameter search must use `RANDOM_STATE` or a recorded deterministic child seed.
- Existing preprocessing encoders are deterministic and must be applied with `transform`; no refitting is part of a training experiment.
- The MLP scaler, if used, is fitted on the encoded Training matrix only and saved with the experiment.

## 5. MLP framework settings

### 5.1 Selected framework in the current environment

`sklearn.neural_network.MLPClassifier` is the only installed MLP implementation. It is the selected MLP framework unless the environment is changed and the change is documented. Use:

```python
MLPClassifier(..., random_state=RANDOM_STATE, shuffle=True)
```

Also record the solver, hidden-layer sizes, activation, alpha, learning rate, batch size, maximum iterations, and early-stopping settings. The existing `onehot` encoder plus a Training-fitted `StandardScaler` remains the recommended input route from report 03.

### 5.2 PyTorch settings if it is added later

PyTorch is not installed today. If it replaces scikit-learn MLP, call `seed_torch()` before model construction and configure:

- `torch.manual_seed(RANDOM_STATE)`.
- `torch.cuda.manual_seed_all(RANDOM_STATE)` when CUDA is used.
- `torch.use_deterministic_algorithms(True)`.
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialisation.
- A seeded `torch.Generator` for every `DataLoader`; use a deterministic `worker_init_fn` if `num_workers > 0`.
- Fixed epoch order, fixed early-stopping tie-breaking, and a recorded checkpoint-selection rule.
- `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False` where bitwise stability matters.
- The same DataLoader order, batch size, and shuffle seed in every run.

GPU kernels, CUDA/cuDNN versions, and non-deterministic reductions can still prevent bitwise reproducibility. If exact reproduction is required, use CPU or a verified deterministic GPU setup, and record hardware and driver versions.

## 6. Gradient Boosting library settings

The final library is not selected. The installed and currently runnable options are the two scikit-learn estimators in Section 4. If an external library is selected later, it must be installed, version-pinned, and recorded with the run.

| Library | Seed and determinism settings to use |
|---|---|
| scikit-learn `GradientBoostingClassifier` | `random_state=RANDOM_STATE`; record `subsample`, `learning_rate`, `n_estimators`, and loss. |
| scikit-learn `HistGradientBoostingClassifier` | `random_state=RANDOM_STATE`; record early-stopping settings; limit OpenMP threads at launch. |
| XGBoost | `random_state=RANDOM_STATE` (alias `seed`), single-threaded CPU execution (`nthread=1` / `n_jobs=1`), `tree_method="hist"` or another deterministic CPU method. GPU training is not accepted as reproducible by default. |
| LightGBM | `random_state=RANDOM_STATE`, `deterministic=True`, `force_col_wise=True`, `num_threads=1`; also record `bagging_seed`, `feature_fraction_seed`, and `data_random_seed` if used. |
| CatBoost | `random_seed=RANDOM_STATE`, `thread_count=1`, CPU training (`task_type="CPU"`); GPU determinism is not guaranteed. |

The external-library rows are conditional and were not executed or API-validated in this environment. If the selected library changes, every model must be re-run; results from different Gradient Boosting libraries are separate experiments.

## 7. Reproducible data loading and split usage

- Use only the frozen files under `data/04_splits/` through `src.data.load_splits.load_split("train"|"val")`.
- Never regenerate, re-split, reshuffle, or reorder the split files. The training seed does not determine the split; the split is an input artifact.
- Never call `train_test_split` on the loaded data, and never use `test` for fitting, tuning, early stopping, calibration, threshold selection, or preprocessing statistics.
- `load_split()` returns `X` in canonical `FEATURES` order, binary `y`, and positive integer `w = count`. Preserve row alignment: every operation must keep `X`, `y`, and `w` aligned.
- Do not drop duplicate rows. The dataset is aggregated; repeated feature/outcome combinations are legitimate frequency records.
- Do not resample, undersample, or oversample rows for official comparisons. `count` is the frequency weight and must be supplied to fitting and evaluation. If a resampling experiment is ever added, it must use a recorded seed and is not directly comparable to the weighted baseline.
- Preprocessing is transform-only. The `onehot`/`ordinal_tree` artifacts and their SHA-256 values must be recorded with the experiment.
- Before each run, verify the Training/Validation row counts, weight sums, and weighted cancer rates against the frozen reference (Train: 152,149 rows / 1,525,868 weight / 0.488378%; Validation: 28,316 rows / 269,271 weight / 0.488356%). A mismatch invalidates the run.
- Internal cross-validation, if used, partitions Training only and uses the fixed stratified folds described in Section 4.

## 8. Deterministic runtime settings

Set these in the shell before starting Python. Python hash randomisation and BLAS/OpenMP thread counts are read at interpreter or library start-up; setting them inside `src/config.py` at import time is too late to be reliable.

PowerShell example:

```powershell
$env:PYTHONHASHSEED = "42"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python -m src.config
```

Rules:

- `PYTHONHASHSEED=42` for every official run.
- Single-threaded BLAS/OpenMP/NumExpr for official comparisons; `N_JOBS=1` in configuration.
- For scikit-learn estimators, pass `n_jobs=N_JOBS` where supported and avoid parallel hyperparameter searches that reuse one RNG without derived seeds.
- Record `environment_snapshot()` in every experiment. Its `thread_environment` field shows whether the launch-time variables were actually set.
- Record CPU/GPU model, OS, Python, and library versions. Do not mix CPU and GPU results in one comparison table.
- Do not rely on joblib file bytes being identical across runs; the fitted model content and metric outputs are the reproducibility targets.

## 9. Experiment configuration and result recording

Use one immutable directory per run:

```text
outputs/training/experiments/<run_id>/
```

Recommended `run_id`:

```text
YYYYMMDDTHHMMSS_<model>_<encoder>_seed42
```

Each run directory must contain:

| File | Contents |
|---|---|
| `config.json` | Model class and full resolved hyperparameters, encoder family, encoder artifact path and SHA-256, scaler details, `RANDOM_STATE`, `N_JOBS`, internal-CV settings, threshold, seed list, and any resampling settings |
| `environment.json` | Output of `src.config.environment_snapshot()`, plus `pip freeze`, git commit/HEAD, git dirty state, and platform/hardware |
| `splits.json` | Paths, SHA-256, row counts, weight sums, and weighted cancer rates for the Training and Validation files used |
| `metrics.json` | Weighted Validation metrics from report 04: ROC-AUC, PR-AUC, prevalence, Brier, and threshold-dependent recall, specificity, precision, weighted TP/FP/TN/FN at `t = 0.5`; undefined values as `null` with a reason |
| `predictions_val.csv` | One row per Validation record: stable row index, `y`, `count`, predicted positive-class probability, and run id; no Test predictions until final evaluation |
| `model.joblib` or native checkpoint | Fitted model artifact; for PyTorch, save state dict and architecture/config |
| `log.txt` | Captured stdout/stderr, warnings, and runtime |
| `config_notes.md` | Short human-readable summary of the run and any deviations from this policy |

Also maintain an append-only experiment registry, for example `outputs/training/experiments/registry.csv`, with one row per run: `run_id`, timestamp, model, encoder, seed, key metrics, model artifact path, environment file, git commit, and status.

Recording rules:

- Store the **resolved** configuration, not only the code defaults; use `estimator.get_params()` where available.
- Hyperparameter searches must record the search space, the search seed, the fold definition, all trial results, and the selected parameters. Do not record only the winning trial.
- A failed or invalid run is recorded with a failure status and reason; it is not deleted or overwritten.
- Never overwrite a completed run directory. Re-runs get a new `run_id`.
- Record the exact encoder artifact and data fingerprints; a later encoder or data change invalidates the result.
- Test evaluation is a separate frozen step after model selection. It must not appear in the model-comparison registry before that point.
- Report the weighted metrics from report 04 only; unweighted diagnostics, if used, are clearly labelled appendices.

## 10. Reproducibility limitations

1. **Unpinned dependencies.** `requirements.txt` is empty and the observed versions are not locked. Installing later versions can change numerical results or defaults. A full `pip freeze` and environment snapshot per run are mandatory; dependency pinning is recommended before final comparison.
2. **Floating-point and threading.** BLAS/OpenMP reduction order, CPU architecture, and thread count can change the last digits even with fixed seeds. Single-threaded CPU execution reduces but does not eliminate this.
3. **No external MLP/boosting libraries installed.** PyTorch, XGBoost, LightGBM, and CatBoost settings are documented conditionally and have not been executed. Their APIs and deterministic behaviour must be verified if adopted.
4. **scikit-learn MLP.** Weight initialisation, shuffling, and early-stopping splits are seeded, but solver and BLAS behaviour may still produce small run-to-run differences across environments.
5. **HistGradientBoosting.** OpenMP parallelism is controlled by launch-time environment variables; setting them after import is unreliable.
6. **GPU execution.** CUDA/cuDNN kernels and some reductions are not bitwise deterministic. CPU is the reproducible default; any GPU run must record hardware, driver, CUDA/cuDNN versions, and deterministic settings, and should not be mixed with CPU results.
7. **Serialized artifact bytes.** Joblib/pickle output may differ byte-for-byte between runs even when the fitted model is equivalent. Compare model content and metrics, not only file hashes.
8. **Frozen split provenance.** The split files have no embedded fingerprint. Their SHA-256 values must be recorded per run; a changed file makes prior results non-comparable.
9. **Pending preprocessing decision.** A future `surgmeno`/`hrt` recode changes the encoded matrix and invalidates comparisons made with the current artifacts.
10. **Exact bitwise reproducibility is not guaranteed.** The achievable target is repeatable results under the same data, code, configuration, seed, hardware, and pinned environment, with any residual numerical differences documented.

## 11. Pre-run checklist

1. Set the launch-time environment variables in Section 8.
2. Run `python -m src.config` and save the snapshot into the run's `environment.json`.
3. Record `git rev-parse HEAD`, `git status --short`, and the split/preprocessing file hashes.
4. Call `set_global_seeds()` before any model or random operation.
5. Load Training and Validation with `load_split()`; never load Test.
6. Apply saved encoders with transform only; fit the MLP scaler on Training only if used.
7. Construct each estimator with `random_state=RANDOM_STATE` and `n_jobs=N_JOBS` where supported.
8. Save `config.json`, metrics, predictions, model artifact, log, and registry row.
9. Verify that no Test file was read and no split or preprocessing artifact was modified.

## 12. Open decisions

1. Select the final Gradient Boosting library; only scikit-learn is installed and verified today.
2. Confirm scikit-learn MLP as the final MLP framework, or install and pin PyTorch and re-validate the deterministic settings.
3. Decide whether multi-seed robustness runs are required and, if so, fix the seed list before comparisons.
4. Pin dependency versions and create a reproducible environment lock before final model comparison.
5. Confirm whether `outputs/training/experiments/registry.csv` is the official registry or whether JSONL is preferred.
6. Resolve the pending `surgmeno`/`hrt` preprocessing recode before any official experiment.

Last updated: 2026-11-04
