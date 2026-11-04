# Architecture

## Main Components

The implemented system is a research pipeline with four stages:

1. Data preparation and splitting
2. Preprocessing and encoding
3. Model training, tuning, calibration, and evaluation
4. Inference from human-readable patient input

## High-Level System Architecture

```mermaid
flowchart TB
    Raw[BCSC raw risk.txt] --> Convert[src/data/risk_to_csv.py]
    Convert --> Filtered[data/02_interim/risk_filtered.csv]
    Filtered --> Split[src/preprocessing/split_original.py / split_validation.py]
    Split --> Splits[data/04_splits/train.csv, val.csv, test.csv]
    Splits --> Loader[src/data/load_splits.py]
    Loader --> Fit[src/preprocessing/fit_preprocessing.py]
    Fit --> Artifacts[raw_input_mapping.json + encoder pipelines]
    Loader --> Train[src/training/train_models.py]
    Artifacts --> Train
    Train --> Models[Calibrated model artifacts]
    Models --> Infer[SUPMT.py]
    Artifacts --> Infer
    Human[Human patient input] --> Infer
    Infer --> Output[Four probabilities + weighted average]
```

## Training Architecture

```mermaid
flowchart LR
    X[Training X] --> Enc[Per-model encoder]
    Enc --> Z[Encoded matrix]
    Z --> CV[Weighted 5-fold CV]
    CV --> Select[Select by weighted PR-AUC]
    Select --> Fit[Final fit on full train]
    Fit --> Cal[Sigmoid calibration, ensemble=False]
    Cal --> Save[Joblib model artifact]
```

Evidence: `src/training/train_models.py`, `src/training/config.py`, `src/training/cv.py`.

## Inference Architecture

```mermaid
flowchart LR
    P[Human patient dict] --> Norm[Normalize missing fields to None]
    Norm --> HumanMap[src/preprocessing/raw_input_mapping.py::human_to_interface_record]
    HumanMap --> IF[Interface-coded DataFrame]
    IF --> Mapper[RawInputMapper.transform_frame]
    Mapper --> BCSC[BCSC-coded DataFrame]
    BCSC --> Pipe[Per-model fitted ColumnTransformer]
    Pipe --> Encoded[Encoded matrix]
    Encoded --> Model[Calibrated model predict_proba]
    Model --> Prob[Positive-class probability]
    Prob --> Agg[Weighted average from SUPMT_CONFIG.MODEL_WEIGHTS]
    Agg --> Out[Printed probabilities]
```

Evidence: `SUPMT.py`, `src/preprocessing/raw_input_mapping.py`, `src/preprocessing/pipeline.py`.

## Component Responsibilities

| Component | Responsibility |
|---|---|
| `src/data/load_splits.py` | Defines `FEATURES`, `TARGET`, `WEIGHT`; loads and validates splits |
| `src/preprocessing/raw_input_mapping.py` | Human/interface mapping and strict validation |
| `src/preprocessing/pipeline.py` | Encoder construction, fitting, transformation, serialization |
| `src/training/config.py` | Model specs, encoder mapping, CV/calibration policy |
| `src/training/cv.py` | Weighted folds, metrics, probability extraction |
| `src/training/train_models.py` | Training, tuning, calibration, validation evaluation |
| `SUPMT.py` | Artifact loading and inference |
| `SUPMT_CONFIG.py` | Inference weights and artifact paths |

## Training vs Inference Separation

- Training fits encoders and models on the Training split only.
- Validation is used for evaluation/comparison, not tuning/calibration.
- Inference loads saved artifacts and calls `transform`/`predict_proba` only.
- `SUPMT.py` does not fit or retrain anything.

## Component Dependencies

```mermaid
flowchart TD
    LS[src/data/load_splits.py] --> PP[src/preprocessing/pipeline.py]
    RM[src/preprocessing/raw_input_mapping.py] --> PP
    PP --> FP[src/preprocessing/fit_preprocessing.py]
    LS --> TC[src/training/config.py]
    PP --> TC
    TC --> TM[src/training/train_models.py]
    CV[src/training/cv.py] --> TM
    TP[src/training/pipeline.py] --> TM
    TM --> Models[models/trained/*_calibrated.joblib]
    FP --> Pre[models/preprocessing/*]
    Models --> SUPMT[SUPMT.py]
    Pre --> SUPMT
    RM --> SUPMT
```

## Known Architecture Limitations

- No service/API layer is implemented.
- Test split was used only for one authorized final evaluation.
- `surgmeno` and `hrt` code 9 combines Unknown and Not Applicable.
- Dependency versions are not pinned in `requirements.txt`.