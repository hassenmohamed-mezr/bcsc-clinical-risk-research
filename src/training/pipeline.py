"""Model-level pipelines and serialization helpers.

The saved encoders remain owned by :mod:`src.preprocessing`. They are applied
with ``transform`` only and are never refitted by training code. What this
module adds is the *model* artifact:

* For the MLP, the required ``StandardScaler`` and the ``MLPClassifier`` are one
  scikit-learn :class:`~sklearn.pipeline.Pipeline`. Fitting that pipeline on a
  CV training fold fits the scaler on that fold's rows only; the fitted scaler
  is then reused unchanged for the fold's validation rows. Refitting on the
  full Training split produces the final artifact, and joblib serializes the
  scaler and model together.
* For the other model families, the estimator itself is the model artifact.

scikit-learn's base ``Pipeline`` only accepts step-prefixed fit parameters when
metadata routing is disabled (``classifier__sample_weight``). The weighted CV
and calibration code pass ``sample_weight`` by name, so
:class:`SampleWeightPipeline` exposes that name explicitly and forwards it to
the final estimator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


class SampleWeightPipeline(Pipeline):
    """A ``Pipeline`` whose ``fit`` accepts ``sample_weight`` explicitly.

    The weight is forwarded to the final estimator as
    ``<final_step>__sample_weight``. Intermediate transformers (the MLP scaler)
    do not receive weights; they are fitted unsupervised on the same training
    slice. This makes the pipeline compatible with callers that pass
    ``sample_weight`` by name, including ``CalibratedClassifierCV`` and the
    project's weighted CV helpers.

    Metadata routing must remain disabled (the scikit-learn default). If it is
    enabled process-wide, use step-prefixed parameters instead.
    """

    def fit(self, X, y=None, sample_weight=None, **params):
        if sample_weight is not None:
            if len(sample_weight) != len(X):
                raise ValueError(
                    "sample_weight length "
                    f"({len(sample_weight)}) does not match X rows ({len(X)})"
                )
            key = f"{self.steps[-1][0]}__sample_weight"
            if key in params:
                raise TypeError(
                    f"sample_weight was provided directly and as {key!r}; pass it once"
                )
            params[key] = sample_weight
        return super().fit(X, y, **params)


def save_model_artifact(model: Any, path: str | Path) -> Path:
    """Serialize one fitted (or unfitted) model artifact with joblib.

    The MLP artifact contains its scaler because the scaler is a pipeline step;
    a calibrated model contains the fitted pipeline and its calibrator.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model_artifact(path: str | Path) -> Any:
    """Load a model artifact and require the probability interface."""
    path = Path(path)
    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"object at {path} is not a probabilistic model artifact")
    return model
