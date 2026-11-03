"""Central reproducibility settings for BCSC model-training experiments.

`RANDOM_STATE` matches the seed used to create the frozen train/validation
split (42). Future training code must import it from here instead of defining
local seeds, so that model seeds, internal cross-validation, and experiment
records stay aligned with the existing split.

The frozen split files are never regenerated or modified by this module.
Data loading must go through :func:`src.data.load_splits.load_split`; no
training code should call ``train_test_split`` or reshuffle the splits.

Thread limits and ``PYTHONHASHSEED`` are documented as launch-time environment
variables because Python and BLAS read them at interpreter start-up. Setting
them at runtime cannot retroactively change an already-imported process.

Usage
-----
    from src.config import RANDOM_STATE, N_JOBS, set_global_seeds

    rng = set_global_seeds()
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

For a run-time snapshot of versions and thread settings, call
:func:`environment_snapshot` and write the result into the experiment record.
"""
from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np

# ---- single source of truth for upcoming training experiments -------------
RANDOM_STATE: int = 42
N_JOBS: int = 1
DEFAULT_THRESHOLD: float = 0.5

# Set these before starting Python for strict single-threaded execution.
# They are reported by environment_snapshot() but cannot be applied reliably
# after numpy/BLAS have been imported.
THREAD_ENV_VARS: tuple[str, ...] = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG",
)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    return seed


def set_global_seeds(seed: int = RANDOM_STATE) -> np.random.Generator:
    """Seed Python's ``random`` and NumPy's legacy global RNG.

    Returns a dedicated ``numpy.random.Generator`` for code that prefers the
    modern NumPy API. Both the legacy global RNG and the returned generator are
    seeded identically, so either style is reproducible; do not mix them for
    one random decision.
    """
    seed = _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def seed_torch(seed: int = RANDOM_STATE, *, deterministic: bool = True) -> None:
    """Seed PyTorch when it is installed and selected as the MLP framework.

    PyTorch is not installed in the current environment. This helper therefore
    raises a clear error if called today instead of silently pretending to seed
    a missing library. If PyTorch is later added, call this before model or
    DataLoader construction.
    """
    seed = _validate_seed(seed)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyTorch is not installed; use the installed scikit-learn MLP or "
            "install and pin PyTorch before calling seed_torch()."
        ) from exc

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Must be set before CUDA work; setting it here is best effort for an
        # already-started process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def environment_snapshot() -> dict[str, Any]:
    """Return the versions and settings needed to interpret an experiment."""
    import joblib
    import pandas as pd
    import scipy
    import sklearn

    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "default_threshold": DEFAULT_THRESHOLD,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }

    try:
        import torch
    except ImportError:
        snapshot["torch"] = None
    else:  # pragma: no cover - depends on environment
        snapshot["torch"] = torch.__version__
        snapshot["torch_cuda_available"] = bool(torch.cuda.is_available())

    return snapshot


if __name__ == "__main__":
    import json

    print(json.dumps(environment_snapshot(), indent=2, sort_keys=True))
