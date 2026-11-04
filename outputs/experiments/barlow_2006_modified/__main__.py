"""Compatibility shim for `python -m outputs.experiments.barlow_2006_modified.train_evaluate`."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
