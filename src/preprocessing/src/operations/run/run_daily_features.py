"""Scheduled or manual daily runner for feature generation."""

import sys
from pathlib import Path
from types import SimpleNamespace

PREPROCESS_ROOT = Path(__file__).resolve().parents[3]
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from src.operations.feature_generation.feature_generation_pipeline import run


if __name__ == "__main__":
    args = SimpleNamespace(
        start="2025-01-01",
        end=None,
        database_url=None,
        disable_window_optimization=False,
        recompute_last_n=2,
    )
    run(args)
