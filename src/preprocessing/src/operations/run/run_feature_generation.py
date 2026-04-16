import argparse
import os
import sys
from pathlib import Path

PREPROCESS_ROOT = Path(__file__).resolve().parents[3]
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from config.env_loader import load_project_env_files
from src.operations.feature_generation.feature_generation_pipeline import run


def _to_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    load_project_env_files()

    month_chunk_default = int(os.getenv("STEP6_MONTH_CHUNK_SIZE", "2"))
    window_chunk_default = int(os.getenv("STEP6_WINDOW_GROUP_SIZE", "2"))
    checkpoint_default = os.getenv("STEP6_CHECKPOINT") or None
    resume_default = _to_bool(os.getenv("RESUME_WINDOW_STEP6", "false"))

    parser = argparse.ArgumentParser(description="Generate churn prediction features")
    parser.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: auto-detect from DB)")
    parser.add_argument("--database-url", default=None, help="Database URL (default: from environment)")
    parser.add_argument("--month-chunk-size", type=int, default=month_chunk_default, help="Step 6 month chunk size (env: MONTH_CHUNK_SIZE)")
    parser.add_argument("--window-group-size", type=int, default=window_chunk_default, help="Step 6 window-size chunk (env: WINDOW_GROUP_SIZE)")
    parser.add_argument("--step6-checkpoint", default=checkpoint_default, help="Step 6 checkpoint path (env: CHECKPOINT)")
    run(parser.parse_args())
