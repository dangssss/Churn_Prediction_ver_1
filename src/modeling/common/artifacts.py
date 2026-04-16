from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

import joblib


def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def save_bundle(
    out_dir: str | Path,
    model: Any,
    *,
    metadata: Dict[str, Any],
    model_filename: str = "model.joblib",
    metadata_filename: str = "metadata.json",
) -> Path:
    """Save model + metadata to a folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / model_filename
    meta_path = out_dir / metadata_filename
    joblib.dump(model, model_path)
    
    # Convert metadata to JSON-serializable format
    metadata_clean = _make_json_serializable(metadata)
    meta_path.write_text(json.dumps(metadata_clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir

def load_bundle(
    in_dir: str | Path,
    model_filename: str = "model.joblib",
    metadata_filename: str = "metadata.json",
) -> tuple[Any, Dict[str, Any]]:
    in_dir = Path(in_dir)
    model = joblib.load(in_dir / model_filename)
    meta = json.loads((in_dir / metadata_filename).read_text(encoding="utf-8"))
    return model, meta
