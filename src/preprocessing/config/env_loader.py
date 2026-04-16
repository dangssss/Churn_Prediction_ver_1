"""Shared environment loading helpers for the Preprocess project."""

from __future__ import annotations

import os
from pathlib import Path

PREPROCESS_ROOT = Path(__file__).resolve().parents[1]
ENV_FILES = [PREPROCESS_ROOT / ".env"]


def load_project_env_files() -> None:
    for file_path in ENV_FILES:
        if not file_path.exists():
            continue

        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def parse_bool(name: str) -> bool:
    value = require_env(name).lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


def parse_int(name: str) -> int:
    return int(require_env(name))


load_project_env_files()