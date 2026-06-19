from __future__ import annotations

import os
import re

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from infra.yymm import shift_yymm

LABEL_SCHEMA = os.getenv("LABEL_SCHEMA", "Label")
LABEL_TBL_REGEX = re.compile(r"^label_(\d{4})$", re.IGNORECASE)


def label_table_for_yymm(engine: Engine, yymm: str | int) -> str | None:
    yymm_str = str(yymm).zfill(4)
    q = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND table_type = 'BASE TABLE'
    """)
    try:
        tables = pd.read_sql(q, engine, params={"schema": LABEL_SCHEMA})["table_name"].tolist()
    except Exception:
        return None

    expected = f"label_{yymm_str}".lower()
    matches = [t for t in tables if LABEL_TBL_REGEX.match(t) and t.lower() == expected]
    if not matches:
        return None
    return sorted(matches)[0]


def label_tables_for_horizon(
    engine: Engine,
    origin_yymm: str | int,
    horizon: int,
    *,
    require_all: bool = False,
) -> list[str] | None:
    """Return available positive-list label tables for t+1..t+h.

    Label.YYMM files are supplemental positive labels. Missing months should
    not disable labels from months that are present; callers can still combine
    available actual positives with rule-based positives.
    """
    tables: list[str] = []
    missing = False
    for offset in range(1, int(horizon) + 1):
        table = label_table_for_yymm(engine, shift_yymm(origin_yymm, offset))
        if table is None:
            missing = True
        else:
            tables.append(table)
    if require_all and missing:
        return None
    return tables


def list_label_tables(engine: Engine) -> list[str]:
    q = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    try:
        tables = pd.read_sql(q, engine, params={"schema": LABEL_SCHEMA})["table_name"].tolist()
    except Exception:
        return []
    return [t for t in tables if LABEL_TBL_REGEX.match(t)]


def _infer_label_value(df: pd.DataFrame) -> pd.Series:
    """Infer optional label values; default to positive-list semantics."""
    # Accept historical column names from uploaded label files.
    label_cols = [
        "actual_label",
        "label",
        "label_value",
        "y",
        "y_churn",
        "churn",
        "is_churn",
        "is_churned",
        "churn_label",
        "target",
        "status",
    ]
    found = next((c for c in label_cols if c in df.columns), None)
    if found is None:
        return pd.Series(1, index=df.index, dtype="int64")

    raw = df[found]
    numeric = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    out.loc[numeric.notna()] = (numeric.loc[numeric.notna()].astype(float) >= 0.5).astype(int)

    text_values = raw.astype(str).str.strip().str.lower()
    positive = {
        "1", "true", "t", "yes", "y", "positive", "pos",
        "churn", "churned", "risk", "high_risk", "high-risk",
    }
    negative = {
        "0", "false", "f", "no", "n", "negative", "neg",
        "active", "non_churn", "non-churn", "not_churn", "not-churn", "normal",
    }
    out.loc[out.isna() & text_values.isin(positive)] = 1
    out.loc[out.isna() & text_values.isin(negative)] = 0

    # If a label column exists but some rows are malformed, keep positive-list
    # semantics for those rows so historical label_YYMM tables keep working.
    return out.fillna(1).astype("int64")


def load_label_keys(engine: Engine, table_name: str) -> pd.DataFrame:
    if not LABEL_TBL_REGEX.match(table_name):
        raise ValueError(f"Invalid label table name: {table_name}")

    q = text(f'SELECT * FROM "{LABEL_SCHEMA}"."{table_name}"')
    df = pd.read_sql(q, engine)
    for col in ("crm_code_enc", "cms_code_enc"):
        if col not in df.columns:
            raise KeyError(f'{LABEL_SCHEMA}."{table_name}" missing {col}')
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "None", "nan", "NaN"]), col] = pd.NA
    df["_label_value"] = _infer_label_value(df)
    df = df.dropna(how="all", subset=["crm_code_enc", "cms_code_enc"]).drop_duplicates()
    if df.empty:
        raise ValueError(f'{LABEL_SCHEMA}."{table_name}" contains no usable label keys')
    return df[["crm_code_enc", "cms_code_enc", "_label_value"]]
