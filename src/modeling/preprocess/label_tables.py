from __future__ import annotations

import os
import re

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

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


def load_label_keys(engine: Engine, table_name: str) -> pd.DataFrame:
    if not LABEL_TBL_REGEX.match(table_name):
        raise ValueError(f"Invalid label table name: {table_name}")

    q = text(f'''
        SELECT crm_code_enc, cms_code_enc
        FROM "{LABEL_SCHEMA}"."{table_name}"
    ''')
    df = pd.read_sql(q, engine)
    for col in ("crm_code_enc", "cms_code_enc"):
        if col not in df.columns:
            raise KeyError(f'{LABEL_SCHEMA}."{table_name}" missing {col}')
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "None", "nan", "NaN"]), col] = pd.NA
    return df.dropna(how="all", subset=["crm_code_enc", "cms_code_enc"]).drop_duplicates()
