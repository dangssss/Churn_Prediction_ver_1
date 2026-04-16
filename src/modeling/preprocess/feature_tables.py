from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

FEATURE_SCHEMA = os.getenv("FEATURE_SCHEMA", "data_window")
FEATURE_TBL_REGEX = re.compile(r"^cus_feature_(\d+)m_(\d{4})_(\d{4})$")

def list_tables_in_schema(engine: Engine, schema: str) -> List[str]:
    q = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema=:schema AND table_type='BASE TABLE'
        ORDER BY table_name
    """)
    return pd.read_sql(q, engine, params={"schema": schema})["table_name"].tolist()

def list_feature_tables(engine: Engine) -> List[str]:
    return [t for t in list_tables_in_schema(engine, FEATURE_SCHEMA) if FEATURE_TBL_REGEX.match(t)]

def parse_feature_table_name(t: str) -> Tuple[int, str, str]:
    m = FEATURE_TBL_REGEX.match(t)
    if not m:
        raise ValueError(f"Invalid feature table name: {t}")
    return int(m.group(1)), m.group(2), m.group(3)

def list_k_available(engine: Engine) -> List[int]:
    return sorted({parse_feature_table_name(t)[0] for t in list_feature_tables(engine)})

def list_tables_for_k(engine: Engine, k: int) -> List[str]:
    tbls = []
    for t in list_feature_tables(engine):
        kk, start, end = parse_feature_table_name(t)
        if kk == k:
            tbls.append(t)
    tbls.sort(key=lambda x: parse_feature_table_name(x)[1:])  # sort by (start,end)
    return tbls

def table_exists(engine: Engine, schema: str, table: str) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema=:schema AND table_name=:table
        LIMIT 1
    """)
    df = pd.read_sql(q, engine, params={"schema": schema, "table": table})
    return not df.empty

def load_feature_table(engine: Engine, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
    if not FEATURE_TBL_REGEX.match(table_name):
        raise ValueError("Table name không match pattern cus_feature_{K}m_YYMM_YYMM")
    lim = f"LIMIT {int(limit)}" if limit else ""
    q = text(f'SELECT * FROM "{FEATURE_SCHEMA}"."{table_name}" {lim}')
    return pd.read_sql(q, engine)

def max_window_end_for_k(engine: Engine, k: int) -> int:
    tbls = list_tables_for_k(engine, k)
    if not tbls:
        raise ValueError(f"No feature tables for K={k}")
    ends = [int(parse_feature_table_name(t)[2]) for t in tbls]
    return max(ends)
