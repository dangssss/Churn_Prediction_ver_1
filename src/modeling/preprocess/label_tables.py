from __future__ import annotations

import os
import re

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from logging_config import get_logger
from infra.yymm import shift_yymm

LABEL_SCHEMA = os.getenv("LABEL_SCHEMA", "Label")
LABEL_TBL_REGEX = re.compile(r"^label_(\d{4})$", re.IGNORECASE)
FEATURE_TBL_REGEX = re.compile(r"^cus_feature_(\d+)m_(\d{4})_(\d{4})$")
_CALIBRATION_CACHE: dict[tuple[int, str, str], float | None] = {}
logger = get_logger(__name__)


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
) -> list[str] | None:
    """Return all actual label tables for t+1..t+h, or None if coverage is partial."""
    tables = [
        label_table_for_yymm(engine, shift_yymm(origin_yymm, offset))
        for offset in range(1, int(horizon) + 1)
    ]
    return None if any(table is None for table in tables) else tables


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
    df = df.dropna(how="all", subset=["crm_code_enc", "cms_code_enc"]).drop_duplicates()
    if df.empty:
        raise ValueError(f'{LABEL_SCHEMA}."{table_name}" contains no usable label keys')
    return df


def estimate_observed_label_rate(engine: Engine, *, feature_schema: str = "data_window") -> float | None:
    cache_key = (id(engine), LABEL_SCHEMA, feature_schema)
    if cache_key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[cache_key]

    labels = list_label_tables(engine)
    if not labels:
        _CALIBRATION_CACHE[cache_key] = None
        return None

    q_feature = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND table_type = 'BASE TABLE'
    """)
    try:
        feature_tables = pd.read_sql(q_feature, engine, params={"schema": feature_schema})["table_name"].tolist()
    except Exception:
        _CALIBRATION_CACHE[cache_key] = None
        return None

    by_end: dict[str, tuple[int, str]] = {}
    for table in feature_tables:
        m = FEATURE_TBL_REGEX.match(table)
        if not m:
            continue
        k = int(m.group(1))
        end = m.group(3)
        if end not in by_end or k > by_end[end][0]:
            by_end[end] = (k, table)

    rates = []
    for label_table in labels:
        label_month = LABEL_TBL_REGEX.match(label_table).group(1)
        feature_info = by_end.get(label_month)
        if not feature_info:
            continue
        feature_table = feature_info[1]
        if not FEATURE_TBL_REGEX.match(feature_table):
            continue

        q = text(f'''
            WITH label_keys AS (
                SELECT DISTINCT
                    NULLIF(TRIM(cms_code_enc), '') AS cms_code_enc,
                    NULLIF(TRIM(crm_code_enc), '') AS crm_code_enc
                FROM "{LABEL_SCHEMA}"."{label_table}"
            ),
            population AS (
                SELECT
                    f.cms_code_enc,
                    ci.crm_code_enc
                FROM "{feature_schema}"."{feature_table}" f
                LEFT JOIN public.cas_info ci
                  ON ci.cms_code_enc = f.cms_code_enc
            )
            SELECT
                COUNT(*) AS population_count,
                COUNT(*) FILTER (
                    WHERE EXISTS (
                        SELECT 1
                        FROM label_keys lk
                        WHERE (lk.cms_code_enc IS NOT NULL AND lk.cms_code_enc = population.cms_code_enc)
                           OR (lk.crm_code_enc IS NOT NULL AND lk.crm_code_enc = population.crm_code_enc)
                    )
                ) AS churn_count
            FROM population
        ''')
        try:
            row = pd.read_sql(q, engine).iloc[0]
            population_count = int(row["population_count"] or 0)
            churn_count = int(row["churn_count"] or 0)
        except Exception:
            continue
        if population_count > 0 and churn_count > 0:
            rates.append(churn_count / population_count)
        elif population_count > 0:
            logger.warning(
                'Ignoring empty or unmatched calibration label table %s.%s',
                LABEL_SCHEMA,
                label_table,
            )

    if not rates:
        _CALIBRATION_CACHE[cache_key] = None
        return None

    rate = float(pd.Series(rates).median())
    _CALIBRATION_CACHE[cache_key] = rate
    return rate
