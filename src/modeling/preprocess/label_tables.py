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
_CALIBRATION_CACHE: dict[tuple[int, str, str, int], float | None] = {}
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


def _infer_actual_label(df: pd.DataFrame) -> pd.Series:
    """Infer optional CSKH label values; default to positive-list semantics."""
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
    df["_actual_label"] = _infer_actual_label(df)
    df = df.dropna(how="all", subset=["crm_code_enc", "cms_code_enc"]).drop_duplicates()
    if df.empty:
        raise ValueError(f'{LABEL_SCHEMA}."{table_name}" contains no usable label keys')
    return df[["crm_code_enc", "cms_code_enc", "_actual_label"]]


def estimate_observed_label_rate(
    engine: Engine,
    *,
    horizon: int,
    feature_schema: str = "data_window",
) -> float | None:
    cache_key = (id(engine), LABEL_SCHEMA, feature_schema, int(horizon))
    if cache_key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[cache_key]

    if not list_label_tables(engine):
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
    for origin_month, feature_info in sorted(by_end.items()):
        label_tables = label_tables_for_horizon(engine, origin_month, horizon)
        if not label_tables:
            continue
        feature_table = feature_info[1]
        if not FEATURE_TBL_REGEX.match(feature_table):
            continue
        label_union_sql = "\nUNION\n".join(
            f'''
                SELECT
                    NULLIF(TRIM(cms_code_enc), '') AS cms_code_enc,
                    NULLIF(TRIM(crm_code_enc), '') AS crm_code_enc
                FROM "{LABEL_SCHEMA}"."{label_table}"
            '''
            for label_table in label_tables
        )

        q = text(f'''
            WITH label_keys AS (
                {label_union_sql}
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
                ",".join(label_tables),
            )

    if not rates:
        _CALIBRATION_CACHE[cache_key] = None
        return None

    rate = float(pd.Series(rates).median())
    _CALIBRATION_CACHE[cache_key] = rate
    return rate
