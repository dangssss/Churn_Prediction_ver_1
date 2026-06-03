from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from psycopg2 import sql
from tenacity import retry, stop_after_attempt, wait_fixed

# In Docker this file is mounted at /churn_source/ingestion/validate_ingest.py.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from Data_pull.logging_config import get_logger
from Data_pull.resources import PostgresConfig, get_pg_conn

logger = get_logger("validate_ingest")

ORDER_TABLE_RE = re.compile(r"^bccp_orderitem_(\d{4})$")
LABEL_TABLE_RE = re.compile(r"^label_(\d{4})$", re.IGNORECASE)
SNAPSHOT_TABLES = ("cas_customer", "cas_info", "cms_complaint")
VALIDATION_STATUSES = {"PASS", "DEGRADED", "FAIL"}


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _month_index(yymm: str) -> int:
    yy, mm = int(yymm[:2]), int(yymm[2:])
    if mm < 1 or mm > 12:
        raise ValueError(f"Invalid YYMM: {yymm}")
    return yy * 12 + (mm - 1)


def _shift_yymm(yymm: str, offset: int) -> str:
    index = _month_index(yymm) + offset
    yy, month_zero_based = divmod(index, 12)
    return f"{yy:02d}{month_zero_based + 1:02d}"


def _previous_calendar_month() -> str:
    now = datetime.now(timezone.utc)
    return _shift_yymm(f"{now.year % 100:02d}{now.month:02d}", -1)


def _table_count(cur, schema: str, table: str) -> int:
    cur.execute(
        sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
            sql.Identifier(schema),
            sql.Identifier(table),
        )
    )
    return int(cur.fetchone()[0])


def _list_tables(cur, schema: str) -> list[str]:
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
        """,
        (schema,),
    )
    return [row[0] for row in cur.fetchall()]


def _ensure_validation_table(cur) -> None:
    cur.execute("CREATE SCHEMA IF NOT EXISTS ingest")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest.validation_status (
            id bigserial PRIMARY KEY,
            checked_at timestamptz NOT NULL DEFAULT now(),
            status text NOT NULL,
            details jsonb NOT NULL
        )
        """
    )


def _store_status(cur, status: str, details: dict) -> None:
    if status not in VALIDATION_STATUSES:
        raise ValueError(f"Unsupported validation status: {status}")
    _ensure_validation_table(cur)
    cur.execute(
        "INSERT INTO ingest.validation_status(status, details) VALUES (%s, %s::jsonb)",
        (status, json.dumps(details, ensure_ascii=False, default=str)),
    )


def _latest_success_age_days(cur, base: str) -> float | None:
    cur.execute("SELECT to_regclass('ingest.ingest_log')")
    if cur.fetchone()[0] is None:
        return None
    cur.execute(
        """
        SELECT EXTRACT(EPOCH FROM (now() - MAX(finished_at))) / 86400.0
        FROM ingest.ingest_log
        WHERE base = %s AND status = 'success'
        """,
        (base,),
    )
    value = cur.fetchone()[0]
    return float(value) if value is not None else None


def _validate_core(cur, details: dict) -> list[str]:
    failures: list[str] = []
    public_tables = _list_tables(cur, "public")
    order_tables = sorted(
        (table for table in public_tables if ORDER_TABLE_RE.fullmatch(table)),
        key=lambda table: _month_index(ORDER_TABLE_RE.fullmatch(table).group(1)),
    )
    if not order_tables:
        return ["No public.bccp_orderitem_YYMM table found"]

    latest_order = order_tables[-1]
    latest_yymm = ORDER_TABLE_RE.fullmatch(latest_order).group(1)
    latest_count = _table_count(cur, "public", latest_order)
    details["latest_order_table"] = latest_order
    details["latest_order_yymm"] = latest_yymm
    details["latest_order_rows"] = latest_count
    if latest_count <= 0:
        failures.append(f"public.{latest_order} is empty")

    expected_yymm = _previous_calendar_month()
    lag_months = _month_index(expected_yymm) - _month_index(latest_yymm)
    details["expected_order_yymm"] = expected_yymm
    details["order_lag_months"] = lag_months
    if lag_months > _env_int("FRESHNESS_BCCP_MAX_LAG_MONTHS", 1):
        failures.append(
            f"Latest order table {latest_order} is stale by {lag_months} month(s)"
        )

    if len(order_tables) >= 2:
        previous_order = order_tables[-2]
        previous_count = _table_count(cur, "public", previous_order)
        ratio = latest_count / previous_count if previous_count else 0.0
        details["previous_order_table"] = previous_order
        details["previous_order_rows"] = previous_count
        details["latest_previous_order_row_ratio"] = round(ratio, 6)
        if previous_count and ratio < _env_float("FRESHNESS_MIN_ROW_RATIO", 0.80):
            failures.append(
                f"public.{latest_order} row ratio {ratio:.3f} is below minimum"
            )

    max_snapshot_age = _env_int("FRESHNESS_SNAPSHOT_MAX_AGE_DAYS", 14)
    snapshots: dict[str, dict] = {}
    for table in SNAPSHOT_TABLES:
        if table not in public_tables:
            failures.append(f"Missing public.{table}")
            continue
        row_count = _table_count(cur, "public", table)
        age_days = _latest_success_age_days(cur, table)
        snapshots[table] = {"rows": row_count, "success_log_age_days": age_days}
        if row_count <= 0:
            failures.append(f"public.{table} is empty")
        if age_days is None:
            failures.append(f"No successful ingest log found for {table}")
        elif age_days > max_snapshot_age:
            failures.append(f"Latest {table} ingest log is stale ({age_days:.1f} days)")
    details["snapshots"] = snapshots
    return failures


def _validate_labels(cur, details: dict) -> list[str]:
    warnings: list[str] = []
    schema = os.getenv("LABEL_SCHEMA", "Label")
    try:
        label_tables = sorted(
            (table for table in _list_tables(cur, schema) if LABEL_TABLE_RE.fullmatch(table)),
            key=lambda table: _month_index(LABEL_TABLE_RE.fullmatch(table).group(1)),
        )
    except Exception as exc:
        details["label_schema"] = schema
        return [f"Cannot inspect label schema {schema}: {exc}"]

    usable: list[tuple[str, int]] = []
    for table in label_tables:
        count = _table_count(cur, schema, table)
        if count > 0:
            usable.append((table, count))

    details["label_schema"] = schema
    details["usable_label_tables"] = [
        {"table": table, "rows": count} for table, count in usable
    ]
    if not usable:
        return ["No non-empty actual label table is available; rule-based fallback required"]

    latest_label_table, latest_label_rows = usable[-1]
    latest_label_yymm = LABEL_TABLE_RE.fullmatch(latest_label_table).group(1)
    latest_order_yymm = details.get("latest_order_yymm")
    horizon_months = _env_int("LABEL_EXPECTED_HORIZON_MONTHS", 2)
    details["latest_label_table"] = latest_label_table
    details["latest_label_rows"] = latest_label_rows
    details["latest_complete_actual_origin_yymm"] = _shift_yymm(
        latest_label_yymm,
        -horizon_months,
    )
    if latest_order_yymm:
        expected_label_yymm = _shift_yymm(
            latest_order_yymm,
            -horizon_months,
        )
        details["expected_actual_label_yymm"] = expected_label_yymm
        if _month_index(latest_label_yymm) < _month_index(expected_label_yymm):
            warnings.append(
                f"Latest actual label {latest_label_table} is older than expected "
                f"label_{expected_label_yymm}; rule-based fallback may be used"
            )
    return warnings


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5), reraise=True)
def validate_ingestion() -> str:
    logger.info("Starting ingest freshness validation")
    pg_cfg = PostgresConfig.from_env()
    details: dict = {
        "policy": {
            "core_failure": "FAIL",
            "label_unavailable_or_stale": "DEGRADED",
        }
    }

    with get_pg_conn(pg_cfg) as conn:
        with conn.cursor() as cur:
            try:
                core_failures = _validate_core(cur, details)
                label_warnings = _validate_labels(cur, details)
            except Exception as exc:
                conn.rollback()
                with conn.cursor() as status_cur:
                    details["core_failures"] = [f"Validator exception: {exc}"]
                    _store_status(status_cur, "FAIL", details)
                conn.commit()
                raise

            details["core_failures"] = core_failures
            details["label_warnings"] = label_warnings
            status = "FAIL" if core_failures else "DEGRADED" if label_warnings else "PASS"
            _store_status(cur, status, details)
        conn.commit()

    log_method = logger.error if status == "FAIL" else logger.warning if status == "DEGRADED" else logger.info
    log_method("Ingest freshness validation status=%s details=%s", status, details)
    if status == "FAIL":
        raise RuntimeError("Ingest freshness validation failed: " + "; ".join(core_failures))
    return status


if __name__ == "__main__":
    try:
        validate_ingestion()
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
