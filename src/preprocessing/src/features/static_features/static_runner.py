import pandas as pd
from sqlalchemy import text
import re

from config.app_config import get_config
from libs.db_utils import build_bccp_src
from libs.db_utils import create_bccp_indexes
from libs.db_utils import execute_sql
from logging_config import get_logger
from src.features.template_engine import render_template

logger = get_logger('static_runner')


SNAPSHOT_TABLE_RE = re.compile(r"^cus_lifetime_(\d{4})$")


def _resolve_static_date_range(end_date=None):
    cfg = get_config()
    start_date = cfg.features.static_data_start_date
    end_date = pd.Timestamp(end_date or pd.Timestamp.today()).strftime('%Y-%m-%d')
    return start_date, end_date


def _count_static_rows(engine) -> int:
    with engine.begin() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM data_static.cus_lifetime;"))
        return int(result.scalar() or 0)


def _render_lifetime_sql(engine, *, target_table: str, start_date: str, end_date: str) -> str:
    bccp_src = build_bccp_src(engine, start_date, end_date)
    return render_template(
        'lifetime_aggregate',
        TARGET_TABLE=target_table,
        BCCP_SRC=bccp_src,
        START_DATE=start_date,
        END_DATE=end_date,
    )


def _ensure_snapshot_table(engine, table_name: str) -> None:
    if not SNAPSHOT_TABLE_RE.fullmatch(table_name):
        raise ValueError(f"Invalid lifetime snapshot table name: {table_name}")
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE TABLE IF NOT EXISTS data_static.{table_name} "
            "(LIKE data_static.cus_lifetime INCLUDING ALL);"
        ))


def run_static_aggregate(engine, *, end_date=None):
    logger.info("Starting static feature aggregation...")

    start_date, end_date = _resolve_static_date_range(end_date)
    logger.info(f"  Static data range: {start_date} to {end_date}")

    logger.info("  Creating indexes on bccp_orderitem tables...")
    create_bccp_indexes(engine, analyze_sources=True, use_concurrently=True)
    logger.info("  [OK] Indexes created")

    sql = _render_lifetime_sql(
        engine,
        target_table="data_static.cus_lifetime",
        start_date=start_date,
        end_date=end_date,
    )

    logger.info("  Executing lifetime_aggregate SQL...")
    execute_sql(engine, sql)

    try:
        count = _count_static_rows(engine)
        logger.info(f"  [OK] Inserted {count:,} customer records into cus_lifetime")
    except Exception as exc:
        logger.warning(f"  Could not verify insertion count: {exc}")

    logger.info("Static feature aggregation complete")


def run_static_snapshots(engine, months, *, recompute_last_n: int = 2):
    months_list = list(months)
    if not months_list:
        return

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'data_static'
              AND table_name LIKE 'cus_lifetime_%'
        """)).fetchall()
    existing = {row[0] for row in rows if SNAPSHOT_TABLE_RE.fullmatch(row[0])}

    planned = []
    for month in months_list:
        table_name = f"cus_lifetime_{month.strftime('%y%m')}"
        if table_name not in existing:
            planned.append(month)
    planned.extend(months_list[-max(0, recompute_last_n):])

    unique_months = sorted({pd.Timestamp(month) for month in planned})
    logger.info(
        "Lifetime snapshot plan: existing=%d, compute=%d",
        len(existing),
        len(unique_months),
    )
    for month in unique_months:
        table_name = f"cus_lifetime_{month.strftime('%y%m')}"
        month_end = (month + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        logger.info("  Building data_static.%s through %s", table_name, month_end)
        _ensure_snapshot_table(engine, table_name)
        execute_sql(
            engine,
            f"TRUNCATE TABLE data_static.{table_name};\n"
            + _render_lifetime_sql(
                engine,
                target_table=f"data_static.{table_name}",
                start_date=get_config().features.static_data_start_date,
                end_date=month_end,
            ),
        )
