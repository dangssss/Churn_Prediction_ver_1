import pandas as pd
from sqlalchemy import text

from config.app_config import get_config
from libs.db_utils import build_bccp_src
from libs.db_utils import create_bccp_indexes
from libs.db_utils import execute_sql
from logging_config import get_logger
from src.features.template_engine import render_template

logger = get_logger('static_runner')


def _resolve_static_date_range():
    cfg = get_config()
    start_date = cfg.features.static_data_start_date
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    return start_date, end_date


def _count_static_rows(engine) -> int:
    with engine.begin() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM data_static.cus_lifetime;"))
        return int(result.scalar() or 0)


def run_static_aggregate(engine):
    logger.info("Starting static feature aggregation...")

    start_date, end_date = _resolve_static_date_range()
    logger.info(f"  Static data range: {start_date} to {end_date}")

    logger.info("  Creating indexes on bccp_orderitem tables...")
    create_bccp_indexes(engine, analyze_sources=True, use_concurrently=True)
    logger.info("  [OK] Indexes created")

    bccp_src = build_bccp_src(engine, start_date, end_date)
    sql = render_template(
        'lifetime_aggregate',
        BCCP_SRC=bccp_src,
        START_DATE=start_date,
        END_DATE=end_date,
    )

    logger.info("  Executing lifetime_aggregate SQL...")
    execute_sql(engine, sql)

    try:
        count = _count_static_rows(engine)
        logger.info(f"  [OK] Inserted {count:,} customer records into cus_lifetime")
    except Exception as exc:
        logger.warning(f"  Could not verify insertion count: {exc}")

    logger.info("Static feature aggregation complete")
