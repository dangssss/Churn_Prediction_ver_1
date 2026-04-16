"""
Database utilities: connection, index creation, table discovery. Handles all database operations except query execution.
"""
from sqlalchemy import text, inspect
import pandas as pd
from pathlib import Path
import sys

# Add parent to path for logging_config import
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from logging_config import get_logger

logger = get_logger('db_utils')


def execute_sql(engine, sql_str: str):
    with engine.begin() as conn:
        try:
            conn.exec_driver_sql(sql_str)
        except Exception as e:
            logger.error(f'Error executing SQL: {e}')
            raise


def build_bccp_src(engine, start_date: str, end_date: str) -> str:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    months = pd.date_range(start, end, freq='MS')
    
    # Get all tables once and convert to set for O(1) lookup
    inspector = inspect(engine)
    all_tables = set(inspector.get_table_names(schema='public'))
    
    # Filter tables with single pass
    selects = []
    for m in months:
        tbl = f"bccp_orderitem_{m.strftime('%y%m')}"
        if tbl in all_tables:
            selects.append(f"SELECT * FROM public.{tbl}")
    
    if not selects:
        logger.warning(f"No bccp_orderitem tables found for {start_date} to {end_date}, using fallback")
        return 'public.bccp_orderitem'
    
    result = '(' + ' UNION ALL '.join(selects) + ') AS bccp'
    logger.debug(f"Built bccp_src with {len(selects)} tables")
    return result


def create_bccp_indexes(
    engine,
    analyze_sources: bool = True,
    use_concurrently: bool = False,
):
    inspector = inspect(engine)
    bccp_tables = [
        t for t in inspector.get_table_names(schema='public')
        if t.startswith('bccp_orderitem_') and len(t) == len('bccp_orderitem_YYMM')
    ]
    
    if not bccp_tables:
        logger.info("No bccp_orderitem tables found, skipping bccp indexes")
        return
    
    logger.info(
        f"Creating indexes on {len(bccp_tables)} bccp_orderitem tables "
        f"(concurrently={use_concurrently})..."
    )

    sorted_tables = sorted(bccp_tables)
    for idx, table in enumerate(sorted_tables, start=1):
        idx_name = f"idx_{table}_code_time"
        create_stmt = (
            f"CREATE INDEX {'CONCURRENTLY ' if use_concurrently else ''}IF NOT EXISTS {idx_name} "
            f"ON public.{table}(cms_code_enc, sending_time)"
        )
        logger.info(f"  [*] [{idx}/{len(sorted_tables)}] Indexing public.{table}...")
        try:
            if use_concurrently:
                with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    conn.exec_driver_sql(create_stmt)
            else:
                with engine.begin() as conn:
                    conn.exec_driver_sql(create_stmt)
        except Exception as e:
            logger.warning(f"Warning creating index on {table}: {e}")

    if analyze_sources:
        try:
            with engine.begin() as conn:
                logger.debug("Running ANALYZE on source tables...")
                conn.execute(text("ANALYZE public.cas_customer"))
                conn.execute(text("ANALYZE public.cas_info"))
                conn.execute(text("ANALYZE public.cms_complaint"))
        except Exception as e:
            logger.warning(f"Warning analyzing tables: {e}")
    
    logger.info(f"Created {len(bccp_tables)} bccp indexes")


def ensure_public_tables_exist(engine, required_tables=None):
    inspector = inspect(engine)
    available = set(inspector.get_table_names(schema='public'))

    if required_tables is None:
        required_tables = ['cas_customer', 'cms_complaint', 'cas_info', 'bccp_orderitem']

    missing = []
    for tbl in required_tables:
        if tbl == 'bccp_orderitem':
            # Accept either a base table or partitioned monthly tables
            has_base = 'bccp_orderitem' in available
            has_partition = any(t.startswith('bccp_orderitem_') for t in available)
            if not (has_base or has_partition):
                missing.append(tbl)
        else:
            if tbl not in available:
                missing.append(tbl)

    if missing:
        msg = f"Missing required public tables: {', '.join(missing)}"
        logger.error(msg)
        raise RuntimeError(msg)


def ensure_public_table_columns_exist(engine, required_columns: dict):
    inspector = inspect(engine)
    available = set(inspector.get_table_names(schema='public'))

    errors = []
    for table, cols in required_columns.items():
        if table not in available:
            errors.append(f"Missing table: {table}")
            continue

        cols_info = inspector.get_columns(table, schema='public')
        existing_cols = {c['name'] for c in cols_info}
        missing_cols = [c for c in cols if c not in existing_cols]
        if missing_cols:
            errors.append(f"Table {table} missing columns: {', '.join(missing_cols)}")

    if errors:
        msg = '; '.join(errors)
        logger.error(msg)
        raise RuntimeError(msg)
