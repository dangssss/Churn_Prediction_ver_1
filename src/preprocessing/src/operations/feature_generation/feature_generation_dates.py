"""Date-range and window-size helpers for feature generation."""

import datetime as dt
import re

import pandas as pd
from sqlalchemy import text
from config.app_config import get_config
from logging_config import get_logger

logger = get_logger('feature_generation_dates')

DEFAULT_START_DATE = pd.Timestamp("2025-01-01")


def _to_date(value) -> dt.date | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.date()


def _yymm_to_date(yymm: str) -> dt.date:
    year = int(yymm[:2])
    month = int(yymm[2:])
    return dt.date(2000 + year, month, 1)


def resolve_month_plan(engine, start_value=None, end_value=None):
    """Calculate month range and window sizes for feature generation.
    
    Args:
        engine: Database engine
        start_value: Override start date (YYYY-MM-DD), default from config STATIC_DATA_START
        end_value: Override end date (YYYY-MM-DD), default: auto-detect from DB max(cas_customer.report_month)
    
    Returns:
        (start_date, end_date, months_range, window_sizes)
    """
    cfg = get_config()
    
    # Determine start date
    if start_value:
        start_date = pd.Timestamp(start_value)
    else:
        start_date = pd.Timestamp(cfg.features.static_data_start_date)
    
    # Determine end date
    if end_value:
        end_date = pd.to_datetime(end_value)
    else:
        end_date = _auto_end_date_from_db(engine, fallback_end=dt.date.today())
    
    # Normalize to first day of month
    start_date = pd.Timestamp(start_date.year, start_date.month, 1)
    end_date = pd.Timestamp(end_date.year, end_date.month, 1)
    
    # Generate all months in range
    months = pd.date_range(start_date, end_date, freq="MS")
    if len(months) == 0:
        raise SystemExit("No months in given range")
    
    # Calculate window sizes using config
    available_months = len(months)
    min_window = cfg.features.window_sizes_min
    max_window_cfg = cfg.features.window_sizes_max
    
    # Rule: max window size = number of months - 2.
    auto_max_window = max(1, available_months - 2)
    if max_window_cfg is None:
        max_window = auto_max_window
    else:
        max_window = min(max_window_cfg, auto_max_window)
    
    # Ensure min_window doesn't exceed available months
    if min_window > available_months:
        logger.warning(
            f"WINDOW_SIZES_MIN ({min_window}) > available months ({available_months}). "
            f"Reducing min to {available_months}."
        )
        min_window = available_months
    
    if min_window > max_window:
        logger.warning(
            f"No valid window range after applying max=months-2 rule (min={min_window}, max={max_window})."
        )
        window_sizes = []
    else:
        window_sizes = list(range(min_window, max_window + 1))
    
    logger.info(f"Date plan: {len(months)} months available ({start_date.date()} to {end_date.date()})")
    logger.info(f"Window config: min={cfg.features.window_sizes_min}, max={cfg.features.window_sizes_max}")
    logger.info(f"Calculated window sizes: {len(window_sizes)} sizes ({min_window} to {max_window})")
    
    return start_date, end_date, months, window_sizes


def _auto_end_date_from_db(db_engine, fallback_end=None) -> dt.date:
    """Auto-detect end date from bccp_orderitem_YYMM tables.
    
    Only uses bccp tables as source of truth - directly queries public schema
    for all tables matching bccp_orderitem_* pattern.
    """
    # Query information_schema directly for bccp tables
    bccp_sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name LIKE 'bccp_orderitem_%'
        ORDER BY table_name DESC
    """
    
    try:
        with db_engine.connect() as conn:
            result = conn.execute(text(bccp_sql))
            rows = result.fetchall()
            yymm_list = []
            
            # Extract YYMM from table names
            for row in rows:
                table_name = row[0]
                matched = re.match(r"bccp_orderitem_(\d{4})$", table_name)
                if matched:
                    yymm_list.append(matched.group(1))
            
            if yymm_list:
                max_yymm = max(yymm_list)
                bccp_max = _yymm_to_date(max_yymm)
                logger.info(f"[BCCP] Found {len(yymm_list)} bccp_orderitem tables, max YYMM: {max_yymm} = {bccp_max}")
                return bccp_max
            else:
                logger.warning(f"[BCCP] Query returned {len(rows)} table rows but no regex matches")
    
    except Exception as e:
        logger.error(f"[BCCP] Failed to query bccp_orderitem tables: {type(e).__name__}: {e}")
    
    # Only fallback to today if bccp query completely fails
    logger.warning("[BCCP] Could not determine end date from bccp tables, using today")
    fallback_date = _to_date(fallback_end) if fallback_end is not None else None
    final_date = fallback_date if fallback_date is not None else dt.date.today()
    return final_date