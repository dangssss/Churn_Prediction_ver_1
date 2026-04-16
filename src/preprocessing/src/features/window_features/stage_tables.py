from libs.db_utils import build_bccp_src
from libs.db_utils import create_bccp_indexes
from logging_config import get_logger

logger = get_logger('window_stage_tables')

WINDOW_SOURCE_TABLE = 'data_window._window_source'
WINDOW_COMPLAINTS_TABLE = 'data_window._window_complaints'
WINDOW_BCCP_TABLE = 'data_window._window_bccp'


def _drop_stage_tables(engine):
    with engine.begin() as conn:
        for table_name in (WINDOW_BCCP_TABLE, WINDOW_COMPLAINTS_TABLE, WINDOW_SOURCE_TABLE):
            try:
                conn.exec_driver_sql(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            except Exception as exc:
                logger.warning(f"Failed to drop staging table {table_name}: {exc}")


def _ensure_source_indexes(engine):
    with engine.begin() as conn:
        statements = [
            "CREATE INDEX IF NOT EXISTS idx_cms_complaint_code_date ON public.cms_complaint(cms_code_enc, create_complaint_date);",
            "CREATE INDEX IF NOT EXISTS idx_cms_complaint_date_range ON public.cms_complaint(create_complaint_date);",
        ]
        for statement in statements:
            conn.exec_driver_sql(statement)


def prepare_stage_tables(engine, start_date: str, end_date: str):
    logger.info("[WINDOW] Cleaning old staging tables")
    _drop_stage_tables(engine)

    logger.info("[WINDOW] Ensuring source indexes")
    _ensure_source_indexes(engine)
    create_bccp_indexes(engine, analyze_sources=False, use_concurrently=True)

    logger.info("[WINDOW] Preparing stage tables from BCCP sources")
    bccp_src = build_bccp_src(engine, start_date, end_date)

    with engine.begin() as conn:
        logger.info("[WINDOW] Building stage: _window_source")
        conn.exec_driver_sql(
            f"""
            CREATE TABLE {WINDOW_SOURCE_TABLE} AS
            SELECT
                cms_code_enc,
                DATE_TRUNC('month', sending_time)::date AS report_month,
                to_char(DATE_TRUNC('month', sending_time), 'YYMM')::text AS month_key,
                to_char(DATE_TRUNC('month', sending_time), 'YYMM')::bigint AS month_key_num,
                COUNT(*)::bigint AS item_count,
                COALESCE(SUM(total_fee), 0)::bigint AS total_fee,
                COALESCE(SUM(total_complaint), 0)::bigint AS total_complaint,
                SUM(CASE WHEN COALESCE(delay_day, 0) > 0 THEN 1 ELSE 0 END)::bigint AS delay_count,
                COALESCE(SUM(delay_day), 0)::bigint AS delay_day,
                SUM(CASE WHEN COALESCE(done, 0) = 0 THEN 1 ELSE 0 END)::bigint AS nodone,
                COALESCE(SUM(refunded), 0)::bigint AS refunded,
                COALESCE(SUM(no_accepted), 0)::bigint AS noaccepted,
                COALESCE(SUM(lost_order), 0)::bigint AS lost_order,
                SUM(CASE WHEN COALESCE(is_domestic, 0) = 1 THEN 1 ELSE 0 END)::bigint AS intra_province,
                SUM(CASE WHEN COALESCE(is_domestic, 0) = 0 THEN 1 ELSE 0 END)::bigint AS international,
                COALESCE(AVG(order_score), 0)::double precision AS order_score,
                COALESCE(AVG(rec_success), 0)::double precision AS satisfaction_score,
                COALESCE(AVG(weight_kg), 0)::double precision AS weight_kg,
                COALESCE(AVG(EXTRACT(day FROM sending_time)), 0)::double precision AS lastday,
                SUM(CASE WHEN UPPER(COALESCE(service_code, '')) = 'C' THEN 1 ELSE 0 END)::bigint AS ser_c,
                SUM(CASE WHEN UPPER(COALESCE(service_code, '')) = 'E' THEN 1 ELSE 0 END)::bigint AS ser_e,
                SUM(CASE WHEN UPPER(COALESCE(service_code, '')) = 'M' THEN 1 ELSE 0 END)::bigint AS ser_m,
                SUM(CASE WHEN UPPER(COALESCE(service_code, '')) = 'P' THEN 1 ELSE 0 END)::bigint AS ser_p,
                SUM(CASE WHEN UPPER(COALESCE(service_code, '')) = 'R' THEN 1 ELSE 0 END)::bigint AS ser_r,
                SUM(CASE WHEN UPPER(COALESCE(service_code, '')) = 'U' THEN 1 ELSE 0 END)::bigint AS ser_u,
                SUM(CASE WHEN UPPER(COALESCE(service_code, '')) = 'L' THEN 1 ELSE 0 END)::bigint AS ser_l,
                SUM(CASE WHEN UPPER(COALESCE(service_code, '')) = 'Q' THEN 1 ELSE 0 END)::bigint AS ser_q
            FROM {bccp_src}
            WHERE sending_time >= TIMESTAMP '{start_date}'
              AND sending_time <= TIMESTAMP '{end_date}'
            GROUP BY cms_code_enc, DATE_TRUNC('month', sending_time);
            """
        )

        logger.info("[WINDOW] Indexing stage: _window_source")
        conn.exec_driver_sql(
            f"CREATE INDEX IF NOT EXISTS idx_window_source_code_month ON {WINDOW_SOURCE_TABLE}(cms_code_enc, report_month);"
        )

        logger.info("[WINDOW] Building stage: _window_complaints")
        conn.exec_driver_sql(
            f"""
            CREATE TABLE {WINDOW_COMPLAINTS_TABLE} AS
            SELECT
                cms_code_enc,
                complaint_code,
                create_complaint_date
            FROM public.cms_complaint
            WHERE create_complaint_date >= DATE '{start_date}' AND create_complaint_date <= DATE '{end_date}';
            """
        )
        logger.info("[WINDOW] Indexing stage: _window_complaints")
        conn.exec_driver_sql(
            f"CREATE INDEX IF NOT EXISTS idx_window_complaints_code_date ON {WINDOW_COMPLAINTS_TABLE}(cms_code_enc, create_complaint_date);"
        )

        logger.info("[WINDOW] Building stage: _window_bccp")
        conn.exec_driver_sql(
            f"CREATE TABLE {WINDOW_BCCP_TABLE} AS SELECT * FROM {bccp_src};"
        )
        logger.info("[WINDOW] Indexing stage: _window_bccp")
        conn.exec_driver_sql(
            f"CREATE INDEX IF NOT EXISTS idx_window_bccp_code_time ON {WINDOW_BCCP_TABLE}(cms_code_enc, sending_time);"
        )

        logger.info("[WINDOW] Running ANALYZE on stage tables")
        conn.exec_driver_sql(f"ANALYZE {WINDOW_SOURCE_TABLE};")
        conn.exec_driver_sql(f"ANALYZE {WINDOW_COMPLAINTS_TABLE};")
        conn.exec_driver_sql(f"ANALYZE {WINDOW_BCCP_TABLE};")


def cleanup_stage_tables(engine):
    _drop_stage_tables(engine)
