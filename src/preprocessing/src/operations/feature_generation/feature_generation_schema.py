"""Schema preparation and static table maintenance helpers for feature generation."""

from pathlib import Path

from sqlalchemy import text

from logging_config import get_logger

logger = get_logger("feature_generation_database")


def _truncate_or_delete_in_batches(engine, table_name: str, batch_size: int = 50000) -> None:
    """Try TRUNCATE first; fallback to batched DELETE if lock/contention blocks truncate."""
    try:
        with engine.begin() as conn:
            conn.execute(text("SET LOCAL lock_timeout = '5s';"))
            conn.execute(text(f"TRUNCATE TABLE {table_name};"))
        logger.info("  [OK] Truncated cus_lifetime")
        return
    except Exception as truncate_exc:
        logger.warning(f"  [WARN] TRUNCATE blocked/failed, fallback to batched DELETE: {truncate_exc}")

    deleted_total = 0
    while True:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    f"""
                    WITH rows AS (
                        SELECT ctid
                        FROM {table_name}
                        LIMIT :batch_size
                    )
                    DELETE FROM {table_name} t
                    USING rows
                    WHERE t.ctid = rows.ctid;
                    """
                ),
                {"batch_size": batch_size},
            )
            deleted = int(result.rowcount or 0)

        if deleted == 0:
            break

        deleted_total += deleted
        logger.info(f"  [*] Deleted {deleted_total:,} rows from cus_lifetime...")

    logger.info(f"  [OK] Cleared cus_lifetime with batched DELETE ({deleted_total:,} rows)")


def reset_pipeline_schemas(engine, static_sql_path: Path) -> None:
    logger.info("Resetting pipeline schemas...")
    try:
        logger.info("  [*] Ensuring data_static schema exists...")
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS data_static;"))
        logger.info("  [OK] data_static schema ready")

        logger.info("  [*] Truncating data_static.cus_lifetime if exists...")
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'data_static' AND table_name = 'cus_lifetime');"
                )
            )
            table_exists = bool(result.scalar())

        if table_exists:
            _truncate_or_delete_in_batches(engine, "data_static.cus_lifetime")
        else:
            logger.info("  [OK] cus_lifetime will be created fresh")

        logger.info("  [*] Ensuring data_window schema exists (preserve existing tables)...")
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS data_window;"))
        logger.info("  [OK] data_window schema ready")

        logger.info("  [*] Loading static SQL templates...")
        sql_content = static_sql_path.read_text(encoding="utf-8")
        logger.info(f"  [*] Executing {len(sql_content)} bytes of SQL...")
        with engine.begin() as conn:
            conn.execute(text(sql_content))
        logger.info("  [OK] Static SQL templates loaded successfully")
        logger.info("[OK] Schemas prepared and tables ready")
    except Exception as e:
        logger.error(f"[FAILED] Schema reset failed: {e}")
        raise


def count_static_customers(engine) -> int:
    logger.info("Counting static feature records...")
    try:
        with engine.begin() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM data_static.cus_lifetime;"))
            count = int(result.scalar() or 0)
            logger.info(f"[OK] Found {count:,} customer records in data_static.cus_lifetime")
            return count
    except Exception as e:
        logger.error(f"[FAILED] Failed to count static customers: {e}")
        raise
