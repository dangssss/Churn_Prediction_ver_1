
import os
import sys
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_fixed


# In Docker: file is at /churn_source/ingestion/validate_ingest.py
# We need to add /churn_source/ingestion to sys.path to allow `from Data_pull...`
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from Data_pull.resources import PostgresConfig, get_pg_conn
from Data_pull.logging_config import get_logger

logger = get_logger("validate_ingest")

# TODO: NAMNT Kiem tra lai

def _scalar_count(conn, sql: str) -> int:
    """
    Trả về COUNT(*) từ query.
    Hỗ trợ:
      - SQLAlchemy Connection: conn.execute(...)
      - psycopg2 connection: conn.cursor().execute(...)
    """
    # SQLAlchemy style
    if hasattr(conn, "execute"):
        try:
            # Nếu dùng SQLAlchemy 1.4/2.0 thì text() là chuẩn
            from sqlalchemy import text
            return int(conn.execute(text(sql)).scalar())
        except Exception:
            # Nếu conn.execute nhận string trực tiếp thì cũng ok
            return int(conn.execute(sql).scalar())

    # psycopg2 style (Fix: MUST use cursor)
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    reraise=True,  # quan tr?ng: raise l?i ra ngoài d? Airflow th?y FAIL và retry task
)
def validate_ingestion() -> None:
    """
    Validate ingestion:
      - Ki?m tra b?ng ingest.bccp_orderitem có d? li?u hay không.
    """

    logger.info("Starting validation...")
    
    pg_cfg = PostgresConfig.from_env()

    # 1. Validation Level 1: Check Ingest Log (Schema ingest)
    logger.info("--- [1/2] Checking Ingest Log ---")
    log_sql = "SELECT COUNT(*) FROM ingest.ingest_log WHERE status = 'success'"
    
    with get_pg_conn(pg_cfg) as conn:
        # DB Connection Sanity Check
        with conn.cursor() as cur:
             cur.execute("SELECT 1")
        
        # Check Log Table
        try:
             log_count = _scalar_count(conn, log_sql)
             logger.info(f"Total successful jobs in ingest.ingest_log: {log_count}")
             
             if log_count == 0:
                 logger.warning("No successful ingestions recorded in ingest_log yet.")
                 # Not raising error here, letting data check decide
             else:
                 logger.info("âœ… Found successful ingestion logs.")
        except Exception as e:
             logger.warning(f"Could not read ingest.ingest_log: {e}")

        # 2. Validation Level 2: Check Actual Data (Schema public)
        logger.info("--- [2/2] Checking Public Data Tables (Dynamic) ---")
        
        # Vì tên bảng là dynamic (bccp_orderitem_2401, bccp_orderitem_2402...)
        # Ta sẽ tìm xem có BẤT KỲ bảng bccp_orderitem nào không.
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'bccp_orderitem_%'
                LIMIT 1;
            """)
            found_table = cur.fetchone()
            
        if found_table:
            table_name = found_table[0]
            logger.info(f"Found at least one data table: public.{table_name}")
            
            # Count sample rows from this table
            data_sql = f"SELECT COUNT(*) FROM public.{table_name}"
            row_count = _scalar_count(conn, data_sql)
            logger.info(f"Rows in public.{table_name}: {row_count}")
            
            if row_count > 0:
                logger.info("Data validation PASSED.")
            else:
                logger.warning(f"public.{table_name} exists but is empty.")
                # Empty table is suspicious but technically "ingest success" if file was empty
        else:
            msg = "No table matching 'public.bccp_orderitem_%' found."
            logger.error(msg)
            raise RuntimeError(msg)


if __name__ == "__main__":
    try:
        validate_ingestion()
    except Exception as e:
        logger.error(f"? Validation Failed: {e}")
        sys.exit(1)

