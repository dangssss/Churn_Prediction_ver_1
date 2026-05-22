"""
post_ingest_maintenance.py
==========================
Các tác vụ sau khi ingest xong:
  1. Validate row count (source vs DB)
  2. Kiểm tra duplicate
  3. Di chuyển ZIP đã xử lý sang SAVED_DIR
  4. Log reconciliation report
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..logging_config import get_logger
from ..config.paths import SAVED_DIR, FAIL_DIR
from ..jobs.ingest_zip_job import JobResult, CsvIngestResult, MANIFEST_TABLE

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# 1. Row count validation
# ──────────────────────────────────────────────

def validate_all_tables(
    engine: Engine,
    job_result: JobResult,
) -> dict[str, dict]:
    """
    Với mỗi CSV đã ingest, COUNT(*) trong DB và so sánh với source.
    Trả về dict { csv_name: { source, db_count, diff, ok } }.
    """
    report = {}
    for r in job_result.csv_results:
        if r.status not in ("success",):
            continue
        with engine.connect() as conn:
            db_count = conn.execute(
                text(f"SELECT COUNT(*) FROM {r.target_table}")
            ).scalar()

        diff = db_count - r.rows_in_source
        ok   = diff == 0

        report[r.csv_name] = {
            "source":   r.rows_in_source,
            "db_count": db_count,
            "diff":     diff,
            "ok":       ok,
        }
        level = logger.info if ok else logger.warning
        level(
            "VALIDATE | %-45s | source=%10,d | db=%10,d | diff=%+d",
            r.csv_name, r.rows_in_source, db_count, diff,
        )
    return report


# ──────────────────────────────────────────────
# 2. Duplicate check
# ──────────────────────────────────────────────

def check_duplicates(
    engine: Engine,
    table: str,
    natural_key_cols: List[str],
) -> int:
    """
    Đếm số dòng duplicate theo natural key.
    Trả về số lượng duplicate (0 = sạch).
    """
    key_clause = ", ".join(natural_key_cols)
    sql = text(f"""
        SELECT COUNT(*) - COUNT(DISTINCT ({key_clause})) AS dup_count
        FROM {table}
    """)
    with engine.connect() as conn:
        dup_count = conn.execute(sql).scalar()

    if dup_count == 0:
        logger.info("✓ Không có duplicate trong %s", table)
    else:
        logger.error("✗ %,d dòng duplicate trong %s", dup_count, table)
    return dup_count


def remove_duplicates(
    engine: Engine,
    table: str,
    natural_key_cols: List[str],
) -> int:
    """
    Xóa duplicate giữ lại dòng có ctid cao nhất (dòng được insert sau).
    Trả về số dòng đã xóa.
    """
    key_clause = ", ".join(natural_key_cols)
    sql = text(f"""
        DELETE FROM {table}
        WHERE ctid NOT IN (
            SELECT MAX(ctid)
            FROM {table}
            GROUP BY {key_clause}
        )
    """)
    with engine.begin() as conn:
        result = conn.execute(sql)
    deleted = result.rowcount
    logger.info("Đã xóa %,d dòng duplicate khỏi %s", deleted, table)
    return deleted


# ──────────────────────────────────────────────
# 3. Di chuyển ZIP
# ──────────────────────────────────────────────

def archive_zip(zip_path: str, success: bool) -> str:
    """Di chuyển ZIP vào SAVED_DIR (success) hoặc FAIL_DIR (fail)."""
    dest_dir = Path(SAVED_DIR if success else FAIL_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(zip_path).name
    shutil.move(zip_path, dest)
    logger.info("ZIP moved → %s", dest)
    return str(dest)


# ──────────────────────────────────────────────
# 4. Reconciliation report
# ──────────────────────────────────────────────

def print_reconciliation_report(
    engine: Engine,
    zip_mtime: float,
) -> None:
    """In bảng đối soát từ manifest cho lần chạy theo zip_mtime."""
    sql = text(f"""
        SELECT csv_name, target_table, rows_source, rows_inserted,
               ingested_at,
               CASE WHEN rows_source = rows_inserted THEN 'OK' ELSE 'MISMATCH' END AS status
        FROM {MANIFEST_TABLE}
        WHERE zip_mtime = :mtime
        ORDER BY ingested_at
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"mtime": zip_mtime}).fetchall()

    logger.info("══ RECONCILIATION REPORT (zip_mtime=%s) ══", zip_mtime)
    logger.info("%-45s | %12s | %12s | %8s", "CSV", "source_rows", "inserted", "status")
    logger.info("-" * 85)
    for row in rows:
        logger.info(
            "%-45s | %12,d | %12,d | %8s",
            row.csv_name, row.rows_source, row.rows_inserted, row.status,
        )
    logger.info("═" * 85)
