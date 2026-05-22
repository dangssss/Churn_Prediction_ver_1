"""
ingest_zip_job.py
=================
Job chính để đọc file ZIP, giải nén, ingest từng CSV vào PostgreSQL.

FIX LOG (so với code cũ):
──────────────────────────────────────────────────────────────────────
BUG #1 – TAIL-END CHUNK OVER-COUNTING
  Triệu chứng : log báo 5,200,000 dòng nhưng file chỉ có 5,182,525.
  Nguyên nhân : vòng lặp dùng `offset += CHUNK_SIZE` để tính log thay vì
                cộng `len(chunk)` thực tế → chunk cuối (17,525 dòng) bị
                báo cáo nhầm là 50,000 dòng.
  Fix         : đếm bằng `rows_in_chunk = len(chunk)` thực tế.

BUG #2 – DUPLICATE DO ZIP GHI ĐÈ HÀNG TUẦN
  Triệu chứng : DB thừa ~2M dòng, check duplicate thấy nhiều bản sao.
  Nguyên nhân : Mỗi tuần DE ghi đè file ZIP cũ bằng ZIP mới chứa toàn bộ
                dữ liệu cũ + mới. Nếu không có cơ chế tracking "đã ingest
                file nào", toàn bộ CSV trong ZIP sẽ bị ingest lại từ đầu.
  Fix         : Bảng `ingest_manifest` lưu (zip_mtime, csv_name, sha256,
                row_count). Mỗi lần chạy chỉ ingest file CSV nào chưa có
                record trong manifest HOẶC có sha256 mới.

BUG #3 – KHÔNG TRUNCATE / KHÔNG UPSERT NHẤT QUÁN
  Triệu chứng : Chạy lại job (retry) sau lỗi giữa chừng → duplicate.
  Nguyên nhân : Insert thẳng không có idempotency.
  Fix         : Dùng staging table → TRUNCATE staging → bulk copy vào
                staging → INSERT INTO prod SELECT … WHERE NOT EXISTS
                (hoặc ON CONFLICT DO NOTHING nếu có natural key).
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import io
import os
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ..logging_config import get_logger
from ..config.csv_schema import CSV_SCHEMA_MAP          # {csv_stem: dtype_dict}
from ..config.table_schema import TABLE_MAP             # {csv_stem: pg_table_name}
from ..config.paths import (
    INCOMING_DIR, SAVED_DIR, FAIL_DIR, LOGS_DIR,
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PW,
)

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Hằng số
# ──────────────────────────────────────────────
CHUNK_SIZE: int = 50_000          # dòng / lần đọc
MANIFEST_TABLE: str = "ingest_manifest"

# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────

@dataclass
class CsvIngestResult:
    csv_name: str
    target_table: str
    rows_in_source: int = 0       # số dòng THỰC TẾ đọc được từ file
    rows_inserted: int = 0        # số dòng thực sự ghi vào DB
    rows_skipped_dup: int = 0     # số dòng bị bỏ qua do trùng
    status: str = "pending"       # pending | success | skipped | error
    error_msg: str = ""
    duration_sec: float = 0.0

@dataclass
class JobResult:
    zip_path: str
    zip_mtime: float
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    csv_results: List[CsvIngestResult] = field(default_factory=list)
    total_rows_source: int = 0
    total_rows_inserted: int = 0

    def summary_log(self) -> str:
        lines = [
            f"╔══ INGEST SUMMARY ══════════════════════════════",
            f"║ ZIP       : {self.zip_path}",
            f"║ ZIP mtime : {datetime.utcfromtimestamp(self.zip_mtime).isoformat()}",
            f"║ Started   : {self.started_at.isoformat()}",
            f"║ Finished  : {self.finished_at.isoformat() if self.finished_at else 'N/A'}",
            f"╠══ PER-FILE ════════════════════════════════════",
        ]
        for r in self.csv_results:
            match_marker = "✓" if r.rows_in_source == r.rows_inserted + r.rows_skipped_dup else "⚠ MISMATCH"
            lines.append(
                f"║ {r.csv_name:<40} "
                f"status={r.status:<8} "
                f"source={r.rows_in_source:>10,} "
                f"inserted={r.rows_inserted:>10,} "
                f"dup_skip={r.rows_skipped_dup:>8,} "
                f"{match_marker}"
            )
        lines += [
            f"╠══ TOTALS ══════════════════════════════════════",
            f"║ Total source rows : {self.total_rows_source:>12,}",
            f"║ Total inserted    : {self.total_rows_inserted:>12,}",
            f"╚════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _build_engine() -> Engine:
    url = f"postgresql+psycopg2://{PG_USER}:{PG_PW}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(url, pool_pre_ping=True)


def _sha256_of_zip_member(zf: zipfile.ZipFile, member: str) -> str:
    """Tính SHA-256 của một entry bên trong ZIP mà không giải nén ra đĩa."""
    h = hashlib.sha256()
    with zf.open(member) as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ensure_manifest(engine: Engine) -> None:
    """Tạo bảng manifest nếu chưa có."""
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {MANIFEST_TABLE} (
        id              BIGSERIAL PRIMARY KEY,
        zip_path        TEXT        NOT NULL,
        zip_mtime       DOUBLE PRECISION NOT NULL,
        csv_name        TEXT        NOT NULL,
        csv_sha256      TEXT        NOT NULL,
        target_table    TEXT        NOT NULL,
        rows_source     BIGINT,
        rows_inserted   BIGINT,
        ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_manifest UNIQUE (zip_mtime, csv_name, csv_sha256)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Manifest table ensured: %s", MANIFEST_TABLE)


def _is_already_ingested(
    engine: Engine,
    zip_mtime: float,
    csv_name: str,
    csv_sha256: str,
) -> bool:
    """Trả về True nếu đúng combo (zip_mtime, csv_name, sha256) đã có trong manifest."""
    sql = text(f"""
        SELECT 1 FROM {MANIFEST_TABLE}
        WHERE zip_mtime = :mtime
          AND csv_name  = :name
          AND csv_sha256 = :sha
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"mtime": zip_mtime, "name": csv_name, "sha": csv_sha256}).fetchone()
    return row is not None


def _write_manifest(
    engine: Engine,
    zip_path: str,
    zip_mtime: float,
    csv_name: str,
    csv_sha256: str,
    target_table: str,
    rows_source: int,
    rows_inserted: int,
) -> None:
    sql = text(f"""
        INSERT INTO {MANIFEST_TABLE}
            (zip_path, zip_mtime, csv_name, csv_sha256, target_table, rows_source, rows_inserted)
        VALUES
            (:zip_path, :zip_mtime, :csv_name, :csv_sha256, :target_table, :rows_source, :rows_inserted)
        ON CONFLICT ON CONSTRAINT uq_manifest DO UPDATE
            SET rows_source   = EXCLUDED.rows_source,
                rows_inserted = EXCLUDED.rows_inserted,
                ingested_at   = now()
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "zip_path": zip_path, "zip_mtime": zip_mtime,
            "csv_name": csv_name, "csv_sha256": csv_sha256,
            "target_table": target_table,
            "rows_source": rows_source, "rows_inserted": rows_inserted,
        })


# ──────────────────────────────────────────────
# BUG #1 FIX: Chunked CSV reader đếm ĐÚNG
# ──────────────────────────────────────────────

def _iter_csv_chunks(
    zf: zipfile.ZipFile,
    member: str,
    dtype: Optional[dict],
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[tuple[pd.DataFrame, int, int]]:
    """
    Generator đọc CSV theo chunk, trả về (chunk_df, chunk_index, rows_in_chunk).

    QUAN TRỌNG:
      - `rows_in_chunk = len(chunk)` ← đếm THỰC TẾ (fix Bug #1)
      - Không dùng `offset * CHUNK_SIZE` để log vì chunk cuối < CHUNK_SIZE
      - pandas.read_csv với chunksize tự xử lý tail-end đúng; không cần
        làm thủ công skiprows/nrows.
    """
    with zf.open(member) as raw:
        # Wrap trong TextIOWrapper để pandas đọc được
        text_stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        reader = pd.read_csv(
            text_stream,
            chunksize=chunk_size,
            dtype=dtype,
            low_memory=False,
        )
        for chunk_idx, chunk in enumerate(reader):
            rows_in_chunk = len(chunk)          # ← ĐÚNG: đếm dòng thực tế
            yield chunk, chunk_idx, rows_in_chunk


# ──────────────────────────────────────────────
# BUG #3 FIX: Staging → Production idempotent
# ──────────────────────────────────────────────

def _ingest_chunk_to_staging(
    conn,                           # psycopg2 raw connection
    staging_table: str,
    chunk: pd.DataFrame,
) -> int:
    """Bulk-copy một chunk vào staging table, trả về số dòng copy được."""
    cols = list(chunk.columns)
    buf = io.StringIO()
    chunk.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cursor = conn.cursor()
    cursor.copy_expert(
        f"COPY {staging_table} ({', '.join(cols)}) FROM STDIN WITH CSV NULL '\\N'",
        buf,
    )
    return cursor.rowcount


def _merge_staging_to_prod(
    engine: Engine,
    staging_table: str,
    target_table: str,
    natural_key_cols: Optional[List[str]] = None,
) -> tuple[int, int]:
    """
    Merge staging → production.
    - Nếu có natural_key_cols: ON CONFLICT DO NOTHING (dedup by natural key).
    - Nếu không: INSERT … WHERE NOT EXISTS (exact-row dedup).
    Trả về (rows_inserted, rows_skipped).
    """
    with engine.begin() as conn:
        # Đếm staging trước khi merge
        total_staging = conn.execute(text(f"SELECT COUNT(*) FROM {staging_table}")).scalar()

        if natural_key_cols:
            key_clause = ", ".join(natural_key_cols)
            sql = f"""
                INSERT INTO {target_table}
                SELECT * FROM {staging_table}
                ON CONFLICT ({key_clause}) DO NOTHING
            """
        else:
            # Exact-row dedup: dùng hash của toàn bộ row
            sql = f"""
                INSERT INTO {target_table}
                SELECT s.* FROM {staging_table} s
                WHERE NOT EXISTS (
                    SELECT 1 FROM {target_table} t
                    WHERE t::text = s::text
                )
            """

        result = conn.execute(text(sql))
        rows_inserted = result.rowcount
        rows_skipped = total_staging - rows_inserted

    return rows_inserted, rows_skipped


# ──────────────────────────────────────────────
# Core: ingest một CSV
# ──────────────────────────────────────────────

def _ingest_one_csv(
    engine: Engine,
    raw_conn,                       # psycopg2 raw conn (cho COPY)
    zf: zipfile.ZipFile,
    member: str,
    target_table: str,
    dtype: Optional[dict],
    natural_key_cols: Optional[List[str]],
    zip_path: str,
    zip_mtime: float,
    csv_sha256: str,
) -> CsvIngestResult:

    result = CsvIngestResult(csv_name=member, target_table=target_table)
    t0 = time.perf_counter()
    staging_table = f"_staging_{target_table}"

    try:
        logger.info("[%s] Bắt đầu ingest → %s", member, target_table)

        # ── 1. Tạo staging table (clone structure từ prod) ──────────────
        with engine.begin() as conn:
            conn.execute(text(
                f"CREATE TEMP TABLE IF NOT EXISTS {staging_table} "
                f"(LIKE {target_table} INCLUDING DEFAULTS)"
            ))
            conn.execute(text(f"TRUNCATE {staging_table}"))

        # ── 2. Đọc + COPY vào staging theo chunk ────────────────────────
        cumulative_rows = 0
        for chunk, chunk_idx, rows_in_chunk in _iter_csv_chunks(zf, member, dtype):
            _ingest_chunk_to_staging(raw_conn, staging_table, chunk)
            cumulative_rows += rows_in_chunk           # ← cộng dồn ĐÚNG

            # Log mỗi 10 chunk để có audit trail
            if (chunk_idx + 1) % 10 == 0 or rows_in_chunk < CHUNK_SIZE:
                logger.info(
                    "[%s] chunk #%d | dòng trong chunk: %,d | lũy kế: %,d",
                    member, chunk_idx + 1, rows_in_chunk, cumulative_rows,
                )

        raw_conn.commit()
        result.rows_in_source = cumulative_rows
        logger.info("[%s] Đọc xong: %,d dòng thực tế", member, cumulative_rows)

        # ── 3. Merge staging → production (idempotent) ──────────────────
        inserted, skipped = _merge_staging_to_prod(
            engine, staging_table, target_table, natural_key_cols
        )
        result.rows_inserted   = inserted
        result.rows_skipped_dup = skipped

        # ── 4. Kiểm tra số dòng khớp ────────────────────────────────────
        if result.rows_in_source != result.rows_inserted + result.rows_skipped_dup:
            logger.warning(
                "[%s] ⚠ MISMATCH: source=%,d ≠ inserted(%,d) + dup_skip(%,d)",
                member, result.rows_in_source, result.rows_inserted, result.rows_skipped_dup,
            )
        else:
            logger.info(
                "[%s] ✓ Số dòng khớp: source=%,d inserted=%,d dup_skip=%,d",
                member, result.rows_in_source, result.rows_inserted, result.rows_skipped_dup,
            )

        # ── 5. Ghi manifest ─────────────────────────────────────────────
        _write_manifest(
            engine, zip_path, zip_mtime, member, csv_sha256,
            target_table, result.rows_in_source, result.rows_inserted,
        )

        result.status = "success"

    except Exception as exc:
        raw_conn.rollback()
        result.status    = "error"
        result.error_msg = str(exc)
        logger.exception("[%s] LỖI khi ingest: %s", member, exc)

    finally:
        result.duration_sec = time.perf_counter() - t0

    return result


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def run_ingest_zip_job(
    zip_path: Optional[str] = None,
    natural_key_map: Optional[dict] = None,
    force_reingest: bool = False,
) -> JobResult:
    """
    Job chính.

    Args:
        zip_path       : Đường dẫn đến file .zip. Mặc định lấy từ INCOMING_DIR.
        natural_key_map: {csv_stem: [col1, col2, ...]} – khóa tự nhiên để dedup.
                         Nếu None, dùng exact-row dedup.
        force_reingest : Nếu True, bỏ qua manifest và ingest lại từ đầu.
    """
    if zip_path is None:
        # Tìm file zip mới nhất trong INCOMING_DIR
        zips = sorted(Path(INCOMING_DIR).glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if not zips:
            raise FileNotFoundError(f"Không tìm thấy file .zip trong {INCOMING_DIR}")
        zip_path = str(zips[-1])

    zip_mtime = Path(zip_path).stat().st_mtime
    job = JobResult(zip_path=zip_path, zip_mtime=zip_mtime)

    logger.info("══ BẮT ĐẦU JOB INGEST ══")
    logger.info("ZIP: %s  (mtime=%s)", zip_path, datetime.utcfromtimestamp(zip_mtime).isoformat())

    engine = _build_engine()
    _ensure_manifest(engine)

    # Raw psycopg2 connection dùng cho COPY (SQLAlchemy không hỗ trợ copy_expert)
    raw_conn = psycopg2.connect(
        host=PG_HOST, port=int(PG_PORT), dbname=PG_DB,
        user=PG_USER, password=PG_PW,
    )
    raw_conn.autocommit = False

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            logger.info("ZIP chứa %d file CSV: %s", len(csv_members), csv_members)

            for member in csv_members:
                csv_stem = Path(member).stem

                # Lookup schema và table name
                dtype        = CSV_SCHEMA_MAP.get(csv_stem)
                target_table = TABLE_MAP.get(csv_stem)
                if target_table is None:
                    logger.warning("[%s] Không tìm thấy target table → bỏ qua", member)
                    continue

                natural_key_cols = (natural_key_map or {}).get(csv_stem)

                # ── BUG #2 FIX: Kiểm tra manifest trước khi ingest ─────
                csv_sha256 = _sha256_of_zip_member(zf, member)
                logger.info("[%s] SHA-256: %s", member, csv_sha256)

                if not force_reingest and _is_already_ingested(engine, zip_mtime, member, csv_sha256):
                    logger.info("[%s] ✓ Đã ingest với SHA này → BỎ QUA", member)
                    r = CsvIngestResult(
                        csv_name=member, target_table=target_table, status="skipped"
                    )
                    job.csv_results.append(r)
                    continue

                # Ingest
                result = _ingest_one_csv(
                    engine, raw_conn, zf, member, target_table,
                    dtype, natural_key_cols, zip_path, zip_mtime, csv_sha256,
                )
                job.csv_results.append(result)
                job.total_rows_source   += result.rows_in_source
                job.total_rows_inserted += result.rows_inserted

    finally:
        raw_conn.close()

    job.finished_at = datetime.utcnow()
    logger.info("\n%s", job.summary_log())
    return job
