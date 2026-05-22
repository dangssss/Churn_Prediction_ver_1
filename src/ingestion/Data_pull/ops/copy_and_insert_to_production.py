# ops/copy_and_insert_to_production.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import csv
import io
import os

import psycopg2
import pandas as pd

from Data_pull.resources import PostgresConfig, get_pg_conn
from Data_pull.logging_config import get_logger
from Data_pull.ops.data_transformations import SafeTypeCaster

logger = get_logger(__name__)

from Data_pull.config.csv_schema import (
    get_table_config,
    get_text_cols,
    get_datetime_cols,
    SOURCE_HAS_HEADER,
    CSV_INJECTION_GUARD,
    BATCH_ROWS,
)

from Data_pull.ops.data_transformations import (
    CustomerEncryption,
    transform_bccp_orderitem_row,
    transform_cms_complaint_row,
    transform_cas_customer_row,
    transform_cas_info_row,
)


class CsvHeaderMismatchError(Exception):
    """
    Được raise khi header CSV (số lượng cột) không khớp với EXPECTED_HEADERS
    cho base tương ứng. Dùng để dừng ingest và đẩy file vào fail_ingest.
    """
    pass


# ============================================================
# EXPECTED HEADERS (canonical column order per table)
# Map CSV header → canonical header bằng thứ tự (position-based)
# ============================================================

COMPLAINT_CODES = [114, 115, 116, 134, 194, 554, 595,
                   314, 594, 274, 614, 654, 234, 174]

EXPECTED_HEADERS = {
    "bccp_orderitem": [
        "crm_code_enc", "cms_code_enc", "item_code_enc", "service_code",
        "weight_kg", "length_size", "width_size", "height_size",
        "total_fee", "is_domestic", "country_code",
        "send_province_code", "send_district_code", "send_commune_code",
        "rec_province_code", "rec_district_code", "rec_commune_code",
        "region", "sending_time", "ending_time",
        "rec_success", "refunded", "no_accepted", "lost_order",
        "delay_day", "done", "total_complaint",
        *[f"complaint{c}" for c in COMPLAINT_CODES],
        "order_score", "bccp_update_date",
    ],  # CSV có item_code_enc, transform map sang item_code
    "cas_customer": [
        "cms_code_enc", "report_month", "item_count", "weight_kg", "total_fee",
        "intra_province", "international",
        "ser_c", "ser_e", "ser_m", "ser_p", "ser_r", "ser_u", "ser_l", "ser_q",
        "delay_day", "delay_count", "nodone", "refunded", "noaccepted",
        "lost_order", "lastday", "noservice", "dev_item",
        "order_score", "satisfaction_score", "total_complaint",
        *[f"complaint{c}" for c in COMPLAINT_CODES],
        "updated_at",
    ],  # 37 columns (CSV thật không có etl_date)
    "cms_complaint": [
        "cms_code_enc", "item_code",
        "create_complaint_date", "exp_complaint_date", "close_complaint_date",
        "delay_complaint", "complaint_code", "complaint_content",
        "complaint_content_bit", "complaint_update_date", "etl_date",
    ],  # CSV có cms_code, map theo position sang cms_code_enc (canonical)
    "cas_info": [
        "cms_code_enc", "crm_code_enc", "cus_province",
        "contract_service", "tenure", "custype",
        "customer_update_date", "contract_classify",
        "contract_sig_first", "contract_mgr_org", "cus_poscode",
    ],
}

# Transform function dispatch để tránh if-elif chain
TRANSFORM_DISPATCH = {
    "bccp_orderitem": transform_bccp_orderitem_row,
    "cms_complaint": transform_cms_complaint_row,
    "cas_customer": transform_cas_customer_row,
    "cas_info": transform_cas_info_row,
}

# ============================================================
# DEDUP KEYS — khoá tự nhiên để phát hiện duplicate sau COPY
# Nếu có khoá tự nhiên → dùng để dedup (nhanh hơn)
# Nếu không có → so sánh toàn bộ row (chậm hơn, dùng làm fallback)
# ============================================================
DEDUP_KEYS: Dict[str, List[str]] = {
    # bccp_orderitem: mỗi đơn hàng có item_code_enc duy nhất
    "bccp_orderitem": ["item_code_enc"],
    # cms_complaint: 1 khiếu nại = 1 (customer, item, ngày tạo, mã KN)
    "cms_complaint": ["cms_code_enc", "item_code", "create_complaint_date", "complaint_code"],
    # cas_customer: 1 customer chỉ có 1 bản ghi mỗi tháng
    "cas_customer": ["cms_code_enc", "report_month"],
    # cas_info: 1 customer chỉ có 1 bản ghi thông tin
    "cas_info": ["cms_code_enc"],
}


def _dedup_exact_duplicates(cur, conn, prod_tbl: str, base: str, columns_list: Optional[List[str]] = None) -> int:
    """
    Phát hiện và xoá duplicate rows sau khi COPY.
    
    Quy trình 2 bước:
      1. Xoá trùng lặp Y HỆT NHAU (tất cả các cột giống hệt nhau):
         - Lấy danh sách cột thực tế của bảng từ information_schema.
         - Sử dụng ctid để chỉ giữ lại 1 row đầu tiên cho mỗi nhóm trùng toàn bộ cột.
      2. Xoá trùng lặp theo khoá tự nhiên DEDUP_KEYS nếu có cấu hình:
         - Giữ lại row có ctid nhỏ nhất cho mỗi nhóm trùng key (vd: item_code_enc).
    
    Returns:
        Tổng số dòng đã bị xoá (0 = không có duplicate).
    """
    # Parse schema và table name từ prod_tbl (vd: public."bccp_orderitem_2501")
    parts = prod_tbl.split('.')
    if len(parts) == 2:
        schema = parts[0].strip().strip('"')
        tbl_name = parts[1].strip().strip('"')
    else:
        schema = 'public'
        tbl_name = prod_tbl.strip().strip('"')

    # Bước 1: Xoá trùng lặp y hệt nhau (tất cả các cột trùng nhau)
    if columns_list is not None:
        columns = [f'"{col}"' for col in columns_list]
    else:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
        """, (schema, tbl_name))
        columns = [f'"{r[0]}"' for r in cur.fetchall()]

    deleted_exact = 0
    if columns:
        cols_joined = ", ".join(columns)
        count_exact_sql = f"""
            SELECT COALESCE(SUM(cnt - 1), 0)
            FROM (
                SELECT COUNT(*) AS cnt
                FROM {prod_tbl}
                GROUP BY {cols_joined}
                HAVING COUNT(*) > 1
            ) sub;
        """
        cur.execute(count_exact_sql)
        row = cur.fetchone()
        exact_dup_count = int(row[0]) if row else 0

        if exact_dup_count > 0:
            logger.warning(
                f"[DEDUP] {prod_tbl}: phát hiện {exact_dup_count:,} dòng trùng lặp Y HỆT NHAU (tất cả các cột). Đang xoá..."
            )
            delete_exact_sql = f"""
                DELETE FROM {prod_tbl}
                WHERE ctid NOT IN (
                    SELECT MIN(ctid)
                    FROM {prod_tbl}
                    GROUP BY {cols_joined}
                );
            """
            cur.execute(delete_exact_sql)
            deleted_exact = cur.rowcount
            conn.commit()
            logger.warning(
                f"[DEDUP] {prod_tbl}: đã xoá {deleted_exact:,} dòng trùng lặp y hệt nhau."
            )
        else:
            logger.info(f"[DEDUP] {prod_tbl}: không phát hiện dòng trùng lặp y hệt nhau.")
    else:
        logger.warning(f"[DEDUP] {prod_tbl}: không lấy được danh sách cột từ schema, bỏ qua bước dedup trùng lặp y hệt.")

    # Bước 2: Xoá trùng lặp theo khoá tự nhiên
    deleted_key = 0
    dedup_cols = DEDUP_KEYS.get(base)

    if dedup_cols:
        key_cols_sql = ', '.join([f'"{c}"' for c in dedup_cols])
        count_key_sql = f"""
            SELECT COALESCE(SUM(cnt - 1), 0)
            FROM (
                SELECT COUNT(*) AS cnt
                FROM {prod_tbl}
                GROUP BY {key_cols_sql}
                HAVING COUNT(*) > 1
            ) sub;
        """
        cur.execute(count_key_sql)
        row = cur.fetchone()
        key_dup_count = int(row[0]) if row else 0

        if key_dup_count > 0:
            logger.warning(
                f"[DEDUP] {prod_tbl}: phát hiện {key_dup_count:,} dòng trùng khóa tự nhiên {dedup_cols} (khác biệt ở các cột khác). Đang xoá..."
            )
            delete_key_sql = f"""
                DELETE FROM {prod_tbl}
                WHERE ctid NOT IN (
                    SELECT MIN(ctid)
                    FROM {prod_tbl}
                    GROUP BY {key_cols_sql}
                );
            """
            cur.execute(delete_key_sql)
            deleted_key = cur.rowcount
            conn.commit()
            logger.warning(
                f"[DEDUP] {prod_tbl}: đã xoá {deleted_key:,} dòng trùng khóa tự nhiên."
            )
        else:
            logger.info(f"[DEDUP] {prod_tbl}: không phát hiện trùng khóa tự nhiên (key={dedup_cols}).")

    return deleted_exact + deleted_key


def get_csv_header(csv_file: Path) -> List[str]:
    """
    Đọc dòng đầu của CSV file để lấy header.
    Delimiter: ';', Encoding: utf-8-sig
    """
    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        first_row = next(reader, None)
        if not first_row:
            raise ValueError(f"CSV file is empty: {csv_file}")
        return first_row


def copy_and_insert_to_production(
    meta: Dict[str, Any],
    pg_cfg: PostgresConfig,
    *,
    batch_rows: int = BATCH_ROWS,
    source_has_header: bool = SOURCE_HAS_HEADER,
    injection_mode: str = CSV_INJECTION_GUARD,
    use_encryption: bool = True,
    encryption_mapping_file: Optional[str] = None,
) -> int:
    """
    Read CSV files and insert directly into production table with data transformation.
    
    Supports 4 tables:
    - bccp_orderitem (monthly mode)
    - cms_complaint (monthly mode)
    - cas_customer (snapshot mode)
    - cas_info (snapshot mode)
    """
    from Data_pull.config.table_schema import get_prod_table_ddl
    
    base = meta["base"]
    table_name = meta["table_name"]  # vd: bccp_orderitem_2501, cas_customer
    csv_files: List[Path] = meta.get("csv_files", [])
    mode = meta.get("mode", "monthly")
    prod_schema = "public"
    
    if not csv_files:
        logger.warning(f"No CSV files to ingest for {table_name}")
        return 0
    
    # Get table config (text_cols, datetime_cols, mode)
    try:
        table_cfg = get_table_config(base)
    except ValueError as e:
        logger.error(f"{e}")
        raise
    
    text_cols = table_cfg.get("text_cols", set())
    datetime_cols = table_cfg.get("datetime_cols", set())
    
    prod_tbl = f'{prod_schema}."{table_name}"'
    
    # Setup encryption nếu cần
    encrypto = None
    if use_encryption:
        encrypto = CustomerEncryption()
        if encryption_mapping_file and Path(encryption_mapping_file).exists():
            try:
                encrypto.load_mapping(encryption_mapping_file)
                logger.info(f"Loaded encryption mapping from {encryption_mapping_file}")
            except Exception as e:
                logger.warning(f"Could not load encryption mapping: {e}")
    
    logger.info(f"COPY & CAST -> production: {prod_tbl} | base={base} | mode={mode} | files={len(csv_files)}")
    
    # ===== Lấy header từ CSV file đầu tiên =====
    first_csv = csv_files[0]
    try:
        headers_raw = get_csv_header(first_csv)
        headers_raw = [h.strip() for h in headers_raw]
        logger.info(f"Read header from {first_csv.name}: {len(headers_raw)} columns (raw: {headers_raw[:5]}...)")
    except Exception as e:
        logger.error(f"Failed to read header from {first_csv.name}: {e}")
        raise

    # ===== Map CSV header → canonical header (STRICT, position-based) =====
    expected = EXPECTED_HEADERS.get(base)
    header_map: Dict[str, str] = {}

    if expected is not None:
        # Bắt buộc số cột khớp với schema
        if len(expected) != len(headers_raw):
            msg = (
                f"Header count mismatch for base={base}: "
                f"expected {len(expected)} columns, got {len(headers_raw)}. "
                f"CSV đang bị thiếu hoặc thừa cột so với schema EXPECTED_HEADERS.\n"
                f"Expected: {expected[:10]}{'...' if len(expected) > 10 else ''}\n"
                f"Got:      {headers_raw[:10]}{'...' if len(headers_raw) > 10 else ''}"
            )
            logger.error(msg)
            # Raise để ingest_zip_job bắt được và move ZIP sang fail_ingest
            raise CsvHeaderMismatchError(msg)

        # Số cột khớp → map theo vị trí: cột i file → cột i canonical
        header_map = {headers_raw[i]: expected[i] for i in range(len(expected))}
        headers = expected[:]  # canonical order

        mismatches = [
            (headers_raw[i], expected[i])
            for i in range(len(expected))
            if headers_raw[i] != expected[i]
        ]
        if mismatches:
            logger.warning(
                "Column name mismatches detected (will map by position for base=%s):",
                base,
            )
            for csv_col, canonical_col in mismatches:
                logger.warning("  CSV: '%s' → Canonical: '%s'", csv_col, canonical_col)

        logger.info(
            "Using canonical header order for base=%s (%d columns)",
            base,
            len(headers),
        )
    else:
        # Base chưa có EXPECTED_HEADERS → cho phép dùng header raw
        header_map = {h: h for h in headers_raw}
        headers = headers_raw
        logger.info(
            "Using raw header from CSV (no EXPECTED_HEADERS for base=%s): %d columns",
            base,
            len(headers),
        )
    
    # ===== Kết nối DB =====
    conn = get_pg_conn(pg_cfg)
    conn.autocommit = False
    cur = conn.cursor()
    
    try:
        # 0) Đảm bảo schema production tồn tại
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS {prod_schema};')
        conn.commit()
        logger.info(f"Ensured production schema: {prod_schema}")
        
        # 1) Tạo bảng production với ĐÚNG kiểu dữ liệu (INT, TIMESTAMPTZ, etc.)
        ddl = get_prod_table_ddl(base, table_name, prod_schema)
        cur.execute(ddl)
        conn.commit()
        logger.info(f"Ensured production table: {prod_tbl}")
        
        # 2) Đảm bảo bảng manifest tồn tại và lấy danh sách csv_hashes
        try:
            from unzip_and_discover import ensure_manifest_table
        except ImportError:
            # Fallback if imported differently
            def ensure_manifest_table(cur) -> None:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS public.ingestion_manifest (
                    zip_name text NOT NULL,
                    csv_name text NOT NULL,
                    file_sha256 varchar(64) NOT NULL PRIMARY KEY,
                    ingested_at timestamptz DEFAULT now()
                );
                """)
        ensure_manifest_table(cur)
        conn.commit()

        # 3) Đọc CSV và transform data
        total_rows = 0
        audit_log = []
        total_read_all = 0
        total_skipped_all = 0
        total_inserted_all = 0
        total_deleted_all = 0
        
        csv_hashes = meta.get("csv_hashes", {})

        for csv_file in csv_files:
            logger.info(f"Reading {csv_file.name}")
            
            # Tính hoặc lấy SHA256
            file_sha256 = csv_hashes.get(str(csv_file))
            if not file_sha256:
                import hashlib
                sha = hashlib.sha256()
                with open(csv_file, "rb") as f:
                    for chunk in iter(lambda: f.read(4096 * 1024), b""):
                        sha.update(chunk)
                file_sha256 = sha.hexdigest()

            # Kiểm tra xem đã ingest chưa
            cur.execute("SELECT 1 FROM public.ingestion_manifest WHERE file_sha256 = %s LIMIT 1;", (file_sha256,))
            if cur.fetchone():
                logger.info(f"CSV file {csv_file.name} (SHA256: {file_sha256}) already in manifest database. Skipping.")
                continue

            file_rows_read = 0
            file_rows_skipped = 0
            file_rows_inserted = 0
            file_rows_staged = 0
            file_deleted = 0

            # Tạo bảng Staging tạm thời
            # Dùng prefix _stg_ để tránh trùng tên với production table
            staging_tbl = f"_stg_{table_name}"
            cur.execute(f"DROP TABLE IF EXISTS {staging_tbl};")
            cur.execute(f"CREATE TEMP TABLE {staging_tbl} (LIKE {prod_tbl});")
            conn.commit()

            columns_to_insert = []

            try:
                # Đọc CSV theo chunk
                chunks = pd.read_csv(
                    csv_file,
                    sep=";",
                    chunksize=batch_rows,
                    dtype=str,
                    keep_default_na=False,
                    encoding="utf-8-sig"
                )
                
                first_chunk = True
                for chunk in chunks:
                    chunk_len = len(chunk)
                    file_rows_read += chunk_len
                    
                    if first_chunk:
                        chunk_cols = [h.strip() for h in chunk.columns]
                        if len(chunk_cols) != len(headers_raw):
                            msg = (
                                f"Header mismatch in {csv_file.name}: "
                                f"expected {len(headers_raw)} columns (from first CSV), "
                                f"got {len(chunk_cols)} columns."
                            )
                            logger.error(msg)
                            raise CsvHeaderMismatchError(msg)
                        first_chunk = False

                    rows_buffer = []
                    records = chunk.to_dict('records')
                    for raw_row in records:
                        raw_row = {
                            (k.strip() if isinstance(k, str) else k): v
                            for k, v in raw_row.items()
                        }
                        
                        normalized_row = {}
                        for k, v in raw_row.items():
                            canonical_key = header_map.get(k, k)
                            normalized_row[canonical_key] = v
                        
                        transform_func = TRANSFORM_DISPATCH.get(base)
                        if transform_func is None:
                            file_rows_skipped += 1
                            continue
                            
                        transformed = transform_func(normalized_row, encrypto)
                        if transformed is None:
                            file_rows_skipped += 1
                            continue
                            
                        rows_buffer.append(transformed)

                    if rows_buffer:
                        if not columns_to_insert:
                            columns_to_insert = list(rows_buffer[0].keys())
                        _bulk_insert_rows(cur, staging_tbl, rows_buffer, base)
                        # KHÔNG commit giữa chừng: nếu chunk sau bị lỗi,
                        # rollback sẽ cuốn toàn bộ staging trong transaction này.
                        file_rows_staged += len(rows_buffer)
                        logger.info(f"[{base}] Staged {file_rows_staged:,} rows so far from {csv_file.name}...")

            except pd.errors.EmptyDataError:
                logger.warning(f"CSV file {csv_file.name} is empty. Skipping.")
                try:
                    cur.execute(f"DROP TABLE IF EXISTS {staging_tbl};")
                    conn.commit()
                except Exception:
                    pass
                continue
            except Exception as e:
                logger.error(f"Error reading CSV file {csv_file.name}: {e}")
                try:
                    cur.execute(f"DROP TABLE IF EXISTS {staging_tbl};")
                    conn.commit()
                except Exception:
                    pass
                raise

            # Commit toàn bộ dữ liệu staging 1 lần duy nhất sau khi đọc xong file
            conn.commit()

            # Thực hiện Dedup trên Staging Table
            if file_rows_staged > 0:
                try:
                    file_deleted = _dedup_exact_duplicates(cur, conn, staging_tbl, base, columns_to_insert)
                    logger.info(f"[DEDUP] {staging_tbl}: cleaned {file_deleted:,} duplicate rows from staging.")
                except Exception as dedup_exc:
                    logger.warning(
                        f"[DEDUP] {staging_tbl}: staging dedup failed: {dedup_exc}. Proceeding with raw staged data."
                    )
                    conn.rollback()

                # Insert từ staging_tbl vào prod_tbl WHERE NOT EXISTS
                col_str = ", ".join([f'"{c}"' for c in columns_to_insert])
                dedup_cols = DEDUP_KEYS.get(base)
                if dedup_cols:
                    key_conditions = " AND ".join([f'p."{col}" IS NOT DISTINCT FROM s."{col}"' for col in dedup_cols])
                    insert_sql = f"""
                        INSERT INTO {prod_tbl} ({col_str})
                        SELECT {col_str} FROM {staging_tbl} s
                        WHERE NOT EXISTS (
                            SELECT 1 FROM {prod_tbl} p
                            WHERE {key_conditions}
                        );
                    """
                else:
                    all_cols_cond = " AND ".join([f'p."{col}" IS NOT DISTINCT FROM s."{col}"' for col in columns_to_insert])
                    insert_sql = f"""
                        INSERT INTO {prod_tbl} ({col_str})
                        SELECT {col_str} FROM {staging_tbl} s
                        WHERE NOT EXISTS (
                            SELECT 1 FROM {prod_tbl} p
                            WHERE {all_cols_cond}
                        );
                    """

                try:
                    cur.execute(insert_sql)
                    file_rows_inserted = cur.rowcount
                    conn.commit()
                    logger.info(f"[{base}] Inserted {file_rows_inserted:,} new rows from {csv_file.name} into {prod_tbl}.")
                except Exception as insert_exc:
                    logger.error(f"Failed to insert from staging to production: {insert_exc}")
                    conn.rollback()
                    raise

            # Ghi nhận SHA256 vào manifest
            try:
                cur.execute(
                    """
                    INSERT INTO public.ingestion_manifest (zip_name, csv_name, file_sha256, ingested_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (file_sha256) DO NOTHING;
                    """,
                    (meta.get("zip_name") or "", csv_file.name, file_sha256)
                )
                conn.commit()
                logger.info(f"Recorded SHA256 of {csv_file.name} in ingestion_manifest.")
            except Exception as manifest_exc:
                logger.warning(f"Failed to write manifest for {csv_file.name}: {manifest_exc}")
                conn.rollback()

            # Xoá Staging Table
            try:
                cur.execute(f"DROP TABLE IF EXISTS {staging_tbl};")
                conn.commit()
            except Exception:
                pass

            # Track audit cho file này
            audit_log.append({
                "file": csv_file.name,
                "read": file_rows_read,
                "skipped": file_rows_skipped,
                "inserted": file_rows_inserted
            })
            total_read_all += file_rows_read
            total_skipped_all += file_rows_skipped
            total_inserted_all += file_rows_inserted
            total_deleted_all += file_deleted

        # Tính tổng số rows cuối cùng trong bảng Production
        cur.execute(f"SELECT COUNT(*) FROM {prod_tbl};")
        total_rows = cur.fetchone()[0]
        total_deleted = total_deleted_all

        # Save encryption mapping nếu sử dụng
        if use_encryption and encrypto and encryption_mapping_file:
            try:
                encrypto.save_mapping(encryption_mapping_file)
                logger.info(f"Saved encryption mapping to {encryption_mapping_file}")
            except Exception as e:
                logger.warning(f"Could not save encryption mapping: {e}")
        
        # --- AUDIT SUMMARY ---
        logger.info(f"==================================================")
        logger.info(f"AUDIT SUMMARY: {prod_tbl}")
        logger.info(f"==================================================")
        for stat in audit_log:
            logger.info(f"File: {stat['file']} | Read: {stat['read']:,} | Skipped: {stat['skipped']:,} | Inserted: {stat['inserted']:,}")
        logger.info(f"--------------------------------------------------")
        logger.info(f"TOTAL CSV ROWS READ:      {total_read_all:,}")
        logger.info(f"TOTAL CSV ROWS SKIPPED:   {total_skipped_all:,}")
        logger.info(f"TOTAL ROWS INSERTED:      {total_inserted_all:,}")
        logger.info(f"TOTAL ROWS DEDUPLICATED:  {total_deleted:,}")
        logger.info(f"FINAL DB ROW COUNT:       {total_rows:,}")
        logger.info(f"==================================================")

        return total_rows
        
    except Exception as e:
        conn.rollback()
        logger.error(f"copy_and_insert_to_production base={base}, table={table_name}: {e}")
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def _bulk_insert_rows(cur, prod_tbl: str, rows: List[Dict[str, Any]], base: str) -> None:
    """
    Insert rows vào production table bằng COPY FROM trực tiếp.
    Do bảng đích luôn được TRUNCATE trước mỗi lần chạy, ta không cần UPSERT.
    """
    if not rows:
        return

    # --- EXTRA GUARD cho cas_info: normalize lại 2 cột datetime ---
    if base == "cas_info":
        ts_fields = ["customer_update_date", "contract_sig_first"]
        for row in rows:
            for field in ts_fields:
                val = row.get(field)
                if isinstance(val, str) and val.strip():
                    # Ép lại qua SafeTypeCaster.to_timestamp 1 lần nữa
                    fixed = SafeTypeCaster.to_timestamp(val)
                    row[field] = fixed  # có thể ra 'YYYY-MM-DD ...' hoặc None

    # Lấy danh sách cột từ row đầu
    columns = list(rows[0].keys())
    col_str = ", ".join([f'"{col}"' for col in columns])

    # Tạo CSV data trong memory
    from io import StringIO
    buffer = StringIO()
    
    for row in rows:
        line_values = []
        for col in columns:
            val = row.get(col)
            if val is None:
                line_values.append('\\N')  # PostgreSQL NULL marker
            else:
                # Escape special characters for COPY
                val_str = str(val).replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')
                line_values.append(val_str)
        buffer.write('\t'.join(line_values) + '\n')
    
    buffer.seek(0)

    try:
        # COPY trực tiếp đối với các bảng
        cur.copy_expert(
            f"COPY {prod_tbl} ({col_str}) FROM STDIN WITH (FORMAT TEXT, NULL '\\N')",
            buffer
        )
    except Exception as e:
        logger.error(f"COPY failed for base={base}: {e}")
        logger.debug(f"Columns: {columns}")
        logger.debug(f"First row sample: {rows[0] if rows else 'N/A'}")
        raise
