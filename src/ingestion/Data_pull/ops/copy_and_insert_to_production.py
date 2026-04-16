# ops/copy_and_insert_to_production.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import csv
import io
import os

import psycopg2

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
        
        # 2) TRUNCATE production nếu mode = snapshot
        if mode == "snapshot":
            cur.execute(f"TRUNCATE TABLE {prod_tbl};")
            conn.commit()
            logger.info(f"Truncated {prod_tbl} (snapshot mode)")
        
        # 3) Đọc CSV và transform data
        total_rows = 0
        for csv_file in csv_files:
            logger.info(f"Reading {csv_file.name}")
            
            with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
                # CSV files sử dụng delimiter ';'
                delimiter = ";"
                reader = csv.DictReader(f, delimiter=delimiter)
                source_fields = [h.strip() for h in (reader.fieldnames or [])]
                if len(source_fields) != len(headers_raw):
                    msg = (
                        f"Header mismatch in {csv_file.name}: "
                        f"expected {len(headers_raw)} columns (from first CSV), "
                        f"got {len(source_fields)} columns. "
                        f"All CSVs in ZIP must have same header structure."
                    )
                    logger.error(msg)
                    raise CsvHeaderMismatchError(msg)
                
                logger.debug(f"Processing: {len(source_fields)} columns from {csv_file.name}")

                # Buffer rows để batch insert
                rows_buffer: List[Dict[str, Any]] = []
                for raw_row in reader:
                    # ===== NORMALIZE ROW: Apply header_map (position-based) =====
                    # Chuẩn hoá key: strip khoảng trắng 2 đầu
                    raw_row = {
                        (k.strip() if isinstance(k, str) else k): v
                        for k, v in raw_row.items()
                    }
                    
                    # Áp map header: CSV key → canonical key (nếu khác tên)
                    normalized_row: Dict[str, Any] = {}
                    for k, v in raw_row.items():
                        canonical_key = header_map.get(k, k)
                        normalized_row[canonical_key] = v
                    
                    raw_row = normalized_row
                    # ===== HẾT NORMALIZE =====

                    # Transform row theo table type (sử dụng dispatch dict)
                    transform_func = TRANSFORM_DISPATCH.get(base)
                    if transform_func is None:
                        logger.warning(f"No transform function for base={base}, skipping row")
                        continue
                    
                    transformed = transform_func(raw_row, encrypto)
                    if transformed is None:
                        continue  # Skip invalid row

                    rows_buffer.append(transformed)

                    # Batch insert
                    if len(rows_buffer) >= batch_rows:
                        _bulk_insert_rows(cur, prod_tbl, rows_buffer, base)
                        conn.commit()
                        total_rows += len(rows_buffer)
                        logger.info(f"[{base}] Inserted {total_rows:,} rows so far...")
                        rows_buffer = []
                
                # Insert remaining rows
                if rows_buffer:
                    _bulk_insert_rows(cur, prod_tbl, rows_buffer, base)
                    conn.commit()
                    total_rows += len(rows_buffer)
        
        # Save encryption mapping nếu sử dụng
        if use_encryption and encrypto and encryption_mapping_file:
            try:
                encrypto.save_mapping(encryption_mapping_file)
                logger.info(f"Saved encryption mapping to {encryption_mapping_file}")
            except Exception as e:
                logger.warning(f"Could not save encryption mapping: {e}")
        
        logger.info(f"Production {prod_tbl}: {total_rows:,} rows inserted")
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
    Insert rows vào production table bằng COPY FROM với StringIO.
    Nhanh hơn executemany() rất nhiều lần.
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
        cur.copy_expert(
            f"COPY {prod_tbl} ({col_str}) FROM STDIN WITH (FORMAT TEXT, NULL '\\N')",
            buffer
        )
    except Exception as e:
        logger.error(f"COPY FROM failed for base={base}: {e}")
        logger.debug(f"Columns: {columns}")
        logger.debug(f"First row sample: {rows[0] if rows else 'N/A'}")
        raise
