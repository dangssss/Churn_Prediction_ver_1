# ops/post_ingest_maintenance.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from Data_pull.resources import PostgresConfig, get_pg_conn
from Data_pull.logging_config import get_logger

logger = get_logger(__name__)


def post_ingest_maintenance(
    meta: Dict[str, Any],
    pg_cfg: PostgresConfig,
    *,
    prod_schema: str = "public",
    ingest_schema: str = "ingest",
    stg_rows: Optional[int] = None,
    prod_rows: Optional[int] = None,
    zip_path: Optional[Path] = None,
) -> None:
    """
    Hậu xử lý sau khi nạp xong 1 ZIP:
      - ANALYZE bảng prod theo tháng
      - Ghi log vào ingest.ingest_log (tự tạo schema + table nếu chưa có)

    Input:
      - meta: dict từ unzip_and_discover (cần base, table_name, period_key_month)
      - pg_cfg: PostgresConfig
      - prod_schema: schema prod, vd 'public'
      - ingest_schema: schema log, vd 'ingest'
      - stg_rows: số dòng staging (để log)
      - prod_rows: số dòng prod (để log)
      - zip_path: Path tới file ZIP (để log zip_name và file_size)
    """
    base = meta.get("base", "")
    table_name = meta.get("table_name", "")
    period_key_month = meta.get("period_key_month")  # vd: "202403"
    zip_name = zip_path.name if zip_path is not None else None
    file_size = zip_path.stat().st_size if zip_path is not None else 0
    file_mtime = zip_path.stat().st_mtime if zip_path is not None else 0.0

    conn = get_pg_conn(pg_cfg)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        logger.info(f"Running post_ingest_maintenance for {prod_schema}.\"{table_name}\"")

        # 1) Đảm bảo schema ingest tồn tại
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS { ingest_schema };")

        # 2) Tạo bảng ingest_log nếu chưa có
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {ingest_schema}.ingest_log (
          id                bigserial PRIMARY KEY,
          zip_name          text        NOT NULL,
          base              text        NOT NULL,
          table_name        text        NOT NULL,
          period_key_month  varchar(6),
          prod_schema       text        NOT NULL,
          staging_rows      bigint,
          prod_rows         bigint,
          file_size         bigint,
          file_mtime        double precision,
          status            text        NOT NULL,
          started_at        timestamptz DEFAULT now(),
          finished_at       timestamptz DEFAULT now()
        );
        """)
        
        # 2.1) Thêm các cột mới nếu chưa có (safe migration)
        for col_name, col_type in [('file_size', 'bigint'), ('file_mtime', 'double precision')]:
            try:
                cur.execute(f"ALTER TABLE {ingest_schema}.ingest_log ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
            except Exception:
                # Column đã tồn tại hoặc lỗi khác - bỏ qua
                pass
        conn.commit()

        # 3) ANALYZE bảng prod để tối ưu query plan
        if table_name:
            cur.execute(f'ANALYZE {prod_schema}."{table_name}";')
            logger.info(f"Analyzed {prod_schema}.\"{table_name}\"")
        # 4) Ghi log success
        cur.execute(
            f"""
            INSERT INTO {ingest_schema}.ingest_log(
              zip_name, base, table_name, period_key_month,
              prod_schema, staging_rows, prod_rows, file_size, file_mtime, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                zip_name or "",
                base,
                table_name,
                period_key_month,
                prod_schema,
                stg_rows,
                prod_rows,
                file_size,
                file_mtime,
                "success",
            ),
        )   
        

        conn.commit()
        logger.info(
            f"Logged success for zip={zip_name}, "
            f"table={prod_schema}.\"{table_name}\", prod_rows={prod_rows}"
        )

    except Exception as e:
        conn.rollback()
        logger.error(f"post_ingest_maintenance for {table_name}: {e}")
        # tuỳ ý: có thể raise để job fail cứng, hoặc chỉ in lỗi
        # ở đây mình raise cho dễ debug:
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()
