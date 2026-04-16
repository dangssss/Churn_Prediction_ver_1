# jobs/ingest_zip_job.py (refactored for 4 new tables + error handling)
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

from Data_pull.resources import PostgresConfig, FSConfig, get_pg_conn
from Data_pull.ops.unzip_and_discover import unzip_and_discover
from Data_pull.ops.copy_and_insert_to_production import copy_and_insert_to_production
from Data_pull.ops.post_ingest_maintenance import post_ingest_maintenance
from Data_pull.logging_config import get_logger

# Setup logging
logger = get_logger(__name__)


def ingest_zip_job(
    zip_path: Path,
    fs_cfg: FSConfig,
    pg_cfg: PostgresConfig,
    *,
    prod_schema: str = "public",
    ingest_schema: str = "ingest",
    batch_rows: int = 50_000,
    source_has_header: bool = True,
    injection_mode: str = "sanitize",
    use_encryption: bool = True,
    encryption_mapping_file: Optional[str] = None,
    skip_if_logged: bool = False,
    use_test_schema: bool = None,  # Auto-detect from ENV if None
) -> Dict[str, Any]:
    """
    Job xử lý 1 file ZIP: unzip -> transform + insert into production -> maintenance.

    
    Hỗ trợ 4 bảng (monthly + snapshot modes) với error handling:
    - Nếu parse tên ZIP fail -> (unzip_and_discover) copy ZIP vào fail_data folder, log error
    - Nếu CSV header sai -> copy ZIP vào fail_data folder (tại đây)
    - Nếu unzip fail -> (unzip_and_discover) copy ZIP vào fail_data folder, log error
    - Nếu insert fail -> copy ZIP vào fail_data folder
    
    Args:
        use_test_schema: Nếu True, dùng csv_schema_test.py. Nếu None, auto-detect từ env var USE_TEST_SCHEMA.
    
    Returns:
        dict {
            "zip_name": str,
            "meta": meta dict từ unzip_and_discover (hoặc None nếu fail),
            "staging_rows": số dòng được nạp (same as prod_rows, legacy key for compatibility),
            "prod_rows": số dòng insert vào production,
            "skipped": bool,
            "reason": string (nếu skipped),
            "success": bool,
            "error": string (nếu fail),
        }
    """
    # Auto-detect test schema from environment
    # TODO NAMNT check lai
    if use_test_schema is None:
        use_test_schema = os.getenv("USE_TEST_SCHEMA", "").lower() in ("1", "true", "yes")
    
    # Set environment variable để copy_and_insert_to_production biết dùng test schema
    if use_test_schema:
        os.environ["USE_TEST_SCHEMA"] = "1"
    
    zip_name = zip_path.name
    result = {
        "zip_name": zip_name,
        "meta": None,
        "staging_rows": 0,
        "prod_rows": 0,
        "skipped": False,
        "reason": None,
        "success": False,
        "error": None,
    }

    # 0) Optional: skip nếu đã log success
    if skip_if_logged:
        try:
            conn = get_pg_conn(pg_cfg)
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT 1
                FROM {ingest_schema}.ingest_log
                WHERE zip_name = %s AND status = 'success'
                LIMIT 1;
                """,
                (zip_name,),
            )
            if cur.fetchone():
                result["skipped"] = True
                result["reason"] = "already_logged_success"
                logger.info(f"[SKIP] {zip_name} đã được ingest thành công trước đó.")
                return result
        except Exception as e:
            logger.warning(f"[WARN] Không check được ingest_log: {e}")
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    # 1) unzip & discover
    try:
        meta = unzip_and_discover(zip_path, fs_cfg)
        result["meta"] = meta
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False
        logger.error(f"[FAIL] unzip_and_discover for {zip_name}: {e}")
        # unzip_and_discover đã lo copy sang fail_data (nếu cần)
        return result

    # 2) Kiểm tra CSV
    if not meta.get("csv_files"):
        result["skipped"] = True
        result["reason"] = "no_csv_after_unzip"
        logger.warning(f"[SKIP] {zip_name}: không có CSV sau khi unzip.")
        # Có thể copy thêm ZIP sang fail_data nếu muốn log rõ
        return result

    # 3) load data into production (transform + insert directly)
    try:
        prod_rows = copy_and_insert_to_production(
            meta,
            pg_cfg,
            batch_rows=batch_rows,
            source_has_header=source_has_header,
            injection_mode=injection_mode,
            use_encryption=use_encryption,
            encryption_mapping_file=encryption_mapping_file,
        )
        result["prod_rows"] = prod_rows
        result["staging_rows"] = prod_rows  # legacy key, same as prod_rows
    except Exception as e:
        result["error"] = f"copy_and_insert_to_production failed: {str(e)}"
        result["success"] = False
        logger.error(f"[FAIL] copy_and_insert_to_production for {zip_name}: {e}")
        
        # Trước đây: move ZIP vào fail_data
        # Bây giờ: COPY ZIP đang ở extract_dir sang fail_data, KHÔNG move/xoá bản gốc
        try:
            extract_dir = meta.get("extract_dir")
            if extract_dir and extract_dir.exists():
                src_zip = extract_dir / zip_name
                if src_zip.exists():
                    fail_path = fs_cfg.fail_dir / zip_name
                    fail_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src_zip), str(fail_path))
                    logger.info(
                        f"[COPY] Copied {zip_name} from {src_zip} to fail_data {fail_path} "
                        "(kept original & saved copy)."
                    )
                else:
                    # fallback: copy trực tiếp từ incoming_dir nếu cần
                    fail_path = fs_cfg.fail_dir / zip_name
                    fail_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(zip_path), str(fail_path))
                    logger.info(
                        f"[COPY] Copied {zip_name} from {zip_path} to fail_data {fail_path} "
                        "(src_zip not found)."
                    )
        except Exception as move_err:
            logger.warning(f"[WARN] Could not copy ZIP to fail_data: {move_err}")
        
        return result

    # 4) Mark as success (data is already in production)
    result["success"] = True

    # 5) maintenance (ANALYZE, log ingest...)
    try:
        post_ingest_maintenance(
            meta,
            pg_cfg,
            prod_schema=prod_schema,
            ingest_schema=ingest_schema,
            stg_rows=prod_rows,  # Same as prod_rows since no staging
            prod_rows=prod_rows,
            zip_path=zip_path,
        )
    except Exception as e:
        logger.warning(f"[WARN] post_ingest_maintenance failed: {e}")
        # không fail job vì maintenance không critical

    logger.info(f"[SUCCESS] {zip_name}: {prod_rows:,} rows inserted directly to production")
    return result
