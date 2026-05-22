# ops/unzip_and_discover.py
from pathlib import Path
import logging
import shutil
import zipfile

from typing import Optional
from Data_pull.resources import FSConfig, ZIP_RE, list_zip_files, PostgresConfig, get_pg_conn
from Data_pull.ops.naming import (
    parse_zip_and_decide_names,
    order_csvs_chronologically,
)
from Data_pull.logging_config import get_logger

logger = get_logger(__name__)


def ensure_manifest_table(cur) -> None:
    cur.execute("""
    CREATE TABLE IF NOT EXISTS public.ingestion_manifest (
        zip_name text NOT NULL,
        csv_name text NOT NULL,
        file_sha256 varchar(64) NOT NULL PRIMARY KEY,
        ingested_at timestamptz DEFAULT now()
    );
    """)


def unzip_and_discover(
    zip_path: Path,
    fs_cfg: FSConfig,
    pg_cfg: Optional[PostgresConfig] = None,
) -> dict:
    """
    Input:
      - zip_path: đường dẫn tới file ZIP (đang nằm trong incoming_dir, vd: churn_data)
      - fs_cfg: config filesystem (incoming_dir, saved_dir)
      - pg_cfg: config database to check manifest

    Output: dict meta:
      {
        "base": "bccp_orderitem",
        "mode": "monthly" | "snapshot",
        "table_name": "bccp_orderitem_2501" hoặc "cas_customer",
        "month_folder": "bccp_orderitem_2501" hoặc "cas_customer_snapshot_DDMMYY",
        "extract_dir": <Path saved_data/bccp_orderitem/bccp_orderitem_2501>,
        "csv_files": [Path(...), ...],
        "csv_hashes": {str(Path): sha256, ...}
      }

    CSV Format:
      - Monthly: tenbang_mmdd_mmdd_yyyy.csv (ví dụ: bccp_orderitem_0101_0131_2025.csv)
      - Snapshot: tenbang.csv (ví dụ: cas_customer.csv)

    Side-effect (đã sửa):
      - ZIP gốc vẫn nằm ở incoming_dir (churn_data).
      - Tạo 1 bản COPY ZIP ở saved_dir/<base>/<month_folder>/<zip_name> để unzip & xử lý.
      - Khi lỗi parse tên ZIP / unzip, COPY thêm 1 bản sang fail_data, KHÔNG move/xoá file gốc.
    """
    # 1) Decode tên ZIP -> meta
    try:
        meta = parse_zip_and_decide_names(zip_path)
    except Exception as e:
        # ZIP name invalid -> copy ZIP vào fail_data, KHÔNG move
        fail_path = fs_cfg.fail_dir / zip_path.name
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(zip_path), str(fail_path))
            logger.error(f"ZIP name invalid: {zip_path.name} - copied to fail_data")
        except Exception as copy_err:
            logger.error(f"ZIP name invalid: {zip_path.name} - also failed to copy to fail_data: {copy_err}")
        logger.error(f"  Reason: {e}")
        raise RuntimeError(f"Invalid ZIP name: {zip_path.name}") from e

    base = meta["base"]
    table_name = meta["table_name"]
    month_folder = meta["month_folder"]
    mode = meta.get("mode", "monthly")

    # 2) Thư mục giải nén (trong saved_data)
    extract_dir = fs_cfg.saved_dir / base / month_folder
    if extract_dir.exists():
        logger.warning(f"Clearing existing extraction directory to avoid stale files: {extract_dir}")
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 3) COPY ZIP từ incoming_dir sang extract_dir (KHÔNG move/xoá zip_path)
    dest_zip = extract_dir / zip_path.name
    if dest_zip.exists():
        dest_zip.unlink()  # overwrite bản cũ nếu có

    shutil.copy2(str(zip_path), str(dest_zip))
    logger.info(f"Copied ZIP from {zip_path.name} to {extract_dir}")

    # 4) Load ingested hashes from database
    already_ingested_hashes = set()
    if pg_cfg is not None:
        try:
            conn = get_pg_conn(pg_cfg)
            with conn.cursor() as cur:
                ensure_manifest_table(cur)
                conn.commit()
                cur.execute("SELECT file_sha256 FROM public.ingestion_manifest;")
                already_ingested_hashes = {row[0] for row in cur.fetchall()}
            conn.close()
            logger.info(f"Loaded {len(already_ingested_hashes)} hashes from public.ingestion_manifest")
        except Exception as e:
            logger.warning(f"Could not read ingestion_manifest: {e}")

    # 5) Giải nén từ bản trong saved_data (Chỉ giải nén các file chưa ingest)
    import hashlib
    csv_hashes = {}
    extracted_csv_paths = []

    try:
        with zipfile.ZipFile(dest_zip, "r") as zf:
            # Lấy danh sách các file CSV
            members = [
                m for m in zf.infolist()
                if not m.is_dir() and m.filename.lower().endswith(".csv")
            ]
            
            for member in members:
                # Tính SHA256 in-memory của member
                sha = hashlib.sha256()
                with zf.open(member) as f:
                    for chunk in iter(lambda: f.read(4096 * 1024), b""): # 4MB chunks
                        sha.update(chunk)
                file_hash = sha.hexdigest()
                
                member_filename = Path(member.filename).name
                
                # Nếu đã ingest rồi thì bỏ qua
                if file_hash in already_ingested_hashes:
                    logger.info(f"CSV '{member_filename}' (SHA256: {file_hash}) already ingested. Skipping extraction.")
                    continue
                
                # Giải nén file
                extracted_path = Path(zf.extract(member, extract_dir))
                extracted_csv_paths.append(extracted_path)
                csv_hashes[str(extracted_path)] = file_hash
                logger.info(f"Extracted '{member_filename}' (SHA256: {file_hash}) to {extracted_path}")
                
        logger.info(f"Unzipped completed. Extracted {len(extracted_csv_paths)} new CSV files.")
    except Exception as e:
        # Unzip fail -> copy ZIP sang fail_data, KHÔNG move/xoá
        fail_path = fs_cfg.fail_dir / zip_path.name
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dest_zip.exists():
                shutil.copy2(str(dest_zip), str(fail_path))
            else:
                # fallback: copy trực tiếp từ incoming nếu vì lý do gì đó dest_zip không tồn tại
                shutil.copy2(str(zip_path), str(fail_path))
            logger.error(f"Failed to unzip {zip_path.name}: {e} - copied to fail_data")
        except Exception as copy_err:
            logger.error(f"Failed to unzip {zip_path.name}: {e} - also failed to copy to fail_data: {copy_err}")
        raise RuntimeError(f"Failed to unzip: {zip_path.name}") from e

    # 6) Sắp xếp CSV theo thứ tự logic
    csv_unsorted = sorted(extracted_csv_paths)
    
    if not csv_unsorted:
        logger.warning(f"No new CSV files to ingest after unzipping {table_name}")
        return {
            **meta,
            "extract_dir": extract_dir,
            "csv_files": [],
            "csv_hashes": {},
        }

    # Đối với monthly mode: sắp CSV theo thời gian
    if mode == "monthly":
        csv_files = order_csvs_chronologically(
            csv_unsorted,
            expect_base=base,
            expect_year=meta["year"],
            expect_month=meta["month"],
        )
    else:
        # snapshot mode: sắp theo tên file
        csv_files = csv_unsorted

    return {
        **meta,
        "extract_dir": extract_dir,
        "csv_files": csv_files,
        "csv_hashes": csv_hashes,
    }
