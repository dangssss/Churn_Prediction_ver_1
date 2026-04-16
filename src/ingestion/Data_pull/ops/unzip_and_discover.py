# ops/unzip_and_discover.py
from pathlib import Path
import logging
import shutil
import zipfile

from Data_pull.resources import FSConfig, ZIP_RE, list_zip_files
from Data_pull.ops.naming import (
    parse_zip_and_decide_names,
    order_csvs_chronologically,
)
from Data_pull.logging_config import get_logger

logger = get_logger(__name__)


def unzip_and_discover(zip_path: Path, fs_cfg: FSConfig) -> dict:
    """
    Input:
      - zip_path: đường dẫn tới file ZIP (đang nằm trong incoming_dir, vd: churn_data)
      - fs_cfg: config filesystem (incoming_dir, saved_dir)

    Output: dict meta:
      {
        "base": "bccp_orderitem",
        "mode": "monthly" | "snapshot",
        "table_name": "bccp_orderitem_2501" hoặc "cas_customer",
        "month_folder": "bccp_orderitem_2501" hoặc "cas_customer_snapshot_DDMMYY",
        "extract_dir": <Path saved_data/bccp_orderitem/bccp_orderitem_2501>,
        "csv_files": [Path(...), ...]
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
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 3) COPY ZIP từ incoming_dir sang extract_dir (KHÔNG move/xoá zip_path)
    dest_zip = extract_dir / zip_path.name
    if dest_zip.exists():
        dest_zip.unlink()  # overwrite bản cũ nếu có

    shutil.copy2(str(zip_path), str(dest_zip))
    logger.info(f"Copied ZIP from {zip_path.name} to {extract_dir}")

    # 4) Giải nén từ bản trong saved_data
    try:
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall(extract_dir)
        logger.info(f"Extracted {dest_zip.name} to {extract_dir}")
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

    # 5) Tìm CSV đệ quy trong extract_dir
    csv_unsorted = sorted(
        [p for p in extract_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"]
    )
    
    if not csv_unsorted:
        logger.warning(f"No CSV files found after unzipping {table_name}")
        return {
            **meta,
            "extract_dir": extract_dir,
            "csv_files": [],
        }

    # 6) Đối với monthly mode: sắp CSV theo thời gian
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
    }
