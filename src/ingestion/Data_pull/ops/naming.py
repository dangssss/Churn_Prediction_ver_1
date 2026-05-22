# Data_pull/ops/naming.py
from datetime import date
from pathlib import Path
import re

from Data_pull.resources.fs import ZIP_RE  # đã định nghĩa ở resources/fs.py

# CSV: tenbang_mmdd_mmdd_yyyy.csv (monthly)
CSV_RE = re.compile(
    r"(?P<base>[A-Za-z_][A-Za-z0-9_]*)_(?P<start>\d{4})_(?P<end>\d{4})_(?P<year>\d{4})\.csv$",
    re.IGNORECASE,
)

# CSV: tenbang.csv (snapshot - simplified format)
CSV_SNAPSHOT_RE = re.compile(
    r"(?P<base>[A-Za-z_][A-Za-z0-9_]*)\.csv$",
    re.IGNORECASE,
)

def _mmdd_to_date(mmdd: str, year: int) -> date:
    mm = int(mmdd[:2])
    dd = int(mmdd[2:])
    return date(year, mm, dd)

def _parse_yymm(yymm: str):
    """
    yymm (vd: '2403') -> (year=2024, month=3)
    Giả định tất cả là 20xx, nếu sau này có 19xx thì đổi chỗ year = 2000 + yy.
    """
    yy = int(yymm[:2])
    mm = int(yymm[2:])
    assert 1 <= mm <= 12, f"MM không hợp lệ trong yymm={yymm}"
    year = 2000 + yy
    return year, mm

def parse_zip_and_decide_names(zip_path: Path) -> dict:
    m = ZIP_RE.fullmatch(zip_path.name)
    assert m, f"Tên ZIP không hợp lệ: {zip_path.name}"

    base = m.group("base")
    yymm = m.group("yymm")
    base_update = m.group("base_update")
    update_ddmmyy = m.group("update_ddmmyy")  # Format: DDMMYY (ngày-tháng-năm)

    if yymm:
        year, month = _parse_yymm(yymm)
        table_name = f"{base}_{yymm}"  # ví dụ: bccp_orderitem_2501
        month_folder = table_name
        period_key_month = f"{year:04d}{month:02d}"

        # Lấy các file CSV cho tháng (3 đợt)
        csv_files = [
            f"{base}_0101_0110_{year}.csv",
            f"{base}_0111_0120_{year}.csv",
            f"{base}_0121_0131_{year}.csv",
        ]

        return {
            "base": base,
            "year": year,
            "month": month,
            "yymm": yymm,
            "table_name": table_name,
            "month_folder": month_folder,
            "period_key_month": period_key_month,
            "csv_files": csv_files,  # Danh sách các file CSV cho tháng
            "mode": "monthly",  # Dạng theo tháng
        }

    elif update_ddmmyy and base_update:
        # File update (snapshot mode):
        #   - cas_customer_update_250322.zip -> 25/03/2022 -> table_name = cas_customer
        table_name = base_update
        # Note: month_folder dùng string gốc để tạo folder (giữ nguyên ddmmyy cho dễ track)
        month_folder = f"{base_update}_snapshot_{update_ddmmyy}"
        
        # Parse ddmmyy -> date (Format: DDMMYY = Ngày-Tháng-Năm)
        # VD: 250322 = DD=25 (ngày 25) + MM=03 (tháng 3) + YY=22 (năm 2022)
        dd = int(update_ddmmyy[0:2])   # 2 ký tự đầu = ngày
        mm = int(update_ddmmyy[2:4])   # 2 ký tự giữa = tháng
        yy = int(update_ddmmyy[4:6])   # 2 ký tự cuối = năm
        yyyy = 2000 + yy        # Giả định 20xx (2000-2099)
        
        period_key = f"{yyyy:04d}{mm:02d}{dd:02d}"  # YYYYMMDD: 20220325 (để sort đúng)

        return {
            "base": base_update,
            "yymmdd": update_ddmmyy,  # Giữ key này để tương thích nếu cần, giá trị là raw string (DDMMYY)
            "table_name": table_name,
            "month_folder": month_folder,
            "period_key": period_key,  # Format chuẩn YYYYMMDD để sort
            "csv_files": [],  # Sẽ được populate lại sau khi unzip
            "mode": "snapshot",  # Dạng snapshot (truncate & reload)
        }
    
    raise ValueError(f"Không thể parse ZIP: {zip_path.name}")


def order_csvs_chronologically(csv_paths, expect_base: str, expect_year: int, expect_month: int):
    """
    Sắp CSV theo start_mmdd tăng dần.
    Ràng buộc:
      - base phải khớp
      - start_mmdd & end_mmdd phải thuộc đúng (expect_year, expect_month)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    keyed = []    
    for p in csv_paths:
        m = CSV_RE.fullmatch(p.name)
        if not m or m["base"].lower() != expect_base.lower():
            logger.warning(f"File bị loại bỏ vì không khớp format/base: {p.name}")
            continue

        s_mmdd = m["start"]
        e_mmdd = m["end"]
        y = int(m["year"])

        try:
            ps = _mmdd_to_date(s_mmdd, y)
            pe = _mmdd_to_date(e_mmdd, y)
            if not (ps.year == pe.year == expect_year and ps.month == pe.month == expect_month):
                logger.warning(f"File bị loại bỏ vì không thuộc {expect_month:02d}/{expect_year}: {p.name}")
                continue
            keyed.append((int(s_mmdd), p.name, p))
        except Exception as e:
            logger.warning(f"File bị loại bỏ vì lỗi parse thời gian: {p.name} - {e}")
            continue

    keyed.sort(key=lambda x: (x[0], x[1]))
    return [p for _, __, p in keyed]
