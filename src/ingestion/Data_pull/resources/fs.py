# resources/fs.py
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Union

# Regex tên ZIP: 
#   1) <base>_yymm.zip (vd: bccp_orderitem_2501.zip) - monthly
#      Format: yymm = YY (năm 2 chữ số) + MM (tháng)
#      VD: 2501 = tháng 01 năm 2025
#
#   2) <base>_update_yymmdd.zip (vd: cas_customer_update_220325.zip) - snapshot
#      Format: yymmdd = YY (năm) + MM (tháng) + DD (ngày)
#      VD: 220325 = 25/03/2022 (ngày 25 tháng 3 năm 2022)
ZIP_RE = re.compile(
    r"(?P<base>[A-Za-z_][A-Za-z0-9_]*)_(?P<yymm>\d{4})\.zip$"
    r"|(?P<base_update>[A-Za-z_][A-Za-z0-9_]*)_update_(?P<update_ddmmyy>\d{6})\.zip$"
    r"|(?P<base_snapshot>[A-Za-z_][A-Za-z0-9_]*)_(?P<ddmmyy>\d{6})\.zip$",
    re.IGNORECASE,
)

LABEL_ZIP_RE = re.compile(r"label_(?P<yymm>\d{4})\.zip$", re.IGNORECASE)



@dataclass
class FSConfig:
    """
    Config cho filesystem.

    - incoming_dir: nơi chứa các file ZIP đầu vào
    - saved_dir   : nơi lưu raw ZIP + dữ liệu đã giải nén
    - fail_dir    : nơi move file ZIP/CSV bị lỗi
    """

    from Data_pull.config.paths import INCOMING_DATA_DIR, SAVED_DATA_DIR, FAIL_DATA_DIR

    incoming_dir: Path = INCOMING_DATA_DIR
    saved_dir: Path = SAVED_DATA_DIR
    fail_dir: Path = FAIL_DATA_DIR

    # TODO: NAMNT Check lai
    def ensure_dirs(self) -> None:
        """Đảm bảo các thư mục tồn tại."""
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.saved_dir.mkdir(parents=True, exist_ok=True)
        self.fail_dir.mkdir(parents=True, exist_ok=True)


    # TODO: NAMNT Check lai
    @classmethod
    def from_env(cls) -> "FSConfig":
        """
        Factory method dùng trong sensor / job:
        - Đọc env (nếu có)
        - Tạo FSConfig
        - Gọi ensure_dirs()
        """
        cfg = cls()
        cfg.ensure_dirs()
        return cfg


def list_zip_files(src: Union[Path, FSConfig]) -> List[Path]:
    """
    Trả về list các file ZIP trong incoming_dir, lọc theo pattern <base>_yymm.zip.

    - Nếu truyền FSConfig: dùng cfg.incoming_dir
    - Nếu truyền Path: dùng Path đó luôn
    """
    if isinstance(src, FSConfig):
        incoming_dir = src.incoming_dir
    else:
        incoming_dir = src

    return [
        p for p in sorted(incoming_dir.glob("*.zip"))
        if ZIP_RE.fullmatch(p.name)
    ]


def list_label_zip_files(label_dir: Path) -> List[Path]:
    return [
        p for p in sorted(label_dir.glob("*.zip"))
        if LABEL_ZIP_RE.fullmatch(p.name)
    ]
