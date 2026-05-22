"""
unzip_and_discover.py
=====================
Phát hiện file ZIP mới và xác định những CSV nào cần ingest.

Giải quyết BUG #2: "ZIP tuần mới ghi đè ZIP tuần cũ"
─────────────────────────────────────────────────────
Vấn đề: ZIP mới = toàn bộ dữ liệu cũ + dữ liệu mới.
Nếu không có cơ chế tracking, mỗi tuần sẽ ingest lại toàn bộ.

Giải pháp: So sánh SHA-256 của từng CSV member trong zip với manifest.
  - SHA khác  → CSV này có nội dung mới → INGEST
  - SHA giống → CSV này không thay đổi  → BỎ QUA
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sqlalchemy.engine import Engine

from ..logging_config import get_logger
from ..jobs.ingest_zip_job import _is_already_ingested, MANIFEST_TABLE
from ..config.paths import INCOMING_DIR

logger = get_logger(__name__)


@dataclass
class CsvDiscovery:
    member: str          # tên file bên trong zip
    sha256: str
    size_bytes: int
    needs_ingest: bool   # True = cần ingest, False = đã có trong manifest


def discover_zip(
    zip_path: str,
    zip_mtime: float,
    engine: Engine,
    force: bool = False,
) -> List[CsvDiscovery]:
    """
    Mở ZIP và kiểm tra từng CSV:
      - Tính SHA-256 (không giải nén ra đĩa)
      - Tra manifest xem đã ingest chưa
      - Trả về danh sách CsvDiscovery
    """
    discoveries = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        logger.info("Phát hiện %d CSV trong ZIP", len(csv_members))

        for member in csv_members:
            info = zf.getinfo(member)
            sha = _sha256_member(zf, member)
            already = (not force) and _is_already_ingested(engine, zip_mtime, member, sha)

            d = CsvDiscovery(
                member=member,
                sha256=sha,
                size_bytes=info.file_size,
                needs_ingest=not already,
            )
            discoveries.append(d)

            status = "BỎ QUA (đã có)" if already else "CẦN INGEST"
            logger.info(
                "  %-50s size=%8,.0f KB sha=%s... → %s",
                member, info.file_size / 1024, sha[:12], status,
            )

    new_count = sum(1 for d in discoveries if d.needs_ingest)
    logger.info("Tổng: %d CSV, %d cần ingest, %d bỏ qua",
                len(discoveries), new_count, len(discoveries) - new_count)
    return discoveries


def find_latest_zip(incoming_dir: Optional[str] = None) -> Optional[str]:
    """Tìm file .zip mới nhất trong INCOMING_DIR."""
    d = Path(incoming_dir or INCOMING_DIR)
    zips = sorted(d.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        logger.warning("Không tìm thấy file .zip trong %s", d)
        return None
    latest = str(zips[-1])
    logger.info("ZIP mới nhất: %s", latest)
    return latest


def _sha256_member(zf: zipfile.ZipFile, member: str) -> str:
    h = hashlib.sha256()
    with zf.open(member) as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
