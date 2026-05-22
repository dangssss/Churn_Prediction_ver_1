"""
csv_utils.py
============
Tiện ích đọc CSV từ bên trong ZipFile theo chunk.

ROOT CAUSE của Bug #1:
─────────────────────────────────────────────────────
Code cũ (suy đoán từ log + triệu chứng):

    offset = 0
    while True:
        chunk = pd.read_csv(path, skiprows=offset, nrows=CHUNK_SIZE, ...)
        if chunk.empty:
            break
        write_to_db(chunk)
        offset += CHUNK_SIZE          # ← BUG: luôn cộng 50k kể cả chunk cuối
        logger.info("Đã nạp %d dòng", offset)   # ← log SAI

Với file 5,182,525 dòng:
  - 103 chunk đầu: OK (mỗi chunk đúng 50k)
  - Chunk thứ 104: chỉ có 32,525 dòng
  - NHƯNG offset vẫn được cộng thêm 50,000
  - → log báo 5,200,000 thay vì 5,182,525
  - Với skiprows/nrows thủ công: chunk 104 có thể đọc lại 50k dòng bao gồm
    dòng cuối cùng của chunk 103 (boundary overlap) → DUPLICATE

Code mới dưới đây dùng pandas.read_csv(chunksize=...) – pandas tự xử lý
tail-end đúng, không cần skiprows/nrows thủ công.
─────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import zipfile
from typing import Iterator, Optional

import pandas as pd

from ..logging_config import get_logger

logger = get_logger(__name__)

CHUNK_SIZE: int = 50_000


def iter_csv_chunks_from_zip(
    zf: zipfile.ZipFile,
    member: str,
    dtype: Optional[dict] = None,
    chunk_size: int = CHUNK_SIZE,
    encoding: str = "utf-8",
) -> Iterator[tuple[pd.DataFrame, int, int]]:
    """
    Đọc một CSV bên trong ZipFile theo từng chunk.

    Yields:
        (chunk_df, chunk_index, rows_in_chunk)
        - chunk_df      : DataFrame của chunk hiện tại
        - chunk_index   : 0-based index của chunk (dùng cho log)
        - rows_in_chunk : số dòng THỰC TẾ trong chunk (quan trọng: luôn
                          dùng cái này để đếm, KHÔNG dùng chunk_index * chunk_size)

    Đảm bảo:
        sum(rows_in_chunk) == tổng số dòng dữ liệu trong file (trừ header)
    """
    with zf.open(member) as raw_bytes:
        text_stream = io.TextIOWrapper(raw_bytes, encoding=encoding, errors="replace")
        reader = pd.read_csv(
            text_stream,
            chunksize=chunk_size,
            dtype=dtype,
            low_memory=False,
        )
        for chunk_idx, chunk in enumerate(reader):
            rows_in_chunk = len(chunk)          # ← KEY FIX: len() thực tế
            yield chunk, chunk_idx, rows_in_chunk


def count_csv_rows_in_zip(
    zf: zipfile.ZipFile,
    member: str,
    encoding: str = "utf-8",
) -> int:
    """
    Đếm số dòng thực tế của một CSV trong zip (không load vào RAM).
    Dùng để validate sau khi ingest.
    """
    total = 0
    with zf.open(member) as raw:
        text_stream = io.TextIOWrapper(raw, encoding=encoding, errors="replace")
        # skip header
        next(text_stream, None)
        for _ in text_stream:
            total += 1
    return total


def validate_row_count(
    expected: int,
    actual_in_db: int,
    csv_name: str,
    tolerance: int = 0,
) -> bool:
    """
    So sánh số dòng nguồn vs DB.
    Trả về True nếu khớp (trong ngưỡng tolerance).
    """
    diff = abs(expected - actual_in_db)
    if diff <= tolerance:
        logger.info(
            "✓ Row count OK | %s | source=%,d | db=%,d | diff=%,d",
            csv_name, expected, actual_in_db, diff,
        )
        return True
    else:
        logger.error(
            "✗ Row count MISMATCH | %s | source=%,d | db=%,d | diff=%,d",
            csv_name, expected, actual_in_db, diff,
        )
        return False
