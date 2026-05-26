# Walkthrough: Ingestion Deduplication & Row Count Fix

## Vấn đề ban đầu

Hệ thống ingest dữ liệu churn chạy mỗi thứ Sáu gặp 3 bug liên quan:

| Bug | Triệu chứng | Nguyên nhân gốc |
|-----|-------------|----------------|
| **#1** Tail-end chunk over-counting | Log báo 5,200,000 dòng nhưng file chỉ có 5,182,525 | Dùng `offset += CHUNK_SIZE` thay vì `len(chunk)` thực tế |
| **#2** ZIP tuần mới ghi đè ZIP cũ | DB thừa ~2M dòng mỗi tuần do ingest lại toàn bộ | Không có cơ chế tracking file đã ingest |
| **#3** Retry sau lỗi gây duplicate | Chạy lại job sau lỗi giữa chừng → duplicate rows | Insert thẳng vào production, không idempotent |

---

## Kiến trúc sau khi fix

```
ZIP (incoming_dir)
  └─► unzip_and_discover()
        ├─ Hash SHA256 từng CSV member trong ZIP (in-memory, không giải nén)
        ├─ Tra public.ingestion_manifest → bỏ qua nếu đã có hash
        └─ Chỉ giải nén các CSV chưa ingest
              │
              ▼
      copy_and_insert_to_production()
        ├─ pd.read_csv(chunksize=50_000)   ← Fix Bug #1
        ├─ Transform + Encrypt mỗi row
        ├─ COPY vào TEMP TABLE _stg_{table}
        │     └─ 1 transaction duy nhất (commit sau khi staging xong toàn bộ file)
        ├─ _dedup_exact_duplicates() trên staging
        ├─ INSERT INTO prod WHERE NOT EXISTS (dùng DEDUP_KEYS)   ← Fix Bug #3
        └─ INSERT INTO public.ingestion_manifest (SHA256)         ← Fix Bug #2
```

---

## Files đã thay đổi

### [`unzip_and_discover.py`](file:///d:/churn.deployment/src/ingestion/Data_pull/ops/unzip_and_discover.py)

- Thêm `ensure_manifest_table(cur)` — tạo `public.ingestion_manifest` nếu chưa có.
- Khi `pg_cfg` được truyền vào: load toàn bộ SHA256 đã ingest từ manifest trước khi giải nén.
- Với mỗi CSV member trong ZIP: tính SHA256 in-memory (chunks 4MB), nếu đã có trong manifest → **skip extraction** và log `SKIP (Already ingested)`.
- Trả về `csv_hashes: {str(path): sha256}` trong meta dict để downstream dùng lại, không phải hash lại.

### [`copy_and_insert_to_production.py`](file:///d:/churn.deployment/src/ingestion/Data_pull/ops/copy_and_insert_to_production.py)

**Bug #1 fix — chunking đúng:**
```python
# TRƯỚC (sai):
offset += CHUNK_SIZE  # luôn cộng 50k kể cả chunk cuối
logger.info("Đã nạp %d dòng", offset)  # báo sai

# SAU (đúng):
chunks = pd.read_csv(csv_file, chunksize=batch_rows, ...)
for chunk in chunks:
    chunk_len = len(chunk)        # đếm thực tế
    file_rows_read += chunk_len   # cộng dồn đúng
```

**Bug #3 fix — staging idempotent:**
```python
# Tạo TEMP TABLE với prefix _stg_ (tránh trùng tên production table)
staging_tbl = f"_stg_{table_name}"
cur.execute(f"CREATE TEMP TABLE {staging_tbl} (LIKE {prod_tbl})")

# Staging toàn bộ file trong 1 transaction (không commit giữa chunk)
for chunk in chunks:
    _bulk_insert_rows(cur, staging_tbl, rows_buffer, base)
conn.commit()  # 1 lần duy nhất sau khi xong file

# Dedup staging → prod idempotent
INSERT INTO prod ({cols})
SELECT {cols} FROM staging s
WHERE NOT EXISTS (
    SELECT 1 FROM prod p
    WHERE p."key_col" IS NOT DISTINCT FROM s."key_col"
)
```

**Bug #2 fix — manifest tracking:**
```python
# Ghi SHA256 vào manifest sau khi insert thành công
INSERT INTO public.ingestion_manifest (zip_name, csv_name, file_sha256, ingested_at)
VALUES (%s, %s, %s, now())
ON CONFLICT (file_sha256) DO NOTHING
```

**`_dedup_exact_duplicates()` — fix cho TEMP table:**
- Thêm tham số `columns_list: Optional[List[str]]` — khi truyền vào, bỏ qua lookup `information_schema.columns` (TEMP tables không nằm trong `public` schema nên query đó trả về empty).

### [`ingest_zip_job.py`](file:///d:/churn.deployment/src/ingestion/Data_pull/jobs/ingest_zip_job.py)

- Truyền `pg_cfg` vào `unzip_and_discover()` để enable manifest checking.

---

## Kết luận về `fix/` folder

Folder `fix/` là một bản **prototype riêng biệt, KHÔNG tương thích** với codebase hiện tại:

| Vấn đề | Chi tiết |
|--------|---------|
| Thiếu transform pipeline | Bỏ qua `transform_bccp_orderitem_row`, encryption, `EXPECTED_HEADERS` |
| Schema manifest khác | Dùng `zip_mtime DOUBLE PRECISION` thay vì `file_sha256 PRIMARY KEY` |
| Circular import | `fix/unzip_and_discover.py` import từ `fix/ingest_zip_job.py` |

**Kết luận**: Không merge `fix/` vào source. Các bug đã được fix trực tiếp trong source files hiện tại.

---

## Manifest Table DDL

```sql
CREATE TABLE IF NOT EXISTS public.ingestion_manifest (
    zip_name    text           NOT NULL,
    csv_name    text           NOT NULL,
    file_sha256 varchar(64)    NOT NULL PRIMARY KEY,
    ingested_at timestamptz    DEFAULT now()
);
```

---

## Kiểm tra khi chạy thật

```bash
# Lần 1 — phải ingest đầy đủ
python src/ingestion/run_job_now.py
# Log phải thấy: "Inserted X rows" cho từng CSV

# Lần 2 — idempotency test (cùng ZIP)
python src/ingestion/run_job_now.py
# Log phải thấy: "CSV already in manifest database. Skipping."
# Và: "Inserted 0 rows"
```
