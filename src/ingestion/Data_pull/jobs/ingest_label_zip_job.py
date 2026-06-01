from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
import csv
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from psycopg2 import sql

from Data_pull.logging_config import get_logger
from Data_pull.resources import FSConfig, PostgresConfig, get_pg_conn
from Data_pull.resources.fs import LABEL_ZIP_RE

logger = get_logger(__name__)

LABEL_SCHEMA = os.getenv("LABEL_SCHEMA", "Label")
LABEL_COLUMNS = [
    "stt",
    "ma_kh",
    "ma_cms",
    "crm_code_enc",
    "cms_code_enc",
    "ma_don_vi",
    "ten_don_vi",
    "tinh_trang_kh",
]


def _normalize_label_name(stem: str) -> str:
    if not LABEL_ZIP_RE.fullmatch(f"{stem}.zip"):
        raise ValueError(f"Invalid label name: {stem}")
    return stem


def _extract_label_csv(zip_path: Path, fs_cfg: FSConfig, yymm: str) -> tuple[Path, Path, str]:
    table_name = _normalize_label_name(zip_path.stem)
    extract_dir = fs_cfg.saved_dir / "label" / table_name

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    dest_zip = extract_dir / zip_path.name
    shutil.copy2(str(zip_path), str(dest_zip))

    with zipfile.ZipFile(dest_zip, "r") as zf:
        members = [
            m for m in zf.infolist()
            if not m.is_dir() and m.filename.lower().endswith(".csv")
        ]
        expected_stem = f"label_{yymm}".lower()
        matches = [m for m in members if Path(m.filename).stem.lower() == expected_stem]
        if not matches:
            raise RuntimeError(f"{zip_path.name} does not contain {expected_stem}.csv")
        if len(matches) > 1:
            raise RuntimeError(f"{zip_path.name} contains multiple {expected_stem}.csv files")

        member = matches[0]
        sha = hashlib.sha256()
        with zf.open(member) as f:
            for chunk in iter(lambda: f.read(4096 * 1024), b""):
                sha.update(chunk)
        extracted = Path(zf.extract(member, extract_dir))

    return extracted, extract_dir, sha.hexdigest()


def _ensure_ingest_log(cur, ingest_schema: str) -> None:
    cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(ingest_schema)))
    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {}.ingest_log (
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
    """).format(sql.Identifier(ingest_schema)))
    cur.execute(sql.SQL("ALTER TABLE {}.ingest_log ADD COLUMN IF NOT EXISTS file_size bigint;").format(sql.Identifier(ingest_schema)))
    cur.execute(sql.SQL("ALTER TABLE {}.ingest_log ADD COLUMN IF NOT EXISTS file_mtime double precision;").format(sql.Identifier(ingest_schema)))


def _validate_label_header(csv_file: Path) -> str:
    with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()

    detected_headers: dict[str, list[str]] = {}
    for delimiter in (",", ";"):
        reader = csv.reader([first_line], delimiter=delimiter)
        detected_columns = [str(column).strip() for column in next(reader, [])]
        detected_headers[delimiter] = detected_columns
        if all(column in detected_columns for column in LABEL_COLUMNS):
            return delimiter

    raise ValueError(
        f"Missing required label columns in {csv_file.name}. "
        f"Detected columns by delimiter: {detected_headers}. "
        "Expected a comma- or semicolon-delimited CSV with the canonical label header."
    )


def ingest_label_zip_job(
    zip_path: Path,
    fs_cfg: FSConfig,
    pg_cfg: PostgresConfig,
    *,
    ingest_schema: str = "ingest",
    batch_rows: int = 100_000,
) -> Dict[str, Any]:
    m = LABEL_ZIP_RE.fullmatch(zip_path.name)
    if not m:
        raise ValueError(f"Invalid label ZIP name: {zip_path.name}")

    yymm = m.group("yymm")
    table_name = _normalize_label_name(zip_path.stem)
    result = {
        "zip_name": zip_path.name,
        "table_name": table_name,
        "prod_rows": 0,
        "staging_rows": 0,
        "success": False,
        "error": None,
    }

    try:
        csv_file, _, file_sha256 = _extract_label_csv(zip_path, fs_cfg, yymm)
        logger.info("Extracted label CSV %s from %s", csv_file.name, zip_path.name)
        delimiter = _validate_label_header(csv_file)
    except Exception as exc:
        result["error"] = str(exc)
        fail_path = fs_cfg.fail_dir / zip_path.name
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(zip_path), str(fail_path))
        logger.error("[LABEL] Failed to extract %s: %s", zip_path.name, exc)
        return result

    conn = get_pg_conn(pg_cfg)
    conn.autocommit = False
    cur = conn.cursor()
    prod_tbl = sql.SQL("{}.{}").format(sql.Identifier(LABEL_SCHEMA), sql.Identifier(table_name))

    try:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(LABEL_SCHEMA)))
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                stt text,
                ma_kh text,
                ma_cms text,
                crm_code_enc varchar(100),
                cms_code_enc varchar(100),
                ma_don_vi text,
                ten_don_vi text,
                tinh_trang_kh text
            );
        """).format(prod_tbl))
        cur.execute(sql.SQL("TRUNCATE TABLE {};").format(prod_tbl))

        total_inserted = 0
        for chunk in pd.read_csv(
            csv_file,
            sep=delimiter,
            chunksize=batch_rows,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        ):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            missing = [c for c in LABEL_COLUMNS if c not in chunk.columns]
            if missing:
                raise ValueError(
                    f"Missing required label columns in {csv_file.name}: {missing}. "
                    f"Detected columns: {chunk.columns.tolist()}. "
                    "Expected a comma- or semicolon-delimited CSV with the canonical label header."
                )

            rows = chunk[LABEL_COLUMNS].copy()
            rows["crm_code_enc"] = rows["crm_code_enc"].astype(str).str.strip()
            rows["cms_code_enc"] = rows["cms_code_enc"].astype(str).str.strip()

            values = [tuple(None if v == "" else v for v in row) for row in rows.itertuples(index=False, name=None)]
            if not values:
                continue

            from psycopg2.extras import execute_values

            insert_sql = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
                prod_tbl,
                sql.SQL(", ").join(sql.Identifier(c) for c in LABEL_COLUMNS),
            )
            execute_values(cur, insert_sql.as_string(conn), values, page_size=10_000)
            total_inserted += len(values)

        if total_inserted == 0:
            raise ValueError(f"Label CSV {csv_file.name} contains no data rows")

        cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (cms_code_enc);").format(
            sql.Identifier(f"idx_{table_name}_cms_code_enc"),
            prod_tbl,
        ))
        cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (crm_code_enc);").format(
            sql.Identifier(f"idx_{table_name}_crm_code_enc"),
            prod_tbl,
        ))
        cur.execute(sql.SQL("ANALYZE {};").format(prod_tbl))

        _ensure_ingest_log(cur, ingest_schema)
        cur.execute(
            sql.SQL("""
                INSERT INTO {}.ingest_log(
                  zip_name, base, table_name, period_key_month,
                  prod_schema, staging_rows, prod_rows, file_size, file_mtime, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """).format(sql.Identifier(ingest_schema)),
            (
                zip_path.name,
                "label",
                table_name,
                f"20{yymm}",
                LABEL_SCHEMA,
                total_inserted,
                total_inserted,
                zip_path.stat().st_size,
                zip_path.stat().st_mtime,
                "success",
            ),
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.ingestion_manifest (
                zip_name text NOT NULL,
                csv_name text NOT NULL,
                file_sha256 varchar(64) NOT NULL PRIMARY KEY,
                ingested_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            INSERT INTO public.ingestion_manifest (zip_name, csv_name, file_sha256, ingested_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (file_sha256) DO NOTHING;
            """,
            (zip_path.name, csv_file.name, file_sha256),
        )
        conn.commit()

        result["prod_rows"] = total_inserted
        result["staging_rows"] = total_inserted
        result["success"] = True
        logger.info(
            "[LABEL] Loaded %s.%s: %d rows (delimiter=%r)",
            LABEL_SCHEMA,
            table_name,
            total_inserted,
            delimiter,
        )
        return result
    except Exception as exc:
        conn.rollback()
        result["error"] = str(exc)
        fail_path = fs_cfg.fail_dir / zip_path.name
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(zip_path), str(fail_path))
        logger.error("[LABEL] Failed to load %s: %s", zip_path.name, exc)
        return result
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()
