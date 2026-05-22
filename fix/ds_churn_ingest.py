"""
ds_churn_ingest.py  –  Airflow DAG
====================================
Chạy mỗi thứ Sáu để ingest ZIP mới từ đội DE.

Task flow:
  discover_zip → ingest_csv_files → validate_counts → archive_zip
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.ingestion.Data_pull.jobs.ingest_zip_job import run_ingest_zip_job
from src.ingestion.Data_pull.ops.post_ingest_maintenance import (
    validate_all_tables,
    print_reconciliation_report,
    archive_zip,
)
from src.ingestion.Data_pull.config.paths import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PW
from sqlalchemy import create_engine

# Natural key cho từng bảng – chỉnh theo schema thực tế
NATURAL_KEY_MAP = {
    # "ten_csv_stem": ["col1", "col2"],
    # Ví dụ:
    # "customer_data": ["cms_code_enc", "snapshot_date"],
}

DEFAULT_ARGS = {
    "owner": "ds-churn",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,
}


def _get_engine():
    url = f"postgresql+psycopg2://{PG_USER}:{PG_PW}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(url)


def task_ingest(**ctx):
    result = run_ingest_zip_job(natural_key_map=NATURAL_KEY_MAP)
    # Đẩy kết quả sang XCom để task sau dùng
    ctx["ti"].xcom_push(key="job_result", value=result)
    ctx["ti"].xcom_push(key="zip_path",   value=result.zip_path)
    ctx["ti"].xcom_push(key="zip_mtime",  value=result.zip_mtime)

    # Fail DAG nếu có CSV nào lỗi
    errors = [r for r in result.csv_results if r.status == "error"]
    if errors:
        raise RuntimeError(f"{len(errors)} CSV bị lỗi: {[e.csv_name for e in errors]}")


def task_validate(**ctx):
    result   = ctx["ti"].xcom_pull(key="job_result")
    engine   = _get_engine()
    report   = validate_all_tables(engine, result)
    print_reconciliation_report(engine, result.zip_mtime)

    mismatches = {k: v for k, v in report.items() if not v["ok"]}
    if mismatches:
        raise ValueError(f"Row count mismatch: {mismatches}")


def task_archive(**ctx):
    zip_path = ctx["ti"].xcom_pull(key="zip_path")
    archive_zip(zip_path, success=True)


with DAG(
    dag_id="ds_churn_ingest",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 6 * * 5",    # Thứ Sáu 06:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["churn", "ingestion"],
) as dag:

    t_ingest   = PythonOperator(task_id="ingest_zip",     python_callable=task_ingest,   provide_context=True)
    t_validate = PythonOperator(task_id="validate_counts", python_callable=task_validate, provide_context=True)
    t_archive  = PythonOperator(task_id="archive_zip",    python_callable=task_archive,  provide_context=True)

    t_ingest >> t_validate >> t_archive
