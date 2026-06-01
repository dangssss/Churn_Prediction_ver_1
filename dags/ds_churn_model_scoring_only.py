from __future__ import annotations

"""
DAG: ds_churn_model_scoring_only
----------------------------------
Chạy duy nhất bước Scoring & Export CSV — KHÔNG bao giờ kích hoạt Retrain.
Sử dụng model bundle đã được chấp nhận gần nhất và config đã lưu trong DB.

Cách sử dụng:
  - Trigger thủ công trên Airflow UI bất cứ khi nào cần xuất danh sách rủi ro mới nhất.
  - Hoặc có thể lên lịch hàng tuần nếu cần (ví dụ: schedule="0 23 * * 1").
"""

from airflow import DAG
from pendulum import datetime

with DAG(
    dag_id="ds_churn_model_scoring_only",
    description="Chỉ Scoring & Export CSV — không Retrain, không Sweep K",
    start_date=datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule=None,  # Chạy thủ công (trigger on demand) hoặc lên lịch hàng tuần tuỳ nhu cầu
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1},
    tags=["ds_churn", "model", "scoring"],
) as dag:

    from airflow.providers.standard.operators.bash import BashOperator

    # Chỉ chạy lệnh export-risk: nạp model bundle cũ và xuất danh sách rủi ro Churn
    # Không bao giờ kích hoạt bước sweep-k hay train-main
    run_scoring_only = BashOperator(
        task_id="run_export_risk_only",
        bash_command=(
            "cd /churn_source && "
            "python modeling/main.py export-risk "
            "--horizon 2 "
            "--risk-threshold-pct 95"
        ),
        env={
            "TZ": "Asia/Ho_Chi_Minh",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/churn_source/modeling",
        },
        append_env=True,
    )
