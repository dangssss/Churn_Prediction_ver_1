from __future__ import annotations

from airflow import DAG
from pendulum import datetime

with DAG(
    dag_id="ds_churn_model_monthly",
    start_date=datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule=None,          # chỉ chạy khi features trigger
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    tags=["ds_churn", "model"],
) as dag:

    from airflow.operators.bash import BashOperator

    # Run model training locally
    # Assumes code is at /churn_source/modeling/main.py
    # Note: bundle-dir default is now handled in python code -> src.config.paths.CHURN_MODEL_DIR/bundles/latest
    # But we can override if needed.
    run_monthly_churn = BashOperator(
        task_id="run_monthly_churn",
        bash_command="cd /churn_source && python modeling/main.py run-monthly --horizon 2 --risk-threshold-pct 95",
        env={
            "TZ": "Asia/Ho_Chi_Minh", 
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/churn_source/modeling", # Ensure imports work
            # DB Connection is loaded from .env or airflow connection
        },
        append_env=True,
    )
