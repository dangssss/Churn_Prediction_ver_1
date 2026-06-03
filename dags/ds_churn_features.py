from __future__ import annotations

from airflow import DAG
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from pendulum import datetime

with DAG(
    dag_id="ds_churn_features",
    start_date=datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule=None,          # ch? ch?y khi ingest trigger
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1},
    tags=["ds_churn", "features"],
) as dag:

    # GI? NGUYÊN LOGIC: dùng dúng entrypoint hi?n t?i
    from airflow.providers.standard.operators.bash import BashOperator

    # GI? NGUYÊN LOGIC: dùng dúng entrypoint hi?n t?i
    # Assumes code is at /churn_source/Preprocess/src/operations/run/run_feature_generation.py
    run_features = BashOperator(
        task_id="run_features",
        bash_command=(
            "python /churn_source/modeling/ops_lock.py --wait-seconds 0 -- "
            "bash -lc 'cd /churn_source/preprocessing && "
            "python src/operations/run/run_feature_generation.py --start 2025-01-01'"
        ),
        env={
            "WINDOW_SCHEMA": "data_window",
            "TZ": "Asia/Ho_Chi_Minh",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/churn_source/preprocessing", # Ensure imports work
        },
        append_env=True,
    )

    trigger_post_feature = TriggerDagRunOperator(
        task_id="trigger_post_feature",
        trigger_dag_id="ds_churn_model_post_feature",
        conf={
            "upstream_features_run_id": "{{ run_id }}",
            "logical_date": "{{ ds }}",
        },
        wait_for_completion=False,
        reset_dag_run=True,
    )

    run_features >> trigger_post_feature
