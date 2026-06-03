from __future__ import annotations

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from pendulum import datetime

with DAG(
    dag_id="ds_churn_model_post_feature",
    description="After features: bootstrap or retrain when needed, then score using a ready bundle",
    start_date=datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    tags=["ds_churn", "model", "orchestration"],
) as dag:
    prepare_scoring_bundle = BashOperator(
        task_id="prepare_scoring_bundle",
        bash_command=(
            "python /churn_source/modeling/ops_lock.py --wait-seconds 0 -- "
            "bash -lc 'cd /churn_source && python modeling/main.py prepare-scoring "
            "--horizon 2 --risk-threshold-pct 95'"
        ),
        env={
            "TZ": "Asia/Ho_Chi_Minh",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/churn_source/modeling",
        },
        append_env=True,
    )

    trigger_scoring = TriggerDagRunOperator(
        task_id="trigger_scoring",
        trigger_dag_id="ds_churn_model_scoring_only",
        conf={
            "upstream_post_feature_run_id": "{{ run_id }}",
            "logical_date": "{{ ds }}",
        },
        wait_for_completion=False,
        reset_dag_run=True,
    )

    prepare_scoring_bundle >> trigger_scoring
