from __future__ import annotations

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from pendulum import datetime

with DAG(
    dag_id="ds_churn_model_retrain",
    description="Retrain evaluation: train a fresh candidate and promote only if it beats the accepted bundle",
    start_date=datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule="0 1 * * 1",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    tags=["ds_churn", "model", "retrain"],
) as dag:
    run_retrain_if_due = BashOperator(
        task_id="run_retrain_if_due",
        bash_command=(
            "python /churn_source/modeling/ops_lock.py --skip-if-busy -- "
            "bash -lc 'cd /churn_source && python modeling/main.py retrain-if-due --horizon 2 --force-evaluate --tune-hyperparams'"
        ),
        env={
            "TZ": "Asia/Ho_Chi_Minh",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/churn_source/modeling",
            "MAIN_XGB_RANDOM_SEARCH_TRIALS": "20",
            "MAIN_XGB_OPTUNA_TRIALS": "50",
            "MAIN_XGB_OPTUNA_TOP_N_VARIANTS": "1",
        },
        append_env=True,
    )
