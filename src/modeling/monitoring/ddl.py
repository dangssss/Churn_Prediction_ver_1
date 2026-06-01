
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

DEFAULT_SCHEMA = "ml_monitor"

def ensure_monitoring_schema(engine: Engine, schema: str = DEFAULT_SCHEMA) -> None:
    """
    Create monitoring schema + core tables (Postgres).

    Tables:
      - <schema>.churn_ops_runs   (run log / audit)
      - <schema>.feature_drift
      - <schema>.score_drift
      - <schema>.backtest
    """
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};

    CREATE TABLE IF NOT EXISTS {schema}.churn_ops_runs (
        run_id            TEXT PRIMARY KEY,
        started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        finished_at       TIMESTAMPTZ,
        status            TEXT NOT NULL, -- RUNNING/SUCCESS/FAILED

        window_end        INT,
        horizon           INT,
        risk_threshold_pct INT,

        prev_best_k       INT,
        prev_best_f1      DOUBLE PRECISION,

        cand_best_k       INT,
        cand_best_f1      DOUBLE PRECISION,
        cand_is_accepted  BOOLEAN,

        did_retrain       BOOLEAN,
        did_score         BOOLEAN,

        notes             TEXT
    );

    CREATE TABLE IF NOT EXISTS {schema}.feature_drift (
        window_end        INT NOT NULL,
        horizon           INT NOT NULL,
        best_k            INT,
        feature_name      TEXT NOT NULL,

        psi               DOUBLE PRECISION,
        ks_stat           DOUBLE PRECISION,

        severity          TEXT,  -- OK/WARN/ALERT
        is_anomaly        BOOLEAN DEFAULT FALSE,

        details_json      JSONB,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

        PRIMARY KEY (window_end, horizon, feature_name)
    );

    CREATE TABLE IF NOT EXISTS {schema}.score_drift (
        window_end        INT NOT NULL,
        horizon           INT NOT NULL,
        best_k            INT,

        active_cnt        INT,
        churned_now_cnt   INT,

        mean_score        DOUBLE PRECISION,
        p50               DOUBLE PRECISION,
        p90               DOUBLE PRECISION,
        p99               DOUBLE PRECISION,

        risk_threshold_pct INT,
        risk_cnt          INT,
        risk_ratio        DOUBLE PRECISION,

        is_anomaly        BOOLEAN DEFAULT FALSE,
        anomaly_reason    TEXT,

        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

        PRIMARY KEY (window_end, horizon)
    );

    CREATE TABLE IF NOT EXISTS {schema}.backtest (
        pred_window_end   INT NOT NULL,
        label_window_end  INT NOT NULL,
        horizon           INT NOT NULL,

        best_k            INT,
        risk_threshold_pct INT,

        label_source       TEXT,
        label_tables       TEXT,

        active_cnt        INT,
        list_size         INT,
        churn_true_total  INT,
        churn_true_in_list INT,
        actual_churn_total INT,
        actual_churn_in_list INT,

        true_positive      INT,
        false_positive     INT,
        false_negative     INT,
        true_negative      INT,

        actual_churn_rate  DOUBLE PRECISION,
        predicted_risk_rate DOUBLE PRECISION,
        precision_in_list DOUBLE PRECISION,
        recall_in_list    DOUBLE PRECISION,
        specificity       DOUBLE PRECISION,
        f1_in_list        DOUBLE PRECISION,
        lift_vs_random     DOUBLE PRECISION,

        guardrail_status   TEXT,
        blocks_model_promotion BOOLEAN NOT NULL DEFAULT FALSE,
        guardrail_reasons  TEXT,
        recommended_action TEXT,

        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

        PRIMARY KEY (pred_window_end, horizon)
    );

    CREATE INDEX IF NOT EXISTS idx_score_drift_created_at ON {schema}.score_drift (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_feature_drift_created_at ON {schema}.feature_drift (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_backtest_created_at ON {schema}.backtest (created_at DESC);

    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS label_source TEXT;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS label_tables TEXT;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS actual_churn_total INT;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS actual_churn_in_list INT;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS true_positive INT;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS false_positive INT;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS false_negative INT;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS true_negative INT;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS actual_churn_rate DOUBLE PRECISION;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS predicted_risk_rate DOUBLE PRECISION;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS specificity DOUBLE PRECISION;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS lift_vs_random DOUBLE PRECISION;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS guardrail_status TEXT;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS blocks_model_promotion BOOLEAN NOT NULL DEFAULT FALSE;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS guardrail_reasons TEXT;
    ALTER TABLE {schema}.backtest ADD COLUMN IF NOT EXISTS recommended_action TEXT;
    """
    with engine.begin() as conn:
        for stmt in ddl.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
