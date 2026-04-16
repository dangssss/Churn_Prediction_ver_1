from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

def ensure_best_config_table(engine: Engine) -> None:
    create_sql = """
    CREATE SCHEMA IF NOT EXISTS data_static;

    CREATE TABLE IF NOT EXISTS data_static.model_best_config (
        as_of_month       INT NOT NULL,
        horizon           INT NOT NULL,

        best_k            INT NOT NULL,
        use_static        BOOLEAN NOT NULL,

        best_threshold    DOUBLE PRECISION NOT NULL,
        best_spw          DOUBLE PRECISION NOT NULL,

        metric_f1_val     DOUBLE PRECISION,
        metric_pr_auc_val DOUBLE PRECISION,
        val_month         INT,
        target_month      INT,

        created_at        TIMESTAMP DEFAULT now(),
        notes            TEXT,

        -- production gating
        is_accepted       BOOLEAN NOT NULL DEFAULT TRUE,
        prev_accepted_f1  DOUBLE PRECISION,
        accept_rule       TEXT,
        accepted_at       TIMESTAMP,

        PRIMARY KEY (as_of_month, horizon)
    );
    """
    with engine.begin() as conn:
        for stmt in create_sql.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

    # add new columns if table already existed
    alter_sql = """
    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS is_accepted BOOLEAN NOT NULL DEFAULT TRUE;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS prev_accepted_f1 DOUBLE PRECISION;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS accept_rule TEXT;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS accepted_at TIMESTAMP;
    """
    with engine.begin() as conn:
        for stmt in alter_sql.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

def upsert_best_config(engine: Engine, best_config: dict) -> None:
    ensure_best_config_table(engine)
    upsert_sql = """
    INSERT INTO data_static.model_best_config (
        as_of_month, horizon,
        best_k, use_static,
        best_threshold, best_spw,
        metric_f1_val, metric_pr_auc_val, val_month, target_month,
        notes,
        is_accepted, prev_accepted_f1, accept_rule, accepted_at
    )
    VALUES (
        :as_of_month, :horizon,
        :best_k, :use_static,
        :best_threshold, :best_spw,
        :metric_f1_val, :metric_pr_auc_val, :val_month, :target_month,
        :notes,
        :is_accepted, :prev_accepted_f1, :accept_rule, :accepted_at
    )
    ON CONFLICT (as_of_month, horizon) DO UPDATE SET
        best_k=EXCLUDED.best_k,
        use_static=EXCLUDED.use_static,
        best_threshold=EXCLUDED.best_threshold,
        best_spw=EXCLUDED.best_spw,
        metric_f1_val=EXCLUDED.metric_f1_val,
        metric_pr_auc_val=EXCLUDED.metric_pr_auc_val,
        val_month=EXCLUDED.val_month,
        target_month=EXCLUDED.target_month,
        created_at=now(),
        notes=EXCLUDED.notes;
    """
    with engine.begin() as conn:
        conn.execute(text(upsert_sql), best_config)

def load_latest_best_config(engine: Engine, horizon: int) -> dict:
    q = text("""
        SELECT *
        FROM data_static.model_best_config
        WHERE horizon=:h
        ORDER BY as_of_month DESC
        LIMIT 1
    """)
    df = pd.read_sql(q, engine, params={"h": horizon})
    if df.empty:
        raise ValueError(f"Không tìm thấy best_config cho horizon={horizon}")
    return df.iloc[0].to_dict()



def load_latest_accepted_best_config(engine: Engine, horizon: int) -> dict:
    q = text("""
        SELECT *
        FROM data_static.model_best_config
        WHERE horizon=:h AND is_accepted=TRUE
        ORDER BY as_of_month DESC
        LIMIT 1
    """)
    df = pd.read_sql(q, engine, params={"h": horizon})
    if df.empty:
        raise ValueError(f"Không tìm thấy accepted best_config cho horizon={horizon}")
    return df.iloc[0].to_dict()

def load_previous_accepted_best_config(engine: Engine, horizon: int) -> dict | None:
    q = text("""
        SELECT *
        FROM data_static.model_best_config
        WHERE horizon=:h AND is_accepted=TRUE
        ORDER BY as_of_month DESC
        OFFSET 1
        LIMIT 1
    """)
    df = pd.read_sql(q, engine, params={"h": horizon})
    return None if df.empty else df.iloc[0].to_dict()

def ensure_main_columns(engine: Engine) -> None:
    alter_sql = """
    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS main_threshold DOUBLE PRECISION;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS main_f1_val DOUBLE PRECISION;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS main_ap_val DOUBLE PRECISION;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS model_type TEXT;

    -- early stopping meta
    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS xgb_best_iteration INTEGER;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS xgb_best_score DOUBLE PRECISION;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS xgb_es_rounds INTEGER;

    -- sanity / guardrail
    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS val_prevalence DOUBLE PRECISION;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS dummy_ap_const0 DOUBLE PRECISION;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS dummy_ap_random DOUBLE PRECISION;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS dummy_ap_simple2 DOUBLE PRECISION;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS dummy_simple2_features TEXT;

    ALTER TABLE data_static.model_best_config
    ADD COLUMN IF NOT EXISTS guardrail_warning TEXT;
    """
    with engine.begin() as conn:
        for stmt in alter_sql.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

def update_main_metrics(engine: Engine, as_of_month: int, horizon: int, main_report: dict) -> int:
    ensure_main_columns(engine)
    if main_report.get("guardrail_warning"):
        raise RuntimeError(f"Guardrail failed: {main_report['guardrail_warning']}")

    update_sql = """
    UPDATE data_static.model_best_config
    SET main_threshold = :main_thr,
        main_f1_val = :main_f1,
        main_ap_val = :main_ap,
        model_type = :model_type,

        xgb_best_iteration = :xgb_best_iteration,
        xgb_best_score = :xgb_best_score,
        xgb_es_rounds = :xgb_es_rounds,

        val_prevalence = :val_prevalence,
        dummy_ap_const0 = :dummy_ap_const0,
        dummy_ap_random = :dummy_ap_random,
        dummy_ap_simple2 = :dummy_ap_simple2,
        dummy_simple2_features = :dummy_simple2_features,
        guardrail_warning = :guardrail_warning,

        created_at = now()
    WHERE as_of_month = :as_of_month AND horizon = :horizon;
    """
    payload = {
        "main_thr": float(main_report["thr_main_opt"]),
        "main_f1": float(main_report["f1@main_thr"]),
        "main_ap": float(main_report["AP_val"]),
        "model_type": "xgboost",

        "xgb_best_iteration": main_report.get("xgb_best_iteration"),
        "xgb_best_score": main_report.get("xgb_best_score"),
        "xgb_es_rounds": main_report.get("xgb_es_rounds"),

        "val_prevalence": main_report.get("val_prevalence"),
        "dummy_ap_const0": main_report.get("dummy_ap_const0"),
        "dummy_ap_random": main_report.get("dummy_ap_random"),
        "dummy_ap_simple2": main_report.get("dummy_ap_simple2"),
        "dummy_simple2_features": main_report.get("dummy_simple2_features"),
        "guardrail_warning": main_report.get("guardrail_warning"),

        "as_of_month": int(as_of_month),
        "horizon": int(horizon),
    }
    with engine.begin() as conn:
        res = conn.execute(text(update_sql), payload)
        return int(res.rowcount)
