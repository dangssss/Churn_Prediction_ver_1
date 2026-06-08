
from __future__ import annotations

import uuid
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .ddl import ensure_monitoring_schema, DEFAULT_SCHEMA

def new_run_id() -> str:
    return str(uuid.uuid4())

def start_run(
    engine: Engine,
    *,
    run_id: str,
    status: str = "RUNNING",
    horizon: int | None = None,
    risk_threshold_pct: int | None = None,
    prev_best_k: int | None = None,
    prev_best_f1: float | None = None,
    notes: str | None = None,
    schema: str = DEFAULT_SCHEMA,
) -> None:
    ensure_monitoring_schema(engine, schema=schema)
    q = text(f"""
        INSERT INTO {schema}.churn_ops_runs (
            run_id, status, horizon, risk_threshold_pct,
            prev_best_k, prev_best_f1, notes
        )
        VALUES (
            :run_id, :status, :horizon, :risk_threshold_pct,
            :prev_best_k, :prev_best_f1, :notes
        )
        ON CONFLICT (run_id) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(q, {
            "run_id": run_id,
            "status": status,
            "horizon": horizon,
            "risk_threshold_pct": risk_threshold_pct,
            "prev_best_k": prev_best_k,
            "prev_best_f1": prev_best_f1,
            "notes": notes,
        })

def finish_run(
    engine: Engine,
    *,
    run_id: str,
    status: str,
    window_end: int | None = None,
    cand_best_k: int | None = None,
    cand_best_f1: float | None = None,
    cand_is_accepted: bool | None = None,
    did_retrain: bool | None = None,
    did_score: bool | None = None,
    notes: str | None = None,
    schema: str = DEFAULT_SCHEMA,
) -> None:
    ensure_monitoring_schema(engine, schema=schema)
    q = text(f"""
        UPDATE {schema}.churn_ops_runs
        SET finished_at = now(),
            status = :status,
            window_end = COALESCE(:window_end, window_end),
            cand_best_k = COALESCE(:cand_best_k, cand_best_k),
            cand_best_f1 = COALESCE(:cand_best_f1, cand_best_f1),
            cand_is_accepted = COALESCE(:cand_is_accepted, cand_is_accepted),
            did_retrain = COALESCE(:did_retrain, did_retrain),
            did_score = COALESCE(:did_score, did_score),
            notes = COALESCE(:notes, notes)
        WHERE run_id = :run_id
    """)
    with engine.begin() as conn:
        conn.execute(q, {
            "run_id": run_id,
            "status": status,
            "window_end": window_end,
            "cand_best_k": cand_best_k,
            "cand_best_f1": cand_best_f1,
            "cand_is_accepted": cand_is_accepted,
            "did_retrain": did_retrain,
            "did_score": did_score,
            "notes": notes,
        })
