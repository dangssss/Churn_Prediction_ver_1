
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .ddl import ensure_monitoring_schema, DEFAULT_SCHEMA

def score_stats(scores: np.ndarray) -> dict:
    s = np.asarray(scores, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return {"mean": None, "p50": None, "p90": None, "p99": None}
    return {
        "mean": float(np.mean(s)),
        "p50": float(np.quantile(s, 0.50)),
        "p90": float(np.quantile(s, 0.90)),
        "p99": float(np.quantile(s, 0.99)),
    }

def upsert_score_drift(
    engine: Engine,
    *,
    window_end: int,
    horizon: int,
    best_k: int | None,
    active_cnt: int,
    churned_now_cnt: int,
    scores: np.ndarray,
    risk_threshold_pct: int,
    risk_cnt: int,
    anomaly_window: int = 6,
    schema: str = DEFAULT_SCHEMA,
) -> dict:
    """
    Store score drift stats.
    Anomaly detection: compare risk_ratio to median + 3*MAD over last N rows.
    """
    ensure_monitoring_schema(engine, schema=schema)

    stats = score_stats(scores)
    risk_ratio = float(risk_cnt / active_cnt) if active_cnt > 0 else None

    # anomaly detection
    is_anom = False
    reason = None
    if risk_ratio is not None:
        q_hist = text(f"""
            SELECT risk_ratio
            FROM {schema}.score_drift
            WHERE horizon=:h
            ORDER BY window_end DESC
            LIMIT :n
        """)
        df_hist = pd.read_sql(q_hist, engine, params={"h": int(horizon), "n": int(anomaly_window)})
        vals = [float(x) for x in df_hist["risk_ratio"].dropna().tolist() if x is not None]
        if len(vals) >= 4:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(np.array(vals) - med))) + 1e-9
            thresh = med + 3.0 * mad
            if risk_ratio > thresh:
                is_anom = True
                reason = f"risk_ratio {risk_ratio:.4f} > median+3*MAD ({thresh:.4f}) over last {len(vals)}"
    payload = {
        "window_end": int(window_end),
        "horizon": int(horizon),
        "best_k": int(best_k) if best_k is not None else None,
        "active_cnt": int(active_cnt),
        "churned_now_cnt": int(churned_now_cnt),
        "mean_score": stats["mean"],
        "p50": stats["p50"],
        "p90": stats["p90"],
        "p99": stats["p99"],
        "risk_threshold_pct": int(risk_threshold_pct),
        "risk_cnt": int(risk_cnt),
        "risk_ratio": float(risk_ratio) if risk_ratio is not None else None,
        "is_anomaly": bool(is_anom),
        "anomaly_reason": reason,
    }
    q = text(f"""
        INSERT INTO {schema}.score_drift (
            window_end, horizon, best_k,
            active_cnt, churned_now_cnt,
            mean_score, p50, p90, p99,
            risk_threshold_pct, risk_cnt, risk_ratio,
            is_anomaly, anomaly_reason
        )
        VALUES (
            :window_end, :horizon, :best_k,
            :active_cnt, :churned_now_cnt,
            :mean_score, :p50, :p90, :p99,
            :risk_threshold_pct, :risk_cnt, :risk_ratio,
            :is_anomaly, :anomaly_reason
        )
        ON CONFLICT (window_end, horizon)
        DO UPDATE SET
            best_k = EXCLUDED.best_k,
            active_cnt = EXCLUDED.active_cnt,
            churned_now_cnt = EXCLUDED.churned_now_cnt,
            mean_score = EXCLUDED.mean_score,
            p50 = EXCLUDED.p50,
            p90 = EXCLUDED.p90,
            p99 = EXCLUDED.p99,
            risk_threshold_pct = EXCLUDED.risk_threshold_pct,
            risk_cnt = EXCLUDED.risk_cnt,
            risk_ratio = EXCLUDED.risk_ratio,
            is_anomaly = EXCLUDED.is_anomaly,
            anomaly_reason = EXCLUDED.anomaly_reason,
            created_at = now()
    """)
    with engine.begin() as conn:
        conn.execute(q, payload)
    return payload
