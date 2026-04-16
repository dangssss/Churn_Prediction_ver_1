
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .psi import make_numeric_profile, counts_on_profile, psi_from_counts, discrete_ks_from_counts
from .ddl import ensure_monitoring_schema, DEFAULT_SCHEMA

def compute_feature_profile(
    df_train: pd.DataFrame,
    *,
    feat_cols: List[str],
    cat_cols: List[str] | None = None,
    n_bins: int = 10,
    max_features: int = 200,
) -> dict:
    """
    Build a lightweight baseline profile from training data for drift checks.

    Current implementation:
      - numeric only PSI/KS bins for up to max_features columns
      - categorical profile is stored as top-k freq map (optional; not used for PSI yet)
    """
    cat_cols = cat_cols or []
    prof = {"numeric": {}, "categorical": {}, "n_bins": int(n_bins)}
    # numeric
    num_cols = []
    for c in feat_cols:
        if c in cat_cols:
            continue
        if c not in df_train.columns:
            continue
        num_cols.append(c)
    # cap
    num_cols = num_cols[:max_features]
    for c in num_cols:
        p = make_numeric_profile(df_train[c], n_bins=n_bins)
        if p is not None:
            prof["numeric"][c] = p

    # cat top freq
    for c in cat_cols:
        if c not in df_train.columns:
            continue
        s = df_train[c].astype(str).fillna("NA")
        vc = s.value_counts(dropna=False).head(30)
        prof["categorical"][c] = {"top": vc.to_dict(), "n": int(len(s))}
    return prof

def compute_feature_drift(df_current: pd.DataFrame, profile: dict) -> pd.DataFrame:
    rows = []
    num_prof = profile.get("numeric", {}) or {}
    for feat, p in num_prof.items():
        if feat not in df_current.columns:
            rows.append({"feature_name": feat, "psi": None, "ks_stat": None, "severity": "ALERT",
                         "details_json": {"reason": "missing_in_current"}})
            continue
        cur_counts = counts_on_profile(df_current[feat], p)
        train_counts = np.array(p["train_counts"], dtype=float)
        psi = psi_from_counts(train_counts, cur_counts)
        ks = discrete_ks_from_counts(train_counts, cur_counts)
        sev = "OK"
        if psi > 0.2:
            sev = "ALERT"
        elif psi > 0.1:
            sev = "WARN"
        rows.append({
            "feature_name": feat,
            "psi": float(psi),
            "ks_stat": float(ks),
            "severity": sev,
            "details_json": {"train_n": p.get("n"), "cur_n": int(pd.to_numeric(df_current[feat], errors="coerce").dropna().shape[0])}
        })
    return pd.DataFrame(rows)

def upsert_feature_drift(
    engine: Engine,
    *,
    window_end: int,
    horizon: int,
    best_k: int | None,
    drift_df: pd.DataFrame,
    schema: str = DEFAULT_SCHEMA,
) -> int:
    ensure_monitoring_schema(engine, schema=schema)
    if drift_df is None or drift_df.empty:
        return 0

    q = text(f"""
        INSERT INTO {schema}.feature_drift (
            window_end, horizon, best_k, feature_name,
            psi, ks_stat, severity, is_anomaly, details_json
        )
        VALUES (
            :window_end, :horizon, :best_k, :feature_name,
            :psi, :ks_stat, :severity, :is_anomaly, CAST(:details_json AS JSONB)
        )
        ON CONFLICT (window_end, horizon, feature_name)
        DO UPDATE SET
            best_k = EXCLUDED.best_k,
            psi = EXCLUDED.psi,
            ks_stat = EXCLUDED.ks_stat,
            severity = EXCLUDED.severity,
            is_anomaly = EXCLUDED.is_anomaly,
            details_json = EXCLUDED.details_json,
            created_at = now()
    """)
    payload = []
    for _, r in drift_df.iterrows():
        psi = r.get("psi")
        sev = r.get("severity")
        is_anom = bool(sev == "ALERT")
        payload.append({
            "window_end": int(window_end),
            "horizon": int(horizon),
            "best_k": int(best_k) if best_k is not None else None,
            "feature_name": str(r.get("feature_name")),
            "psi": float(psi) if psi is not None and not pd.isna(psi) else None,
            "ks_stat": float(r.get("ks_stat")) if r.get("ks_stat") is not None and not pd.isna(r.get("ks_stat")) else None,
            "severity": str(sev) if sev else None,
            "is_anomaly": is_anom,
            "details_json": json.dumps(r.get("details_json") or {}, ensure_ascii=False),
        })

    with engine.begin() as conn:
        conn.execute(q, payload)
    return int(len(payload))
