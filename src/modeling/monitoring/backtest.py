
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from infra.yymm import shift_yymm
from preprocess.dataset import load_scoring_table_for_k
from .ddl import ensure_monitoring_schema, DEFAULT_SCHEMA

def _risk_hist_table_name(risk_threshold_pct: int) -> str:
    return f"cus_risk_{int(risk_threshold_pct)}_hist"

def run_backtest_precision_in_list(
    engine: Engine,
    *,
    label_window_end: int,
    horizon: int,
    risk_threshold_pct: int,
    best_k_for_population: int = 3,
    schema: str = DEFAULT_SCHEMA,
) -> dict | None:
    """
    Backtest for pred_month = label_month - H using risk history list.
    Computes precision_in_list / recall_in_list on active population at pred_month.
    """
    ensure_monitoring_schema(engine, schema=schema)
    pred_month = int(shift_yymm(str(label_window_end), -int(horizon)))

    hist_tbl = _risk_hist_table_name(risk_threshold_pct)
    # Does history table exist?
    q_exists = text("""
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema='data_static' AND table_name=:t
        LIMIT 1
    """)
    if pd.read_sql(q_exists, engine, params={"t": hist_tbl}).empty:
        return None

    q_list = text(f"""
        SELECT cms_code_enc
        FROM data_static.{hist_tbl}
        WHERE window_end = :pred_month
    """)
    df_list = pd.read_sql(q_list, engine, params={"pred_month": int(pred_month)})
    list_size = int(len(df_list))
    if list_size == 0:
        return None

    # active population at pred_month
    df_pred_pop, _, _ = load_scoring_table_for_k(engine, k=int(best_k_for_population), window_end=int(pred_month))
    df_pred_active = df_pred_pop[df_pred_pop.get("is_active_now", 0) == 1][["cms_code_enc"]].copy()
    active_cnt = int(len(df_pred_active))

    # churn truth at label_month
    df_label, _, _ = load_scoring_table_for_k(engine, k=int(best_k_for_population), window_end=int(label_window_end))
    df_label_truth = df_label[["cms_code_enc", "is_churned_now"]].copy()
    df_label_truth["is_churned_now"] = pd.to_numeric(df_label_truth["is_churned_now"], errors="coerce").fillna(0).astype(int)

    # list join
    df_list2 = df_list.merge(df_label_truth, on="cms_code_enc", how="left")
    df_list2["is_churned_now"] = df_list2["is_churned_now"].fillna(0).astype(int)
    churn_true_in_list = int(df_list2["is_churned_now"].sum())

    # total churn among active population
    df_pred_active_truth = df_pred_active.merge(df_label_truth, on="cms_code_enc", how="left")
    df_pred_active_truth["is_churned_now"] = df_pred_active_truth["is_churned_now"].fillna(0).astype(int)
    churn_true_total = int(df_pred_active_truth["is_churned_now"].sum())

    precision = float(churn_true_in_list / list_size) if list_size > 0 else None
    recall = float(churn_true_in_list / churn_true_total) if churn_true_total > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))

    out = {
        "pred_window_end": int(pred_month),
        "label_window_end": int(label_window_end),
        "horizon": int(horizon),
        "best_k": None,
        "risk_threshold_pct": int(risk_threshold_pct),
        "active_cnt": int(active_cnt),
        "list_size": int(list_size),
        "churn_true_total": int(churn_true_total),
        "churn_true_in_list": int(churn_true_in_list),
        "precision_in_list": precision,
        "recall_in_list": recall,
        "f1_in_list": f1,
    }

    q_upsert = text(f"""
        INSERT INTO {schema}.backtest (
            pred_window_end, label_window_end, horizon,
            best_k, risk_threshold_pct,
            active_cnt, list_size, churn_true_total, churn_true_in_list,
            precision_in_list, recall_in_list, f1_in_list
        )
        VALUES (
            :pred_window_end, :label_window_end, :horizon,
            :best_k, :risk_threshold_pct,
            :active_cnt, :list_size, :churn_true_total, :churn_true_in_list,
            :precision_in_list, :recall_in_list, :f1_in_list
        )
        ON CONFLICT (pred_window_end, horizon)
        DO UPDATE SET
            label_window_end = EXCLUDED.label_window_end,
            best_k = EXCLUDED.best_k,
            risk_threshold_pct = EXCLUDED.risk_threshold_pct,
            active_cnt = EXCLUDED.active_cnt,
            list_size = EXCLUDED.list_size,
            churn_true_total = EXCLUDED.churn_true_total,
            churn_true_in_list = EXCLUDED.churn_true_in_list,
            precision_in_list = EXCLUDED.precision_in_list,
            recall_in_list = EXCLUDED.recall_in_list,
            f1_in_list = EXCLUDED.f1_in_list,
            created_at = now()
    """)
    with engine.begin() as conn:
        conn.execute(q_upsert, out)
    return out
