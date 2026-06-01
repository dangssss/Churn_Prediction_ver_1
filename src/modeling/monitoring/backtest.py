from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config_store.best_config import load_latest_accepted_best_config
from logging_config import get_logger
from preprocess.dataset import load_scoring_table_for_k
from preprocess.label_tables import LABEL_SCHEMA, label_tables_for_horizon, load_label_keys
from .ddl import DEFAULT_SCHEMA, ensure_monitoring_schema

logger = get_logger(__name__)


def _risk_table_name(risk_threshold_pct: int) -> str:
    return f"cus_risk_{int(risk_threshold_pct)}"


def _table_exists(engine: Engine, schema: str, table: str) -> bool:
    q = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema AND table_name = :table
        )
        """
    )
    with engine.connect() as conn:
        return bool(conn.execute(q, {"schema": schema, "table": table}).scalar())


def _resolve_best_k(engine: Engine, *, pred_window_end: int, horizon: int) -> int:
    q = text(
        """
        SELECT best_k
        FROM ml_monitor.score_drift
        WHERE window_end = :window_end AND horizon = :horizon
        LIMIT 1
        """
    )
    try:
        with engine.connect() as conn:
            best_k = conn.execute(
                q,
                {"window_end": int(pred_window_end), "horizon": int(horizon)},
            ).scalar()
        if best_k is not None:
            return int(best_k)
    except Exception:
        pass
    return int(load_latest_accepted_best_config(engine, horizon=int(horizon))["best_k"])


def _truth_for_population(
    engine: Engine,
    population: pd.DataFrame,
    *,
    label_tables: list[str],
) -> pd.DataFrame:
    labels = pd.concat(
        [load_label_keys(engine, table) for table in label_tables],
        ignore_index=True,
    ).drop_duplicates()
    cms_keys = set(labels["cms_code_enc"].dropna().astype(str))
    crm_keys = set(labels["crm_code_enc"].dropna().astype(str))

    out = population[["cms_code_enc"]].copy()
    out["cms_code_enc"] = out["cms_code_enc"].astype(str).str.strip()
    matched = out["cms_code_enc"].isin(cms_keys)
    if crm_keys:
        q = text(
            """
            SELECT cms_code_enc, crm_code_enc
            FROM public.cas_info
            WHERE crm_code_enc IS NOT NULL
            """
        )
        code_map = pd.read_sql(q, engine)
        code_map["cms_code_enc"] = code_map["cms_code_enc"].astype(str).str.strip()
        code_map["crm_code_enc"] = code_map["crm_code_enc"].astype(str).str.strip()
        code_map = code_map.drop_duplicates("cms_code_enc")
        crm_series = out.merge(code_map, on="cms_code_enc", how="left")["crm_code_enc"].fillna("")
        matched = matched | crm_series.isin(crm_keys).to_numpy()
    out["actual_churn"] = matched.astype(int)
    return out.drop_duplicates("cms_code_enc")


def _guardrail(metrics: dict) -> tuple[str, bool, str, str]:
    min_actual = int(os.getenv("BACKTEST_MIN_ACTUAL_CHURN_ROWS", "30"))
    warn_precision = float(os.getenv("BACKTEST_WARN_MIN_PRECISION", "0.05"))
    fail_precision = float(os.getenv("BACKTEST_FAIL_MIN_PRECISION", "0.02"))
    warn_recall = float(os.getenv("BACKTEST_WARN_MIN_RECALL", "0.20"))
    fail_recall = float(os.getenv("BACKTEST_FAIL_MIN_RECALL", "0.10"))
    warn_lift = float(os.getenv("BACKTEST_WARN_MIN_LIFT", "1.50"))
    fail_lift = float(os.getenv("BACKTEST_FAIL_MIN_LIFT", "1.00"))

    if metrics["actual_churn_total"] < min_actual:
        reason = f"actual_churn_total={metrics['actual_churn_total']} < {min_actual}"
        return "WARN", False, reason, "COLLECT_MORE_ACTUAL_LABELS_AND_REVIEW_LABEL_QUALITY"

    precision = metrics["precision_in_list"] or 0.0
    recall = metrics["recall_in_list"] or 0.0
    lift = metrics["lift_vs_random"] or 0.0
    fail_reasons = []
    warn_reasons = []
    if precision < fail_precision:
        fail_reasons.append(f"precision={precision:.4f} < {fail_precision:.4f}")
    elif precision < warn_precision:
        warn_reasons.append(f"precision={precision:.4f} < {warn_precision:.4f}")
    if recall < fail_recall:
        fail_reasons.append(f"recall={recall:.4f} < {fail_recall:.4f}")
    elif recall < warn_recall:
        warn_reasons.append(f"recall={recall:.4f} < {warn_recall:.4f}")
    if lift < fail_lift:
        fail_reasons.append(f"lift={lift:.4f} < {fail_lift:.4f}")
    elif lift < warn_lift:
        warn_reasons.append(f"lift={lift:.4f} < {warn_lift:.4f}")

    if fail_reasons:
        return "FAIL", True, "; ".join(fail_reasons), "HOLD_MODEL_PROMOTION_REVIEW_LABELS_THRESHOLD_AND_RETRAIN"
    if warn_reasons:
        return "WARN", False, "; ".join(warn_reasons), "REVIEW_THRESHOLD_DRIFT_AND_LABEL_QUALITY"
    return "PASS", False, "metrics_within_guardrails", "KEEP_SERVING_CURRENT_BUNDLE"


def run_backtest_for_prediction_origin(
    engine: Engine,
    *,
    pred_window_end: int,
    horizon: int,
    risk_threshold_pct: int,
    schema: str = DEFAULT_SCHEMA,
) -> dict | None:
    """Evaluate one stored prediction origin against actual labels from t+1..t+h."""
    ensure_monitoring_schema(engine, schema=schema)
    label_tables = label_tables_for_horizon(engine, pred_window_end, horizon)
    if not label_tables:
        return None

    risk_table = _risk_table_name(risk_threshold_pct)
    if not _table_exists(engine, "data_static", risk_table):
        return None

    best_k = _resolve_best_k(engine, pred_window_end=pred_window_end, horizon=horizon)
    population, _, _ = load_scoring_table_for_k(
        engine,
        k=best_k,
        window_end=int(pred_window_end),
    )
    if "is_active_now" in population.columns:
        population = population[population["is_active_now"] == 1].copy()
    truth = _truth_for_population(engine, population, label_tables=label_tables)
    active_cnt = int(len(truth))
    if active_cnt == 0:
        return None

    q_list = text(
        f"""
        SELECT DISTINCT cms_code_enc
        FROM data_static.{risk_table}
        WHERE window_end = :pred_window_end
        """
    )
    risk_list = pd.read_sql(q_list, engine, params={"pred_window_end": int(pred_window_end)})
    risk_list["cms_code_enc"] = risk_list["cms_code_enc"].astype(str).str.strip()
    risk_set = set(risk_list["cms_code_enc"])
    truth["predicted_risk"] = truth["cms_code_enc"].isin(risk_set).astype(int)

    tp = int(((truth["predicted_risk"] == 1) & (truth["actual_churn"] == 1)).sum())
    fp = int(((truth["predicted_risk"] == 1) & (truth["actual_churn"] == 0)).sum())
    fn = int(((truth["predicted_risk"] == 0) & (truth["actual_churn"] == 1)).sum())
    tn = int(((truth["predicted_risk"] == 0) & (truth["actual_churn"] == 0)).sum())
    list_size = tp + fp
    actual_churn_total = tp + fn
    precision = tp / list_size if list_size else None
    recall = tp / actual_churn_total if actual_churn_total else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    prevalence = actual_churn_total / active_cnt
    predicted_risk_rate = list_size / active_cnt
    lift = precision / prevalence if precision is not None and prevalence > 0 else None

    out = {
        "pred_window_end": int(pred_window_end),
        "label_window_end": int(max(int(table.rsplit("_", 1)[1]) for table in label_tables)),
        "horizon": int(horizon),
        "best_k": int(best_k),
        "risk_threshold_pct": int(risk_threshold_pct),
        "label_source": "actual",
        "label_tables": ",".join(f"{LABEL_SCHEMA}.{table}" for table in label_tables),
        "active_cnt": active_cnt,
        "list_size": list_size,
        "actual_churn_total": actual_churn_total,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "actual_churn_in_list": tp,
        "actual_churn_rate": prevalence,
        "predicted_risk_rate": predicted_risk_rate,
        "precision_in_list": precision,
        "recall_in_list": recall,
        "specificity": specificity,
        "f1_in_list": f1,
        "lift_vs_random": lift,
    }
    status, blocks_promotion, reasons, action = _guardrail(out)
    out.update(
        {
            "guardrail_status": status,
            "blocks_model_promotion": blocks_promotion,
            "guardrail_reasons": reasons,
            "recommended_action": action,
        }
    )

    q_upsert = text(
        f"""
        INSERT INTO {schema}.backtest (
            pred_window_end, label_window_end, horizon, best_k, risk_threshold_pct,
            label_source, label_tables, active_cnt, list_size,
            churn_true_total, churn_true_in_list, actual_churn_total, actual_churn_in_list,
            true_positive, false_positive, false_negative, true_negative,
            actual_churn_rate, predicted_risk_rate, precision_in_list, recall_in_list,
            specificity, f1_in_list, lift_vs_random, guardrail_status,
            blocks_model_promotion, guardrail_reasons, recommended_action
        )
        VALUES (
            :pred_window_end, :label_window_end, :horizon, :best_k, :risk_threshold_pct,
            :label_source, :label_tables, :active_cnt, :list_size,
            :actual_churn_total, :actual_churn_in_list, :actual_churn_total, :actual_churn_in_list,
            :true_positive, :false_positive, :false_negative, :true_negative,
            :actual_churn_rate, :predicted_risk_rate, :precision_in_list, :recall_in_list,
            :specificity, :f1_in_list, :lift_vs_random, :guardrail_status,
            :blocks_model_promotion, :guardrail_reasons, :recommended_action
        )
        ON CONFLICT (pred_window_end, horizon) DO UPDATE SET
            label_window_end = EXCLUDED.label_window_end,
            best_k = EXCLUDED.best_k,
            risk_threshold_pct = EXCLUDED.risk_threshold_pct,
            label_source = EXCLUDED.label_source,
            label_tables = EXCLUDED.label_tables,
            active_cnt = EXCLUDED.active_cnt,
            list_size = EXCLUDED.list_size,
            churn_true_total = EXCLUDED.churn_true_total,
            churn_true_in_list = EXCLUDED.churn_true_in_list,
            actual_churn_total = EXCLUDED.actual_churn_total,
            actual_churn_in_list = EXCLUDED.actual_churn_in_list,
            true_positive = EXCLUDED.true_positive,
            false_positive = EXCLUDED.false_positive,
            false_negative = EXCLUDED.false_negative,
            true_negative = EXCLUDED.true_negative,
            actual_churn_rate = EXCLUDED.actual_churn_rate,
            predicted_risk_rate = EXCLUDED.predicted_risk_rate,
            precision_in_list = EXCLUDED.precision_in_list,
            recall_in_list = EXCLUDED.recall_in_list,
            specificity = EXCLUDED.specificity,
            f1_in_list = EXCLUDED.f1_in_list,
            lift_vs_random = EXCLUDED.lift_vs_random,
            guardrail_status = EXCLUDED.guardrail_status,
            blocks_model_promotion = EXCLUDED.blocks_model_promotion,
            guardrail_reasons = EXCLUDED.guardrail_reasons,
            recommended_action = EXCLUDED.recommended_action,
            created_at = now()
        """
    )
    with engine.begin() as conn:
        conn.execute(q_upsert, out)
    return out


def run_backtests_for_available_actual_labels(
    engine: Engine,
    *,
    horizon: int,
    risk_threshold_pct: int,
    schema: str = DEFAULT_SCHEMA,
) -> list[dict]:
    """Backtest every stored origin that has complete actual-label coverage."""
    ensure_monitoring_schema(engine, schema=schema)
    risk_table = _risk_table_name(risk_threshold_pct)
    if not _table_exists(engine, "data_static", risk_table):
        return []
    origins = pd.read_sql(
        text(
            f"""
            SELECT DISTINCT window_end
            FROM ml_monitor.score_drift
            WHERE horizon = :horizon
              AND risk_threshold_pct = :risk_threshold_pct
              AND window_end IS NOT NULL
            ORDER BY window_end
            """
        ),
        engine,
        params={
            "horizon": int(horizon),
            "risk_threshold_pct": int(risk_threshold_pct),
        },
    )["window_end"].tolist()
    results = []
    for origin in origins:
        try:
            result = run_backtest_for_prediction_origin(
                engine,
                pred_window_end=int(origin),
                horizon=int(horizon),
                risk_threshold_pct=int(risk_threshold_pct),
                schema=schema,
            )
        except Exception as exc:
            logger.warning("[BACKTEST] Skip origin=%s: %s", origin, exc)
            continue
        if result is not None:
            results.append(result)
    return results


def run_backtest_precision_in_list(
    engine: Engine,
    *,
    label_window_end: int,
    horizon: int,
    risk_threshold_pct: int,
    best_k_for_population: int = 3,
    schema: str = DEFAULT_SCHEMA,
) -> dict | None:
    """Backward-compatible wrapper for the previous monthly pipeline."""
    from infra.yymm import shift_yymm

    del best_k_for_population
    return run_backtest_for_prediction_origin(
        engine,
        pred_window_end=int(shift_yymm(str(label_window_end), -int(horizon))),
        horizon=int(horizon),
        risk_threshold_pct=int(risk_threshold_pct),
        schema=schema,
    )
