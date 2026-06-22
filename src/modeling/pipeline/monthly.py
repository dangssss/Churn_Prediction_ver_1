
from __future__ import annotations

from pathlib import Path
import os
import traceback
import pandas as pd

from sqlalchemy.engine import Engine

from infra.yymm import shift_yymm
from infra.db import smoke_test
from preprocess.feature_tables import max_window_end_for_k
from preprocess.eligibility import ChurnEligibilityConfig
from preprocess.static_features import load_cus_lifetime_snapshots
from baseline.sweep import run_sweep_k
from config_store.best_config import (
    ensure_best_config_table,
    load_latest_accepted_best_config,
    upsert_best_config,
)
from main_model.runner import evaluate_existing_bundle_on_current_folds, run_main_variant
from common.artifacts import save_bundle, load_bundle

from export_risk_mode.runner import run_export_risk_mode

from monitoring.ddl import ensure_monitoring_schema
from monitoring.run_log import new_run_id, start_run, finish_run
from monitoring.score import upsert_score_drift
from monitoring.drift import compute_feature_drift, upsert_feature_drift
from logging_config import get_logger

logger = get_logger(__name__)


def _bundle_lifecycle(cfg: dict | None) -> str:
    if not cfg:
        return "PRODUCTION"
    value = cfg.get("bundle_lifecycle")
    if value is None or pd.isna(value):
        return "PRODUCTION"
    return str(value).upper()


def _supervised_as_of_from_report(report: dict | None) -> int | None:
    """Return the latest labeled/evaluated origin represented in a model report."""
    if not report:
        return None

    candidates: list[int] = []

    def add_yymm(value) -> None:
        if value is None or pd.isna(value):
            return
        try:
            candidates.append(int(value))
        except (TypeError, ValueError):
            return

    add_yymm(report.get("val_month"))

    final_holdout = report.get("final_holdout") or {}
    for month in final_holdout.get("validation_months") or []:
        add_yymm(month)

    for key in ("walk_forward_all_reports", "walk_forward_reports"):
        for fold_report in report.get(key) or []:
            add_yymm((fold_report or {}).get("val_month"))

    for rejected in report.get("rejected_folds") or report.get("walk_forward_rejected_folds") or []:
        for month in (rejected or {}).get("validation_months") or []:
            add_yymm(month)

    return max(candidates) if candidates else None


def _env_first(*names: str) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return None


def _normalize_operating_mode(value) -> str:
    raw = str(value or "percentile").strip().lower().replace("-", "_")
    if raw in {"probability", "probability_threshold", "proba", "proba_threshold"}:
        return "probability"
    if raw in {"percentile", "top_percentile", "rank", "top_tail"}:
        return "percentile"
    logger.warning("Invalid operating mode %r. Falling back to percentile.", value)
    return "percentile"


def _apply_operating_policy(cfg: dict, *, risk_threshold_pct: float) -> dict:
    """Persist the same decision policy used by production scoring into model cfg."""
    out = dict(cfg)
    mode = _normalize_operating_mode(
        _env_first("CHURN_OPERATING_MODE", "MODEL_OPERATING_MODE")
        or out.get("operating_mode")
        or "percentile"
    )
    out["operating_mode"] = mode

    risk_cutoff_raw = (
        _env_first("CHURN_OPERATING_RISK_THRESHOLD_PCT", "MODEL_OPERATING_RISK_THRESHOLD_PCT")
        or risk_threshold_pct
    )
    try:
        out["operating_risk_threshold_pct"] = float(risk_cutoff_raw)
    except (TypeError, ValueError):
        logger.warning("Invalid operating risk threshold %r. Using %.2f.", risk_cutoff_raw, risk_threshold_pct)
        out["operating_risk_threshold_pct"] = float(risk_threshold_pct)

    proba_raw = _env_first("CHURN_OPERATING_PROBABILITY_THRESHOLD", "MODEL_OPERATING_PROBABILITY_THRESHOLD")
    if proba_raw is not None:
        try:
            proba = float(proba_raw)
            if proba > 1.0 and proba <= 100.0:
                proba = proba / 100.0
            out["operating_probability_threshold"] = min(max(proba, 0.0), 1.0)
        except ValueError:
            logger.warning("Invalid operating probability threshold %r. Ignoring.", proba_raw)

    if "xgb_candidate_configs" in out:
        out["xgb_candidate_configs"] = [
            _apply_operating_policy(dict(candidate), risk_threshold_pct=float(risk_threshold_pct))
            for candidate in out.get("xgb_candidate_configs") or []
        ]
    return out


def retrain_due_reason(
    engine: Engine,
    *,
    horizon: int,
    interval_months: int = 3,
    min_freshness_age_hours: float | None = None,
) -> tuple[bool, str]:
    """Return whether a retrain run is due.

    Freshness status is audit context only. The training policy now accepts mixed
    final unified labels, so DEGRADED freshness must not block retraining.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        has_validation_table = conn.execute(
            text("SELECT to_regclass('ingest.validation_status')")
        ).scalar()
        if has_validation_table is None:
            freshness_row = None
        else:
            freshness_row = conn.execute(
                text(
                    """
                    SELECT status,
                           EXTRACT(EPOCH FROM (now() - checked_at)) / 3600.0 AS age_hours
                    FROM ingest.validation_status
                    ORDER BY checked_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).fetchone()
    freshness_status = str(freshness_row[0]).lower() if freshness_row is not None else "missing"
    freshness_age_hours = float(freshness_row[1]) if freshness_row is not None else None
    freshness_note = (
        f"freshness_{freshness_status}_{freshness_age_hours:.1f}h"
        if freshness_age_hours is not None
        else f"freshness_{freshness_status}"
    )

    try:
        cfg = load_latest_accepted_best_config(engine, horizon=int(horizon))
    except Exception:
        return True, f"first_run_no_accepted_bundle_{freshness_note}"
    if _bundle_lifecycle(cfg) == "PROVISIONAL":
        return True, f"provisional_bundle_requires_retrain_{freshness_note}"

    accepted_at = cfg.get("accepted_at")
    if accepted_at is None or pd.isna(accepted_at):
        return True, f"accepted_bundle_missing_timestamp_{freshness_note}"

    accepted_ts = pd.Timestamp(accepted_at)
    if accepted_ts.tzinfo is not None:
        accepted_ts = accepted_ts.tz_convert(None)
    now_ts = pd.Timestamp.utcnow().tz_localize(None)
    if now_ts >= accepted_ts + pd.DateOffset(months=int(interval_months)):
        return True, f"accepted_bundle_age_gte_{interval_months}_months_{freshness_note}"

    ensure_monitoring_schema(engine)
    with engine.connect() as conn:
        has_alert = bool(
            conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM ml_monitor.feature_drift
                        WHERE horizon = :horizon
                          AND severity = 'ALERT'
                          AND created_at > :accepted_at
                    )
                    """
                ),
                {"horizon": int(horizon), "accepted_at": accepted_ts.to_pydatetime()},
            ).scalar()
        )
    return (True, f"feature_drift_alert_{freshness_note}") if has_alert else (False, f"not_due_{freshness_note}")


def is_mandatory_retrain_month(window_end: int, anchor_yymm: int = 2603, interval: int = 3) -> bool:
    """Xác định tháng hiện tại có thuộc chu kỳ 3 tháng bắt buộc retrain cố định hay không."""
    try:
        def yymm_to_months(yymm: int) -> int:
            s = str(yymm).zfill(4)
            yy = int(s[:2])
            mm = int(s[2:])
            return yy * 12 + mm
        
        diff = abs(yymm_to_months(window_end) - yymm_to_months(anchor_yymm))
        return diff % interval == 0
    except Exception as e:
        logger.warning("Lỗi khi kiểm tra chu kỳ retrain bắt buộc cho month=%s: %s", window_end, e)
        return False


def get_active_count_for_month(engine: Engine, k: int, window_end: int) -> int:
    """Lấy số lượng khách hàng active trong một tháng của K xác định bằng SQL COUNT nhanh."""
    from preprocess.dataset import load_scoring_table_for_k
    try:
        # Load chỉ 1 dòng để lấy tên bảng thực tế tương ứng với window_end
        _, table_name, _ = load_scoring_table_for_k(engine, k, window_end, limit_rows=1)
        schema = "data_window"
        
        from sqlalchemy import text
        q = text(f'SELECT COUNT(*) FROM "{schema}"."{table_name}" WHERE COALESCE(item_last, 0) != 0 OR COALESCE(revenue_last, 0) != 0')
        with engine.connect() as conn:
            cnt = conn.execute(q).scalar()
            return int(cnt)
    except Exception as e:
        logger.warning("[GUARD-RAIL] Không thể đếm số lượng active_cnt cho K=%d, month=%d: %s", k, window_end, e)
        return 0



def get_activity_profile_for_month(engine: Engine, k: int, window_end: int) -> dict:
    """Fast month-completeness profile for a scoring feature table."""
    from sqlalchemy import text
    from preprocess.dataset import load_scoring_table_for_k

    try:
        _, table_name, month_used = load_scoring_table_for_k(engine, int(k), int(window_end), limit_rows=1)
        schema = "data_window"
        q = text(
            f"""
            SELECT
                COUNT(*)::bigint AS row_count,
                SUM(CASE WHEN COALESCE(item_last, 0) <> 0 OR COALESCE(revenue_last, 0) <> 0 THEN 1 ELSE 0 END)::bigint AS active_count,
                COALESCE(SUM(COALESCE(item_last, 0)), 0)::numeric AS item_sum,
                COALESCE(SUM(COALESCE(revenue_last, 0)), 0)::numeric AS revenue_sum
            FROM "{schema}"."{table_name}"
            """
        )
        with engine.connect() as conn:
            row = conn.execute(q).mappings().first()
        if row is None:
            raise ValueError(f"No aggregate row for {schema}.{table_name}")
        return {
            "window_end": int(month_used),
            "table_name": str(table_name),
            "row_count": int(row["row_count"] or 0),
            "active_count": int(row["active_count"] or 0),
            "item_sum": float(row["item_sum"] or 0.0),
            "revenue_sum": float(row["revenue_sum"] or 0.0),
        }
    except Exception as e:
        logger.warning(
            "[GUARD-RAIL] Cannot compute activity profile for K=%d, month=%d: %s",
            int(k),
            int(window_end),
            e,
        )
        return {
            "window_end": int(window_end),
            "table_name": None,
            "row_count": 0,
            "active_count": 0,
            "item_sum": 0.0,
            "revenue_sum": 0.0,
            "error": str(e),
        }


def _ratio(cur: float, prev: float) -> float:
    return float(cur) / max(float(prev), 1e-9)


def _passes_month_completeness_guard(cur: dict, prev: dict) -> tuple[bool, dict]:
    min_active_ratio = float(os.getenv("MODEL_RETRAIN_MIN_ACTIVE_RATIO", "0.50"))
    min_row_ratio = float(os.getenv("MODEL_RETRAIN_MIN_ROW_RATIO", "0.80"))
    min_item_ratio = float(os.getenv("MODEL_RETRAIN_MIN_ITEM_RATIO", "0.70"))
    min_revenue_ratio = float(os.getenv("MODEL_RETRAIN_MIN_REVENUE_RATIO", "0.70"))

    ratios = {
        "row_ratio": _ratio(cur.get("row_count", 0), prev.get("row_count", 0)),
        "active_ratio": _ratio(cur.get("active_count", 0), prev.get("active_count", 0)),
        "item_ratio": _ratio(cur.get("item_sum", 0.0), prev.get("item_sum", 0.0)),
        "revenue_ratio": _ratio(cur.get("revenue_sum", 0.0), prev.get("revenue_sum", 0.0)),
    }
    thresholds = {
        "min_row_ratio": min_row_ratio,
        "min_active_ratio": min_active_ratio,
        "min_item_ratio": min_item_ratio,
        "min_revenue_ratio": min_revenue_ratio,
    }
    failures = []
    if ratios["row_ratio"] < min_row_ratio:
        failures.append("row_ratio")
    if ratios["active_ratio"] < min_active_ratio:
        failures.append("active_ratio")
    if ratios["item_ratio"] < min_item_ratio:
        failures.append("item_ratio")
    if ratios["revenue_ratio"] < min_revenue_ratio:
        failures.append("revenue_ratio")
    meta = {
        **ratios,
        **thresholds,
        "failures": failures,
        "current_profile": cur,
        "previous_profile": prev,
    }
    return len(failures) == 0, meta


def _train_main_inline(
    engine: Engine,
    *,
    horizon: int,
    bundle_dir: Path,
    cfg_override: dict | None = None,
    candidate_configs: list[dict] | None = None,
    save_output: bool = True,
    tune_hyperparams: bool | None = None,
    optuna_trials: int | None = None,
    optuna_timeout_seconds: int | None = None,
) -> dict:
    """
    Inline training (no subprocess). Trains both use_static variants (if allowed by cfg) and saves bundle.
    Returns chosen report dict (same as scripts/train_main).
    """
    from config_store.best_config import load_latest_accepted_best_config as load_cfg
    cfg = dict(cfg_override) if cfg_override is not None else dict(load_cfg(engine, horizon=int(horizon)))
    candidates = [dict(c) for c in (candidate_configs or cfg.get("xgb_candidate_configs") or [cfg])]
    df_static = load_cus_lifetime_snapshots(engine)
    variants = []
    seen = set()
    tune_enabled = (
        str(os.getenv("MAIN_XGB_OPTUNA_ENABLED", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
        if tune_hyperparams is None
        else bool(tune_hyperparams)
    )
    for cand in candidates:
        cand = dict(cand)
        for use_static_flag in [False, True]:
            key = (int(cand["best_k"]), bool(use_static_flag))
            if key in seen:
                continue
            seen.add(key)
            logger.info(
                "[XGB CANDIDATE] Training K=%d use_static=%s from LR shortlist",
                int(cand["best_k"]),
                bool(use_static_flag),
            )
            variant = run_main_variant(engine, cand, df_static, use_static_flag=use_static_flag)
            variant["candidate_cfg"] = dict(variant.get("cfg") or cand)
            variants.append(variant)

    ok = [
        v for v in variants
        if "F1_val" in v
        and float(v.get("F1_val") or 0.0) > 0.0
        and "f1@main_thr" in (v.get("report") or {})
    ]
    if not ok:
        rejected = [v for v in variants if "F1_val" in v]
        if rejected:
            best_rejected = rejected[0]
            rejected_cfg = dict(best_rejected.get("candidate_cfg") or cfg)
            rejected_report = dict(best_rejected.get("report") or {})
            rejected_cfg["use_static"] = bool(best_rejected.get("use_static", rejected_cfg.get("use_static", False)))
            rejected_cfg["best_k"] = int(rejected_report.get("K", rejected_cfg.get("best_k")))
            rejected_as_of = _supervised_as_of_from_report(rejected_report)
            if rejected_as_of is None:
                rejected_as_of = int(max_window_end_for_k(engine, int(rejected_cfg["best_k"])))
            rejected_cfg["as_of_month"] = int(rejected_as_of)
            rejected_cfg["target_month"] = int(shift_yymm(str(rejected_cfg["as_of_month"]), int(horizon)))
            rejected_cfg["metric_f1_val"] = 0.0
            rejected_cfg["metric_pr_auc_val"] = 0.0
            rejected_cfg["metric_roc_auc_val"] = 0.0
            rejected_cfg["metric_val_prevalence"] = float(rejected_report.get("val_prevalence", 0.0) or 0.0)
            rejected_cfg["best_threshold"] = float(rejected_cfg.get("best_threshold", 0.5))
            rejected_cfg["notes"] = (
                f"{rejected_cfg.get('notes') or ''}; "
                "XGBoost candidate rejected by walk-forward guard"
            ).strip("; ")
            logger.warning(
                "[XGB SELECTED] No usable XGBoost variant; returning rejected candidate K=%d use_static=%s reason=%s",
                int(rejected_cfg["best_k"]),
                bool(rejected_cfg["use_static"]),
                rejected_report.get("guardrail_warning") or best_rejected.get("guardrail_warning") or "unknown",
            )
            return {
                "cfg": rejected_cfg,
                "main_report": rejected_report,
                "model": None,
                "metadata": {
                    "cfg": rejected_cfg,
                    "bundle_lifecycle": _bundle_lifecycle(rejected_cfg),
                    "main_report": rejected_report,
                },
            }
        raise RuntimeError("No trainable XGBoost variants produced validation metrics.")
    for variant in ok:
        if variant.get("guardrail_warning"):
            logger.warning(
                "[MAIN SANITY WARNING] K=%s use_static=%s: %s",
                variant.get("report", {}).get("K"),
                variant.get("use_static"),
                variant.get("guardrail_warning"),
            )
    if tune_enabled:
        try:
            top_n = max(int(os.getenv("MAIN_XGB_OPTUNA_TOP_N_VARIANTS", "1")), 1)
        except ValueError:
            logger.warning("Invalid MAIN_XGB_OPTUNA_TOP_N_VARIANTS=%r. Using 1.", os.getenv("MAIN_XGB_OPTUNA_TOP_N_VARIANTS"))
            top_n = 1
        ranked_for_tuning = sorted(
            ok,
            key=lambda r: (r["F1_val"], r["AP_val"], r["ROC_AUC_val"]),
            reverse=True,
        )[:top_n]
        tuned_variants = []
        for base in ranked_for_tuning:
            base_cfg = dict(base.get("candidate_cfg") or cfg)
            use_static_flag = bool(base.get("use_static"))
            logger.info(
                "[OPTUNA] Retuning top candidate K=%d use_static=%s base_F1=%.4f",
                int(base_cfg["best_k"]),
                use_static_flag,
                float(base["F1_val"]),
            )
            tuned = run_main_variant(
                engine,
                base_cfg,
                df_static,
                use_static_flag=use_static_flag,
                tune_hyperparams=True,
                optuna_trials=optuna_trials,
                optuna_timeout_seconds=optuna_timeout_seconds,
            )
            tuned["candidate_cfg"] = dict(tuned.get("cfg") or base_cfg)
            tuned["is_optuna_tuned"] = True
            tuned_variants.append(tuned)

        tuned_ok = [
            v for v in tuned_variants
            if "F1_val" in v
            and float(v.get("F1_val") or 0.0) > 0.0
            and "f1@main_thr" in (v.get("report") or {})
        ]
        ok.extend(tuned_ok)
        if tuned_ok:
            logger.info(
                "[OPTUNA] Added %d tuned candidate(s). Best tuned F1=%.4f",
                len(tuned_ok),
                max(float(v["F1_val"]) for v in tuned_ok),
            )
    ok.sort(key=lambda r: (r["F1_val"], r["AP_val"], r["ROC_AUC_val"]), reverse=True)
    best = ok[0]
    if len(ok) == 2:
        f1_gap = ok[0]["F1_val"] - ok[1]["F1_val"]
        if abs(f1_gap) <= 0.005:
            best = next((v for v in ok if v["use_static"] is False), best)

    cfg = dict(best.get("candidate_cfg") or cfg)
    cfg["use_static"] = bool(best["use_static"])
    cfg["best_k"] = int(best["report"]["K"])
    supervised_as_of = _supervised_as_of_from_report(best["report"])
    if supervised_as_of is None:
        supervised_as_of = int(max_window_end_for_k(engine, int(cfg["best_k"])))
    cfg["as_of_month"] = int(supervised_as_of)
    cfg["target_month"] = int(shift_yymm(str(cfg["as_of_month"]), int(horizon)))
    cfg["metric_f1_val"] = float(best["report"].get("f1@operating", best["report"]["f1@main_thr"]))
    cfg["metric_pr_auc_val"] = float(best["report"]["AP_val"])
    cfg["metric_roc_auc_val"] = best["report"].get("ROC_AUC_val")
    cfg["metric_val_prevalence"] = float(best["report"]["val_prevalence"])
    cfg["best_threshold"] = float(best["report"]["thr_main_opt"])
    cfg["operating_mode"] = best["report"].get("operating_mode", cfg.get("operating_mode", "percentile"))
    if best["report"].get("operating_percentile_cutoff") is not None:
        cfg["operating_risk_threshold_pct"] = float(best["report"]["operating_percentile_cutoff"])
    if best["report"].get("operating_mode") == "probability":
        cfg["operating_probability_threshold"] = float(best["report"]["operating_threshold"])
    cfg["main_threshold_min"] = float(best["report"].get("thr_main_min", cfg.get("main_threshold_min", 0.005)))
    cfg["notes"] = (
        f"{cfg.get('notes') or ''}; "
        f"XGBoost selected final K={cfg['best_k']} use_static={cfg['use_static']} "
        f"from LR shortlist"
    ).strip("; ")
    cfg.pop("xgb_candidate_configs", None)
    cfg.pop("xgb_candidate_ks", None)
    final_holdout = best["report"].get("final_holdout") or {}
    final_holdout_status = str(final_holdout.get("status") or "disabled_or_unavailable")
    final_holdout_f1 = final_holdout.get("f1")
    final_holdout_ap = final_holdout.get("ap")
    final_holdout_roc = final_holdout.get("roc_auc")
    logger.info(
        "[XGB SELECTED] K=%d use_static=%s operating_mode=%s operating_F1=%.4f "
        "operating_precision=%.4f operating_recall=%.4f AP=%.4f ROC_AUC=%s "
        "threshold=%.6f latest_operating_F1=%.4f",
        int(cfg["best_k"]),
        bool(cfg["use_static"]),
        best["report"].get("operating_mode"),
        float(best["report"].get("f1@operating", best["report"]["f1@main_thr"])),
        float(best["report"].get("precision@operating", best["report"]["precision@main_thr"])),
        float(best["report"].get("recall@operating", best["report"]["recall@main_thr"])),
        float(best["report"]["AP_val"]),
        f"{best['report'].get('ROC_AUC_val'):.4f}" if best["report"].get("ROC_AUC_val") is not None else "n/a",
        float(best["report"]["thr_main_opt"]),
        float(best["report"].get("f1@operating_latest", best["report"].get("f1@main_thr_latest", best["report"]["f1@main_thr"]))),
    )
    logger.info(
        "[XGB SELECTED DETAIL] selection_folds=%d total_folds=%d holdout_excluded_from_selection=%s "
        "latest_operating_F1=%.4f latest_AP=%.4f latest_ROC_AUC=%s operating_predicted_positive_rate=%.2f%% final_holdout_status=%s "
        "final_holdout_F1=%s final_holdout_AP=%s final_holdout_ROC_AUC=%s threshold_source=%s",
        int(best["report"].get("walk_forward_folds", 0)),
        int(best["report"].get("walk_forward_total_folds", 0)),
        bool(best["report"].get("walk_forward_holdout_excluded_from_selection", False)),
        float(best["report"].get("f1@operating_latest", best["report"].get("f1@main_thr_latest", best["report"]["f1@main_thr"]))),
        float(best["report"].get("AP_val_latest", best["report"]["AP_val"])),
        f"{best['report'].get('ROC_AUC_val_latest'):.4f}"
        if best["report"].get("ROC_AUC_val_latest") is not None else "n/a",
        100.0 * float(best["report"].get("predicted_positive_rate@operating", best["report"].get("predicted_positive_rate@main_thr", 0.0))),
        final_holdout_status,
        f"{float(final_holdout_f1):.4f}" if final_holdout_f1 is not None else "n/a",
        f"{float(final_holdout_ap):.4f}" if final_holdout_ap is not None else "n/a",
        f"{float(final_holdout_roc):.4f}" if final_holdout_roc is not None else "n/a",
        best["report"].get("threshold_source"),
    )
    meta = {
        "cfg": cfg,
        "bundle_lifecycle": _bundle_lifecycle(cfg),
        "churn_eligibility": ChurnEligibilityConfig.from_env().__dict__,
        "main_report": best["report"],
        "feat_cols": best.get("feat_cols"),
        "cat_cols": best.get("cat_cols"),
        "date_cols": best.get("date_cols", []),
        "feature_name_map": best.get("feature_name_map"),
        "feature_profile": best.get("feature_profile"),
    }
    if save_output:
        save_bundle(bundle_dir, best["model"], metadata=meta)
    return {"cfg": cfg, "main_report": best["report"], "model": best["model"], "metadata": meta}


from config.paths import CHURN_MODEL_DIR

def run_monthly_pipeline(
    engine: Engine,
    *,
    horizon: int,
    risk_threshold_pct: int = 70,
    bundle_dir: str | Path = CHURN_MODEL_DIR / "bundles/latest",
    limit_rows_each: int | None = None,
    k_min: int = 3,
    f1_improve_eps: float = 1e-6,
    do_feature_drift: bool = True,
    do_scoring: bool = True,
    force_cycle_retrain: bool = False,
    always_train_candidate: bool = False,
    tune_hyperparams: bool | None = None,
    optuna_trials: int | None = None,
    optuna_timeout_seconds: int | None = None,
) -> dict:
    """
    FULL monthly pipeline (run once):
      1) Sweep K (LR shortlist candidate)
      2) Optionally train the XGBoost candidate before comparison
      3) Compare candidate F1 vs previous accepted F1
         - accept if improved
         - else keep previous accepted config/model
      4) If accepted -> overwrite bundle
      5) Score month (export_risk_mode) + save churned_now + dossier
      6) Monitoring tables: score drift and feature drift (PSI)

    Returns a dict summary for logs.
    """
    ensure_best_config_table(engine)
    ensure_monitoring_schema(engine)

    run_id = new_run_id()

    # previous accepted config (can be None)
    prev_cfg = None
    prev_f1 = None
    prev_k = None
    prev_f1_db = 0.0
    prev_f1_bundle = 0.0
    try:
        prev_cfg = load_latest_accepted_best_config(engine, horizon=int(horizon))
        prev_f1_db = float(prev_cfg.get("metric_f1_val") or 0)
        prev_k = int(prev_cfg.get("best_k")) if prev_cfg.get("best_k") is not None else None
    except Exception:
        prev_cfg = None

    # Cross-check với bundle metadata để tránh DB bị overwrite thủ công
    try:
        _, bundle_meta = load_bundle(bundle_dir)
        bundle_cfg = (bundle_meta or {}).get("cfg", {})
        prev_f1_bundle = float(
            bundle_cfg.get("metric_f1_val") or 0
        )
    except Exception:
        prev_f1_bundle = 0.0

    prev_f1 = max(prev_f1_db, prev_f1_bundle) if (prev_f1_db > 0 or prev_f1_bundle > 0) else None
    logger.info("[PREV_F1] DB=%.4f | bundle=%.4f | using=%s",
                prev_f1_db, prev_f1_bundle,
                f"{prev_f1:.4f}" if prev_f1 is not None else "None")

    start_run(
        engine,
        run_id=run_id,
        horizon=int(horizon),
        risk_threshold_pct=int(risk_threshold_pct),
        prev_best_k=prev_k,
        prev_best_f1=prev_f1,
        notes=f"smoke_test={smoke_test(engine)}",
    )

    did_retrain = False
    did_score = False
    t_current = None
    t_scoring = None
    cand_cfg = None
    accepted = None

    try:
        # 1) Sweep K
        cand_cfg, df_ab = run_sweep_k(
            engine,
            horizon=int(horizon),
            limit_rows_each=limit_rows_each,
            k_min=int(k_min),
        )
        cand_cfg = _apply_operating_policy(cand_cfg, risk_threshold_pct=float(risk_threshold_pct))
        cand_f1 = float(cand_cfg["metric_f1_val"])
        cand_k = int(cand_cfg["best_k"])
        t_current = int(cand_cfg["as_of_month"])
        try:
            latest_scoring_month = int(max_window_end_for_k(engine, cand_k))
            logger.info(
                "[WINDOW POLICY] supervised_as_of=%s latest_scoring_window=%s K=%d "
                "(current scoring month is excluded from train/valid/test)",
                t_current,
                latest_scoring_month,
                cand_k,
            )
        except Exception:
            logger.info(
                "[WINDOW POLICY] supervised_as_of=%s K=%d "
                "(current scoring month is excluded from train/valid/test)",
                t_current,
                cand_k,
            )

        # Phát hiện lần chạy đầu tiên (chưa có bất kỳ accepted config nào trong DB)
        is_first_run = (prev_cfg is None)

        # Prevalence guard: churn_ratio quá cao → labels chưa sẵn sàng
        # Chỉ áp dụng khi đã có model cũ để fallback; lần đầu tiên thì LUÔN chạy mới.
        VAL_PREVALENCE_MAX = 0.45
        prevalence_blocked = False
        if not is_first_run and not df_ab.empty and "churn_ratio_train" in df_ab.columns:
            # Dùng churn_ratio của best K đã chọn (không phải max toàn ablation)
            # để tránh bị block bởi K lớn có ratio cao một cách tự nhiên
            best_row = df_ab[df_ab["K"] == cand_k].iloc[0] if cand_k in df_ab["K"].values else df_ab.iloc[0]
            cand_prev = float(best_row["churn_ratio_train"])
            if cand_prev > VAL_PREVALENCE_MAX:
                prevalence_blocked = True
                logger.warning(
                    "[GUARD] val_month=%d có churn_ratio=%.2f > %.2f (tại best K=%d). "
                    "Labels cho tháng này chưa sẵn sàng. HỦY RETRAIN.",
                    t_current, cand_prev, VAL_PREVALENCE_MAX, cand_k,
                )

        # 2) Decide accept
        # Retrain means "train/evaluate a candidate". Promotion still requires
        # the candidate to beat the accepted bundle, except for first-run cases.
        is_mandatory = bool(force_cycle_retrain)
        pass_guardrail = True
        active_ratio = 1.0
        active_cnt_cur = 0
        active_cnt_prev = 0
        month_guard_meta = {}
        prev_current_eval = None
        acceptance_prev_f1 = prev_f1
        t_prev = None
        candidate_train_out = None

        if always_train_candidate and not prevalence_blocked:
            logger.info(
                "[RETRAIN EVAL] Training fresh XGBoost candidate before acceptance comparison."
            )
            bundle_dir = Path(bundle_dir)
            bundle_dir.mkdir(parents=True, exist_ok=True)
            candidate_train_out = _train_main_inline(
                engine,
                horizon=int(horizon),
                bundle_dir=bundle_dir,
                cfg_override=cand_cfg,
                candidate_configs=cand_cfg.get("xgb_candidate_configs"),
                save_output=False,
                tune_hyperparams=tune_hyperparams,
                optuna_trials=optuna_trials,
                optuna_timeout_seconds=optuna_timeout_seconds,
            )
            cand_cfg = dict(candidate_train_out["cfg"])
            cand_f1 = float(cand_cfg["metric_f1_val"])
            cand_k = int(cand_cfg["best_k"])
            t_current = int(cand_cfg["as_of_month"])
            logger.info(
                "[RETRAIN EVAL] Fresh XGBoost candidate: K=%d F1=%.4f prev_F1=%s",
                cand_k,
                cand_f1,
                f"{prev_f1:.4f}" if prev_f1 is not None else "None",
            )
            if prev_cfg is not None and os.getenv("MODEL_ACCEPT_COMPARE_CURRENT_REEVAL", "1").strip().lower() in {"1", "true", "yes", "y", "on"}:
                try:
                    prev_model, prev_meta = load_bundle(bundle_dir)
                    prev_df_static = load_cus_lifetime_snapshots(engine)
                    prev_eval_cfg = _apply_operating_policy(
                        dict(prev_cfg),
                        risk_threshold_pct=float(risk_threshold_pct),
                    )
                    prev_current_eval = evaluate_existing_bundle_on_current_folds(
                        engine,
                        prev_eval_cfg,
                        prev_df_static,
                        prev_model,
                        prev_meta or {},
                    )
                    acceptance_prev_f1 = float(prev_current_eval["F1_val"])
                    logger.info(
                        "[ACCEPTANCE BASELINE] historical_prev_F1=%s current_re_eval_F1=%.4f",
                        f"{prev_f1:.4f}" if prev_f1 is not None else "None",
                        acceptance_prev_f1,
                    )
                except Exception as e:
                    logger.warning(
                        "[ACCEPTANCE BASELINE] Could not re-evaluate previous bundle on current folds; "
                        "falling back to historical prev_F1=%s: %s",
                        f"{prev_f1:.4f}" if prev_f1 is not None else "None",
                        e,
                    )

        candidate_model_ready = candidate_train_out is None or candidate_train_out.get("model") is not None

        if not candidate_model_ready:
            accepted = False
            rule = "rejected_no_trainable_xgb_candidate"
        elif is_first_run:
            # Lần đầu tiên → chạy mới hoàn toàn, không áp dụng bất kỳ guard nào
            accepted = True
            rule = "accepted_first_run"
            logger.info(
                "[FIRST RUN] Chưa có model nào trong DB. Chấp nhận ngay config mới "
                "(K=%d, F1=%.4f) và train fresh. Bỏ qua mọi guard.",
                cand_k, cand_f1,
            )
        elif prevalence_blocked:
            accepted = False
            rule = f"rejected_high_prevalence_{cand_prev:.2f}"
        elif is_mandatory:
            accepted = True
            rule = "accepted_mandatory_cycle"
            logger.info("[CYCLE] Tháng %d thuộc chu kỳ 3 tháng cố định. BẮT BUỘC RETRAIN (bỏ qua check guardrail).", t_current)
        else:
            # Kiểm tra guardrail dữ liệu (chỉ khi không mandatory và không first_run)
            from infra.yymm import shift_yymm
            try:
                t_prev = int(shift_yymm(str(t_current), -1))
                cur_profile = get_activity_profile_for_month(engine, cand_k, t_current)
                prev_profile = get_activity_profile_for_month(engine, cand_k, t_prev)
                pass_guardrail, month_guard_meta = _passes_month_completeness_guard(cur_profile, prev_profile)
                active_cnt_cur = int(cur_profile.get("active_count") or 0)
                active_cnt_prev = int(prev_profile.get("active_count") or 0)
                active_ratio = float(month_guard_meta.get("active_ratio", 0.0))
                logger.info(
                    "[GUARD-RAIL] month completeness K=%d current=%s previous=%s "
                    "row_ratio=%.3f active_ratio=%.3f item_ratio=%.3f revenue_ratio=%.3f failures=%s",
                    cand_k,
                    t_current,
                    t_prev,
                    float(month_guard_meta.get("row_ratio", 0.0)),
                    float(month_guard_meta.get("active_ratio", 0.0)),
                    float(month_guard_meta.get("item_ratio", 0.0)),
                    float(month_guard_meta.get("revenue_ratio", 0.0)),
                    ",".join(month_guard_meta.get("failures") or []) or "none",
                )
            except Exception as e:
                logger.warning("[GUARD-RAIL] Gặp lỗi khi tính toán guardrail active customers: %s. Chặn retrain để an toàn.", e)
                pass_guardrail = False

            if not pass_guardrail:
                accepted = False
                rule = "rejected_by_guardrail_incomplete_data"
                logger.warning(
                    "[GUARD] Tháng %d chưa hoàn thành dữ liệu (Active: %d vs tháng trước %s: %d, Tỷ lệ: %.2f). "
                    "HỦY RETRAIN, giữ nguyên model cũ và chỉ chạy scoring.",
                    t_current, active_cnt_cur, t_prev, active_cnt_prev, active_ratio
                )
            elif acceptance_prev_f1 is None:
                accepted = True
                rule = "accepted_missing_prev_f1"
            else:
                accepted = bool(cand_f1 > (acceptance_prev_f1 + f1_improve_eps))
                rule = "accepted_f1_improved_current_re_eval" if accepted else "rejected_f1_not_improved"

        cand_cfg["is_accepted"] = bool(accepted)
        cand_cfg["prev_accepted_f1"] = prev_f1
        cand_cfg["prev_current_re_eval_f1"] = acceptance_prev_f1
        cand_cfg["prev_current_re_eval"] = prev_current_eval
        cand_cfg["accept_rule"] = rule
        cand_cfg["accepted_at"] = pd.Timestamp.utcnow().to_pydatetime() if accepted else None

        # Store rejected candidates immediately. Accepted candidates are stored after
        # XGBoost selects the final K/use_static from the LR shortlist.
        if not accepted:
            upsert_best_config(engine, cand_cfg)

        # Choose K/month for serving (updated again after accepted XGBoost training)
        best_k_for_scoring = int(cand_k) if accepted or prev_k is None else int(prev_k)
        t_scoring = int(max_window_end_for_k(engine, best_k_for_scoring))
        logger.info(
            "[WINDOW POLICY] supervised_window=%s scoring_window=%s scoring_k=%d do_scoring=%s",
            t_current,
            t_scoring,
            best_k_for_scoring,
            bool(do_scoring),
        )

        # 3) Promote only if accepted
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        if accepted:
            if candidate_train_out is None:
                train_out = _train_main_inline(
                    engine,
                    horizon=int(horizon),
                    bundle_dir=bundle_dir,
                    cfg_override=cand_cfg,
                    candidate_configs=cand_cfg.get("xgb_candidate_configs"),
                    tune_hyperparams=tune_hyperparams,
                    optuna_trials=optuna_trials,
                    optuna_timeout_seconds=optuna_timeout_seconds,
                )
            else:
                train_out = candidate_train_out
                save_bundle(bundle_dir, train_out["model"], metadata=train_out["metadata"])
            cand_cfg = dict(train_out["cfg"])
            cand_cfg["is_accepted"] = True
            cand_cfg["prev_accepted_f1"] = prev_f1
            cand_cfg["prev_current_re_eval_f1"] = acceptance_prev_f1
            cand_cfg["prev_current_re_eval"] = prev_current_eval
            cand_cfg["accept_rule"] = rule
            cand_cfg["accepted_at"] = pd.Timestamp.utcnow().to_pydatetime()
            upsert_best_config(engine, cand_cfg)
            cand_k = int(cand_cfg["best_k"])
            t_current = int(cand_cfg["as_of_month"])
            did_retrain = True
            t_scoring = int(max_window_end_for_k(engine, cand_k))

        # 4) Monthly scoring — chỉ chạy nếu có accepted config trong DB
        # (trường hợp bị block ngay lần đầu tiên: chưa có model nào được accepted)
        has_accepted_in_db = prev_cfg is not None or accepted
        if not do_scoring:
            logger.info("[SKIP SCORING] Retrain DAG is isolated from business scoring.")
            res = {"status": "skipped_retrain_only", "active_cnt": 0, "risk_cnt": 0, "churned_now_cnt": 0}
        elif not has_accepted_in_db:
            logger.warning(
                "[SKIP SCORING] Không có accepted best_config nào trong DB và tháng này bị block "
                "(prevalence_blocked=%s, accepted=%s). "
                "Bỏ qua bước scoring. Pipeline kết thúc sớm.",
                prevalence_blocked, accepted,
            )
            res = {"status": "skipped_no_accepted_config", "active_cnt": 0, "risk_cnt": 0, "churned_now_cnt": 0}
        else:
            res = run_export_risk_mode(
                engine,
                horizon=int(horizon),
                bundle_dir=bundle_dir,
                risk_threshold=float(risk_threshold_pct),
                t_current=int(t_scoring),
                limit_rows=None,
                make_dossier=True,
            )
            did_score = True

        # 5) Monitoring — chỉ chạy khi scoring thực sự được thực hiện
        import numpy as np
        if did_score:
            # Score drift
            score_stats = res.get("score_stats") or {}
            active_cnt = int(res.get("active_cnt") or 0)
            churned_now_cnt = int(res.get("churned_now_cnt") or 0)
            risk_cnt = int(res.get("risk_cnt") or 0)
            upsert_score_drift(
                engine,
                window_end=int(t_scoring),
                horizon=int(horizon),
                best_k=int(load_latest_accepted_best_config(engine, horizon=int(horizon)).get("best_k", cand_k)),
                active_cnt=active_cnt,
                churned_now_cnt=churned_now_cnt,
                scores=np.array([], dtype=float),
                risk_threshold_pct=int(risk_threshold_pct),
                risk_cnt=risk_cnt,
            )
            # overwrite stored quantiles with export runner's (more accurate)
            if score_stats:
                from sqlalchemy import text
                q = text(f"""
                    UPDATE ml_monitor.score_drift
                    SET mean_score=:m, p50=:p50, p90=:p90, p99=:p99
                    WHERE window_end=:w AND horizon=:h
                """)
                with engine.begin() as conn:
                    conn.execute(q, {
                        "m": score_stats.get("mean"),
                        "p50": score_stats.get("p50"),
                        "p90": score_stats.get("p90"),
                        "p99": score_stats.get("p99"),
                        "w": int(t_scoring),
                        "h": int(horizon),
                    })

            # Feature drift (PSI) if baseline profile exists in bundle
            if do_feature_drift:
                try:
                    _, meta = load_bundle(bundle_dir)
                    prof = (meta or {}).get("feature_profile")
                    if prof:
                        best_k_used = int(load_latest_accepted_best_config(engine, horizon=int(horizon)).get("best_k", cand_k))
                        from preprocess.dataset import load_scoring_table_for_k
                        df_cur, _, _ = load_scoring_table_for_k(engine, k=best_k_used, window_end=int(t_scoring))
                        drift_df = compute_feature_drift(df_cur, prof)
                        upsert_feature_drift(engine, window_end=int(t_scoring), horizon=int(horizon), best_k=best_k_used, drift_df=drift_df)
                except Exception:
                    # do not fail whole pipeline
                    pass

        guardrail_meta = {
            "is_mandatory_cycle": bool(is_mandatory),
            "pass_guardrail": bool(pass_guardrail),
            "active_ratio": round(float(active_ratio), 4),
            "active_cnt_cur": int(active_cnt_cur),
            "active_cnt_prev": int(active_cnt_prev),
            "month_completeness": month_guard_meta,
        }
        finish_run(
            engine,
            run_id=run_id,
            status="SUCCESS",
            window_end=int(t_scoring if did_score and t_scoring is not None else t_current),
            cand_best_k=int(cand_k),
            cand_best_f1=float(cand_f1),
            cand_is_accepted=bool(accepted),
            did_retrain=bool(did_retrain),
            did_score=bool(did_score),
            notes=(
                f"accept_rule={rule}; "
                f"F1={cand_f1:.4f}; "
                f"PR_AUC={float(cand_cfg.get('metric_pr_auc_val') or 0.0):.4f}; "
                f"mandatory={is_mandatory}; "
                f"guardrail={'pass' if pass_guardrail else 'blocked'}; "
                f"active_ratio={active_ratio:.2f}"
            ),
        )

        return {
            "run_id": run_id,
            "window_end": int(t_scoring if did_score and t_scoring is not None else t_current),
            "supervised_window_end": int(t_current),
            "scoring_window_end": int(t_scoring) if t_scoring is not None else None,
            "candidate": cand_cfg,
            "accepted": bool(accepted),
            "did_retrain": bool(did_retrain),
            "guardrail": guardrail_meta,
            "export": res,
        }

    except Exception as e:
        finish_run(
            engine,
            run_id=run_id,
            status="FAILED",
            window_end=int(t_current) if t_current is not None else None,
            cand_best_k=int(cand_cfg["best_k"]) if cand_cfg else None,
            cand_best_f1=float(cand_cfg["metric_f1_val"]) if cand_cfg else None,
            cand_is_accepted=bool(accepted) if accepted is not None else None,
            did_retrain=bool(did_retrain),
            did_score=bool(did_score),
            notes=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )
        raise
