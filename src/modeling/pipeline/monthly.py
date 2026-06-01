
from __future__ import annotations

from pathlib import Path
import traceback
import pandas as pd

from sqlalchemy.engine import Engine

from infra.yymm import shift_yymm
from infra.db import smoke_test
from preprocess.feature_tables import max_window_end_for_k
from preprocess.static_features import load_cus_lifetime_snapshots
from baseline.sweep import run_sweep_k
from config_store.best_config import (
    ensure_best_config_table,
    load_latest_accepted_best_config,
    load_previous_accepted_best_config,
    upsert_best_config,
)
from scripts.train_main import main as _train_main_cli  # fallback
from main_model.runner import run_main_variant
from common.artifacts import save_bundle, load_bundle

from export_risk_mode.runner import run_export_risk_mode

from monitoring.ddl import ensure_monitoring_schema
from monitoring.run_log import new_run_id, start_run, finish_run
from monitoring.score import upsert_score_drift
from monitoring.drift import compute_feature_drift, upsert_feature_drift
from monitoring.backtest import run_backtest_precision_in_list
from logging_config import get_logger

logger = get_logger(__name__)


def retrain_due_reason(engine: Engine, *, horizon: int, interval_months: int = 3) -> tuple[bool, str]:
    """Return whether a retrain run may start and the audit reason."""
    from sqlalchemy import text

    with engine.connect() as conn:
        has_validation_table = conn.execute(
            text("SELECT to_regclass('ingest.validation_status')")
        ).scalar()
        if has_validation_table is None:
            return False, "blocked_missing_freshness_status"
        freshness_status = conn.execute(
            text(
                """
                SELECT status
                FROM ingest.validation_status
                ORDER BY checked_at DESC, id DESC
                LIMIT 1
                """
            )
        ).scalar()
    if freshness_status != "PASS":
        return False, f"blocked_freshness_{str(freshness_status).lower()}"

    try:
        cfg = load_latest_accepted_best_config(engine, horizon=int(horizon))
    except Exception:
        return True, "first_run_no_accepted_bundle"

    accepted_at = cfg.get("accepted_at")
    if accepted_at is None or pd.isna(accepted_at):
        return True, "accepted_bundle_missing_timestamp"

    accepted_ts = pd.Timestamp(accepted_at)
    if accepted_ts.tzinfo is not None:
        accepted_ts = accepted_ts.tz_convert(None)
    now_ts = pd.Timestamp.utcnow().tz_localize(None)
    if now_ts >= accepted_ts + pd.DateOffset(months=int(interval_months)):
        return True, f"accepted_bundle_age_gte_{interval_months}_months"

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
    return (True, "feature_drift_alert") if has_alert else (False, "not_due")


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



def _train_main_inline(engine: Engine, *, horizon: int, bundle_dir: Path) -> dict:
    """
    Inline training (no subprocess). Trains both use_static variants (if allowed by cfg) and saves bundle.
    Returns chosen report dict (same as scripts/train_main).
    """
    from config_store.best_config import load_latest_accepted_best_config as load_cfg
    cfg = load_cfg(engine, horizon=int(horizon))
    cfg = dict(cfg)
    df_static = load_cus_lifetime_snapshots(engine)
    variants = [run_main_variant(engine, cfg, df_static, use_static_flag=False),
                run_main_variant(engine, cfg, df_static, use_static_flag=True)]
    ok = [v for v in variants if not v.get("guardrail_warning")]
    if not ok:
        raise RuntimeError("All variants failed guardrail. Stop training.")
    ok.sort(key=lambda r: (r["F1_val"], r["AP_val"]), reverse=True)
    best = ok[0]
    if len(ok) == 2:
        f1_gap = ok[0]["F1_val"] - ok[1]["F1_val"]
        if abs(f1_gap) <= 0.002:
            best = next((v for v in ok if v["use_static"] is False), best)

    cfg["use_static"] = bool(best["use_static"])
    meta = {
        "cfg": cfg,
        "main_report": best["report"],
        "feat_cols": best.get("feat_cols"),
        "cat_cols": best.get("cat_cols"),
        "date_cols": best.get("date_cols", []),
        "feature_name_map": best.get("feature_name_map"),
        "feature_profile": best.get("feature_profile"),
    }
    save_bundle(bundle_dir, best["model"], metadata=meta)
    return {"cfg": cfg, "main_report": best["report"]}


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
    do_backtest: bool = True,
    do_feature_drift: bool = True,
    do_scoring: bool = True,
    force_cycle_retrain: bool = False,
) -> dict:
    """
    FULL monthly pipeline (run once):
      1) Sweep K (candidate)
      2) Compare candidate F1 vs previous accepted F1
         - accept if improved
         - else keep previous accepted config/model
      3) If accepted -> retrain main model + overwrite bundle
      4) Score month (export_risk_mode) + save churned_now + dossier
      5) Monitoring tables: score drift, feature drift (PSI), backtest

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
        prev_f1_bundle = float(
            (bundle_meta or {}).get("cfg", {}).get("metric_f1_val") or 0
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
        cand_f1 = float(cand_cfg["metric_f1_val"])
        cand_k = int(cand_cfg["best_k"])
        t_current = int(cand_cfg["as_of_month"])

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
        is_mandatory = force_cycle_retrain or is_mandatory_retrain_month(t_current)
        pass_guardrail = True
        active_ratio = 1.0
        active_cnt_cur = 0
        active_cnt_prev = 0
        t_prev = None

        if is_first_run:
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
                active_cnt_cur = get_active_count_for_month(engine, cand_k, t_current)
                active_cnt_prev = get_active_count_for_month(engine, cand_k, t_prev)
                if active_cnt_prev > 0:
                    active_ratio = active_cnt_cur / active_cnt_prev
                    if active_ratio < 0.80:
                        pass_guardrail = False
                else:
                    logger.warning("[GUARD-RAIL] Không có active customers ở tháng trước %s. Tự động kích hoạt guardrail.", t_prev)
                    pass_guardrail = False
            except Exception as e:
                logger.warning("[GUARD-RAIL] Gặp lỗi khi tính toán guardrail active customers: %s. Chặn retrain để an toàn.", e)
                pass_guardrail = False

            if not pass_guardrail:
                accepted = False
                rule = "rejected_by_guardrail_incomplete_data"
                logger.warning(
                    "[GUARD] Tháng %d chưa hoàn thành dữ liệu (Active: %d vs tháng trước %s: %d, Tỷ lệ: %.2f < 0.80). "
                    "HỦY RETRAIN, giữ nguyên model cũ và chỉ chạy scoring.",
                    t_current, active_cnt_cur, t_prev, active_cnt_prev, active_ratio
                )
            elif prev_f1 is None:
                accepted = True
                rule = "accepted_no_prev"
            else:
                accepted = bool(cand_f1 > (prev_f1 + f1_improve_eps))
                rule = "accepted_f1_improved" if accepted else "rejected_f1_not_improved"

        cand_cfg["is_accepted"] = bool(accepted)
        cand_cfg["prev_accepted_f1"] = prev_f1
        cand_cfg["accept_rule"] = rule
        cand_cfg["accepted_at"] = pd.Timestamp.utcnow().to_pydatetime()

        # Store candidate config (accepted or rejected)
        upsert_best_config(engine, cand_cfg)

        # Choose K/month for serving (if rejected -> keep previous accepted K)
        best_k_for_scoring = int(cand_k) if accepted or prev_k is None else int(prev_k)
        t_current = int(max_window_end_for_k(engine, best_k_for_scoring))

        # 3) Retrain only if accepted
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        if accepted:
            try:
                _train_main_inline(engine, horizon=int(horizon), bundle_dir=bundle_dir)
                did_retrain = True
            except RuntimeError as e:
                if "All variants failed guardrail" not in str(e):
                    raise

                accepted = False
                rule = "rejected_main_guardrail_all_variants_failed"
                cand_cfg["is_accepted"] = False
                cand_cfg["accept_rule"] = rule
                cand_cfg["accepted_at"] = None
                cand_cfg["notes"] = (
                    f"{cand_cfg.get('notes') or ''}; main_train_guardrail={str(e)}"
                ).strip("; ")
                upsert_best_config(engine, cand_cfg)
                logger.warning(
                    "[GUARD] Main model retrain failed guardrail for K=%d, month=%d. "
                    "Reject candidate and keep previous accepted model/config if available. Reason: %s",
                    cand_k,
                    t_current,
                    e,
                )

        # 4) Monthly scoring — chỉ chạy nếu có accepted config trong DB
        # (trường hợp bị block ngay lần đầu tiên: chưa có model nào được accepted)
        has_accepted_in_db = prev_cfg is not None or accepted
        if not do_scoring:
            logger.info("[SKIP SCORING] Retrain DAG is isolated from business scoring.")
            res = {"status": "skipped_retrain_only", "active_cnt": 0, "risk_cnt": 0, "churned_now_cnt": 0}
            bt = None
        elif not has_accepted_in_db:
            logger.warning(
                "[SKIP SCORING] Không có accepted best_config nào trong DB và tháng này bị block "
                "(prevalence_blocked=%s, accepted=%s). "
                "Bỏ qua bước scoring. Pipeline kết thúc sớm.",
                prevalence_blocked, accepted,
            )
            res = {"status": "skipped_no_accepted_config", "active_cnt": 0, "risk_cnt": 0, "churned_now_cnt": 0}
            bt = None
        else:
            res = run_export_risk_mode(
                engine,
                horizon=int(horizon),
                bundle_dir=bundle_dir,
                risk_threshold=float(risk_threshold_pct),
                t_current=int(t_current),
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
            payload = upsert_score_drift(
                engine,
                window_end=int(t_current),
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
                        "w": int(t_current),
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
                        df_cur, _, _ = load_scoring_table_for_k(engine, k=best_k_used, window_end=int(t_current))
                        drift_df = compute_feature_drift(df_cur, prof)
                        upsert_feature_drift(engine, window_end=int(t_current), horizon=int(horizon), best_k=best_k_used, drift_df=drift_df)
                except Exception:
                    # do not fail whole pipeline
                    pass

            # Backtest (precision-in-list) when label month exists (t_current acts as label month)
            bt = None
            if do_backtest:
                bt = run_backtest_precision_in_list(
                    engine,
                    label_window_end=int(t_current),
                    horizon=int(horizon),
                    risk_threshold_pct=int(risk_threshold_pct),
                    best_k_for_population=int(k_min),
                )
        else:
            # Scoring bị skip → không có monitoring data
            bt = None

        guardrail_meta = {
            "is_mandatory_cycle": bool(is_mandatory),
            "pass_guardrail": bool(pass_guardrail),
            "active_ratio": round(float(active_ratio), 4),
            "active_cnt_cur": int(active_cnt_cur),
            "active_cnt_prev": int(active_cnt_prev),
        }
        finish_run(
            engine,
            run_id=run_id,
            status="SUCCESS",
            window_end=int(t_current),
            cand_best_k=int(cand_k),
            cand_best_f1=float(cand_f1),
            cand_is_accepted=bool(accepted),
            did_retrain=bool(did_retrain),
            did_score=bool(did_score),
            notes=(
                f"accept_rule={rule}; "
                f"mandatory={is_mandatory}; "
                f"guardrail={'pass' if pass_guardrail else 'blocked'}; "
                f"active_ratio={active_ratio:.2f}; "
                f"backtest={'yes' if bt else 'no'}"
            ),
        )

        return {
            "run_id": run_id,
            "window_end": int(t_current),
            "candidate": cand_cfg,
            "accepted": bool(accepted),
            "did_retrain": bool(did_retrain),
            "guardrail": guardrail_meta,
            "export": res,
            "backtest": bt,
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
