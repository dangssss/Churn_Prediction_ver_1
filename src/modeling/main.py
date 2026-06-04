from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

# Ensure `infra`, `baseline`, ... can be imported when running from repo root
# Ensure `infra`, `baseline`, ... can be imported when running from repo root
# Ensure `infra`, `baseline`, ... can be imported when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Add project root (d:\ds_churn) for absolute imports like `from modeling.config...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra.db import get_engine, smoke_test
from baseline.sweep import run_sweep_k
from config_store.best_config import ensure_best_config_table, upsert_best_config
from pipeline.monthly import retrain_due_reason, run_monthly_pipeline
from export_risk_mode.runner import run_export_risk_mode
from config_store.best_config import load_latest_accepted_best_config
from config.paths import CHURN_MODEL_DIR
from preprocess.static_features import load_cus_lifetime_snapshots
from main_model.runner import run_main_variant
from common.artifacts import save_bundle
from logging_config import get_logger

logger = get_logger(__name__)


def _bundle_is_ready(bundle_dir: str | Path) -> bool:
    bundle_path = Path(bundle_dir)
    return (bundle_path / "model.joblib").is_file()


def _latest_freshness_status(engine) -> str | None:
    from sqlalchemy import text

    with engine.connect() as conn:
        has_table = conn.execute(text("SELECT to_regclass('ingest.validation_status')")).scalar()
        if has_table is None:
            return None
        return conn.execute(
            text(
                """
                SELECT status
                FROM ingest.validation_status
                ORDER BY checked_at DESC, id DESC
                LIMIT 1
                """
            )
        ).scalar()


def cmd_run_monthly(args) -> None:
    engine = get_engine()
    logger.info("DB: %s", smoke_test(engine))
    out = run_monthly_pipeline(
        engine,
        horizon=int(args.horizon),
        risk_threshold_pct=int(args.risk_threshold_pct),
        bundle_dir=args.bundle_dir,
        limit_rows_each=args.limit_rows_each,
        k_min=int(args.k_min),
    )
    logger.info("DONE run-monthly: %s", out)


def cmd_retrain_if_due(args) -> None:
    engine = get_engine()
    logger.info("DB: %s", smoke_test(engine))
    due, reason = retrain_due_reason(
        engine,
        horizon=int(args.horizon),
        interval_months=int(args.interval_months),
    )
    if not due:
        logger.info("SKIP retrain: %s", reason)
        return

    logger.info("START retrain: %s", reason)
    out = run_monthly_pipeline(
        engine,
        horizon=int(args.horizon),
        bundle_dir=args.bundle_dir,
        limit_rows_each=args.limit_rows_each,
        k_min=int(args.k_min),
        do_scoring=False,
        force_cycle_retrain=reason.startswith("accepted_bundle_age_gte_"),
    )
    logger.info("DONE retrain-if-due: %s", out)


def cmd_prepare_scoring(args) -> None:
    """Bootstrap/retrain when needed so the scoring DAG can run with a ready bundle."""
    engine = get_engine()
    logger.info("DB: %s", smoke_test(engine))
    bundle_dir = Path(args.bundle_dir)
    bundle_ready = _bundle_is_ready(bundle_dir)
    freshness_status = _latest_freshness_status(engine)

    if not bundle_ready:
        if freshness_status not in {"PASS", "DEGRADED"}:
            raise RuntimeError(
                "Cannot bootstrap model without a ready bundle: "
                f"freshness_status={freshness_status!r}"
            )
        logger.warning(
            "[BOOTSTRAP] No model bundle found. Starting first sweep/train with freshness=%s. "
            "DEGRADED bootstrap may use weighted rule-based labels.",
            freshness_status,
        )
        run_monthly_pipeline(
            engine,
            horizon=int(args.horizon),
            risk_threshold_pct=int(args.risk_threshold_pct),
            bundle_dir=bundle_dir,
            limit_rows_each=args.limit_rows_each,
            k_min=int(args.k_min),
            do_scoring=False,
            force_cycle_retrain=True,
        )
        if not _bundle_is_ready(bundle_dir):
            raise RuntimeError("Bootstrap completed without producing a ready model bundle")
    else:
        if bool(args.force_retrain):
            if freshness_status not in {"PASS", "DEGRADED"}:
                raise RuntimeError(
                    "Cannot force retrain with unsafe freshness status: "
                    f"freshness_status={freshness_status!r}"
                )
            due = True
            reason = f"manual_force_retrain_freshness_{str(freshness_status).lower()}"
            logger.warning(
                "[POST FEATURE] Force retrain requested. Bypass due/freshness gate: %s",
                reason,
            )
        else:
            due, reason = retrain_due_reason(
                engine,
                horizon=int(args.horizon),
                interval_months=int(args.interval_months),
                min_freshness_age_hours=0.0,
            )
        if due:
            logger.info("[POST FEATURE] Retrain before scoring: %s", reason)
            if bool(args.force_retrain):
                run_monthly_pipeline(
                    engine,
                    horizon=int(args.horizon),
                    risk_threshold_pct=int(args.risk_threshold_pct),
                    bundle_dir=bundle_dir,
                    limit_rows_each=args.limit_rows_each,
                    k_min=int(args.k_min),
                    do_scoring=False,
                    force_cycle_retrain=True,
                    strict_main_guardrail=True,
                )
            else:
                try:
                    run_monthly_pipeline(
                        engine,
                        horizon=int(args.horizon),
                        risk_threshold_pct=int(args.risk_threshold_pct),
                        bundle_dir=bundle_dir,
                        limit_rows_each=args.limit_rows_each,
                        k_min=int(args.k_min),
                        do_scoring=False,
                        force_cycle_retrain=reason.startswith("accepted_bundle_age_gte_"),
                    )
                except Exception:
                    logger.exception(
                        "[POST FEATURE] Retrain failed. Continue scoring with the last ready bundle."
                )
        else:
            logger.info("[POST FEATURE] Retrain skipped: %s", reason)

    logger.info("DONE prepare-scoring: ready_bundle=%s", bundle_dir)


def cmd_sweep_k(args) -> None:
    engine = get_engine()
    logger.info("DB: %s", smoke_test(engine))
    ensure_best_config_table(engine)

    best_cfg, df_ab = run_sweep_k(
        engine,
        horizon=int(args.horizon),
        limit_rows_each=args.limit_rows_each,
        k_min=int(args.k_min),
    )

    # Log only — KHÔNG ghi DB để tránh ghi đè config production
    logger.info("Sweep result (NOT saved to DB): %s", best_cfg)
    cols = [
        c for c in [
            "K", "use_static", "val_month", "ranking_top_n",
            "hits_at_n", "precision_at_n", "recall_at_n", "lift_at_n",
            "rule_hits_at_n", "rule_precision_at_n", "rule_recall_at_n", "rule_lift_at_n",
            "combined_weighted_hits_at_n", "combined_weighted_precision_at_n",
            "combined_weighted_recall_at_n", "combined_weighted_lift_at_n",
            "f1", "PR_AUC_val", "best_threshold", "spw_used",
        ] if c in df_ab.columns
    ]
    logger.info("TOP-10:\n%s", df_ab[cols].head(10).to_string(index=False))


def cmd_train_main(args) -> None:
    engine = get_engine()
    logger.info("DB: %s", smoke_test(engine))
    cfg = load_latest_accepted_best_config(engine, horizon=int(args.horizon))
    df_static = load_cus_lifetime_snapshots(engine)

    variants = [
        run_main_variant(engine, cfg, df_static, use_static_flag=False),
        run_main_variant(engine, cfg, df_static, use_static_flag=True),
    ]
    ok = [v for v in variants if not v.get("guardrail_warning")]
    if not ok:
        raise RuntimeError("All variants failed guardrail. Stop training.")
    ok.sort(key=lambda r: (r["Lift_at_n"], r["AP_val"], r["F1_val"]), reverse=True)
    best = ok[0]

    cfg = dict(cfg)
    cfg["use_static"] = bool(best["use_static"])

    bundle_dir = Path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "cfg": cfg,
        "bundle_lifecycle": str(cfg.get("bundle_lifecycle") or "PRODUCTION").upper(),
        "validation_label_source": cfg.get("validation_label_source"),
        "main_report": best["report"],
        "feat_cols": best.get("feat_cols"),
        "cat_cols": best.get("cat_cols"),
        "date_cols": best.get("date_cols", []),
        "feature_name_map": best.get("feature_name_map"),
        "feature_profile": best.get("feature_profile"),
    }
    save_bundle(bundle_dir, best["model"], metadata=meta)
    logger.info("Saved bundle to: %s", bundle_dir)


def cmd_export_risk(args) -> None:
    engine = get_engine()
    logger.info("DB: %s", smoke_test(engine))
    res = run_export_risk_mode(
        engine,
        horizon=int(args.horizon),
        bundle_dir=Path(args.bundle_dir),
        risk_threshold=float(args.risk_threshold_pct),
        t_current=int(args.t_current) if args.t_current else None,
        limit_rows=args.limit_rows,
        make_dossier=bool(args.make_dossier),
    )
    logger.info("DONE export-risk: %s", res)


def cmd_backtest_actual(args) -> None:
    from monitoring.backtest import run_backtests_for_available_actual_labels

    engine = get_engine()
    logger.info("DB: %s", smoke_test(engine))
    results = run_backtests_for_available_actual_labels(
        engine,
        horizon=int(args.horizon),
        risk_threshold_pct=int(args.risk_threshold_pct),
    )
    logger.info("DONE backtest-actual: evaluated_origins=%d", len(results))
    for result in results:
        logger.info(
            "[BACKTEST] origin=%s labels=%s precision=%s recall=%s lift=%s status=%s action=%s",
            result["pred_window_end"],
            result["label_tables"],
            result["precision_in_list"],
            result["recall_in_list"],
            result["lift_vs_random"],
            result["guardrail_status"],
            result["recommended_action"],
        )



def load_config_defaults() -> dict:
    """Load defaults from config/job_config.json if exists."""
    defaults = {"horizon": None, "risk_threshold_pct": 70}
    try:
        cfg_path = Path(__file__).parent / "config/job_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                defaults.update(cfg)
    except Exception as e:
        logger.warning("Failed to load config defaults: %s", e)
    return defaults


def build_parser() -> argparse.ArgumentParser:
    defaults = load_config_defaults()
    
    ap = argparse.ArgumentParser(description="ChurnPrediction CLI (single entrypoint)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run-monthly", help="Run full pipeline once (sweepK -> conditional retrain -> monthly score -> monitoring)")
    # Horizon is now optional if defined in config
    p.add_argument("--horizon", type=int, default=defaults.get("horizon"), help=f"Default: {defaults.get('horizon')}")
    p.add_argument("--risk-threshold-pct", type=int, default=defaults.get("risk_threshold_pct"))
    p.add_argument("--bundle-dir", type=str, default=str(CHURN_MODEL_DIR / "bundles/latest"))
    p.add_argument("--limit-rows-each", type=int, default=None)
    p.add_argument("--k-min", type=int, default=3)
    p.set_defaults(func=cmd_run_monthly)

    p = sub.add_parser("retrain-if-due", help="Retrain-only gate: freshness PASS and interval/drift trigger")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--interval-months", type=int, default=3)
    p.add_argument("--bundle-dir", type=str, default=str(CHURN_MODEL_DIR / "bundles/latest"))
    p.add_argument("--limit-rows-each", type=int, default=None)
    p.add_argument("--k-min", type=int, default=3)
    p.set_defaults(func=cmd_retrain_if_due)

    p = sub.add_parser("prepare-scoring", help="Bootstrap/retrain when needed before triggering the scoring DAG")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--interval-months", type=int, default=3)
    p.add_argument("--risk-threshold-pct", type=int, default=70)
    p.add_argument("--bundle-dir", type=str, default=str(CHURN_MODEL_DIR / "bundles/latest"))
    p.add_argument("--limit-rows-each", type=int, default=None)
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument(
        "--force-retrain",
        action="store_true",
        help="Manual override: retrain before scoring even when due/freshness gate would skip. Allows PASS/DEGRADED only.",
    )
    p.set_defaults(func=cmd_prepare_scoring)

    p = sub.add_parser("sweep-k", help="Sweep K (debug). Saves best_config (accepted=True).")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--limit-rows-each", type=int, default=None)
    p.add_argument("--k-min", type=int, default=3)
    p.set_defaults(func=cmd_sweep_k)

    p = sub.add_parser("train-main", help="Train main model using latest accepted best_config, save bundle")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--bundle-dir", type=str, default=str(CHURN_MODEL_DIR / "bundles/latest"))
    p.set_defaults(func=cmd_train_main)

    p = sub.add_parser("export-risk", help="Monthly scoring using current bundle + accepted best_config")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--bundle-dir", type=str, default=str(CHURN_MODEL_DIR / "bundles/latest"))
    p.add_argument("--risk-threshold-pct", type=int, default=70)
    p.add_argument("--t-current", type=int, default=None, help="YYYYMM; nếu bỏ trống sẽ tự lấy tháng mới nhất của best_k")
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--make-dossier", action="store_true")
    p.set_defaults(func=cmd_export_risk)

    p = sub.add_parser("backtest-actual", help="Evaluate stored scoring origins with complete actual-label coverage")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--risk-threshold-pct", type=int, default=70)
    p.set_defaults(func=cmd_backtest_actual)

    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
