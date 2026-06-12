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
from preprocess.eligibility import ChurnEligibilityConfig
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
        tune_hyperparams=args.tune_hyperparams,
        optuna_trials=args.optuna_trials,
        optuna_timeout_seconds=args.optuna_timeout_seconds,
    )
    logger.info("DONE run-monthly: %s", out)


def cmd_retrain_if_due(args) -> None:
    engine = get_engine()
    logger.info("DB: %s", smoke_test(engine))
    if bool(args.force_evaluate):
        due = True
        reason = "manual_or_scheduled_force_retrain_evaluation"
    else:
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
        force_cycle_retrain=False,
        always_train_candidate=True,
        tune_hyperparams=args.tune_hyperparams,
        optuna_trials=args.optuna_trials,
        optuna_timeout_seconds=args.optuna_timeout_seconds,
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
        logger.warning(
            "[BOOTSTRAP] No model bundle found. Starting first sweep/train with freshness=%s. "
            "Mixed actual/rule-based labels are allowed by training policy.",
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
            tune_hyperparams=args.tune_hyperparams,
            optuna_trials=args.optuna_trials,
            optuna_timeout_seconds=args.optuna_timeout_seconds,
        )
        if not _bundle_is_ready(bundle_dir):
            raise RuntimeError("Bootstrap completed without producing a ready model bundle")
    else:
        if bool(args.force_retrain):
            due = True
            reason = f"manual_force_retrain_freshness_{str(freshness_status).lower()}"
            logger.warning(
                "[POST FEATURE] Force retrain requested. Bypass due/freshness gate: %s",
                reason,
            )
        elif bool(args.skip_due_retrain):
            due = False
            reason = f"skip_due_retrain_after_features_freshness_{str(freshness_status).lower()}"
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
                    tune_hyperparams=args.tune_hyperparams,
                    optuna_trials=args.optuna_trials,
                    optuna_timeout_seconds=args.optuna_timeout_seconds,
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
                        tune_hyperparams=args.tune_hyperparams,
                        optuna_trials=args.optuna_trials,
                        optuna_timeout_seconds=args.optuna_timeout_seconds,
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
            "K", "use_static", "val_month",
            "f1", "precision", "recall", "PR_AUC_val", "ROC_AUC_val",
            "val_prevalence", "best_threshold", "spw_used",
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
    ok = [v for v in variants if "F1_val" in v]
    if not ok:
        raise RuntimeError("No trainable XGBoost variants produced validation metrics.")
    for variant in ok:
        if variant.get("guardrail_warning"):
            logger.warning(
                "[MAIN SANITY WARNING] use_static=%s: %s",
                variant.get("use_static"),
                variant.get("guardrail_warning"),
            )
    ok.sort(key=lambda r: (r["F1_val"], r["AP_val"], r["ROC_AUC_val"]), reverse=True)
    best = ok[0]

    cfg = dict(cfg)
    cfg["use_static"] = bool(best["use_static"])

    bundle_dir = Path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "cfg": cfg,
        "bundle_lifecycle": str(cfg.get("bundle_lifecycle") or "PRODUCTION").upper(),
        "validation_label_source": cfg.get("validation_label_source"),
        "churn_eligibility": ChurnEligibilityConfig.from_env().__dict__,
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


def add_tuning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tune-hyperparams",
        dest="tune_hyperparams",
        action="store_true",
        default=None,
        help="Enable Optuna tuning for the top XGBoost retrain candidate(s).",
    )
    parser.add_argument(
        "--no-tune-hyperparams",
        dest="tune_hyperparams",
        action="store_false",
        help="Disable Optuna tuning even if MAIN_XGB_OPTUNA_ENABLED is set.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=None,
        help="Override MAIN_XGB_OPTUNA_TRIALS for this run.",
    )
    parser.add_argument(
        "--optuna-timeout-seconds",
        type=int,
        default=None,
        help="Override MAIN_XGB_OPTUNA_TIMEOUT_SECONDS for this run.",
    )


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
    add_tuning_args(p)
    p.set_defaults(func=cmd_run_monthly)

    p = sub.add_parser("retrain-if-due", help="Retrain-only gate: interval/drift trigger; freshness is audit-only")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--interval-months", type=int, default=3)
    p.add_argument("--bundle-dir", type=str, default=str(CHURN_MODEL_DIR / "bundles/latest"))
    p.add_argument("--limit-rows-each", type=int, default=None)
    p.add_argument("--k-min", type=int, default=3)
    add_tuning_args(p)
    p.add_argument(
        "--force-evaluate",
        action="store_true",
        help="Always train a fresh XGBoost candidate and compare it with the accepted bundle.",
    )
    p.set_defaults(func=cmd_retrain_if_due)

    p = sub.add_parser("prepare-scoring", help="Bootstrap/retrain when needed before triggering the scoring DAG")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--interval-months", type=int, default=3)
    p.add_argument("--risk-threshold-pct", type=int, default=70)
    p.add_argument("--bundle-dir", type=str, default=str(CHURN_MODEL_DIR / "bundles/latest"))
    p.add_argument("--limit-rows-each", type=int, default=None)
    p.add_argument("--k-min", type=int, default=3)
    add_tuning_args(p)
    p.add_argument(
        "--force-retrain",
        action="store_true",
        help="Manual override: retrain before scoring even when the due gate would skip.",
    )
    p.add_argument(
        "--skip-due-retrain",
        action="store_true",
        help="Do not auto-retrain from the post-feature path; still bootstrap if no bundle exists.",
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

    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
