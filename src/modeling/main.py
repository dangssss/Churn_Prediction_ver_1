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
from pipeline.monthly import run_monthly_pipeline
from export_risk_mode.runner import run_export_risk_mode
from config_store.best_config import load_latest_accepted_best_config
from config.paths import CHURN_MODEL_DIR
from preprocess.static_features import load_cus_lifetime
from main_model.runner import run_main_variant
from common.artifacts import save_bundle
from logging_config import get_logger

logger = get_logger(__name__)


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

    # manual sweep: mark accepted=True by default (debug)
    best_cfg["is_accepted"] = True
    best_cfg["accept_rule"] = "manual_sweep"
    upsert_best_config(engine, best_cfg)

    logger.info("Saved best_config: %s", best_cfg)
    cols = [c for c in ["k", "use_static", "val_month", "f1", "PR_AUC_val", "best_threshold", "spw_used"] if c in df_ab.columns]
    logger.info("TOP-10:\n%s", df_ab[cols].head(10).to_string(index=False))


def cmd_train_main(args) -> None:
    engine = get_engine()
    logger.info("DB: %s", smoke_test(engine))
    cfg = load_latest_accepted_best_config(engine, horizon=int(args.horizon))
    df_static = load_cus_lifetime(engine)

    variants = [
        run_main_variant(engine, cfg, df_static, use_static_flag=False),
        run_main_variant(engine, cfg, df_static, use_static_flag=True),
    ]
    ok = [v for v in variants if not v.get("guardrail_warning")]
    if not ok:
        raise RuntimeError("All variants failed guardrail. Stop training.")
    ok.sort(key=lambda r: (r["F1_val"], r["AP_val"]), reverse=True)
    best = ok[0]

    cfg = dict(cfg)
    cfg["use_static"] = bool(best["use_static"])

    bundle_dir = Path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
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
