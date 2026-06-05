from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.db import get_engine, smoke_test
from preprocess.static_features import load_cus_lifetime_snapshots
from config_store.best_config import load_latest_accepted_best_config as load_latest_best_config, update_main_metrics
from main_model.runner import run_main_variant
from common.artifacts import save_bundle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--main-es-rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--choose-static", choices=["false","true","both"], default="both",
                   help="Train use_static variants. 'both' will train False/True and auto choose by Lift@N.")
    ap.add_argument("--save-bundle", type=str, default=None,
                   help="Folder to save model + metadata (joblib + json).")
    args = ap.parse_args()

    engine = get_engine()
    print("DB:", smoke_test(engine))
    df_static = load_cus_lifetime_snapshots(engine)

    cfg = load_latest_best_config(engine, horizon=args.horizon)
    cfg["main_es_rounds"] = int(args.main_es_rounds)
    cfg["seed"] = int(args.seed)

    # train variants
    variants = []
    if args.choose_static in ("false","both"):
        variants.append(run_main_variant(engine, cfg, df_static, use_static_flag=False))
    if args.choose_static in ("true","both"):
        variants.append(run_main_variant(engine, cfg, df_static, use_static_flag=True))

    # filter guardrail fails
    ok = [v for v in variants if not v.get("guardrail_warning")]
    if not ok:
        raise SystemExit("All variants failed guardrail. Stop.")

    ok.sort(key=lambda r: (r["F1_val"], r["AP_val"], r["Lift_at_n"]), reverse=True)
    best = ok[0]
    if len(ok) == 2:
        f1_gap = ok[0]["F1_val"] - ok[1]["F1_val"]
        if abs(f1_gap) <= 0.005:
            # If F1 is effectively tied, prefer no_static (simpler & less leakage risk).
            best = next((v for v in ok if v["use_static"] is False), best)

    cfg["use_static"] = bool(best["use_static"])
    main_report = best["report"]
    main_model = best["model"]

    print("==> CHOSEN use_static =", cfg["use_static"])
    print("Lift@N:", best["Lift_at_n"], "| AP_val:", best["AP_val"], "| F1_val:", best["F1_val"])
    print("Warnings:", best.get("guardrail_warning"))

    # update DB
    as_of_month = int(cfg["as_of_month"])
    rowcount = update_main_metrics(engine, as_of_month=as_of_month, horizon=int(args.horizon), main_report=main_report)
    print("DB update rowcount=", rowcount)

    # save bundle (optional)
    if args.save_bundle:
        meta = {
            "cfg": cfg,
            "bundle_lifecycle": str(cfg.get("bundle_lifecycle") or "PRODUCTION").upper(),
            "validation_label_source": cfg.get("validation_label_source"),
            "main_report": main_report,
            "feat_cols": best.get("feat_cols"),
            "cat_cols": best.get("cat_cols"),
            "feature_name_map": best.get("feature_name_map"),
            "feature_profile": best.get("feature_profile"),
        }
        save_bundle(Path(args.save_bundle), main_model, metadata=meta)
        print("Saved bundle to:", args.save_bundle)

if __name__ == "__main__":
    main()
