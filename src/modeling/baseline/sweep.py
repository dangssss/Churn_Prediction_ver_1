
from __future__ import annotations

import os
import pandas as pd
from sqlalchemy.engine import Engine

from preprocess.feature_tables import list_k_available, max_window_end_for_k
from infra.yymm import shift_yymm
from preprocess.static_features import load_cus_lifetime_snapshots
from preprocess.dataset import build_dataset_for_k, preflight_purged_train_val_for_k
from baseline.runner import SparseChurnLabelsError, eval_one_k_train_val
from logging_config import get_logger

logger = get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int env %s=%r. Using default %d.", name, raw, default)
        return int(default)


def _config_from_ablation_row(engine: Engine, row: pd.Series, horizon: int) -> dict:
    best_k = int(row["K"])
    as_of_month = int(max_window_end_for_k(engine, best_k))
    target_month = int(shift_yymm(str(as_of_month), int(horizon)))
    return {
        "as_of_month": as_of_month,
        "target_month": target_month,
        "horizon": int(horizon),
        "best_k": best_k,
        "use_static": bool(row["use_static"]),
        "best_threshold": float(row["best_threshold"]),
        "best_spw": float(row["spw_used"]),
        "metric_f1_val": float(row["f1"]),
        "metric_pr_auc_val": float(row["PR_AUC_val"]),
        "metric_roc_auc_val": float(row.get("ROC_AUC_val", 0.0)),
        "metric_val_prevalence": float(row["val_prevalence"]),
        "val_month": int(row["val_month"]),
        "bundle_lifecycle": str(row["bundle_lifecycle"]),
        "notes": "LR shortlisted by F1, PR_AUC, ROC_AUC; XGBoost selects final K",
    }

def run_sweep_k(
    engine: Engine,
    *,
    horizon: int,
    limit_rows_each: int | None = None,
    k_min: int = 3,
) -> tuple[dict, pd.DataFrame]:
    """
    Always sweep K to find the best K for current data (picked by F1 first).
    Returns:
      (best_config_candidate, df_ablation_sorted)
    """
    ks = [int(k) for k in list_k_available(engine) if int(k) >= int(k_min)]
    if not ks:
        raise ValueError("No K available in feature tables.")

    df_static = load_cus_lifetime_snapshots(engine)

    ablation = []
    for k in ks:
        try:
            preflight_purged_train_val_for_k(
                engine,
                int(k),
                horizon=int(horizon),
            )
        except ValueError as exc:
            logger.warning("Skipping K=%d during purged preflight: %s", k, exc)
            continue
        df_k = build_dataset_for_k(
            engine,
            int(k),
            horizon=int(horizon),
            limit_rows_each=limit_rows_each,
        )
        for use_static in [False, True]:
            try:
                out = eval_one_k_train_val(
                    engine,
                    k=int(k),
                    horizon=int(horizon),
                    use_static=bool(use_static),
                    df_static=df_static,
                    limit_rows_each=limit_rows_each,
                    df_k=df_k,
                )
            except SparseChurnLabelsError as exc:
                logger.warning("Skipping K=%d: %s", k, exc)
                break
            if out is None:
                continue
            if out.get('degenerate'):
                logger.warning("Skipping degenerate K=%d use_static=%s (predict-all-positive)", k, use_static)
                continue
            ablation.append(out)
            logger.info(
                "K=%d | use_static=%s | val=%s | F1=%.4f | PR_AUC=%.4f | ROC_AUC=%.4f | threshold=%.4f",
                k,
                use_static,
                out.get("val_month"),
                out["f1"],
                out["PR_AUC_val"],
                out.get("ROC_AUC_val", 0.0),
                out["best_threshold"],
            )

    if not ablation:
        raise ValueError("Ablation produced no result.")

    df_ab = (
        pd.DataFrame(ablation)
        .sort_values(
            ["f1", "PR_AUC_val", "ROC_AUC_val"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_config = _config_from_ablation_row(engine, df_ab.iloc[0], int(horizon))

    shortlist_size = max(_env_int("MODEL_XGB_K_CANDIDATES", 3), 1)
    candidate_configs = []
    seen_k = set()
    for _, row in df_ab.iterrows():
        k = int(row["K"])
        if k in seen_k:
            continue
        seen_k.add(k)
        candidate_configs.append(_config_from_ablation_row(engine, row, int(horizon)))
        if len(candidate_configs) >= shortlist_size:
            break

    best_config["xgb_candidate_configs"] = candidate_configs
    best_config["xgb_candidate_ks"] = [int(c["best_k"]) for c in candidate_configs]
    best_config["notes"] = (
        f"{best_config.get('notes')}; "
        f"LR shortlist for XGBoost K={best_config['xgb_candidate_ks']}"
    )
    logger.info(
        "[LR SHORTLIST] top_%d distinct K for XGBoost: %s",
        shortlist_size,
        best_config["xgb_candidate_ks"],
    )
    return best_config, df_ab
