from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy.engine import Engine

from preprocess.dataset import load_scoring_table_for_k
from preprocess.eligibility import filter_churn_eligible
from preprocess.static_features import load_cus_lifetime, attach_static, LIFETIME_RATIO_REQUIRED_COLS
from common.artifacts import load_bundle
from config_store.best_config import load_latest_accepted_best_config as load_latest_best_config

from .create_risk_table import ensure_risk_table_schema
from .insert_predictions import (
    make_predictions,
    insert_predictions_to_risk_table,
    compute_simple_reasons,
    compute_shap_reasons,
    filter_risk_predictions,
)
import os


def engineer_features_step6(df_window: pd.DataFrame, df_static: pd.DataFrame, *, use_static: bool) -> pd.DataFrame:
    """Merge window features + lifetime and create ratio features."""
    if use_static:
        return attach_static(df_window, df_static, cols=None, keep_static_cols=True, add_ratios=True)
    return attach_static(df_window, df_static, cols=LIFETIME_RATIO_REQUIRED_COLS, keep_static_cols=False, add_ratios=True)


def run_export_risk(
    engine: Engine,
    horizon: int = 2,
    t_current: int | None = None,
    limit_rows: int | None = None,
    risk_threshold: float = 90,
    bundle_path: str | Path = None,
    enrich_dossier: bool = False,  # Ignored in simplified version
) -> dict:
    # [1/6] Load config & static data
    print("\n[1/6] Load config & data...")
    cfg = load_latest_best_config(engine, horizon=horizon)
    df_static = load_cus_lifetime(engine)
    print(f"? Loaded config (K={cfg['best_k']}, horizon={horizon}, use_static={cfg.get('use_static')})")

    # [2/6] Load model
    print("\n[2/6] Load trained model...")
    if bundle_path is None:
        raise ValueError("bundle_path is required (folder contains model.joblib + metadata.json)")
    bundle_path = Path(bundle_path)
    model, metadata = load_bundle(bundle_path)
    print(f"? Loaded model from {bundle_path}")
    bundle_lifecycle = str(
        (metadata or {}).get("bundle_lifecycle")
        or (metadata or {}).get("cfg", {}).get("bundle_lifecycle")
        or cfg.get("bundle_lifecycle")
        or "PRODUCTION"
    ).upper()
    if bundle_lifecycle == "PROVISIONAL":
        print(
            "WARNING: Scoring uses a PROVISIONAL bundle validated with rule-based labels only. "
            "Keep output under business review until actual-label validation promotes a PRODUCTION bundle."
        )

    # [3/6] Load scoring feature table...
    print("\n[3/6] Load scoring feature table...")
    k = int(cfg["best_k"])
    
    # CRITICAL: If model was trained with a different K than config,
    # use the model's K to avoid feature_names mismatch
    model_k = None
    if metadata and "cfg" in metadata:
        model_k = metadata["cfg"].get("best_k")
    if model_k is not None and int(model_k) != k:
        print(f"   ? Config DB has K={k} but Model bundle was trained with K={model_k}")
        print(f"   ? Using K={model_k} from Model bundle to match feature columns")
        k = int(model_k)
    
    df_raw, table_t, month_used = load_scoring_table_for_k(engine, k, window_end=t_current, limit_rows=limit_rows)

    # Filter active customers (item_last > 0 OR revenue_last > 0)
    df_active = df_raw[df_raw.get("is_active_now", 1) == 1].copy()
    active_before_eligibility = len(df_active)
    df_active = filter_churn_eligible(df_active, k=k, context=f"scoring_k{k}_month{month_used}")
    print(
        f"? Table: {table_t} | month={month_used} | active customers={active_before_eligibility} "
        f"| churn-eligible={len(df_active)}"
    )

    if df_active.empty:
        print("No active customers to score")
        return {"status": "warning", "message": "No active customers"}

    # [4/6] Feature engineering
    print("\n[4/6] Engineer features...")
    df_engineered = engineer_features_step6(df_active, df_static, use_static=bool(cfg.get("use_static", False)))
    print(f"? Engineered dataset shape: {df_engineered.shape}")

    # [5/6] Make predictions
    print("\n[5/6] Make predictions...")
    df_pred = make_predictions(
        model,
        df_engineered,
        cfg,
        metadata,
        risk_threshold=float(risk_threshold),
    )
    print("? Predictions completed")

    # Compute reasons — dùng SHAP làm cốt lõi, fallback sang rule-based
    print("      Compute reasons (SHAP)...")
    try:
        from main_model.xgb_utils import date_col_to_ordinal, is_date_like_col, safe_to_category

        # Tái tạo X_scored từ df_engineered (cùng pipeline type-conversion với make_predictions)
        _feat_cols_meta = metadata.get("feat_cols") or []
        _feat_cols = [c for c in _feat_cols_meta if c in df_engineered.columns]
        if not _feat_cols:
            drop_cols = {
                "cms_code_enc", "window_size", "window_start", "window_end",
                "source_table_t", "source_table_t_plus_h",
                "is_active_now", "is_churned_now", "gate_group",
                "is_churn_eligible", "churn_ineligible_reason",
                "churn_active_months_in_window", "churn_required_active_months",
                "churn_item_sum_for_eligibility", "churn_revenue_sum_for_eligibility",
                "churn_avg_revenue_per_item_for_eligibility",
            }
            _feat_cols = [c for c in df_engineered.columns if c not in drop_cols]

        X_scored = df_engineered[_feat_cols].copy()
        _cat_cols  = list(metadata.get("cat_cols")  or [])
        _date_cols = list(metadata.get("date_cols") or [])
        _extra_dc  = [
            c for c in _cat_cols
            if c in X_scored.columns and c not in _date_cols and is_date_like_col(X_scored[c])
        ]
        _date_cols = _date_cols + _extra_dc
        _cat_cols  = [c for c in _cat_cols if c not in _extra_dc]
        for c in _date_cols:
            if c in X_scored.columns:
                X_scored[c] = date_col_to_ordinal(X_scored[c])
        for c in _cat_cols:
            if c in X_scored.columns:
                X_scored[c] = safe_to_category(X_scored[c])
        for c in _feat_cols:
            if c not in _cat_cols and c not in _date_cols:
                X_scored[c] = pd.to_numeric(X_scored[c], errors="coerce")
        _fmap = metadata.get("feature_name_map") or {}
        if _fmap:
            X_scored = X_scored.rename(columns=_fmap)
        _model_feats = getattr(model, "feature_names_in_", None)
        if _model_feats is not None:
            for c in _model_feats:
                if c not in X_scored.columns:
                    X_scored[c] = 0
            X_scored = X_scored[list(_model_feats)]

        df_pred, df_shap_raw = compute_shap_reasons(model, X_scored, df_pred, df_static)
        
        # In kết quả chuẩn của SHAP (lúc chưa map) ra 1 folder riêng
        if df_shap_raw is not None:
            output_dir_env = os.environ.get("OUTPUT_DIR", "/churn_data/output_prediction")
            shap_dir = Path(output_dir_env) / "shap_raw"
            shap_dir.mkdir(parents=True, exist_ok=True)
            shap_csv_path = shap_dir / f"shap_raw_{month_used}.csv"
            df_shap_raw.to_csv(shap_csv_path, index=False, encoding='utf-8-sig')
            print(f"   ? Saved raw SHAP values to {shap_csv_path}")
            
    except Exception as _shap_err:
        print(f"   ? SHAP reasons gặp lỗi: {_shap_err}. Fallback sang rule-based.")
        df_pred = compute_simple_reasons(df_pred, df_static)
        df_shap_raw = None
    print("? Reasons computed")

    # Score stats
    scores = pd.to_numeric(df_pred.get("churn_probability"), errors="coerce").to_numpy()
    scores = scores[~np.isnan(scores)]
    score_stats = {
        "mean": float(np.mean(scores)) if scores.size else None,
        "p50": float(np.quantile(scores, 0.50)) if scores.size else None,
        "p90": float(np.quantile(scores, 0.90)) if scores.size else None,
        "p99": float(np.quantile(scores, 0.99)) if scores.size else None,
    }
    threshold_pct = float(risk_threshold) / 100.0
    score_cutoff = float(np.quantile(scores, threshold_pct)) if scores.size else None

    # [6/6] Insert to risk table
    print("\n[6/6] Insert to risk table...")
    table_name = ensure_risk_table_schema(engine, risk_threshold=risk_threshold)
    num_customers = insert_predictions_to_risk_table(
        engine, df_pred, risk_threshold=risk_threshold, horizon=horizon
    )

    # Store the scoring origin and K used by scoring-only for drift monitoring.
    try:
        from monitoring.score import upsert_score_drift

        upsert_score_drift(
            engine,
            window_end=int(month_used),
            horizon=int(horizon),
            best_k=int(k),
            active_cnt=int(len(df_active)),
            churned_now_cnt=int(
                pd.to_numeric(
                    df_raw.get("is_churned_now", pd.Series(0, index=df_raw.index)),
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),
            scores=scores,
            risk_threshold_pct=int(risk_threshold),
            risk_cnt=int(num_customers),
        )
    except Exception as score_drift_error:
        print(f"? WARNING: Score drift monitoring skipped: {score_drift_error}")

    # Training validation uses mixed actual/rule labels from recent origins.

    df_ins = filter_risk_predictions(df_pred, risk_threshold)
    num_with_reasons = int(df_ins['reason_1'].notna().sum()) if 'reason_1' in df_ins.columns else 0

    # CSV Export requirement
    print("\n[7/7] Export to CSV file...")
    try:
        output_dir_env = os.environ.get("OUTPUT_DIR", "/churn_data/output_prediction")
        csv_dir = Path(output_dir_env)
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate predict_period
        dt_window = pd.to_datetime(str(month_used), format='%y%m')
        dt_predict = dt_window + pd.DateOffset(months=horizon - 1)
        predict_period_str = dt_predict.strftime('%y%m')
        
        df_csv = df_ins.copy()
        df_csv['window_end'] = str(month_used)
        df_csv['predict_period'] = predict_period_str
        df_csv['updated_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Ensure target columns exist safely
        required_csv_cols = [
            'cms_code_enc', 'window_end', 'predict_period', 
            'item_last', 'revenue_last', 'complaint_last', 'delay_last', 
            'nodone_last', 'order_score_last', 'satisfaction_last', 
            'churn_rate', 'model_probability_pct', 'reason_1', 'reason_2', 'reason_3',
            'reason_1_code', 'reason_1_metric', 'reason_1_baseline',
            'reason_1_delta', 'reason_1_delta_pct', 'reason_1_severity',
            'reason_2_code', 'reason_2_metric', 'reason_2_baseline',
            'reason_2_delta', 'reason_2_delta_pct', 'reason_2_severity',
            'reason_3_code', 'reason_3_metric', 'reason_3_baseline',
            'reason_3_delta', 'reason_3_delta_pct', 'reason_3_severity',
            'updated_at'
        ]
        
        for col in required_csv_cols:
            if col not in df_csv.columns:
                df_csv[col] = None
                
        df_csv = df_csv[required_csv_cols]
        
        today_str = pd.Timestamp.today().strftime('%y%m%d')
        csv_filename = f"churn_predict_update_{today_str}.csv"
        csv_path = csv_dir / csv_filename
        
        df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"? Saved {len(df_csv)} predicted churn profiles directly to {csv_path}")
    except Exception as e:
        print(f"? WARNING: Failed to export CSV: {e}")

    # Summary
    print("\nEXPORT RISK TABLE COMPLETED")
    print(f"Table:              data_static.{table_name}")
    print(f"Month scored:       {month_used}")
    print(f"Risk threshold:     score percentile >= {risk_threshold}")
    print(f"Active customers:   {len(df_active)}")
    print(f"Inserted:           {num_customers}")
    print(f"With reasons:       {num_with_reasons}")
    print(f"Score stats:        p50={score_stats['p50']:.2%}, p90={score_stats['p90']:.2%}, p99={score_stats['p99']:.2%}")
    if score_cutoff is not None:
        print(f"Score cutoff:       p{int(risk_threshold)}={score_cutoff:.2%}")
    print("="*70 + "\n")

    return {
        "table_name": table_name,
        "month_scored": int(month_used),
        "source_table": table_t,
        "active_cnt": int(len(df_active)),
        "risk_cnt": int(num_customers),
        "score_stats": score_stats,
        "score_cutoff": score_cutoff,
        "num_inserted": int(num_customers),
        "num_with_reasons": int(num_with_reasons),
        "status": "success" if num_customers > 0 else "warning",
    }


def run_export_risk_mode(
    engine: Engine,
    *,
    horizon: int,
    bundle_dir: Path,
    risk_threshold: float,
    t_current: int | None = None,
    limit_rows: int | None = None,
    make_dossier: bool = False,
) -> dict:
    """Back-compat wrapper."""
    return run_export_risk(
        engine,
        horizon=int(horizon),
        t_current=int(t_current) if t_current is not None else None,
        limit_rows=limit_rows,
        risk_threshold=float(risk_threshold),
        bundle_path=Path(bundle_dir),
        enrich_dossier=bool(make_dossier),
    )
