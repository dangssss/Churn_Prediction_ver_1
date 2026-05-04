from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from main_model.xgb_utils import (
    safe_to_category,
    predict_proba_best_iteration,
    date_col_to_ordinal,
    is_date_like_col,
)


def make_predictions(
    model: any,
    df_data: pd.DataFrame,
    cfg: dict,
    metadata: dict,
) -> pd.DataFrame:
    """Make predictions using trained XGBoost model."""
    h = int(cfg["horizon"])
    label_col = f"y_churn_t_plus_{h}"
    
    # Prepare features
    drop_cols = {
        "cms_code_enc", "window_size", "window_start", "window_end",
        "source_table_t", "source_table_t_plus_h",
        "is_active_now", "is_churned_now", "gate_group",
        label_col
    }
    meta_feat_cols = metadata.get("feat_cols")
    if isinstance(meta_feat_cols, list) and meta_feat_cols:
        feat_cols = [c for c in meta_feat_cols if c in df_data.columns]
    else:
        feat_cols = [c for c in df_data.columns if c not in drop_cols]
    
    X = df_data[feat_cols].copy()
    
    # Get metadata
    cat_cols = list(metadata.get("cat_cols") or [])
    date_cols = list(metadata.get("date_cols") or [])
    feature_name_map = metadata.get("feature_name_map") or {}

    # Detect date-like cols that were mistakenly put in cat_cols by older model bundles
    # (handles bundles trained before the date_cols fix)
    extra_date_cols = [
        c for c in cat_cols
        if c in X.columns and c not in date_cols and is_date_like_col(X[c])
    ]
    if extra_date_cols:
        print(f"   Auto-detected {len(extra_date_cols)} date col(s) in cat_cols, converting to ordinal: {extra_date_cols}")
        date_cols = date_cols + extra_date_cols
        cat_cols = [c for c in cat_cols if c not in extra_date_cols]

    # Type conversion — date cols (ordinal numeric), then categorical, then numeric
    for c in date_cols:
        if c in X.columns:
            X[c] = date_col_to_ordinal(X[c])

    for c in cat_cols:
        if c in X.columns:
            X[c] = safe_to_category(X[c])

    for c in feat_cols:
        if c not in cat_cols and c not in date_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Predict
    try:
        # Pad missing columns: if model was trained with K=12 but scoring data only has K=11,
        # fill missing *_Nm_ago columns with 0 so XGBoost doesn't crash
        _feat_names_raw = getattr(model, "feature_names_in_", None)
        if _feat_names_raw is not None:
            model_features = list(_feat_names_raw)
        elif feature_name_map:
            model_features = [feature_name_map.get(c, c) for c in metadata.get("feat_cols", [])]
        else:
            model_features = metadata.get("feat_cols") or None

        if model_features is not None:
            X_pred = X.rename(columns=feature_name_map) if feature_name_map else X.copy()
            missing = [c for c in model_features if c not in X_pred.columns]
            if missing:
                print(f"   Padding {len(missing)} missing feature(s): {missing[:5]}...")
                for c in missing:
                    X_pred[c] = 0
            X_pred = X_pred[list(model_features)]
            prob = predict_proba_best_iteration(model, X_pred)[:, 1]
        elif feature_name_map:
            X_renamed = X.rename(columns=feature_name_map)
            prob = predict_proba_best_iteration(model, X_renamed)[:, 1]
        else:
            prob = predict_proba_best_iteration(model, X)[:, 1]
    except Exception as e:
        print(f"Prediction error: {e}")
        print("   Fallback to standard predict_proba")
        prob = model.predict_proba(X)[:, 1]
    
    # Combine results
    df_out = df_data.copy()
    df_out["churn_probability"] = prob
    df_out["churn_rate"] = (prob * 100).round(2)

    # Threshold from config
    thr = cfg.get("main_threshold", cfg.get("best_threshold", 0.5))
    df_out["risk_score"] = prob
    df_out["risk_flag"] = (prob >= float(thr)).astype(int)

    return df_out


def compute_simple_reasons(df: pd.DataFrame, df_static: pd.DataFrame) -> pd.DataFrame:
    """
    Tính reasons chi ti?t theo yêu c?u:
    Ch? xét khách hàng có phát sinh don trong 2 tháng g?n nh?t (item_last > 0 AND item_1m_ago > 0)
    
    1. S? buu g?i tháng hi?n t?i < 60% trung bình 3 tháng (ch? tính tháng có don)
    2. Khi?u n?i tang > 115% so v?i trung bình 3 tháng
    3. Giao mu?n tang > 115% so v?i trung bình 3 tháng
    4. Không hoàn thành tang > 115% so v?i trung bình 3 tháng
    5. Bi?n d?ng don hàng cao (CV > 0.7)
    6. Giá tr? don hàng trung bình gi?m so v?i 3 tháng tru?c
    7. Gi?m da d?ng d?ch v? (s? lo?i gi?m)
    8. Khách hàng m?i (tenure < 6 tháng)
    """
    d = df.copy()
    
    # Merge tenure t? static
    if "tenure" not in d.columns and "cms_code_enc" in d.columns and "tenure" in df_static.columns:
        tenure_map = df_static[["cms_code_enc", "tenure"]].drop_duplicates("cms_code_enc")
        d = d.merge(tenure_map, on="cms_code_enc", how="left")
    
    def _num(s):
        return pd.to_numeric(s, errors="coerce").fillna(0)
    
    # Helper: tính avg 3 tháng tru?c (ch? tính các tháng có phát sinh)
    def _avg_prev_3m_active(base_col: str) -> pd.Series:
        """T\u00ednh average c\u1ee7a 3 th\u00e1ng tr\u01b0\u1edbc, ch\u1ec9 c\u00e1c th\u00e1ng c\u00f3 ph\u00e1t sinh > 0"""
        cols = [f"{base_col}_{i}m_ago" for i in [1, 2, 3]]
        available = [c for c in cols if c in d.columns]
        if not available:
            return pd.Series(0, index=d.index)
        mat = d[available].apply(pd.to_numeric, errors="coerce")
        # Ch? tính average c?a các giá tr? > 0
        mat_active = mat.where(mat > 0)
        return mat_active.mean(axis=1, skipna=True).fillna(0)
    
    # Current values
    item_last = _num(d.get("item_last", 0))
    item_1m_ago = _num(d.get("item_1m_ago", 0))
    complaint_last = _num(d.get("complaint_last", 0))
    delay_last = _num(d.get("delay_last", 0))
    nodone_last = _num(d.get("nodone_last", 0))
    revenue_last = _num(d.get("revenue_last", 0))
    
    # Tính avg 3 tháng tru?c (ch? tháng có don)
    avg_item_3m = _avg_prev_3m_active("item")
    avg_complaint_3m = _avg_prev_3m_active("complaint")
    avg_delay_3m = _avg_prev_3m_active("delay")
    avg_nodone_3m = _avg_prev_3m_active("nodone")
    avg_revenue_3m = _avg_prev_3m_active("revenue")
    
    # Tính avg_revenue_per_item
    rpi_last = np.where(item_last > 0, revenue_last / item_last, 0)
    rpi_3m = np.where(avg_item_3m > 0, avg_revenue_3m / avg_item_3m, 0)
    
    # cv_item, service_types_used, tenure
    cv_item = _num(d.get("cv_item", 0))
    service_types = _num(d.get("service_types_used", 0))
    service_types_prev = _num(d.get("service_types_used_prev", service_types))  # previous month
    tenure = _num(d.get("tenure", 999))
    
    # Filter: ch? xét khách hàng có don trong 2 tháng g?n nh?t
    active_mask = (item_last > 0) & (item_1m_ago > 0)
    
    # Ðánh giá t?ng reason v?i score uu tiên
    reason_scores = []
    for i in range(len(d)):
        scores = []
        
        # Skip if not active
        if not active_mask.iloc[i]:
            reason_scores.append([])
            continue
        
        # 1. S? buu g?i gi?m (priority 10) - < 60% trung bình
        if avg_item_3m.iloc[i] > 0 and item_last.iloc[i] < 0.6 * avg_item_3m.iloc[i]:
            decrease_pct = (1 - item_last.iloc[i] / avg_item_3m.iloc[i]) * 100
            reason_text = f"S\u1ed1 b\u01b0u g\u1eedi th\u00e1ng hi\u1ec7n t\u1ea1i th\u1ea5p h\u01a1n {decrease_pct:.0f}% so v\u1edbi trung b\u00ecnh 3 th\u00e1ng li\u1ec1n tr\u01b0\u1edbc"
            scores.append((10, reason_text))
        
        # 2. Khi?u n?i tang (priority 9)
        if avg_complaint_3m.iloc[i] > 0 and complaint_last.iloc[i] > 1.15 * avg_complaint_3m.iloc[i]:
            increase_pct = (complaint_last.iloc[i] / avg_complaint_3m.iloc[i] - 1) * 100
            reason_text = f"S\u1ed1 l\u01b0\u1ee3ng khi\u1ebfu n\u1ea1i nh\u1eadn \u0111\u01b0\u1ee3c t\u0103ng {increase_pct:.0f}% so v\u1edbi trung b\u00ecnh 3 th\u00e1ng li\u1ec1n tr\u01b0\u1edbc"
            scores.append((9, reason_text))
        
        # 3. Giao mu?n tang (priority 8)
        if avg_delay_3m.iloc[i] > 0 and delay_last.iloc[i] > 1.15 * avg_delay_3m.iloc[i]:
            increase_pct = (delay_last.iloc[i] / avg_delay_3m.iloc[i] - 1) * 100
            reason_text = f"T\u1ef7 l\u1ec7 s\u1ed1 \u0111\u01a1n giao mu\u1ed9n t\u0103ng {increase_pct:.0f}% so v\u1edbi trung b\u00ecnh 3 th\u00e1ng li\u1ec1n tr\u01b0\u1edbc"
            scores.append((8, reason_text))
        
        # 4. Không hoàn thành tang (priority 7)
        if avg_nodone_3m.iloc[i] > 0 and nodone_last.iloc[i] > 1.15 * avg_nodone_3m.iloc[i]:
            increase_pct = (nodone_last.iloc[i] / avg_nodone_3m.iloc[i] - 1) * 100
            reason_text = f"T\u1ef7 l\u1ec7 s\u1ed1 \u0111\u01a1n kh\u00f4ng ho\u00e0n th\u00e0nh t\u0103ng {increase_pct:.0f}% so v\u1edbi trung b\u00ecnh 3 th\u00e1ng li\u1ec1n tr\u01b0\u1edbc"
            scores.append((7, reason_text))
        
        # 5. Bi?n d?ng cao (priority 6) - CV > 0.7
        if cv_item.iloc[i] > 0.7:
            reason_text = f"Bi\u1ebfn \u0111\u1ed9ng s\u1ed1 l\u01b0\u1ee3ng b\u01b0u g\u1eedi cao (CV={cv_item.iloc[i]:.2f})"
            scores.append((6, reason_text))
        
        # 6. Giá tr? don gi?m (priority 5)
        if rpi_3m[i] > 0 and rpi_last[i] < rpi_3m[i]:
            decrease_pct = (1 - rpi_last[i] / rpi_3m[i]) * 100
            reason_text = f"Gi\u00e1 tr\u1ecb \u0111\u01a1n h\u00e0ng trung b\u00ecnh gi\u1ea3m {decrease_pct:.0f}% theo th\u1eddi gian"
            scores.append((5, reason_text))
        
        # 7. Gi?m da d?ng d?ch v? (priority 4)
        if service_types_prev.iloc[i] > 0 and service_types.iloc[i] < service_types_prev.iloc[i]:
            old_count = int(service_types_prev.iloc[i])
            new_count = int(service_types.iloc[i])
            reason_text = f"Gi\u1ea3m \u0111a d\u1ea1ng d\u1ecbch v\u1ee5 (gi\u1ea3m t\u1eeb {old_count} c\u00f2n {new_count} lo\u1ea1i)"
            scores.append((4, reason_text))
        
        # 8. Khách m?i (priority 3)
        if tenure.iloc[i] < 6:
            tenure_months = int(tenure.iloc[i])
            reason_text = f"Kh\u00e1ch h\u00e0ng m\u1edbi, m\u1ee9c \u0111\u1ed9 g\u1eafn b\u00f3 th\u1ea5p ({tenure_months} th\u00e1ng)"
            scores.append((3, reason_text))
        
        # Sort by priority (desc) và l?y top 3
        scores.sort(reverse=True)
        reason_scores.append([r[1] for r in scores[:3]])
    
    # Assign reasons (m?i khách hàng c?n ít nh?t 1 reason)
    d["reason_1"] = [rs[0] if len(rs) > 0 else None for rs in reason_scores]
    d["reason_2"] = [rs[1] if len(rs) > 1 else None for rs in reason_scores]
    d["reason_3"] = [rs[2] if len(rs) > 2 else None for rs in reason_scores]
    
    # Keep only active customers
    d = d[active_mask].copy()
    
    return d


def insert_predictions_to_risk_table(
    engine: Engine,
    df_predictions: pd.DataFrame,
    risk_threshold: float = 90.0,
    horizon: int = 1,
) -> int:
    """Insert predictions v\u00e0o risk table (ch\u1ec9 nh\u1eefng kh\u00e1ch h\u00e0ng c\u00f3 churn_rate >= threshold)."""
    risk_pct = int(risk_threshold)
    table_name = f"cus_risk_{risk_pct}"
    
    if "churn_rate" not in df_predictions.columns:
        raise KeyError("Missing 'churn_rate' column in predictions")
    
    # Filter theo threshold
    df_risk = df_predictions[df_predictions["churn_rate"] >= float(risk_threshold)].copy()
    
    if df_risk.empty:
        print(f"??  No customers with churn_rate >= {risk_threshold}%")
        return 0
    
    # Normalize column names
    if "satisfaction_last" not in df_risk.columns and "satisfation_last" in df_risk.columns:
        df_risk["satisfaction_last"] = df_risk["satisfation_last"]
        
    # Select columns needed
    cols_needed = [
        "cms_code_enc",
        "predict_period",
        "window_end",
        "item_last",
        "revenue_last",
        "complaint_last",
        "delay_last",
        "nodone_last",
        "order_score_last",
        "satisfaction_last",
        "churn_rate",
        "reason_1",
        "reason_2",
        "reason_3",
    ]
    
    df_insert = df_risk.copy()
    for col in cols_needed:
        if col not in df_insert.columns and col != "predict_period":
            df_insert[col] = None
            
    df_insert["cms_code_enc"] = df_insert["cms_code_enc"].astype(str)
    df_insert["window_end"] = pd.to_numeric(df_insert["window_end"], errors="coerce").fillna(0).astype("int64")
    
    # Calculate predict_period: yymm
    y = df_insert["window_end"] // 100
    m = df_insert["window_end"] % 100
    write_horizon = horizon - 1
    m = m + write_horizon
    y = y + (m - 1) // 12
    m = (m - 1) % 12 + 1
    df_insert["predict_period"] = (y * 100 + m).astype("int64")
    
    df_insert = df_insert[cols_needed].copy()
    
    # Load SQL templates
    sql_dir = Path(__file__).parent / "sql"
    upsert_sql = (sql_dir / "insert_risk_upsert.sql").read_text()
    
    # Use temporary table + INSERT ON CONFLICT approach
    temp_table = f"_temp_{table_name}"
    
    upsert_sql = upsert_sql.replace("{TABLE_NAME}", table_name).replace("{TEMP_TABLE}", temp_table)
    
    with engine.begin() as conn:
        # Create temp table
        conn.execute(text(f"""
            CREATE TEMP TABLE {temp_table} (LIKE data_static.{table_name} INCLUDING ALL)
        """))
        
        # Insert to temp table using pandas (use the same connection `conn`)
        df_insert.to_sql(
            temp_table,
            con=conn,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
        
        # UPSERT from temp to main table
        conn.execute(text(upsert_sql))
        # Drop temp table
        conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
    
    print(f"? Inserted {len(df_insert)} customers to {table_name}")
    return len(df_insert)