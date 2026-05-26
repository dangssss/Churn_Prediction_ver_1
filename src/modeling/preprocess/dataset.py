from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from infra.yymm import shift_yymm
from .feature_tables import (
    FEATURE_SCHEMA,
    parse_feature_table_name,
    table_exists,
    load_feature_table,
    list_tables_for_k,
)
from .gating import apply_gate

logger = logging.getLogger(__name__)

def clip_and_log_outliers(df: pd.DataFrame, percentile_lower: float = 0.1, percentile_upper: float = 99.9) -> pd.DataFrame:
    df_clipped = df.copy()
    outlier_summary = []
    
    # Bỏ qua các cột metadata và nhãn
    skip_cols = {"cms_code_enc", "window_size", "window_start", "window_end", 
                 "source_table_t", "source_table_t_plus_h", 
                 "is_active_now", "is_churned_now", "gate_group"}
    
    num_cols = []
    for c in df_clipped.columns:
        if c in skip_cols or c.startswith("y_churn_"):
            continue
        if pd.api.types.is_numeric_dtype(df_clipped[c]):
            num_cols.append(c)
            
    for col in num_cols:
        vals = pd.to_numeric(df_clipped[col], errors='coerce')
        if vals.isna().all():
            continue
            
        lower_bound = np.nanpercentile(vals, percentile_lower)
        upper_bound = np.nanpercentile(vals, percentile_upper)
        
        if lower_bound == upper_bound:
            continue
            
        v_max = vals.max()
        v_min = vals.min()
        
        should_clip_high = False
        should_clip_low = False
        
        # Chỉ thực sự clip nếu giá trị max/min vượt quá xa ngưỡng 99.9% / 0.1% để tránh làm phẳng các cột thưa (sparse) hoặc nhị phân
        if upper_bound > 0 and v_max > 5 * upper_bound:
            should_clip_high = True
            
        if lower_bound < 0 and v_min < 5 * lower_bound:
            should_clip_low = True
            
        clip_low = lower_bound if should_clip_low else v_min
        clip_high = upper_bound if should_clip_high else v_max
        
        if clip_low == clip_high:
            continue
            
        low_mask = vals < clip_low
        high_mask = vals > clip_high
        
        low_count = low_mask.sum()
        high_count = high_mask.sum()
        
        if low_count > 0 or high_count > 0:
            df_clipped[col] = vals.clip(clip_low, clip_high)
            outlier_summary.append({
                "column": col,
                "low_count": low_count,
                "high_count": high_count,
                "clip_low": clip_low,
                "clip_high": clip_high
            })
            
    if outlier_summary:
        logger.info(
            "[OUTLIER DETECTION] Đã cắt ngoại lai cho %d cột có giá trị cực đại dị thường. Một số cột ví dụ: %s",
            len(outlier_summary),
            ", ".join(f"{x['column']} (high_clip={x['clip_high']:.2f}, count={x['high_count']})" for x in outlier_summary[:5])
        )
        
    return df_clipped

def future_table_for_pair(k: int, start: str, end: str, horizon: int) -> str:
    start_h = shift_yymm(start, horizon)
    end_h   = shift_yymm(end, horizon)
    return f"cus_feature_{k}m_{start_h}_{end_h}"

def build_labeled_pair(
    engine: Engine,
    k: int,
    table_t: str,
    horizon: int = 1,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    kk, start, end = parse_feature_table_name(table_t)
    if kk != k:
        raise ValueError("table_t không thuộc K")

    table_tp = future_table_for_pair(k, start, end, horizon)
    if not table_exists(engine, FEATURE_SCHEMA, table_tp):
        return pd.DataFrame()  # censor window này (không có label tương lai)

    df_t  = load_feature_table(engine, table_t,  limit=limit)
    df_tp = load_feature_table(engine, table_tp, limit=limit)

    if "cms_code_enc" not in df_t.columns or "cms_code_enc" not in df_tp.columns:
        raise KeyError("Thiếu cms_code_enc để join label")

    # label y = churn at t+h based on item/revenue at t+h
    from .gating import resolve_now_cols
    cols_tp = resolve_now_cols(df_tp)
    item_tp = pd.to_numeric(df_tp[cols_tp["item_now"]], errors="coerce").fillna(0)
    rev_tp  = pd.to_numeric(df_tp[cols_tp["rev_now"]],  errors="coerce").fillna(0)
    y = ((item_tp == 0) & (rev_tp == 0)).astype(int)
    
    n_pos = int(y.sum())
    n_total = len(y)
    logger.info("Đã sinh nhãn y_churn_t_plus_%d từ %s: %d Churn (1), %d Active (0) trên tổng %d rows.",
                horizon, table_tp, n_pos, n_total - n_pos, n_total)

    lab = df_tp[["cms_code_enc"]].copy()
    lab[f"y_churn_t_plus_{horizon}"] = y.values

    out = df_t.merge(lab, on="cms_code_enc", how="left")

    # enforce window_end exists
    if "window_end" not in out.columns:
        out["window_end"] = end

    out["source_table_t"] = table_t
    out["source_table_t_plus_h"] = table_tp
    return out

def build_dataset_for_k(engine: Engine, k: int, horizon: int = 1, limit_rows_each: Optional[int] = None) -> pd.DataFrame:
    tbls = list_tables_for_k(engine, k)
    frames = []
    for t in tbls:
        df = build_labeled_pair(engine, k, t, horizon=horizon, limit=limit_rows_each)
        if not df.empty:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = apply_gate(out)
        out = clip_and_log_outliers(out)
    return out


def load_scoring_table_for_k(
    engine: Engine,
    k: int,
    window_end: int | None = None,
    limit_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, str, int]:
    """
    Load the single feature table for scoring at a specific month (window_end).

    If window_end is None, it uses the latest available month for that K.
    Returns: (df, table_name, window_end_used)
    """
    from .feature_tables import max_window_end_for_k  # local import to avoid cycles

    k = int(k)
    if window_end is None:
        window_end = int(max_window_end_for_k(engine, k))
    else:
        window_end = int(window_end)

    tbls = list_tables_for_k(engine, k)
    cands = []
    want_start = shift_yymm(window_end, -(k - 1))
    for t in tbls:
        kk, start, end = parse_feature_table_name(t)
        if int(end) == window_end:
            # prefer the correct start aligned with K
            priority = 0 if int(start) == int(want_start) else 1
            cands.append((priority, int(start), t))

    if not cands:
        raise ValueError(f"No feature table for K={k} with window_end={window_end}")

    cands.sort(key=lambda z: (z[0], z[1]))
    table_t = cands[0][2]

    df = load_feature_table(engine, table_t, limit=limit_rows)
    kk, start, end = parse_feature_table_name(table_t)

    # ensure window columns exist
    if "window_size" not in df.columns:
        df["window_size"] = int(kk)
    if "window_start" not in df.columns:
        df["window_start"] = int(start)
    if "window_end" not in df.columns:
        df["window_end"] = int(end)

    df["source_table_t"] = table_t

    # gate now (active_now vs churned_now)
    df = apply_gate(df)
    df = clip_and_log_outliers(df)
    return df, table_t, window_end
