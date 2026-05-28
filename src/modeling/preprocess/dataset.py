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

    t_plus_h = shift_yymm(end, horizon)

    # -------------------------------------------------------------------------
    # SỬA LỖI LOGIC: Không dùng `k` để tìm bảng tương lai (table_tp), vì số lượng 
    # cột lịch sử (item_1m_ago,...) sẽ thay đổi theo `k`, làm nhãn bị thay đổi.
    # Giải pháp: Tìm bảng có end == t_plus_h và có K LỚN NHẤT để đảm bảo luôn 
    # có đủ 3 tháng lịch sử tính nhãn (nhãn sẽ đồng nhất cho mọi vòng sweep K).
    # -------------------------------------------------------------------------
    from .feature_tables import list_feature_tables
    all_tbls = list_feature_tables(engine)
    future_cands = []
    for t in all_tbls:
        k_f, st_f, en_f = parse_feature_table_name(t)
        if en_f == t_plus_h:
            future_cands.append((k_f, t))
            
    if not future_cands:
        return pd.DataFrame()  # censor window này (không có label tương lai)
        
    future_cands.sort(key=lambda x: x[0], reverse=True)
    best_k_f, table_tp = future_cands[0]

    df_t  = load_feature_table(engine, table_t,  limit=limit)
    df_tp = load_feature_table(engine, table_tp, limit=limit)

    if "cms_code_enc" not in df_t.columns or "cms_code_enc" not in df_tp.columns:
        raise KeyError("Thiếu cms_code_enc để join label")

    # ---------- Labeling đa tín hiệu (C0 OR C1 OR C2) ----------
    # Tất cả tín hiệu tính từ df_tp (tháng t+h) — không leakage từ df_t

    from .gating import resolve_now_cols
    cols_tp = resolve_now_cols(df_tp)
    item_tp_col = cols_tp["item_now"]
    rev_tp_col  = cols_tp["rev_now"]

    item_tp = pd.to_numeric(df_tp[item_tp_col], errors="coerce").fillna(0)
    rev_tp  = pd.to_numeric(df_tp[rev_tp_col],  errors="coerce").fillna(0)

    # C0: churn hoàn toàn (item=0 và revenue=0)
    c0 = (item_tp == 0) & (rev_tp == 0)

    # Lấy các cột DE đã tính sẵn
    freq_tp = pd.to_numeric(df_tp.get("frequency", 0), errors="coerce").fillna(0)
    monetary_tp = pd.to_numeric(df_tp.get("monetary", 0), errors="coerce").fillna(0)
    rev_slope_tp = pd.to_numeric(df_tp.get("revenue_slope", 0), errors="coerce").fillna(0)

    rev_1m = pd.to_numeric(df_tp.get("revenue_1m_ago", 0), errors="coerce").fillna(0)
    item_1m = pd.to_numeric(df_tp.get("item_1m_ago", 0), errors="coerce").fillna(0)

    # FIX: freq_tp và monetary_tp là TỔNG của best_k_f tháng (window của bảng label).
    # Phải chia cho best_k_f để ra mức BÌNH QUÂN 1 tháng trước khi so sánh với
    # item_tp / rev_tp là giá trị của DUY NHẤT 1 tháng (tháng t+h).
    # Nếu không normalize: khách hàng đều đặn bình thường cũng bị đánh nhầm là Churn.
    k_months_label = max(int(best_k_f), 1)
    avg_freq_per_month     = freq_tp     / k_months_label
    avg_monetary_per_month = monetary_tp / k_months_label

    # C1: Tần suất gửi hàng giảm mạnh (Giảm > 50% so với bình quân 1 tháng)
    # HOẶC giảm > 50% so với tháng liền kề
    c1_drop_avg = (avg_freq_per_month > 0) & (item_tp < 0.50 * avg_freq_per_month)
    c1_drop_1m  = (item_1m > 0) & (item_tp < 0.50 * item_1m)
    c1 = c1_drop_avg | c1_drop_1m

    # C2: Doanh thu giảm mạnh (Giảm > 50% so với bình quân 1 tháng)
    # HOẶC giảm > 50% so với tháng liền kề (revenue_1m_ago)
    c2_drop_avg = (avg_monetary_per_month > 0) & (rev_tp < 0.50 * avg_monetary_per_month)
    c2_drop_1m  = (rev_1m > 0) & (rev_tp < 0.50 * rev_1m)
    c2 = c2_drop_avg | c2_drop_1m

    # C3: Xu hướng doanh thu cắm đầu (revenue_slope âm) kết hợp doanh thu tháng này thấp
    # Dùng avg_monetary_per_month để so sánh cùng hệ quy chiếu 1 tháng
    c3 = (rev_slope_tp < -1000) & (rev_tp < 0.80 * avg_monetary_per_month)

    y = (c0 | c1 | c2 | c3).astype(int)

    n_c0    = int(c0.sum())
    n_c1    = int(c1.sum())
    n_c2    = int(c2.sum())
    n_c3    = int(c3.sum())
    n_total = len(y)
    n_pos   = int(y.sum())
    logger.info(
        "Đã sinh nhãn y_churn_t_plus_%d từ %s (k_label=%d tháng): "
        "Tổng Churn=%d (%.1f%%) | C0(hoàn toàn)=%d | C1(tần suất)=%d | C2(doanh thu)=%d | C3(slope)=%d | Active=%d | Total=%d",
        horizon, table_tp, k_months_label,
        n_pos, 100.0 * n_pos / max(n_total, 1),
        n_c0, n_c1, n_c2, n_c3,
        n_total - n_pos, n_total,
    )
    # -----------------------------------------------------------



    lab = df_tp[["cms_code_enc"]].copy()
    lab[f"y_churn_t_plus_{horizon}"] = y.values

    out = df_t.merge(lab, on="cms_code_enc", how="left")

    # Khách hàng có ở tháng t nhưng biến mất khỏi bảng tương lai (t+h)
    # Vì bảng t+h chỉ chứa khách hàng CÓ GIAO DỊCH trong window K đó,
    # nên vắng mặt ở đây CHẮC CHẮN nghĩa là họ không mua gì -> C0 (Churn hoàn toàn).
    missing_mask = out[f"y_churn_t_plus_{horizon}"].isna()
    if missing_mask.any():
        out.loc[missing_mask, f"y_churn_t_plus_{horizon}"] = 1

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
