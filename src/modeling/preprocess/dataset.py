from __future__ import annotations

import os
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
from .label_tables import LABEL_SCHEMA, estimate_observed_label_rate, label_tables_for_horizon, load_label_keys
from logging_config import get_logger

logger = get_logger(__name__)


def _load_post_origin_activity(
    engine: Engine,
    df_origin: pd.DataFrame,
    *,
    origin_yymm: str,
    horizon: int,
) -> tuple[pd.DataFrame, str]:
    """Build fallback outcome signals strictly from raw order months t+1..t+h."""
    from sqlalchemy import text
    from .gating import resolve_now_cols

    future_yymms = [shift_yymm(origin_yymm, offset) for offset in range(1, horizon + 1)]
    future_tables = [f"bccp_orderitem_{yymm}" for yymm in future_yymms]
    with engine.connect() as conn:
        for table in future_tables:
            exists = conn.execute(
                text("SELECT to_regclass(:table_name)"),
                {"table_name": f"public.{table}"},
            ).scalar()
            if exists is None:
                logger.info(
                    "Censor fallback origin=%s: missing required post-origin table public.%s",
                    origin_yymm,
                    table,
                )
                return pd.DataFrame(), ""

    origin_cols = resolve_now_cols(df_origin)
    origin = df_origin[["cms_code_enc"]].copy()
    origin["cms_code_enc"] = origin["cms_code_enc"].astype(str).str.strip()
    origin["origin_item"] = pd.to_numeric(
        df_origin[origin_cols["item_now"]], errors="coerce"
    ).fillna(0)
    origin["origin_revenue"] = pd.to_numeric(
        df_origin[origin_cols["rev_now"]], errors="coerce"
    ).fillna(0)
    origin = origin.drop_duplicates("cms_code_enc")

    monthly_frames = []
    for table in future_tables:
        query = text(
            f"""
            SELECT cms_code_enc,
                   COUNT(*)::bigint AS item_count,
                   COALESCE(SUM(total_fee), 0)::double precision AS revenue
            FROM public."{table}"
            WHERE cms_code_enc IS NOT NULL
            GROUP BY cms_code_enc
            """
        )
        frame = pd.read_sql(query, engine)
        frame["cms_code_enc"] = frame["cms_code_enc"].astype(str).str.strip()
        monthly_frames.append(frame)

    activity = pd.concat(monthly_frames, ignore_index=True)
    activity = (
        activity.groupby("cms_code_enc", as_index=False)
        .agg(item_count=("item_count", "sum"), revenue=("revenue", "sum"))
    )
    out = origin.merge(activity, on="cms_code_enc", how="left")
    out[["item_count", "revenue"]] = out[["item_count", "revenue"]].fillna(0)
    out["item_last"] = out["item_count"] / max(int(horizon), 1)
    out["revenue_last"] = out["revenue"] / max(int(horizon), 1)
    out["frequency"] = out["origin_item"]
    out["monetary"] = out["origin_revenue"]
    out["item_1m_ago"] = out["origin_item"]
    out["revenue_1m_ago"] = out["origin_revenue"]
    out["revenue_slope"] = out["revenue_last"] - out["origin_revenue"]
    return out, ",".join(f"public.{table}" for table in future_tables)


def clip_and_log_outliers(df: pd.DataFrame, percentile_lower: float = 0.1, percentile_upper: float = 99.9) -> pd.DataFrame:
    df_clipped = df.copy()
    outlier_summary = []
    
    # Bỏ qua các cột metadata và nhãn
    skip_cols = {"cms_code_enc", "window_size", "window_start", "window_end", 
                 "source_table_t", "source_table_t_plus_h", 
                 "is_active_now", "is_churned_now", "gate_group",
                 "label_source", "label_weight"}
    
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

    df_t = load_feature_table(engine, table_t, limit=limit)

    label_tables = label_tables_for_horizon(engine, end, horizon)
    if label_tables:
        if "cms_code_enc" not in df_t.columns:
            raise KeyError("Missing cms_code_enc to join label")

        label_col = f"y_churn_t_plus_{horizon}"
        d = df_t.copy()
        d["cms_code_enc"] = d["cms_code_enc"].astype(str).str.strip()
        labels = pd.concat(
            [load_label_keys(engine, label_table) for label_table in label_tables],
            ignore_index=True,
        ).drop_duplicates()

        cms_keys = set(labels["cms_code_enc"].dropna().astype(str))
        crm_keys = set(labels["crm_code_enc"].dropna().astype(str))

        matched = d["cms_code_enc"].isin(cms_keys)
        if crm_keys:
            if "crm_code_enc" in d.columns:
                crm_series = d["crm_code_enc"].astype(str).str.strip()
            else:
                try:
                    from sqlalchemy import text

                    q = text("""
                        SELECT cms_code_enc, crm_code_enc
                        FROM public.cas_info
                        WHERE crm_code_enc IS NOT NULL
                    """)
                    code_map = pd.read_sql(q, engine)
                    code_map["cms_code_enc"] = code_map["cms_code_enc"].astype(str).str.strip()
                    code_map["crm_code_enc"] = code_map["crm_code_enc"].astype(str).str.strip()
                    code_map = code_map.drop_duplicates("cms_code_enc")
                    crm_series = d[["cms_code_enc"]].merge(code_map, on="cms_code_enc", how="left")["crm_code_enc"].fillna("")
                except Exception as exc:
                    logger.warning("Could not load public.cas_info for crm_code_enc label matching: %s", exc)
                    crm_series = pd.Series([""] * len(d), index=d.index)

            matched = matched | crm_series.isin(crm_keys).to_numpy()

        d[label_col] = matched.astype(int)
        if "window_end" not in d.columns:
            d["window_end"] = end
        d["source_table_t"] = table_t
        label_sources = ",".join(f"{LABEL_SCHEMA}.{label_table}" for label_table in label_tables)
        d["source_table_t_plus_h"] = label_sources
        d["label_source"] = "actual"
        d["label_weight"] = 1.0

        n_pos = int(d[label_col].sum())
        logger.info(
            "Generated %s from actual labels [%s] for window %s: Churn=%d (%.1f%%) | Active=%d | Total=%d",
            label_col,
            label_sources,
            table_t,
            n_pos,
            100.0 * n_pos / max(len(d), 1),
            len(d) - n_pos,
            len(d),
        )
        return d

    # Rule fallback uses raw order tables strictly after the prediction origin.
    df_tp, table_tp = _load_post_origin_activity(
        engine,
        df_t,
        origin_yymm=end,
        horizon=horizon,
    )
            
    if df_tp.empty:
        return pd.DataFrame()  # censor window này (không có label tương lai)
        
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

    # C1: Tần suất gửi hàng giảm mạnh (Giảm > 50% so với tần suất bình quân frequency)
    # HOẶC giảm > 50% so với tháng liền kề
    c1_drop_avg = (freq_tp > 0) & (item_tp < 0.50 * freq_tp)
    c1_drop_1m  = (item_1m > 0) & (item_tp < 0.50 * item_1m)
    c1 = c1_drop_avg | c1_drop_1m

    # C2: Doanh thu giảm mạnh (Giảm > 50% so với doanh thu bình quân monetary)
    # HOẶC giảm > 50% so với tháng liền kề (revenue_1m_ago)
    c2_drop_avg = (monetary_tp > 0) & (rev_tp < 0.50 * monetary_tp)
    c2_drop_1m  = (rev_1m > 0) & (rev_tp < 0.50 * rev_1m)
    c2 = c2_drop_avg | c2_drop_1m

    # C3: Xu hướng doanh thu cắm đầu (revenue_slope âm) kết hợp doanh thu tháng này thấp
    # Dành cho các khách hàng rớt từ từ nhưng rõ rệt
    c3 = (rev_slope_tp < -1000) & (rev_tp < 0.80 * monetary_tp)  # threshold -1000 để lọc nhiễu nhẹ

    rule_y = (c0 | c1 | c2 | c3).astype(int)

    n_c0    = int(c0.sum())
    n_c1    = int(c1.sum())
    n_c2    = int(c2.sum())
    n_c3    = int(c3.sum())
    n_total = len(rule_y)
    n_pos   = int(rule_y.sum())
    logger.debug(
        "Đã sinh nhãn y_churn_t_plus_%d từ %s: "
        "Tổng Churn=%d (%.1f%%) | C0(hoàn toàn)=%d | C1(tần suất)=%d | C2(doanh thu)=%d | C3(slope)=%d | Active=%d | Total=%d",
        horizon, table_tp,
        n_pos, 100.0 * n_pos / max(n_total, 1),
        n_c0, n_c1, n_c2, n_c3,
        n_total - n_pos, n_total,
    )
    # -----------------------------------------------------------



    baseline_item = pd.concat([freq_tp, item_1m], axis=1).max(axis=1)
    baseline_rev = pd.concat([monetary_tp, rev_1m], axis=1).max(axis=1)

    item_drop = (1.0 - item_tp / baseline_item.replace(0, np.nan)).clip(lower=0, upper=1).fillna(0)
    rev_drop = (1.0 - rev_tp / baseline_rev.replace(0, np.nan)).clip(lower=0, upper=1).fillna(0)
    zero_activity = ((item_tp == 0) & (rev_tp == 0)).astype(float)

    neg_slope = (-rev_slope_tp).clip(lower=0)
    slope_score = pd.Series(0.0, index=df_tp.index)
    if float(neg_slope.max()) > 0:
        slope_score = (neg_slope.rank(pct=True) * (neg_slope > 0)).astype(float)

    baseline_strength = (
        baseline_item.rank(pct=True).fillna(0) * 0.5
        + baseline_rev.rank(pct=True).fillna(0) * 0.5
    )
    eligible = (baseline_item > 0) | (baseline_rev > 0)

    risk_score = (
        0.35 * item_drop
        + 0.35 * rev_drop
        + 0.20 * zero_activity
        + 0.10 * slope_score
    ) * baseline_strength
    risk_score = risk_score.where(eligible, np.nan)

    target_rate_env = os.getenv("RULE_LABEL_TARGET_CHURN_RATE")
    target_source = "observed_label_rate"
    if target_rate_env:
        target_rate = float(target_rate_env)
        target_source = "env_RULE_LABEL_TARGET_CHURN_RATE"
    else:
        target_rate = estimate_observed_label_rate(
            engine,
            horizon=horizon,
            feature_schema=FEATURE_SCHEMA,
        )
        if target_rate is None:
            target_rate = float(os.getenv("RULE_LABEL_FALLBACK_TARGET_CHURN_RATE", "0.15"))
            target_source = "fallback_RULE_LABEL_FALLBACK_TARGET_CHURN_RATE"

    uncertain_band = float(os.getenv("RULE_LABEL_UNCERTAIN_BAND_RATE", "0.20"))
    target_rate = max(0.001, min(float(target_rate), 0.50))
    uncertain_band = max(0.0, min(float(uncertain_band), 0.80))

    scored = risk_score.dropna()
    uncertain_mask = pd.Series(False, index=df_tp.index)
    if scored.empty:
        y = pd.Series(np.nan, index=df_tp.index, dtype="float64")
        pos_cutoff = np.nan
        neg_cutoff = np.nan
    else:
        pos_cutoff = float(scored.quantile(1.0 - target_rate))
        neg_quantile = max(0.0, 1.0 - target_rate - uncertain_band)
        neg_cutoff = float(scored.quantile(neg_quantile))

        y = pd.Series(np.nan, index=df_tp.index, dtype="float64")
        y.loc[risk_score <= neg_cutoff] = 0.0
        y.loc[risk_score >= pos_cutoff] = 1.0
        # Resolve the mid-band with explicit business rules instead of dropping it.
        uncertain_mask = y.isna() & risk_score.notna()
        y.loc[uncertain_mask] = rule_y.loc[uncertain_mask].astype(float)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_uncertain = int(y.isna().sum())
    n_rule_resolved = int(uncertain_mask.sum())
    logger.info(
        "Override fallback labels y_churn_t_plus_%d from %s using adaptive risk-score rules: "
        "Churn=%d (%.1f%% of labeled) | Active=%d | Rule-resolved(mid-band)=%d | Unlabeled(drop)=%d | eligible=%d | total=%d | "
        "target_rate=%.4f (%s) | positive_cutoff=%.6f | negative_cutoff=%.6f | uncertain_band=%.2f",
        horizon, table_tp,
        n_pos, 100.0 * n_pos / max(n_pos + n_neg, 1),
        n_neg, n_rule_resolved, n_uncertain, int(eligible.sum()), len(y),
        target_rate, target_source, pos_cutoff, neg_cutoff, uncertain_band,
    )

    lab = df_tp[["cms_code_enc"]].copy()
    lab[f"y_churn_t_plus_{horizon}"] = y.values

    out = df_t.merge(lab, on="cms_code_enc", how="left")

    # Khách hàng có ở tháng t nhưng KHÔNG xuất hiện trong bảng tương lai (t+h):
    # Không thể xác định được nhãn chính xác — bảng t+h chỉ chứa những người
    # có GIAO DỊCH trong window K của t+h. Vắng mặt có thể do:
    #   (a) Thực sự churn (không giao dịch), hoặc
    #   (b) Dữ liệu chưa được ingest đủ cho window t+h (data lag).
    # Để tránh nhiễu nhãn và inflate churn_ratio ở K lớn → LOẠI BỎ khỏi training.
    missing_mask = out[f"y_churn_t_plus_{horizon}"].isna()
    if missing_mask.any():
        n_dropped = int(missing_mask.sum())
        logger.info(
            "Loại bỏ %d/%d khách hàng không có nhãn trong bảng tương lai %s "
            "(không thể xác định churn/active — tránh inflate churn_ratio).",
            n_dropped, len(out), table_tp,
        )
        out = out[~missing_mask].copy()

    # enforce window_end exists
    if "window_end" not in out.columns:
        out["window_end"] = end

    out["source_table_t"] = table_t
    out["source_table_t_plus_h"] = table_tp
    out["label_source"] = "rule_based"
    out["label_weight"] = float(os.getenv("RULE_LABEL_SAMPLE_WEIGHT", "0.20"))
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
        label_col = f"y_churn_t_plus_{horizon}"
        if label_col in out.columns and "source_table_t_plus_h" in out.columns:
            group_cols = ["source_table_t_plus_h"]
            if "label_source" in out.columns:
                group_cols.append("label_source")
            audit = (
                out.groupby(group_cols, dropna=False)[label_col]
                .agg(total_rows="size", churn_rows="sum")
                .reset_index()
            )
            audit["churn_rate_pct"] = (
                100.0 * audit["churn_rows"] / audit["total_rows"].clip(lower=1)
            )
            logger.info(
                "[LABEL AUDIT K=%d H=%d]\n%s",
                k,
                horizon,
                audit.to_string(index=False),
            )
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
