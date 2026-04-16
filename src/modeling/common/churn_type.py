from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# ---------------------------
# YYMM helpers
# ---------------------------

def yymm_to_ym(yymm: int) -> tuple[int,int]:
    y = yymm // 100
    m = yymm % 100
    return y, m

def ym_to_yymm(y: int, m: int) -> int:
    return y * 100 + m

def add_months_yymm(yymm: int, n: int) -> int:
    """Add n months to an integer YYMM (e.g., 2510). Handles year boundaries."""
    y, m = yymm_to_ym(int(yymm))
    total = (y * 12 + (m - 1)) + int(n)
    y2 = total // 12
    m2 = (total % 12) + 1
    return ym_to_yymm(y2, m2)

def prev_yymm(yymm: int, n: int = 1) -> int:
    return add_months_yymm(yymm, -int(n))

# ---------------------------
# Column utilities
# ---------------------------

def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _mean_of_cols(df: pd.DataFrame, cols: List[str]) -> tuple[pd.Series, pd.Series]:
    """Return (mean, n_used) per row across available columns."""
    avail = [c for c in cols if c in df.columns]
    if not avail:
        nan = pd.Series(np.nan, index=df.index)
        zero = pd.Series(0, index=df.index)
        return nan, zero
    mat = df[avail].apply(pd.to_numeric, errors="coerce")
    n_used = mat.notna().sum(axis=1).astype(int)
    mean = mat.mean(axis=1, skipna=True)
    return mean, n_used

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce").astype(float)
    b = pd.to_numeric(b, errors="coerce").astype(float)
    out = a / b.replace({0.0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)

def _as_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.fillna(default)

# ---------------------------
# Rule configuration
# ---------------------------

@dataclass
class ChurnTypeThresholds:
    # spike thresholds (reactive)
    spike_pct: float = 0.15          # +15%
    # drop thresholds (proactive / experience)
    drop_item_pct: float = 0.50      # item_last < 50% avg prev
    drop_revenue_pct: float = 0.50
    drop_experience_pct: float = 0.15  # satisfaction/order_score drop 15%
    # minimum baseline to consider
    min_prev_item: float = 5.0
    min_prev_events: float = 1.0

    # proactive behavior thresholds
    cv_item_high: float = 1.0
    dominant_service_ratio_high: float = 0.85
    service_types_low: int = 1
    tenure_low: int = 6

    # value drop (avg revenue per item)
    drop_value_pct: float = 0.15

def _monthly_cols(base: str, months: List[int]) -> List[str]:
    return [f"{base}_{i}m_ago" for i in months]

def _resolve_metric_base(metric: str) -> str:
    # normalize common typo for satisfaction
    if metric == "satisfaction":
        return "satisfaction"
    return metric

# ---------------------------
# Core analysis
# ---------------------------

def analyze_proact_react(
    df: pd.DataFrame,
    *,
    best_k: int,
    mode: str,
    thresholds: ChurnTypeThresholds | None = None,
) -> pd.DataFrame:
    """Compute proactive/reactive flags + reasons.

    mode:
      - 'active': reference month is *_last, baseline uses {1,2,3}m_ago (bounded by K-1)
      - 'churned': reference month is *_1m_ago (pre-churn), baseline uses {2,3,4}m_ago (bounded by K-1)
    """
    if thresholds is None:
        thresholds = ChurnTypeThresholds()

    d = df.copy()

    if "window_end" in d.columns:
        d["window_end"] = pd.to_numeric(d["window_end"], errors="coerce").astype("Int64")

    # -------- choose reference + baseline month indices
    k = int(best_k)
    max_ago = max(0, k - 1)

    if mode not in ("active", "churned"):
        raise ValueError("mode must be 'active' or 'churned'")

    if mode == "active":
        ref_suffix = "last"
        prev_months = [i for i in [1, 2, 3] if i <= max_ago]
        if "window_end" in d.columns:
            d["ref_month"] = d["window_end"].astype("Int64")
        else:
            d["ref_month"] = pd.Series([pd.NA] * len(d), index=d.index, dtype="Int64")
    else:
        ref_suffix = "1m_ago"
        prev_months = [i for i in [2, 3, 4] if i <= max_ago]
        if "window_end" in d.columns:
            d["ref_month"] = d["window_end"].apply(lambda x: prev_yymm(int(x), 1) if pd.notna(x) else pd.NA).astype("Int64")
        else:
            d["ref_month"] = pd.Series([pd.NA] * len(d), index=d.index, dtype="Int64")

    # -------- resolve satisfaction column family (typo tolerant)
    # If dataset contains satisfation_*, treat as satisfaction_*
    for suf in ["last"] + [f"{i}m_ago" for i in range(1, max_ago + 1)]:
        bad = f"satisfation_{suf}"
        good = f"satisfaction_{suf}"
        if good not in d.columns and bad in d.columns:
            d[good] = d[bad]

    # Metrics used for month-over-month rules
    metrics = ["item", "revenue", "complaint", "delay", "nodone", "order_score", "satisfaction"]

    # Compute reference values and baseline means
    for m in metrics:
        base = _resolve_metric_base(m)
        ref_col = f"{base}_{ref_suffix}"
        if ref_col not in d.columns:
            d[ref_col] = np.nan

        prev_cols = _monthly_cols(base, prev_months)
        mean_prev, n_prev = _mean_of_cols(d, prev_cols)
        d[f"avg_prev_{base}"] = mean_prev
        d[f"n_prev_{base}"] = n_prev

    # Derived monthly rates (ref) + baseline rates
    item_ref = _as_num(d[f"item_{ref_suffix}"], 0.0)
    d["_item_ref"] = item_ref

    def _rate_ref(n: pd.Series) -> pd.Series:
        return _safe_div(_as_num(n, 0.0), item_ref.where(item_ref > 0, 1.0))

    d["_pct_complaint_ref"] = _rate_ref(d[f"complaint_{ref_suffix}"])
    d["_pct_delay_ref"] = _rate_ref(d[f"delay_{ref_suffix}"])
    d["_pct_nodone_ref"] = _rate_ref(d[f"nodone_{ref_suffix}"])

    # Baseline rates (mean counts / mean items)
    item_prev = _as_num(d["avg_prev_item"], np.nan)
    d["_pct_complaint_prev"] = _safe_div(_as_num(d["avg_prev_complaint"], np.nan), item_prev)
    d["_pct_delay_prev"] = _safe_div(_as_num(d["avg_prev_delay"], np.nan), item_prev)
    d["_pct_nodone_prev"] = _safe_div(_as_num(d["avg_prev_nodone"], np.nan), item_prev)

    # Value per item (ref vs baseline)
    revenue_ref = _as_num(d[f"revenue_{ref_suffix}"], 0.0)
    d["_rpi_ref"] = _safe_div(revenue_ref, item_ref.where(item_ref > 0, 1.0))

    # baseline rpi as mean of revenue_i/item_i across prev months (where item_i>0)
    rpi_cols = []
    for i in prev_months:
        i_col = f"item_{i}m_ago"
        r_col = f"revenue_{i}m_ago"
        if i_col in d.columns and r_col in d.columns:
            rpi_cols.append((_safe_div(_as_num(d[r_col], np.nan), _as_num(d[i_col], np.nan).where(_as_num(d[i_col], np.nan) > 0, np.nan))))
    if rpi_cols:
        mat = pd.concat(rpi_cols, axis=1)
        d["_rpi_prev"] = mat.mean(axis=1, skipna=True)
    else:
        d["_rpi_prev"] = np.nan

    # ---------------------------
    # Evaluate rules -> reasons
    # ---------------------------

    reactive_reasons: List[List[str]] = [[] for _ in range(len(d))]
    proactive_reasons: List[List[str]] = [[] for _ in range(len(d))]

    def _append(mask: pd.Series, bucket: List[List[str]], code: str):
        idx = np.where(mask.fillna(False).to_numpy())[0]
        for j in idx:
            bucket[j].append(code)

    # --- Reactive: A/B/C groups
    # A: complaint spike
    _append(
        (_as_num(d[f"complaint_{ref_suffix}"], 0.0) > (1 + thresholds.spike_pct) * _as_num(d["avg_prev_complaint"], np.nan))
        & (_as_num(d["avg_prev_complaint"], 0.0) >= thresholds.min_prev_events),
        reactive_reasons,
        "R_COMPLAINT_SPIKE",
    )
    # B: delay spike
    _append(
        (_as_num(d[f"delay_{ref_suffix}"], 0.0) > (1 + thresholds.spike_pct) * _as_num(d["avg_prev_delay"], np.nan))
        & (_as_num(d["avg_prev_delay"], 0.0) >= thresholds.min_prev_events),
        reactive_reasons,
        "R_DELAY_SPIKE",
    )
    # B: nodone spike
    _append(
        (_as_num(d[f"nodone_{ref_suffix}"], 0.0) > (1 + thresholds.spike_pct) * _as_num(d["avg_prev_nodone"], np.nan))
        & (_as_num(d["avg_prev_nodone"], 0.0) >= thresholds.min_prev_events),
        reactive_reasons,
        "R_NODONE_SPIKE",
    )
    # C: satisfaction drop
    _append(
        (_as_num(d[f"satisfaction_{ref_suffix}"], np.nan) < (1 - thresholds.drop_experience_pct) * _as_num(d["avg_prev_satisfaction"], np.nan)),
        reactive_reasons,
        "R_SATISFACTION_DROP",
    )
    # C: order_score drop
    _append(
        (_as_num(d[f"order_score_{ref_suffix}"], np.nan) < (1 - thresholds.drop_experience_pct) * _as_num(d["avg_prev_order_score"], np.nan)),
        reactive_reasons,
        "R_ORDER_SCORE_DROP",
    )

    # Supportive reactive: complaint diversity high
    if "complaint_diversity" in d.columns:
        _append(_as_num(d["complaint_diversity"], 0.0) >= 3.0, reactive_reasons, "R_COMPLAINT_DIVERSITY_HIGH")

    # --- Proactive: D/E/F/G groups
    # D: item drop vs avg prev
    _append(
        (item_ref < thresholds.drop_item_pct * _as_num(d["avg_prev_item"], np.nan))
        & (_as_num(d["avg_prev_item"], 0.0) >= thresholds.min_prev_item),
        proactive_reasons,
        "P_ITEM_DROP",
    )
    # D: revenue drop vs avg prev
    _append(
        (revenue_ref < thresholds.drop_revenue_pct * _as_num(d["avg_prev_revenue"], np.nan))
        & (_as_num(d["avg_prev_revenue"], 0.0) > 0.0),
        proactive_reasons,
        "P_REVENUE_DROP",
    )
    # F: value per item drop
    _append(
        (_as_num(d["_rpi_ref"], np.nan) < (1 - thresholds.drop_value_pct) * _as_num(d["_rpi_prev"], np.nan)),
        proactive_reasons,
        "P_VALUE_PER_ITEM_DROP",
    )

    # E: trends / volatility
    if "item_slope" in d.columns:
        _append(_as_num(d["item_slope"], 0.0) < 0.0, proactive_reasons, "P_ITEM_TREND_DOWN")
    if "revenue_slope" in d.columns:
        _append(_as_num(d["revenue_slope"], 0.0) < 0.0, proactive_reasons, "P_REVENUE_TREND_DOWN")
    if "cv_item" in d.columns:
        _append(_as_num(d["cv_item"], 0.0) >= thresholds.cv_item_high, proactive_reasons, "P_CV_ITEM_HIGH")

    # F: service diversity low / dominance high
    if "service_types_used" in d.columns:
        _append(_as_num(d["service_types_used"], 0.0) <= thresholds.service_types_low, proactive_reasons, "P_SERVICE_TYPES_LOW")
    if "dominant_service_ratio" in d.columns:
        _append(_as_num(d["dominant_service_ratio"], 0.0) >= thresholds.dominant_service_ratio_high, proactive_reasons, "P_DOMINANT_SERVICE_HIGH")

    # G: tenure low (needs lifetime join)
    if "tenure" in d.columns:
        _append(_as_num(d["tenure"], 999.0) < thresholds.tenure_low, proactive_reasons, "P_TENURE_LOW")

    # Build flags + churn_type label
    reactive_flag = pd.Series([1 if len(x) > 0 else 0 for x in reactive_reasons], index=d.index)
    proactive_flag = pd.Series([1 if len(x) > 0 else 0 for x in proactive_reasons], index=d.index)

    # Optional: enforce proactive as "no strong reactive" (pure proactive). We expose both.
    proactive_pure_flag = ((proactive_flag == 1) & (reactive_flag == 0)).astype(int)

    churn_type = []
    for r, p in zip(reactive_flag.tolist(), proactive_flag.tolist()):
        if r == 1 and p == 1:
            churn_type.append("mixed")
        elif r == 1:
            churn_type.append("reactive")
        elif p == 1:
            churn_type.append("proactive")
        else:
            churn_type.append(None)

    d["reactive_flag"] = reactive_flag.astype(int)
    d["proactive_flag"] = proactive_flag.astype(int)
    d["proactive_pure_flag"] = proactive_pure_flag.astype(int)
    d["churn_type"] = churn_type

    # Join reason strings for DB / downstream
    d["reactive_reasons"] = ["|".join(x) if x else None for x in reactive_reasons]
    d["proactive_reasons"] = ["|".join(x) if x else None for x in proactive_reasons]

    # Top-3 reason_1..3: prioritize reactive then proactive
    top_reasons = []
    for rr, pr in zip(reactive_reasons, proactive_reasons):
        merged = rr + [x for x in pr if x not in rr]
        top_reasons.append(merged[:3])

    d["reason_1"] = [xs[0] if len(xs) > 0 else None for xs in top_reasons]
    d["reason_2"] = [xs[1] if len(xs) > 1 else None for xs in top_reasons]
    d["reason_3"] = [xs[2] if len(xs) > 2 else None for xs in top_reasons]

    # Clean up temporary columns (keep avg_prev_* for evidence)
    # We keep avg_prev_* + n_prev_* because very useful for audit.
    for c in ["_item_ref","_pct_complaint_ref","_pct_delay_ref","_pct_nodone_ref","_pct_complaint_prev","_pct_delay_prev","_pct_nodone_prev","_rpi_ref","_rpi_prev"]:
        if c in d.columns:
            d.drop(columns=[c], inplace=True)

    return d
