from __future__ import annotations

from typing import Iterable, Optional, List

import os
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

STATIC_SCHEMA = os.getenv("STATIC_SCHEMA", "data_static")
STATIC_TABLE  = os.getenv("STATIC_TABLE", "cus_lifetime")

# ---- Minimal lifetime columns needed to compute requested ratio features.
# We keep names as "preferred"; attach_static will only select existing columns anyway.
LIFETIME_RATIO_REQUIRED_COLS: List[str] = [
    "lifetime_total_items",
    "lifetime_total_revenue",
    # complaint column sometimes spelled 'complant' in some datasets
    "lifetime_total_complant",
    "lifetime_total_complaint",
    "lifetime_pct_delay",
    "lifetime_pct_successful_item",
    "lifetime_avg_order_score",
    # satisfaction sometimes spelled 'satisfation'
    "lifetime_avg_satisfation",
    "lifetime_avg_satisfaction",
]

def load_cus_lifetime(engine: Engine) -> pd.DataFrame:
    q = text(f'SELECT * FROM "{STATIC_SCHEMA}"."{STATIC_TABLE}"')
    df = pd.read_sql(q, engine)
    if "cms_code_enc" not in df.columns:
        raise KeyError("cus_lifetime thiếu cột cms_code_enc")
    df["cms_code_enc"] = df["cms_code_enc"].astype(str)
    return df

def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
        # allow static_ prefix from collision rename
        sc = f"static_{c}"
        if sc in df.columns:
            return sc
    return None

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _normalize_pct(p: pd.Series) -> pd.Series:
    """Best-effort: allow lifetime_pct_* stored as 0-1 or 0-100."""
    x = _to_num(p).astype(float)
    try:
        mx = x.max(skipna=True)
    except Exception:
        mx = None
    if mx is not None and pd.notna(mx) and mx > 1.5:
        x = x / 100.0
    return x

def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    n = _to_num(numer).astype(float)
    d = _to_num(denom).astype(float)
    d = d.replace([0.0, -0.0], np.nan)
    d = d.where(np.abs(d) > 1e-12, np.nan)
    return n / d

def add_lifetime_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered ratio features requested by user.

    Window ("last") features are expected in the dataset built from data_window tables.
    Lifetime features come from data_static.cus_lifetime and must be merged first (attach_static).

    If any required column is missing, the corresponding ratio feature is skipped.
    """
    d = df.copy()

    # window cols (support common aliases)
    item_last = _find_col(d, ["item_last", "items_last", "total_items_last", "total_item_last"])
    revenue_last = _find_col(d, ["revenue_last", "rev_last", "total_revenue_last"])
    complaint_last = _find_col(d, ["complaint_last", "complant_last", "total_complaint_last", "total_complant_last"])
    delay_last = _find_col(d, ["delay_last", "delays_last", "total_delay_last"])
    nodone_last = _find_col(d, ["nodone_last", "no_done_last", "not_done_last", "undone_last"])
    order_score_last = _find_col(d, ["order_score_last", "avg_order_score_last"])
    satisfaction_last = _find_col(d, ["satisfaction_last", "satisfation_last", "avg_satisfaction_last"])

    # lifetime cols (support spelling variants)
    lt_total_items = _find_col(d, ["lifetime_total_items"])
    lt_total_revenue = _find_col(d, ["lifetime_total_revenue"])
    lt_total_complaint = _find_col(d, ["lifetime_total_complant", "lifetime_total_complaint"])
    lt_pct_delay = _find_col(d, ["lifetime_pct_delay"])
    lt_pct_success = _find_col(d, ["lifetime_pct_successful_item", "lifetime_pct_success_item"])
    lt_avg_order_score = _find_col(d, ["lifetime_avg_order_score"])
    lt_avg_satisfaction = _find_col(d, ["lifetime_avg_satisfation", "lifetime_avg_satisfaction"])

    # 1) item_last / lifetime_total_items
    if item_last and lt_total_items:
        d["ratio_item_last__lifetime_total_items"] = _safe_div(d[item_last], d[lt_total_items])

    # 2) revenue_last / lifetime_total_revenue
    if revenue_last and lt_total_revenue:
        d["ratio_revenue_last__lifetime_total_revenue"] = _safe_div(d[revenue_last], d[lt_total_revenue])

    # 3) complaint_last / lifetime_total_complant
    if complaint_last and lt_total_complaint:
        d["ratio_complaint_last__lifetime_total_complaint"] = _safe_div(d[complaint_last], d[lt_total_complaint])

    # 4) delay_last / (lifetime_pct_delay * lifetime_total_items)
    if delay_last and lt_pct_delay and lt_total_items:
        denom = _normalize_pct(d[lt_pct_delay]) * _to_num(d[lt_total_items])
        d["ratio_delay_last__lifetime_pct_delay_x_total_items"] = _safe_div(d[delay_last], denom)

    # 5) nodone_last / (lifetime_total_items - lifetime_pct_successful_item * lifetime_total_items)
    if nodone_last and lt_pct_success and lt_total_items:
        total = _to_num(d[lt_total_items])
        pct_s = _normalize_pct(d[lt_pct_success])
        denom = total - (pct_s * total)
        d["ratio_nodone_last__lifetime_nodone_items"] = _safe_div(d[nodone_last], denom)

    # 6) order_score_last / lifetime_avg_order_score
    if order_score_last and lt_avg_order_score:
        d["ratio_order_score_last__lifetime_avg_order_score"] = _safe_div(d[order_score_last], d[lt_avg_order_score])

    # 7) satisfaction_last / lifetime_avg_satisfaction
    if satisfaction_last and lt_avg_satisfaction:
        d["ratio_satisfaction_last__lifetime_avg_satisfaction"] = _safe_div(d[satisfaction_last], d[lt_avg_satisfaction])
        # backward-compat alias (typo)
        d["ratio_satisfation_last__lifetime_avg_satisfation"] = d["ratio_satisfaction_last__lifetime_avg_satisfaction"]

    return d

def attach_static(
    df: pd.DataFrame,
    df_static: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
    *,
    keep_static_cols: bool = True,
    add_ratios: bool = False,
) -> pd.DataFrame:
    """Merge cus_lifetime onto dataset by cms_code_enc.

    Params
    - cols: if provided, only attach these lifetime columns (+cms_code_enc).
    - keep_static_cols: if False and cols is provided, drop the attached lifetime columns after creating ratios.
    - add_ratios: if True, create the engineered ratio features after merge.

    Backward compatible: old call attach_static(df, df_static) still works.
    """
    d = df.copy()
    d["cms_code_enc"] = d["cms_code_enc"].astype(str)

    ds = df_static.copy()
    ds["cms_code_enc"] = ds["cms_code_enc"].astype(str)

    if cols is not None:
        wanted = ["cms_code_enc"] + [c for c in cols if c != "cms_code_enc" and c in ds.columns]
        ds = ds[wanted].copy()

    # rename collisions
    static_cols = [c for c in ds.columns if c != "cms_code_enc"]
    static_ren = {}
    for c in static_cols:
        if c in d.columns:
            static_ren[c] = f"static_{c}"
    ds = ds.rename(columns=static_ren)

    out = d.merge(ds, on="cms_code_enc", how="left")

    if add_ratios:
        out = add_lifetime_ratio_features(out)

    if (not keep_static_cols) and (cols is not None):
        # drop the attached lifetime cols (post-rename), keep engineered ratios
        drop_cols: List[str] = []
        for c in cols:
            if c == "cms_code_enc":
                continue
            if c in static_ren:
                drop_cols.append(static_ren[c])
            elif c in out.columns:
                drop_cols.append(c)
            else:
                sc = f"static_{c}"
                if sc in out.columns:
                    drop_cols.append(sc)
        if drop_cols:
            out = out.drop(columns=list(dict.fromkeys(drop_cols)), errors="ignore")

    return out
