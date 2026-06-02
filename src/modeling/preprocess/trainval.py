from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from .dataset import build_dataset_for_k, select_train_val_tables_for_k
from baseline.runner import time_split_train_val_last_month
from .static_features import attach_static, LIFETIME_RATIO_REQUIRED_COLS

def build_train_val_for_main(
    engine: Engine,
    cfg: dict,
    df_static: pd.DataFrame,
    use_static_override=None
) -> tuple:
    k = int(cfg["best_k"])
    h = int(cfg["horizon"])
    use_static_cfg = bool(cfg.get("use_static", False))
    use_static = use_static_cfg if use_static_override is None else bool(use_static_override)
    label_col = f"y_churn_t_plus_{h}"

    tables, _, _ = select_train_val_tables_for_k(engine, k, horizon=h)
    df = build_dataset_for_k(engine, k, horizon=h, limit_rows_each=None, tables=tables)
    if df is None or df.empty:
        raise ValueError(f"Dataset empty for K={k}, H={h}")

    # train churn-risk chỉ trên active_now + có label
    if "is_active_now" in df.columns:
        df = df[df["is_active_now"] == 1].copy()
    df = df.dropna(subset=[label_col]).copy()
    if df.empty or df[label_col].nunique() < 2:
        raise ValueError("Không đủ dữ liệu labeled để train main")

    # ---- (NEW) create lifetime ratio features before main model
    add_ratios = bool(cfg.get("add_lifetime_ratio_features", True))
    if add_ratios:
        if df_static is None:
            raise ValueError("add_lifetime_ratio_features=True nhưng df_static=None")

        before = len(df)
        if use_static:
            # keep all lifetime cols (+ ratios)
            df = attach_static(df, df_static, cols=None, keep_static_cols=True, add_ratios=True)
        else:
            # only attach minimal lifetime cols needed to compute ratios, then drop them
            df = attach_static(df, df_static, cols=LIFETIME_RATIO_REQUIRED_COLS, keep_static_cols=False, add_ratios=True)
        after = len(df)
        if after != before:
            raise ValueError(f"attach_static changed row count: before={before}, after={after}")
    else:
        # legacy behavior: only merge static when config yêu cầu
        if use_static:
            if df_static is None:
                raise ValueError("use_static=True nhưng df_static=None")
            before = len(df)
            df = attach_static(df, df_static)
            after = len(df)
            if after != before:
                raise ValueError(f"attach_static changed row count: before={before}, after={after}")

    # split train/val theo tháng (val = tháng labeled cuối)
    df_tr, df_va, val_month = time_split_train_val_last_month(
        df,
        time_col="window_end",
        horizon=h,
    )
    if df_tr is None or df_tr.empty or df_va is None or df_va.empty:
        raise ValueError("Không đủ tháng để split train/val cho main")

    return df, df_tr, df_va, int(val_month)
