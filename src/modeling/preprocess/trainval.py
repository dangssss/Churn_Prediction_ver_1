from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from .dataset import build_dataset_for_k
from baseline.runner import time_split_train_val_last_month
from baseline.runner import time_series_purged_splits
from .static_features import attach_static, LIFETIME_RATIO_REQUIRED_COLS


def _build_main_dataset(
    engine: Engine,
    cfg: dict,
    df_static: pd.DataFrame,
    use_static_override=None,
) -> tuple[pd.DataFrame, str]:
    k = int(cfg["best_k"])
    h = int(cfg["horizon"])
    use_static_cfg = bool(cfg.get("use_static", False))
    use_static = use_static_cfg if use_static_override is None else bool(use_static_override)
    label_col = f"y_churn_t_plus_{h}"

    df = build_dataset_for_k(engine, k, horizon=h, limit_rows_each=None)
    if df is None or df.empty:
        raise ValueError(f"Dataset empty for K={k}, H={h}")

    if "is_active_now" in df.columns:
        df = df[df["is_active_now"] == 1].copy()
    df = df.dropna(subset=[label_col]).copy()
    if df.empty or df[label_col].nunique() < 2:
        raise ValueError("Not enough labeled data to train main model")

    add_ratios = bool(cfg.get("add_lifetime_ratio_features", True))
    if add_ratios:
        if df_static is None:
            raise ValueError("add_lifetime_ratio_features=True but df_static=None")

        before = len(df)
        if use_static:
            df = attach_static(df, df_static, cols=None, keep_static_cols=True, add_ratios=True)
        else:
            df = attach_static(
                df,
                df_static,
                cols=LIFETIME_RATIO_REQUIRED_COLS,
                keep_static_cols=False,
                add_ratios=True,
            )
        after = len(df)
        if after != before:
            raise ValueError(f"attach_static changed row count: before={before}, after={after}")
    elif use_static:
        if df_static is None:
            raise ValueError("use_static=True but df_static=None")
        before = len(df)
        df = attach_static(df, df_static)
        after = len(df)
        if after != before:
            raise ValueError(f"attach_static changed row count: before={before}, after={after}")

    return df, label_col


def build_train_val_for_main(
    engine: Engine,
    cfg: dict,
    df_static: pd.DataFrame,
    use_static_override=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    h = int(cfg["horizon"])
    df, _label_col = _build_main_dataset(
        engine,
        cfg,
        df_static,
        use_static_override=use_static_override,
    )

    df_tr, df_va, val_month = time_split_train_val_last_month(
        df,
        time_col="window_end",
        horizon=h,
    )
    if df_tr is None or df_tr.empty or df_va is None or df_va.empty:
        raise ValueError("Not enough months for main train/val split")

    return df, df_tr, df_va, int(val_month)


def build_walk_forward_for_main(
    engine: Engine,
    cfg: dict,
    df_static: pd.DataFrame,
    use_static_override=None,
) -> tuple[pd.DataFrame, list[dict]]:
    h = int(cfg["horizon"])
    df, _label_col = _build_main_dataset(
        engine,
        cfg,
        df_static,
        use_static_override=use_static_override,
    )
    folds = time_series_purged_splits(
        df,
        time_col="window_end",
        horizon=h,
    )
    if not folds:
        raise ValueError("Not enough months for main walk-forward validation")
    return df, folds
