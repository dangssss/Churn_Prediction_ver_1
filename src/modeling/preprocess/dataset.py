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
from .feature_columns import non_feature_columns
from .label_tables import LABEL_SCHEMA, label_tables_for_horizon, load_label_keys
from logging_config import get_logger

logger = get_logger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float env %s=%r. Using default %.4f.", name, raw, default)
        return float(default)


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


def _has_post_origin_activity_tables(
    engine: Engine,
    *,
    origin_yymm: str | int,
    horizon: int,
) -> bool:
    """Return whether raw order tables exist for every required future month."""
    from sqlalchemy import text

    future_yymms = [shift_yymm(origin_yymm, offset) for offset in range(1, int(horizon) + 1)]
    with engine.connect() as conn:
        return all(
            conn.execute(
                text("SELECT to_regclass(:table_name)"),
                {"table_name": f"public.bccp_orderitem_{yymm}"},
            ).scalar()
            is not None
            for yymm in future_yymms
        )


def preflight_purged_train_val_for_k(
    engine: Engine,
    k: int,
    *,
    horizon: int,
) -> tuple[int, int]:
    """Validate that a K candidate has train and validation origins after purging."""
    tables = list_tables_for_k(engine, int(k))
    if not tables:
        raise ValueError(f"No feature tables for K={k}")

    labelable = []
    for table in tables:
        _, _, end = parse_feature_table_name(table)
        has_fallback = _has_post_origin_activity_tables(
            engine,
            origin_yymm=end,
            horizon=int(horizon),
        )
        if has_fallback:
            labelable.append((table, int(end)))

    if not labelable:
        raise ValueError(f"No labelable feature tables for K={k}, H={horizon}")

    validation_origin_count = max(1, int(os.getenv("VALIDATION_ORIGIN_COUNT", "2")))
    validation_months = sorted({end for _, end in labelable})[-validation_origin_count:]
    val_month = max(validation_months)
    train_max_month = int(shift_yymm(str(min(validation_months)), -int(horizon)))
    train_tables = [
        table
        for table, end in labelable
        if end <= train_max_month
    ]
    min_train_origins = int(os.getenv("BASELINE_MIN_PURGED_TRAIN_ORIGINS", "2"))
    train_origins = {
        end
        for _, end in labelable
        if end <= train_max_month
    }
    if len(train_origins) < min_train_origins:
        raise ValueError(
            f"Insufficient purged training origins for K={k}, H={horizon}: "
            f"origins={len(train_origins)}, required>={min_train_origins}, "
            f"val_month={val_month}, train_origin_max={train_max_month}"
        )

    logger.info(
        "[PURGED PREFLIGHT] K=%d H=%d val_month=%d train_origin_max=%d "
        "validation_months=%s available_tables=%d train_tables=%d train_origins=%d "
        "required_origins=%d",
        k,
        horizon,
        val_month,
        train_max_month,
        ",".join(str(m) for m in validation_months),
        len(tables),
        len(train_tables),
        len(train_origins),
        min_train_origins,
    )
    return val_month, train_max_month


def clip_and_log_outliers(df: pd.DataFrame, percentile_lower: float = 0.1, percentile_upper: float = 99.9) -> pd.DataFrame:
    df_clipped = df.copy()
    outlier_summary = []

    # Skip metadata and label columns.
    skip_cols = non_feature_columns()

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

        # Clip only extreme outliers to avoid flattening sparse or binary columns.
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
            "[OUTLIER DETECTION] Clipped extreme outliers for %d columns. Examples: %s",
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
        raise ValueError("table_t does not belong to K")

    df_t = load_feature_table(engine, table_t, limit=limit)

    label_col = f"y_churn_t_plus_{horizon}"
    label_tables = label_tables_for_horizon(engine, end, horizon)
    supplemental_labels: pd.DataFrame | None = None
    if label_tables:
        if "cms_code_enc" not in df_t.columns:
            raise KeyError("Missing cms_code_enc to join label")

        d = df_t.copy()
        d["cms_code_enc"] = d["cms_code_enc"].astype(str).str.strip()
        labels = pd.concat(
            [load_label_keys(engine, label_table) for label_table in label_tables],
            ignore_index=True,
        ).drop_duplicates()

        label_value = pd.Series(np.nan, index=d.index, dtype="float64")
        cms_map = (
            labels.dropna(subset=["cms_code_enc"])
            .assign(cms_code_enc=lambda x: x["cms_code_enc"].astype(str).str.strip())
            .groupby("cms_code_enc")["_label_value"]
            .max()
        )
        cms_match = d["cms_code_enc"].map(cms_map)
        label_value = label_value.combine_first(cms_match.astype("float64"))

        if labels["crm_code_enc"].notna().any():
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

            crm_map = (
                labels.dropna(subset=["crm_code_enc"])
                .assign(crm_code_enc=lambda x: x["crm_code_enc"].astype(str).str.strip())
                .groupby("crm_code_enc")["_label_value"]
                .max()
            )
            crm_match = crm_series.map(crm_map)
            label_value = pd.concat(
                [label_value, crm_match.astype("float64")],
                axis=1,
            ).max(axis=1, skipna=True)

        supplemental_labels = d[["cms_code_enc"]].copy()
        supplemental_labels["_label_value"] = label_value
        supplemental_labels = (
            supplemental_labels.dropna(subset=["_label_value"])
            .groupby("cms_code_enc", as_index=False)["_label_value"]
            .max()
        )
        n_matched = int(len(supplemental_labels))
        n_pos = int(supplemental_labels["_label_value"].astype(int).sum()) if n_matched else 0
        logger.info(
            "Loaded supplemental label keys for %s on window %s: matched_rows=%d | "
            "positive_rows=%d (%.1f%% of matched, %.1f%% of total) | unmatched_rows=%d | total_rows=%d",
            label_col,
            table_t,
            n_matched,
            n_pos,
            100.0 * n_pos / max(n_matched, 1),
            100.0 * n_pos / max(len(d), 1),
            len(d) - n_matched,
            len(d),
        )

    # Rule fallback uses raw order tables strictly after the prediction origin.
    df_tp, table_tp = _load_post_origin_activity(
        engine,
        df_t,
        origin_yymm=end,
        horizon=horizon,
    )

    if df_tp.empty:
        return pd.DataFrame()  # censor this window: no future labels/signals

    if "cms_code_enc" not in df_t.columns or "cms_code_enc" not in df_tp.columns:
        raise KeyError("Missing cms_code_enc to join label")

    # ---------- Multi-signal rule labels (C0 OR C1 OR C2 OR C3) ----------
    # All signals are computed from post-origin activity only; no leakage from df_t.

    from .gating import resolve_now_cols
    cols_tp = resolve_now_cols(df_tp)
    item_tp_col = cols_tp["item_now"]
    rev_tp_col  = cols_tp["rev_now"]

    item_tp = pd.to_numeric(df_tp[item_tp_col], errors="coerce").fillna(0)
    rev_tp  = pd.to_numeric(df_tp[rev_tp_col],  errors="coerce").fillna(0)

    # Precomputed feature-engineering columns.
    freq_tp = pd.to_numeric(df_tp.get("frequency", 0), errors="coerce").fillna(0)
    monetary_tp = pd.to_numeric(df_tp.get("monetary", 0), errors="coerce").fillna(0)
    rev_slope_tp = pd.to_numeric(df_tp.get("revenue_slope", 0), errors="coerce").fillna(0)

    rev_1m = pd.to_numeric(df_tp.get("revenue_1m_ago", 0), errors="coerce").fillna(0)
    item_1m = pd.to_numeric(df_tp.get("item_1m_ago", 0), errors="coerce").fillna(0)

    baseline_item = pd.concat([freq_tp, item_1m], axis=1).max(axis=1)
    baseline_rev = pd.concat([monetary_tp, rev_1m], axis=1).max(axis=1)
    item_drop_ratio = (
        1.0 - item_tp / baseline_item.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1)
    rev_drop_ratio = (
        1.0 - rev_tp / baseline_rev.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1)

    min_base_item = _env_float("RULE_LABEL_MIN_BASE_ITEM", 5.0)
    min_base_revenue = _env_float("RULE_LABEL_MIN_BASE_REVENUE", 500_000.0)
    item_drop_ratio_min = _env_float("RULE_LABEL_ITEM_DROP_RATIO", 0.70)
    revenue_drop_ratio_min = _env_float("RULE_LABEL_REVENUE_DROP_RATIO", 0.70)
    max_current_item = _env_float("RULE_LABEL_MAX_CURRENT_ITEM", 2.0)
    max_current_revenue_ratio = _env_float("RULE_LABEL_MAX_CURRENT_REVENUE_RATIO", 0.30)
    slope_drop_ratio_min = _env_float("RULE_LABEL_SLOPE_DROP_RATIO", 0.60)
    slope_min_abs = _env_float("RULE_LABEL_REVENUE_SLOPE_MIN_ABS", 100_000.0)

    has_meaningful_item_base = baseline_item.ge(min_base_item)
    has_meaningful_revenue_base = baseline_rev.ge(min_base_revenue)

    # C0: no future activity from a customer that had a meaningful prior base.
    c0 = (item_tp == 0) & (rev_tp == 0) & (has_meaningful_item_base | has_meaningful_revenue_base)

    # C1: severe order contraction, not just a small relative movement.
    c1 = (
        has_meaningful_item_base
        & item_drop_ratio.ge(item_drop_ratio_min)
        & item_tp.le(max_current_item)
    )

    # C2: severe revenue contraction from a meaningful revenue base.
    c2 = (
        has_meaningful_revenue_base
        & rev_drop_ratio.ge(revenue_drop_ratio_min)
        & rev_tp.le(max_current_revenue_ratio * baseline_rev)
    )

    # C3: trend deterioration is only a churn signal when both trend and actual drop agree.
    c3 = (
        has_meaningful_revenue_base
        & rev_slope_tp.le(-slope_min_abs)
        & rev_drop_ratio.ge(slope_drop_ratio_min)
        & item_drop_ratio.ge(0.50)
    )

    business_drop = (c1 & c2) | c3
    rule_y = (c0 | business_drop).astype(int)

    y = rule_y.astype(float)
    n_c0 = int(c0.sum())
    n_c1 = int(c1.sum())
    n_c2 = int(c2.sum())
    n_c3 = int(c3.sum())
    n_c1_c2 = int((c1 & c2).sum())
    n_c1_only = int((c1 & ~c2 & ~c3 & ~c0).sum())
    n_c2_only = int((c2 & ~c1 & ~c3 & ~c0).sum())
    n_c3_only = int((c3 & ~c1 & ~c2 & ~c0).sum())
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_uncertain = int(y.isna().sum())
    logger.info(
        "Generated rule-base labels y_churn_t_plus_%d from %s: "
        "positive=%d (%.1f%% of labeled) | negative=%d | unlabeled=%d | total=%d | "
        "C0(no activity)=%d | C1(item drop)=%d | C2(revenue drop)=%d | C3(slope)=%d | "
        "C1&C2=%d | C1_only=%d | C2_only=%d | C3_only=%d | "
        "min_base_item=%.2f min_base_revenue=%.2f item_drop_ratio=%.2f revenue_drop_ratio=%.2f",
        horizon,
        table_tp,
        n_pos, 100.0 * n_pos / max(n_pos + n_neg, 1),
        n_neg,
        n_uncertain,
        len(y),
        n_c0,
        n_c1,
        n_c2,
        n_c3,
        n_c1_c2,
        n_c1_only,
        n_c2_only,
        n_c3_only,
        min_base_item,
        min_base_revenue,
        item_drop_ratio_min,
        revenue_drop_ratio_min,
    )

    lab = df_tp[["cms_code_enc"]].copy()
    lab[label_col] = y.values

    out = df_t.merge(lab, on="cms_code_enc", how="left")

    if supplemental_labels is not None:
        out = out.merge(supplemental_labels, on="cms_code_enc", how="left")
        label_known_mask = out["_label_value"].notna()
        out.loc[label_known_mask, label_col] = out.loc[label_known_mask, "_label_value"].astype(int)
        final_labeled = int(out[label_col].notna().sum())
        final_positive = int((out[label_col] == 1).sum())
        final_negative = int((out[label_col] == 0).sum())
        final_churn_rate = final_positive / max(final_labeled, 1)
        logger.info(
            "Final unified labels for %s on %s: labeled=%d positive=%d negative=%d churn_rate=%.2f%%",
            label_col,
            table_t,
            final_labeled,
            final_positive,
            final_negative,
            100.0 * final_churn_rate,
        )

    # Customers that cannot be matched to post-origin activity are unlabeled.
    # Drop them to avoid noisy fallback labels and inflated churn ratios.
    missing_mask = out[label_col].isna()
    if missing_mask.any():
        n_dropped = int(missing_mask.sum())
        logger.info(
            "Dropped %d/%d customers without post-origin labels from %s "
            "(cannot determine churn/active without inflating churn_ratio).",
            n_dropped, len(out), table_tp,
        )
        out = out[~missing_mask].copy()

    # enforce window_end exists
    if "window_end" not in out.columns:
        out["window_end"] = end

    out["source_table_t"] = table_t
    out["source_table_t_plus_h"] = table_tp

    if supplemental_labels is not None:
        out = out.drop(columns=["_label_value"])

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
