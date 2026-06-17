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


def _positive_quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric[(numeric > 0) & numeric.notna()]
    if numeric.empty:
        return 0.0
    return float(numeric.quantile(float(q)))


def _quantile_or_default(values: pd.Series, q: float, default: float = 0.0) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric[numeric.notna()]
    if numeric.empty:
        return float(default)
    return float(numeric.quantile(float(q)))


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
    if "active_months" in df_origin.columns:
        origin["origin_active_months"] = pd.to_numeric(
            df_origin["active_months"], errors="coerce"
        ).fillna(0)
    else:
        origin["origin_active_months"] = 0
    origin = origin.drop_duplicates("cms_code_enc")

    monthly_frames = []
    for offset, table in enumerate(future_tables, start=1):
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
        frame["future_month_offset"] = int(offset)
        monthly_frames.append(frame)

    activity_raw = pd.concat(monthly_frames, ignore_index=True)
    monthly_totals = (
        activity_raw.groupby("future_month_offset", as_index=False)
        .agg(
            active_customers=("cms_code_enc", "nunique"),
            item_count=("item_count", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_values("future_month_offset")
    )
    if not monthly_totals.empty:
        logger.info(
            "[POST-ORIGIN ACTIVITY AUDIT] origin=%s horizon=%d tables=%s\n%s",
            origin_yymm,
            horizon,
            ",".join(f"public.{table}" for table in future_tables),
            monthly_totals.to_string(index=False),
        )

    activity = (
        activity_raw.groupby("cms_code_enc", as_index=False)
        .agg(item_count=("item_count", "sum"), revenue=("revenue", "sum"))
    )
    if activity_raw.empty:
        monthly_wide = pd.DataFrame({"cms_code_enc": pd.Series(dtype=str)})
    else:
        monthly_wide = activity_raw.pivot_table(
            index="cms_code_enc",
            columns="future_month_offset",
            values=["item_count", "revenue"],
            aggfunc="sum",
            fill_value=0,
        )
        monthly_wide.columns = [
            f"{metric}_future_m{int(offset)}"
            for metric, offset in monthly_wide.columns.to_flat_index()
        ]
        monthly_wide = monthly_wide.reset_index()

    out = origin.merge(activity, on="cms_code_enc", how="left")
    out[["item_count", "revenue"]] = out[["item_count", "revenue"]].fillna(0)
    out = out.merge(monthly_wide, on="cms_code_enc", how="left")
    for offset in range(1, horizon + 1):
        for prefix in ("item_count", "revenue"):
            col = f"{prefix}_future_m{offset}"
            if col not in out.columns:
                out[col] = 0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    future_item_cols = [f"item_count_future_m{offset}" for offset in range(1, horizon + 1)]
    future_rev_cols = [f"revenue_future_m{offset}" for offset in range(1, horizon + 1)]
    out["future_active_months"] = (
        (out[future_item_cols].gt(0) | out[future_rev_cols].gt(0)).sum(axis=1).astype(int)
    )
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

    # ---------- Adaptive business rule labels ----------
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
    baseline_aov = (
        baseline_rev / baseline_item.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    current_aov = (
        rev_tp / item_tp.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    item_drop_ratio = (
        1.0 - item_tp / baseline_item.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1)
    rev_drop_ratio = (
        1.0 - rev_tp / baseline_rev.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1)
    aov_drop_ratio = (
        1.0 - current_aov / baseline_aov.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1)

    min_base_item = _positive_quantile(baseline_item, 0.50)
    min_base_revenue = _positive_quantile(baseline_rev, 0.50)
    has_meaningful_item_base = (
        baseline_item.ge(min_base_item) if min_base_item > 0 else baseline_item.gt(0)
    )
    has_meaningful_revenue_base = (
        baseline_rev.ge(min_base_revenue) if min_base_revenue > 0 else baseline_rev.gt(0)
    )
    has_meaningful_value_base = (
        has_meaningful_item_base & has_meaningful_revenue_base & baseline_aov.gt(0)
    )

    item_drop_ratio_min = _positive_quantile(item_drop_ratio[has_meaningful_item_base], 0.75)
    revenue_drop_ratio_min = _positive_quantile(rev_drop_ratio[has_meaningful_revenue_base], 0.75)
    aov_drop_ratio_min = _positive_quantile(aov_drop_ratio[has_meaningful_value_base], 0.75)
    if item_drop_ratio_min <= 0:
        item_drop_ratio_min = _positive_quantile(item_drop_ratio, 0.75) or 1.0
    if revenue_drop_ratio_min <= 0:
        revenue_drop_ratio_min = _positive_quantile(rev_drop_ratio, 0.75) or 1.0
    if aov_drop_ratio_min <= 0:
        aov_drop_ratio_min = _positive_quantile(aov_drop_ratio, 0.75) or 1.0

    low_current_item = _quantile_or_default(item_tp[has_meaningful_item_base], 0.25)
    low_current_revenue = _quantile_or_default(rev_tp[has_meaningful_revenue_base], 0.25)
    negative_slope_cutoff = _quantile_or_default(
        rev_slope_tp[has_meaningful_revenue_base & rev_slope_tp.lt(0)],
        0.25,
        default=0.0,
    )

    future_active_months = pd.to_numeric(
        df_tp.get("future_active_months", 0), errors="coerce"
    ).fillna(0).astype(int)
    origin_active_months = pd.to_numeric(
        df_tp.get("origin_active_months", 0), errors="coerce"
    ).fillna(0).astype(int)
    expected_future_active_months = np.minimum(
        origin_active_months.clip(lower=1),
        max(int(horizon), 1),
    )
    persistent_activity_drop = future_active_months.lt(expected_future_active_months)

    # C0: no future activity from a customer that had a meaningful prior base.
    c0 = (item_tp == 0) & (rev_tp == 0) & (has_meaningful_item_base | has_meaningful_revenue_base)

    # C1: severe order contraction, not just a small relative movement.
    c1 = (
        has_meaningful_item_base
        & item_drop_ratio.ge(item_drop_ratio_min)
        & (item_tp.le(low_current_item) | persistent_activity_drop)
    )

    # C2: severe revenue contraction from a meaningful revenue base.
    c2 = (
        has_meaningful_revenue_base
        & rev_drop_ratio.ge(revenue_drop_ratio_min)
        & (rev_tp.le(low_current_revenue) | persistent_activity_drop)
    )

    # C3: trend deterioration is an audit signal, not a standalone churn label.
    c3 = (
        has_meaningful_revenue_base
        & rev_slope_tp.le(negative_slope_cutoff)
        & rev_drop_ratio.ge(revenue_drop_ratio_min)
    )

    # C4: value collapse catches customers whose average order value falls sharply.
    c4 = (
        has_meaningful_value_base
        & aov_drop_ratio.ge(aov_drop_ratio_min)
        & c2
    )

    # Business churn should be confirmed by both volume and revenue contraction.
    # AOV collapse can confirm revenue churn when future activity also deteriorates.
    business_drop = (c1 & c2) | (c2 & c4 & persistent_activity_drop)
    rule_y = (c0 | business_drop).astype(int)
    label_rule_reason = pd.Series("stable_or_active", index=df_tp.index, dtype="object")
    label_rule_reason.loc[c1 & ~c2 & ~c0] = "item_drop_audit_only"
    label_rule_reason.loc[c2 & ~c1 & ~c0] = "revenue_drop_audit_only"
    label_rule_reason.loc[c3 & ~business_drop & ~c0] = "revenue_slope_audit_only"
    label_rule_reason.loc[c4 & ~business_drop & ~c0] = "aov_drop_audit_only"
    label_rule_reason.loc[business_drop & c1 & c2] = "order_and_revenue_drop"
    label_rule_reason.loc[business_drop & ~(c1 & c2)] = "revenue_value_activity_drop"
    label_rule_reason.loc[c0] = "no_future_activity"

    y = rule_y.astype(float)
    n_c0 = int(c0.sum())
    n_c1 = int(c1.sum())
    n_c2 = int(c2.sum())
    n_c3 = int(c3.sum())
    n_c4 = int(c4.sum())
    n_c1_c2 = int((c1 & c2).sum())
    n_c1_only = int((c1 & ~c2 & ~c3 & ~c4 & ~c0).sum())
    n_c2_only = int((c2 & ~c1 & ~c3 & ~c4 & ~c0).sum())
    n_c3_only = int((c3 & ~c1 & ~c2 & ~c0).sum())
    n_c4_only = int((c4 & ~c1 & ~c2 & ~c0).sum())
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_uncertain = int(y.isna().sum())
    logger.info(
        "Generated rule-base labels y_churn_t_plus_%d from %s: "
        "positive=%d (%.1f%% of labeled) | negative=%d | unlabeled=%d | total=%d | "
        "C0(no activity)=%d | C1(item drop)=%d | C2(revenue drop)=%d | "
        "C3(slope audit)=%d | C4(aov audit)=%d | C1&C2=%d | "
        "C1_only=%d | C2_only=%d | C3_only=%d | C4_only=%d | "
        "adaptive_min_base_item=%.2f adaptive_min_base_revenue=%.2f "
        "adaptive_item_drop_q75=%.2f adaptive_revenue_drop_q75=%.2f "
        "adaptive_aov_drop_q75=%.2f low_current_item_q25=%.2f "
        "low_current_revenue_q25=%.2f negative_slope_q25=%.2f",
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
        n_c4,
        n_c1_c2,
        n_c1_only,
        n_c2_only,
        n_c3_only,
        n_c4_only,
        min_base_item,
        min_base_revenue,
        item_drop_ratio_min,
        revenue_drop_ratio_min,
        aov_drop_ratio_min,
        low_current_item,
        low_current_revenue,
        negative_slope_cutoff,
    )
    reason_audit = (
        pd.DataFrame({"label_rule_reason": label_rule_reason, "label": y})
        .groupby("label_rule_reason", dropna=False)["label"]
        .agg(rows="size", churn_rows="sum")
        .reset_index()
    )
    reason_audit["churn_rate_pct"] = (
        100.0 * reason_audit["churn_rows"] / reason_audit["rows"].clip(lower=1)
    )
    logger.info(
        "[RULE LABEL REASON AUDIT] y_churn_t_plus_%d from %s\n%s",
        horizon,
        table_tp,
        reason_audit.to_string(index=False),
    )

    lab = df_tp[["cms_code_enc"]].copy()
    lab[label_col] = y.values
    lab["label_rule_reason"] = label_rule_reason.values

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
            rates = audit["churn_rate_pct"].astype(float)
            q1 = float(rates.quantile(0.25))
            q2 = float(rates.quantile(0.50))
            q3 = float(rates.quantile(0.75))
            iqr = q3 - q1
            max_idx = rates.idxmax()
            max_row = audit.loc[max_idx]
            logger.info(
                "[LABEL AUDIT K=%d H=%d]\n%s",
                k,
                horizon,
                audit.to_string(index=False),
            )
            logger.info(
                "[LABEL AUDIT SUMMARY K=%d H=%d] windows=%d churn_rate_pct: "
                "min=%.2f q1=%.2f median=%.2f q3=%.2f max=%.2f iqr=%.2f "
                "max_window=%s max_to_median_ratio=%.2f",
                k,
                horizon,
                len(audit),
                float(rates.min()),
                q1,
                q2,
                q3,
                float(rates.max()),
                iqr,
                str(max_row["source_table_t_plus_h"]),
                float(rates.max() / max(q2, 1e-9)),
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
