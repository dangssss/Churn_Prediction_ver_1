from __future__ import annotations

import os
import re
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
from .eligibility import filter_churn_eligible
from common.business_churn_score import add_business_churn_score_features
from logging_config import get_logger

logger = get_logger(__name__)

_LAGGED_SIGNAL_RE = re.compile(r"^(?P<base>item|revenue|complaint|delay|nodone|order_score|satisfaction)_(?P<lag>\d+)m_ago$")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float env %s=%r. Using default %.4f.", name, raw, default)
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int env %s=%r. Using default %d.", name, raw, default)
        return int(default)


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


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _coalesce_numeric(df: pd.DataFrame, columns: list[str], default: float = 0.0) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            out = out.combine_first(values)
    return out.fillna(float(default))


def _pl1_baseline_3_periods(df: pd.DataFrame, base: str, fallback: pd.Series) -> pd.Series:
    """Return PL1 baseline: mean of the three immediately preceding periods."""
    cols = [f"{base}_last", f"{base}_1m_ago", f"{base}_2m_ago"]
    available = [col for col in cols if col in df.columns]
    if available:
        baseline = df[available].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    elif f"{base}_avg" in df.columns:
        baseline = pd.to_numeric(df[f"{base}_avg"], errors="coerce")
    else:
        baseline = pd.Series(np.nan, index=df.index, dtype="float64")
    return baseline.combine_first(pd.to_numeric(fallback, errors="coerce")).fillna(0.0)


_PL1_SERVICE_CODES = ("c", "e", "m", "p", "r", "u", "l", "q")
_PL1_SERVICE_FAMILY_DEFAULTS = {
    "C": "postal_traditional",
    "E": "domestic_logistics",
    "M": "unknown",
    "P": "domestic_logistics",
    "R": "postal_traditional",
    "U": "international_logistics",
    "L": "international_logistics",
    "Q": "unknown",
}
_PL1_VALID_SERVICE_FAMILIES = {
    "postal_traditional",
    "domestic_logistics",
    "international_logistics",
    "value_added",
    "unknown",
}


def _normalize_pl1_service_family(value: str | None) -> str:
    family = str(value or "unknown").strip().lower()
    if family in _PL1_VALID_SERVICE_FAMILIES:
        return family
    logger.warning("Invalid PL1 service family %r. Falling back to unknown.", value)
    return "unknown"


def _pl1_service_family_for_code(service_code: str | None) -> str:
    code = str(service_code or "").strip().upper()
    default = _PL1_SERVICE_FAMILY_DEFAULTS.get(code, "unknown")
    override = os.getenv(f"CHURN_PL1_SERVICE_FAMILY_{code}") if code else None
    return _normalize_pl1_service_family(override or default)


def _pl1_recency_threshold_days_for_family(family: str) -> int:
    defaults = {
        "postal_traditional": 90,
        "domestic_logistics": 45,
        "international_logistics": 60,
        "value_added": 120,
        "unknown": 60,
    }
    env_names = {
        "postal_traditional": "CHURN_PL1_RECENCY_POSTAL_DAYS",
        "domestic_logistics": "CHURN_PL1_RECENCY_DOMESTIC_LOGISTICS_DAYS",
        "international_logistics": "CHURN_PL1_RECENCY_INTERNATIONAL_LOGISTICS_DAYS",
        "value_added": "CHURN_PL1_RECENCY_VALUE_ADDED_DAYS",
        "unknown": "CHURN_PL1_RECENCY_UNKNOWN_DAYS",
    }
    normalized = _normalize_pl1_service_family(family)
    return max(_env_int(env_names[normalized], defaults[normalized]), 1)


def _dominant_service_from_window(df: pd.DataFrame) -> pd.Series:
    if "dominant_service" in df.columns:
        dominant = df["dominant_service"].astype(str).str.strip().str.upper()
        valid_codes = {code.upper() for code in _PL1_SERVICE_CODES}
        dominant = dominant.where(dominant.isin(valid_codes), "")
    else:
        dominant = pd.Series("", index=df.index, dtype="object")

    unresolved = dominant.eq("")
    service_values: dict[str, pd.Series] = {}
    for code in _PL1_SERVICE_CODES:
        service_values[code.upper()] = _coalesce_numeric(df, [f"ser_{code}_sum", f"ser_{code}"], default=0.0)

    if unresolved.any():
        service_frame = pd.DataFrame(service_values, index=df.index)
        max_service = service_frame.max(axis=1)
        computed = service_frame.idxmax(axis=1).where(max_service.gt(0), "U")
        dominant = dominant.mask(unresolved, computed)

    return dominant.astype(str).str.upper()


def _end_of_yymm(yymm: str | int) -> pd.Timestamp:
    yymm_str = str(yymm).zfill(4)
    yy = int(yymm_str[:2])
    mm = int(yymm_str[2:])
    return pd.Timestamp(year=2000 + yy, month=mm, day=1) + pd.offsets.MonthEnd(0)


def _post_origin_observation_days(origin_yymm: str | int, horizon: int) -> int:
    origin_end = _end_of_yymm(origin_yymm)
    future_end = _end_of_yymm(shift_yymm(origin_yymm, int(horizon)))
    return max(int((future_end - origin_end).days), 0)


def _month_index_yymm(yymm: str | int) -> int:
    yymm_str = str(yymm).zfill(4)
    yy = int(yymm_str[:2])
    mm = int(yymm_str[2:])
    if mm < 1 or mm > 12:
        raise ValueError(f"Invalid YYMM: {yymm}")
    return yy * 12 + (mm - 1)


def _latest_complete_rule_label_yymm() -> int | None:
    """Return latest closed month allowed for rule-based outcome labels."""
    override = (
        os.getenv("CHURN_RULE_LABEL_MAX_FUTURE_YYMM")
        or os.getenv("CHURN_MAX_LABEL_YYMM")
    )
    if override is not None and str(override).strip():
        return int(str(override).strip())
    if _env_bool("CHURN_RULE_LABEL_ALLOW_CURRENT_MONTH", False):
        return None

    tz = os.getenv("CHURN_BUSINESS_TZ", "Asia/Ho_Chi_Minh")
    try:
        now = pd.Timestamp.now(tz=tz)
    except Exception:
        now = pd.Timestamp.utcnow()
    current_yymm = f"{now.year % 100:02d}{now.month:02d}"
    return int(shift_yymm(current_yymm, -1))


def _future_yymms_labelable(future_yymms: list[str]) -> tuple[bool, str]:
    latest_complete = _latest_complete_rule_label_yymm()
    if latest_complete is None:
        return True, "current_month_allowed"
    max_future = max(int(yymm) for yymm in future_yymms)
    if _month_index_yymm(max_future) <= _month_index_yymm(latest_complete):
        return True, f"latest_complete={latest_complete:04d}"
    return False, f"max_future={max_future:04d} latest_complete={latest_complete:04d}"


def _add_temporal_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-side trend/ratio features from observed pre-origin windows only."""
    if df.empty:
        return df
    out = df.copy()
    lagged: dict[str, list[tuple[int, str]]] = {}
    for col in out.columns:
        matched = _LAGGED_SIGNAL_RE.match(str(col))
        if matched:
            lagged.setdefault(matched.group("base"), []).append((int(matched.group("lag")), col))

    for base, pairs in lagged.items():
        pairs.sort()
        lag_cols = [col for _lag, col in pairs]
        mat = out[lag_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if mat.empty:
            continue
        hist_mean = mat.mean(axis=1)
        hist_std = mat.std(axis=1).fillna(0.0)
        hist_max = mat.max(axis=1)
        hist_min = mat.min(axis=1)
        first_col = lag_cols[-1]
        recent_col = lag_cols[0]
        recent = pd.to_numeric(out[recent_col], errors="coerce").fillna(0.0)
        oldest = pd.to_numeric(out[first_col], errors="coerce").fillna(0.0)

        out[f"{base}_hist_mean"] = hist_mean
        out[f"{base}_hist_std"] = hist_std
        out[f"{base}_hist_range"] = hist_max - hist_min
        out[f"{base}_recent_vs_hist_mean"] = _safe_ratio(recent, hist_mean)
        out[f"{base}_recent_drop_from_hist_mean"] = (1.0 - _safe_ratio(recent, hist_mean)).clip(lower=0.0, upper=1.0)
        out[f"{base}_trend_recent_minus_oldest"] = recent - oldest

        last_col = f"{base}_last"
        if last_col in out.columns:
            last = pd.to_numeric(out[last_col], errors="coerce").fillna(0.0)
            out[f"{base}_last_vs_hist_mean"] = _safe_ratio(last, hist_mean)
            out[f"{base}_last_drop_from_hist_mean"] = (1.0 - _safe_ratio(last, hist_mean)).clip(lower=0.0, upper=1.0)

    if {"revenue_last", "item_last"}.issubset(out.columns):
        out["aov_last"] = _safe_ratio(out["revenue_last"], out["item_last"])
    if {"revenue_hist_mean", "item_hist_mean"}.issubset(out.columns):
        out["aov_hist_mean"] = _safe_ratio(out["revenue_hist_mean"], out["item_hist_mean"])
    if {"aov_last", "aov_hist_mean"}.issubset(out.columns):
        out["aov_last_vs_hist_mean"] = _safe_ratio(out["aov_last"], out["aov_hist_mean"])
        out["aov_last_drop_from_hist_mean"] = (1.0 - out["aov_last_vs_hist_mean"]).clip(lower=0.0, upper=1.0)

    return out


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
    labelable, reason = _future_yymms_labelable(future_yymms)
    if not labelable:
        logger.info(
            "Censor fallback origin=%s horizon=%d: future months are not closed for rule labels (%s)",
            origin_yymm,
            horizon,
            reason,
        )
        return pd.DataFrame(), ""
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
    origin["origin_item_baseline_3m"] = _pl1_baseline_3_periods(
        df_origin,
        "item",
        origin["origin_item"],
    )
    origin["origin_revenue_baseline_3m"] = _pl1_baseline_3_periods(
        df_origin,
        "revenue",
        origin["origin_revenue"],
    )
    origin["origin_recency_days"] = _coalesce_numeric(df_origin, ["recency", "recency_days"], default=0.0)
    dominant_service = _dominant_service_from_window(df_origin)
    origin["origin_dominant_service"] = dominant_service
    origin["origin_service_family"] = dominant_service.map(_pl1_service_family_for_code)
    origin["origin_service_recency_threshold_days"] = origin["origin_service_family"].map(
        _pl1_recency_threshold_days_for_family
    )
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
    labelable, reason = _future_yymms_labelable(future_yymms)
    if not labelable:
        logger.info(
            "[PURGED PREFLIGHT] origin=%s H=%d is not labelable yet (%s)",
            origin_yymm,
            horizon,
            reason,
        )
        return False
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
    df_t = add_business_churn_score_features(df_t)
    before_scope = len(df_t)
    df_t = apply_gate(df_t)
    df_t = df_t[df_t["is_active_now"].astype(int).eq(1)].copy()
    after_active = len(df_t)
    df_t = filter_churn_eligible(df_t, k=k, context=f"label_build_k{k}_{end}")
    if df_t.empty:
        logger.info(
            "[LABEL BUILD] %s kept no active eligible rows after active/eligibility filtering "
            "(before=%d after_active=%d)",
            table_t,
            before_scope,
            after_active,
        )
        return pd.DataFrame()

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
    else:
        future_yymms = [shift_yymm(end, offset) for offset in range(1, int(horizon) + 1)]
        logger.info(
            "No supplemental %s label tables available for %s on future months=%s; "
            "using rule-base labels for positives and negatives.",
            LABEL_SCHEMA,
            table_t,
            ",".join(str(yymm) for yymm in future_yymms),
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

    # ---------- PL1 business rule labels ----------
    # PL1 section I/II thresholds are fixed business rules, not adaptive window quantiles.
    from .gating import resolve_now_cols
    cols_tp = resolve_now_cols(df_tp)
    item_tp_col = cols_tp["item_now"]
    rev_tp_col = cols_tp["rev_now"]

    item_tp = pd.to_numeric(df_tp[item_tp_col], errors="coerce").fillna(0.0)
    rev_tp = pd.to_numeric(df_tp[rev_tp_col], errors="coerce").fillna(0.0)

    baseline_item = pd.to_numeric(
        df_tp.get("origin_item_baseline_3m", df_tp.get("origin_item", 0)),
        errors="coerce",
    ).fillna(0.0)
    baseline_rev = pd.to_numeric(
        df_tp.get("origin_revenue_baseline_3m", df_tp.get("origin_revenue", 0)),
        errors="coerce",
    ).fillna(0.0)

    min_base_item = _env_float("CHURN_PL1_MIN_BASE_ITEM", 1.0)
    min_base_revenue = _env_float("CHURN_PL1_MIN_BASE_REVENUE", 0.0)
    has_meaningful_item_base = baseline_item.ge(float(min_base_item))
    has_meaningful_revenue_base = baseline_rev.gt(float(min_base_revenue))
    has_meaningful_base = has_meaningful_item_base | has_meaningful_revenue_base

    item_drop_ratio = (
        1.0 - item_tp / baseline_item.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    rev_drop_ratio = (
        1.0 - rev_tp / baseline_rev.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)

    freq_drop_warning_min = _env_float("CHURN_PL1_FREQUENCY_DROP_WARNING", 0.20)
    freq_drop_high_min = _env_float("CHURN_PL1_FREQUENCY_DROP_HIGH", 0.40)
    revenue_drop_warning_min = _env_float("CHURN_PL1_REVENUE_DROP_WARNING", 0.15)
    revenue_drop_high_min = _env_float("CHURN_PL1_REVENUE_DROP_HIGH", 0.35)

    freq_drop_warning = has_meaningful_item_base & item_drop_ratio.ge(freq_drop_warning_min)
    freq_drop_high = has_meaningful_item_base & item_drop_ratio.ge(freq_drop_high_min)
    revenue_drop_warning = has_meaningful_revenue_base & rev_drop_ratio.ge(revenue_drop_warning_min)
    revenue_drop_high = has_meaningful_revenue_base & rev_drop_ratio.ge(revenue_drop_high_min)
    joint_warning_drop = freq_drop_warning & revenue_drop_warning

    future_active_months = pd.to_numeric(
        df_tp.get("future_active_months", 0), errors="coerce"
    ).fillna(0).astype(int)
    origin_active_months = pd.to_numeric(
        df_tp.get("origin_active_months", 0), errors="coerce"
    ).fillna(0).astype(int)
    expected_future_active_months = origin_active_months.clip(
        lower=1,
        upper=max(int(horizon), 1),
    )
    persistent_activity_drop = future_active_months.lt(expected_future_active_months)

    no_future_activity = item_tp.eq(0) & rev_tp.eq(0) & has_meaningful_base
    observed_gap_days = (
        pd.to_numeric(df_tp.get("origin_recency_days", 0), errors="coerce").fillna(0.0)
        + float(_post_origin_observation_days(end, int(horizon)))
    )
    service_recency_threshold_days = pd.to_numeric(
        df_tp.get("origin_service_recency_threshold_days", 60),
        errors="coerce",
    ).fillna(60).clip(lower=1)
    recency_churn = no_future_activity & observed_gap_days.ge(service_recency_threshold_days)
    recency_audit_only = no_future_activity & ~recency_churn

    warning_drop = freq_drop_warning | revenue_drop_warning
    persistent_warning_drop = warning_drop & persistent_activity_drop
    single_warning_as_positive = _env_bool("CHURN_PL1_SINGLE_WARNING_AS_POSITIVE", True)
    business_drop = (
        freq_drop_high
        | revenue_drop_high
        | joint_warning_drop
        | persistent_warning_drop
        | (single_warning_as_positive & warning_drop)
    )
    rule_y = (recency_churn | business_drop).astype(int)

    label_rule_reason = pd.Series("stable_or_active", index=df_tp.index, dtype="object")
    label_rule_reason.loc[recency_audit_only] = "pl1_no_future_activity_below_recency_threshold_audit_only"
    label_rule_reason.loc[freq_drop_warning & ~business_drop & ~recency_churn] = "pl1_frequency_drop_audit_only"
    label_rule_reason.loc[revenue_drop_warning & ~business_drop & ~recency_churn] = "pl1_revenue_drop_audit_only"
    label_rule_reason.loc[freq_drop_high & ~revenue_drop_high] = "pl1_frequency_drop_high"
    label_rule_reason.loc[revenue_drop_high & ~freq_drop_high] = "pl1_revenue_drop_high"
    label_rule_reason.loc[freq_drop_high & revenue_drop_high] = "pl1_frequency_and_revenue_drop_high"
    label_rule_reason.loc[joint_warning_drop & ~(freq_drop_high | revenue_drop_high)] = (
        "pl1_frequency_and_revenue_drop_warning"
    )
    label_rule_reason.loc[
        persistent_warning_drop & ~(freq_drop_high | revenue_drop_high | joint_warning_drop)
    ] = "pl1_warning_drop_persistent_activity_drop"
    label_rule_reason.loc[recency_churn] = "pl1_recency_churn"

    y = rule_y.astype(float)
    audit_only_mask = label_rule_reason.isin(
        {
            "pl1_frequency_drop_audit_only",
            "pl1_revenue_drop_audit_only",
            "pl1_no_future_activity_below_recency_threshold_audit_only",
        }
    )
    if _env_bool("CHURN_LABEL_DROP_AUDIT_ONLY", False):
        y.loc[audit_only_mask] = np.nan

    churn_label_type = pd.Series("active_or_stable", index=df_tp.index, dtype="object")
    churn_label_type.loc[audit_only_mask] = "uncertain_audit_only"
    churn_label_type.loc[business_drop] = "pl1_business_drop"
    churn_label_type.loc[freq_drop_high | revenue_drop_high] = "pl1_high_drop"
    churn_label_type.loc[joint_warning_drop & ~(freq_drop_high | revenue_drop_high)] = "pl1_joint_warning_drop"
    churn_label_type.loc[recency_churn] = "pl1_recency_churn"
    label_source = pd.Series("rule_negative", index=df_tp.index, dtype="object")
    label_source.loc[rule_y.eq(1)] = "rule_positive"
    label_source.loc[y.isna()] = "excluded_uncertain"

    n_no_future = int(no_future_activity.sum())
    n_recency_churn = int(recency_churn.sum())
    n_recency_audit = int(recency_audit_only.sum())
    n_freq_warning = int(freq_drop_warning.sum())
    n_freq_high = int(freq_drop_high.sum())
    n_revenue_warning = int(revenue_drop_warning.sum())
    n_revenue_high = int(revenue_drop_high.sum())
    n_joint_warning = int(joint_warning_drop.sum())
    n_persistent_warning = int(persistent_warning_drop.sum())
    n_business_drop = int(business_drop.sum())
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_uncertain = int(y.isna().sum())
    logger.info(
        "Generated PL1 rule-base labels y_churn_t_plus_%d from %s: "
        "positive=%d (%.1f%% of labeled) | negative=%d | unlabeled=%d | total=%d | "
        "no_future_activity=%d | recency_churn=%d | recency_audit_only=%d | "
        "frequency_warning=%d | frequency_high=%d | revenue_warning=%d | revenue_high=%d | "
        "joint_warning=%d | persistent_warning=%d | business_drop=%d | "
        "thresholds: freq_warning=%.2f freq_high=%.2f revenue_warning=%.2f revenue_high=%.2f "
        "min_base_item=%.2f min_base_revenue=%.2f single_warning_as_positive=%s",
        horizon,
        table_tp,
        n_pos, 100.0 * n_pos / max(n_pos + n_neg, 1),
        n_neg,
        n_uncertain,
        len(y),
        n_no_future,
        n_recency_churn,
        n_recency_audit,
        n_freq_warning,
        n_freq_high,
        n_revenue_warning,
        n_revenue_high,
        n_joint_warning,
        n_persistent_warning,
        n_business_drop,
        freq_drop_warning_min,
        freq_drop_high_min,
        revenue_drop_warning_min,
        revenue_drop_high_min,
        min_base_item,
        min_base_revenue,
        single_warning_as_positive,
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
    lab["churn_label_type"] = churn_label_type.values
    lab["churn_label_is_uncertain"] = audit_only_mask.astype(int).values
    lab["label_source"] = label_source.values

    out = df_t.merge(lab, on="cms_code_enc", how="left")

    actual_positive_count = 0
    rule_positive_count = int((out[label_col] == 1).sum())
    both_positive_count = 0
    actual_only_count = 0
    rule_only_count = rule_positive_count
    if supplemental_labels is not None:
        out = out.merge(supplemental_labels, on="cms_code_enc", how="left")
        rule_label = pd.to_numeric(out[label_col], errors="coerce")
        actual_label = pd.to_numeric(out["_label_value"], errors="coerce")
        rule_positive_mask = rule_label.eq(1)
        actual_positive_mask = actual_label.eq(1)
        actual_known_mask = actual_label.notna()
        final_positive_mask = rule_positive_mask | actual_positive_mask

        out.loc[final_positive_mask, label_col] = 1
        out.loc[
            actual_known_mask & actual_label.eq(0) & ~rule_positive_mask & out[label_col].isna(),
            label_col,
        ] = 0

        both_mask = actual_positive_mask & rule_positive_mask
        actual_only_mask = actual_positive_mask & ~rule_positive_mask
        rule_only_mask = rule_positive_mask & ~actual_positive_mask

        out.loc[actual_only_mask, "label_source"] = "actual_positive"
        out.loc[rule_only_mask, "label_source"] = "rule_positive"
        out.loc[both_mask, "label_source"] = "actual_and_rule_positive"
        out.loc[actual_only_mask, "churn_label_type"] = "actual_positive"
        out.loc[both_mask, "churn_label_type"] = "actual_and_rule_positive"
        out.loc[actual_positive_mask, "churn_label_is_uncertain"] = 0

        actual_positive_count = int(actual_positive_mask.sum())
        rule_positive_count = int(rule_positive_mask.sum())
        both_positive_count = int(both_mask.sum())
        actual_only_count = int(actual_only_mask.sum())
        rule_only_count = int(rule_only_mask.sum())

    final_labeled = int(out[label_col].notna().sum())
    final_positive = int((out[label_col] == 1).sum())
    final_negative = int((out[label_col] == 0).sum())
    final_churn_rate = final_positive / max(final_labeled, 1)
    logger.info(
        "Final unified labels for %s on %s: labeled=%d positive=%d negative=%d "
        "churn_rate=%.2f%% | actual_positive=%d rule_positive=%d both=%d "
        "actual_only=%d rule_only=%d",
        label_col,
        table_t,
        final_labeled,
        final_positive,
        final_negative,
        100.0 * final_churn_rate,
        actual_positive_count,
        rule_positive_count,
        both_positive_count,
        actual_only_count,
        rule_only_count,
    )

    # Customers still without a final label are outside the usable supervised set.
    missing_mask = out[label_col].isna()
    if missing_mask.any():
        n_dropped = int(missing_mask.sum())
        logger.info(
            "Dropped %d/%d customers without final labels from %s "
            "(typically excluded uncertain rows or censored post-origin activity).",
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
        out = _add_temporal_signal_features(out)
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
    df = add_business_churn_score_features(df)
    kk, start, end = parse_feature_table_name(table_t)

    # ensure window columns exist
    if "window_size" not in df.columns:
        df["window_size"] = int(kk)
    if "window_start" not in df.columns:
        df["window_start"] = int(start)
    if "window_end" not in df.columns:
        df["window_end"] = int(end)

    df["source_table_t"] = table_t

    df = _add_temporal_signal_features(df)

    # gate now (active_now vs churned_now)
    df = apply_gate(df)
    df = clip_and_log_outliers(df)
    return df, table_t, window_end
