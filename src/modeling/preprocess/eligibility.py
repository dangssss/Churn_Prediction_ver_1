from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from logging_config import get_logger

logger = get_logger(__name__)

_ITEM_AGO_RE = re.compile(r"^item_(\d+)m_ago$")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int env %s=%r. Using default %d.", name, raw, default)
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float env %s=%r. Using default %.4f.", name, raw, default)
        return float(default)


@dataclass(frozen=True)
class ChurnEligibilityConfig:
    enabled: bool = True
    min_active_months: int = 3
    require_full_window_for_k_le: int = 3
    min_item_sum: float = 3.0
    min_revenue_sum: float = 0.0
    min_avg_revenue_per_item: float = 0.0
    high_value_min_active_months: int = 2
    high_value_min_item_sum: float = 20.0
    high_value_min_revenue_sum: float = 0.0
    high_value_min_avg_revenue_per_item: float = 0.0

    @classmethod
    def from_env(cls) -> "ChurnEligibilityConfig":
        return cls(
            enabled=_env_bool("CHURN_ELIGIBILITY_ENABLED", True),
            min_active_months=max(_env_int("CHURN_ELIGIBILITY_MIN_ACTIVE_MONTHS", 3), 1),
            require_full_window_for_k_le=max(_env_int("CHURN_ELIGIBILITY_REQUIRE_FULL_WINDOW_FOR_K_LE", 3), 0),
            min_item_sum=max(_env_float("CHURN_ELIGIBILITY_MIN_ITEM_SUM", 3.0), 0.0),
            min_revenue_sum=max(_env_float("CHURN_ELIGIBILITY_MIN_REVENUE_SUM", 0.0), 0.0),
            min_avg_revenue_per_item=max(_env_float("CHURN_ELIGIBILITY_MIN_AVG_REVENUE_PER_ITEM", 0.0), 0.0),
            high_value_min_active_months=max(_env_int("CHURN_ELIGIBILITY_HIGH_VALUE_MIN_ACTIVE_MONTHS", 2), 1),
            high_value_min_item_sum=max(_env_float("CHURN_ELIGIBILITY_HIGH_VALUE_MIN_ITEM_SUM", 20.0), 0.0),
            high_value_min_revenue_sum=max(_env_float("CHURN_ELIGIBILITY_HIGH_VALUE_MIN_REVENUE_SUM", 0.0), 0.0),
            high_value_min_avg_revenue_per_item=max(
                _env_float("CHURN_ELIGIBILITY_HIGH_VALUE_MIN_AVG_REVENUE_PER_ITEM", 0.0),
                0.0,
            ),
        )


def _monthly_item_cols(df: pd.DataFrame) -> list[str]:
    lagged: list[tuple[int, str]] = []
    for col in df.columns:
        matched = _ITEM_AGO_RE.match(str(col))
        if matched:
            lagged.append((int(matched.group(1)), col))
    lagged.sort(reverse=True)
    cols = [col for _months_ago, col in lagged]
    if "item_last" in df.columns:
        cols.append("item_last")
    return cols


def _window_size_series(df: pd.DataFrame, k: int | None) -> pd.Series:
    if "window_size" in df.columns:
        return pd.to_numeric(df["window_size"], errors="coerce").fillna(k or 0).astype(int)
    if k is not None:
        return pd.Series(int(k), index=df.index)
    return pd.Series(0, index=df.index)


def add_churn_eligibility_columns(
    df: pd.DataFrame,
    *,
    k: int | None = None,
    config: ChurnEligibilityConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ChurnEligibilityConfig.from_env()
    out = df.copy()
    if out.empty:
        out["churn_active_months_in_window"] = pd.Series(dtype="int64")
        out["churn_required_active_months"] = pd.Series(dtype="int64")
        out["churn_item_sum_for_eligibility"] = pd.Series(dtype="float64")
        out["churn_revenue_sum_for_eligibility"] = pd.Series(dtype="float64")
        out["churn_avg_revenue_per_item_for_eligibility"] = pd.Series(dtype="float64")
        out["is_churn_eligible"] = pd.Series(dtype="int64")
        out["churn_ineligible_reason"] = pd.Series(dtype="object")
        return out

    item_cols = _monthly_item_cols(out)
    if item_cols:
        monthly_items = out[item_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        active_months_calc = monthly_items.gt(0).sum(axis=1).astype(int)
    elif "active_months" in out.columns:
        active_months_calc = pd.to_numeric(out["active_months"], errors="coerce").fillna(0).astype(int)
    else:
        active_months_calc = pd.Series(0, index=out.index)

    window_size = _window_size_series(out, k)
    active_months_feature = (
        pd.to_numeric(out["active_months"], errors="coerce").fillna(active_months_calc).astype(int)
        if "active_months" in out.columns
        else active_months_calc
    )
    active_months = pd.concat([active_months_calc, active_months_feature], axis=1).max(axis=1).astype(int)

    item_sum = (
        pd.to_numeric(out["item_sum"], errors="coerce").fillna(0.0)
        if "item_sum" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    if float(item_sum.max()) <= 0 and item_cols:
        item_sum = monthly_items.sum(axis=1).astype(float)

    revenue_sum = (
        pd.to_numeric(out["revenue_sum"], errors="coerce").fillna(0.0)
        if "revenue_sum" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    avg_revenue_per_item = (
        pd.to_numeric(out["avg_revenue_per_item"], errors="coerce").fillna(0.0)
        if "avg_revenue_per_item" in out.columns
        else revenue_sum / item_sum.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    required_active_months = pd.Series(int(cfg.min_active_months), index=out.index)
    if int(cfg.require_full_window_for_k_le) > 0:
        full_window_mask = window_size.le(int(cfg.require_full_window_for_k_le)) & window_size.gt(0)
        required_active_months = required_active_months.mask(full_window_mask, window_size)
    required_active_months = required_active_months.clip(lower=1)

    regular_active_ok = active_months.ge(required_active_months)
    regular_item_ok = item_sum.ge(float(cfg.min_item_sum))
    regular_revenue_ok = revenue_sum.ge(float(cfg.min_revenue_sum))
    regular_avg_order_ok = avg_revenue_per_item.ge(float(cfg.min_avg_revenue_per_item))
    regular_ok = regular_active_ok & regular_item_ok & regular_revenue_ok & regular_avg_order_ok

    high_value_active_ok = active_months.ge(int(cfg.high_value_min_active_months))
    high_value_item_ok = item_sum.ge(float(cfg.high_value_min_item_sum))
    high_value_revenue_ok = revenue_sum.ge(float(cfg.high_value_min_revenue_sum))
    high_value_avg_order_ok = avg_revenue_per_item.ge(float(cfg.high_value_min_avg_revenue_per_item))
    high_value_ok = (
        high_value_active_ok
        & high_value_item_ok
        & high_value_revenue_ok
        & high_value_avg_order_ok
    )
    eligible = regular_ok | high_value_ok

    default_reasons = np.where(eligible, "eligible", "mixed_eligibility_constraints")
    reasons = np.select(
        [
            high_value_ok & ~regular_ok,
            ~regular_active_ok & ~high_value_active_ok,
            ~regular_item_ok & ~high_value_item_ok,
            ~regular_revenue_ok & ~high_value_revenue_ok,
            ~regular_avg_order_ok & ~high_value_avg_order_ok,
        ],
        [
            "eligible_high_value_exception",
            "insufficient_active_months",
            "insufficient_item_sum",
            "insufficient_revenue_sum",
            "insufficient_avg_revenue_per_item",
        ],
        default=default_reasons,
    )

    out["churn_active_months_in_window"] = active_months.astype(int)
    out["churn_required_active_months"] = required_active_months.astype(int)
    out["churn_item_sum_for_eligibility"] = item_sum.astype(float)
    out["churn_revenue_sum_for_eligibility"] = revenue_sum.astype(float)
    out["churn_avg_revenue_per_item_for_eligibility"] = avg_revenue_per_item.astype(float)
    out["is_churn_eligible"] = eligible.astype(int)
    out["churn_ineligible_reason"] = reasons
    return out


def filter_churn_eligible(
    df: pd.DataFrame,
    *,
    k: int | None = None,
    context: str,
    config: ChurnEligibilityConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ChurnEligibilityConfig.from_env()
    out = add_churn_eligibility_columns(df, k=k, config=cfg)
    if not cfg.enabled or out.empty:
        return out

    eligible_mask = out["is_churn_eligible"].astype(int).eq(1)
    before = len(out)
    after = int(eligible_mask.sum())
    dropped = before - after
    if before:
        reason_counts = (
            out.loc[~eligible_mask, "churn_ineligible_reason"]
            .value_counts(dropna=False)
            .to_dict()
        )
        logger.info(
            "[CHURN ELIGIBILITY][%s] kept=%d/%d dropped=%d min_active_months=%d "
            "full_window_for_k<=%d min_item_sum=%.2f min_revenue_sum=%.2f "
            "min_avg_revenue_per_item=%.2f high_value_min_active_months=%d "
            "high_value_min_item_sum=%.2f high_value_min_revenue_sum=%.2f "
            "high_value_min_avg_revenue_per_item=%.2f high_value_kept=%d reasons=%s",
            context,
            after,
            before,
            dropped,
            int(cfg.min_active_months),
            int(cfg.require_full_window_for_k_le),
            float(cfg.min_item_sum),
            float(cfg.min_revenue_sum),
            float(cfg.min_avg_revenue_per_item),
            int(cfg.high_value_min_active_months),
            float(cfg.high_value_min_item_sum),
            float(cfg.high_value_min_revenue_sum),
            float(cfg.high_value_min_avg_revenue_per_item),
            int((out["churn_ineligible_reason"] == "eligible_high_value_exception").sum()),
            reason_counts,
        )
    return out.loc[eligible_mask].copy()
