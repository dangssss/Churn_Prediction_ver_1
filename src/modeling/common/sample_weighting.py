from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


def _cfg_value(cfg: dict | None, key: str):
    if not cfg:
        return None
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _cfg_float(cfg: dict | None, key: str, env_name: str, default: float) -> float:
    value = _cfg_value(cfg, key)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
    return _env_float(env_name, default)


def _cfg_bool(cfg: dict | None, key: str, env_name: str, default: bool) -> bool:
    value = _cfg_value(cfg, key)
    if value is not None:
        if isinstance(value, bool):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    return _env_bool(env_name, default)


def _yymm_to_month_index(value) -> float:
    try:
        yymm = int(value)
        year = yymm // 100
        month = yymm % 100
        if month < 1 or month > 12:
            return np.nan
        return float(year * 12 + month)
    except (TypeError, ValueError):
        return np.nan


def label_uncertain_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    mask = pd.Series(False, index=df.index)
    if "churn_label_is_uncertain" in df.columns:
        uncertain = pd.to_numeric(df["churn_label_is_uncertain"], errors="coerce").fillna(0)
        mask = mask | uncertain.astype(float).gt(0)
    if "churn_label_type" in df.columns:
        mask = mask | df["churn_label_type"].astype(str).eq("uncertain_audit_only")
    if "label_rule_reason" in df.columns:
        mask = mask | df["label_rule_reason"].astype(str).str.endswith("_audit_only")
    return mask.fillna(False)


def non_uncertain_label_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    return ~label_uncertain_mask(df).reindex(df.index, fill_value=False)


def build_label_sample_weights(
    df: pd.DataFrame,
    *,
    label_col: str,
    time_col: str = "window_end",
    cfg: dict | None = None,
) -> pd.Series:
    """Build reliability and recency weights without changing labels."""
    weights = pd.Series(
        _cfg_float(cfg, "churn_label_default_sample_weight", "CHURN_LABEL_DEFAULT_SAMPLE_WEIGHT", 1.0),
        index=df.index,
        dtype="float64",
    )

    y = pd.to_numeric(df.get(label_col), errors="coerce")
    label_source = (
        df.get("label_source", pd.Series("", index=df.index))
        .astype(str)
        .str.strip()
        .str.lower()
    )

    actual_w = _cfg_float(
        cfg,
        "churn_label_actual_positive_sample_weight",
        "CHURN_LABEL_ACTUAL_POSITIVE_SAMPLE_WEIGHT",
        1.00,
    )
    both_w = _cfg_float(
        cfg,
        "churn_label_actual_and_rule_positive_sample_weight",
        "CHURN_LABEL_ACTUAL_AND_RULE_POSITIVE_SAMPLE_WEIGHT",
        1.00,
    )
    rule_pos_w = _cfg_float(
        cfg,
        "churn_label_rule_positive_sample_weight",
        "CHURN_LABEL_RULE_POSITIVE_SAMPLE_WEIGHT",
        1.00,
    )
    rule_neg_w = _cfg_float(
        cfg,
        "churn_label_rule_negative_sample_weight",
        "CHURN_LABEL_RULE_NEGATIVE_SAMPLE_WEIGHT",
        1.00,
    )
    audit_w = _cfg_float(
        cfg,
        "churn_label_audit_only_sample_weight",
        "CHURN_LABEL_AUDIT_ONLY_SAMPLE_WEIGHT",
        1.00,
    )

    negative_mask = y.eq(0)
    positive_mask = y.eq(1)
    audit_mask = label_uncertain_mask(df).reindex(df.index, fill_value=False)

    weights.loc[negative_mask & label_source.eq("rule_negative")] = rule_neg_w
    weights.loc[positive_mask & label_source.eq("rule_positive")] = rule_pos_w
    weights.loc[positive_mask & label_source.eq("actual_positive")] = actual_w
    weights.loc[positive_mask & label_source.eq("actual_and_rule_positive")] = both_w
    weights.loc[audit_mask] = audit_w

    if _cfg_bool(cfg, "churn_recency_weight_enabled", "CHURN_RECENCY_WEIGHT_ENABLED", False) and time_col in df.columns:
        months = df[time_col].map(_yymm_to_month_index).astype(float)
        if months.notna().any():
            newest = float(months.max())
            age = (newest - months).clip(lower=0).fillna(0.0)
            half_life = max(
                _cfg_float(
                    cfg,
                    "churn_recency_weight_halflife_months",
                    "CHURN_RECENCY_WEIGHT_HALFLIFE_MONTHS",
                    6.0,
                ),
                1e-6,
            )
            recency = np.power(0.5, age / half_life)
            recency = pd.Series(recency, index=df.index, dtype="float64")
            recency = recency.clip(
                lower=_cfg_float(cfg, "churn_recency_weight_min", "CHURN_RECENCY_WEIGHT_MIN", 0.35),
                upper=_cfg_float(cfg, "churn_recency_weight_max", "CHURN_RECENCY_WEIGHT_MAX", 2.50),
            )
            weights = weights * recency

    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0)
    if _cfg_bool(cfg, "churn_sample_weight_normalize", "CHURN_SAMPLE_WEIGHT_NORMALIZE", True):
        mean_weight = float(weights.mean()) if len(weights) else 1.0
        if mean_weight > 0:
            weights = weights / mean_weight
    return weights.astype("float64")


def _group_summary(
    df: pd.DataFrame,
    *,
    label_col: str,
    weights: pd.Series,
    group_col: str,
) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    tmp = pd.DataFrame({
        group_col: df[group_col].astype(str).fillna(""),
        "label": y,
        "sample_weight": weights.reindex(df.index).astype(float),
    })
    out = (
        tmp.groupby(group_col, dropna=False)
        .agg(
            rows=("label", "size"),
            positives=("label", "sum"),
            weight_mean=("sample_weight", "mean"),
            weight_sum=("sample_weight", "sum"),
        )
        .reset_index()
        .sort_values(["rows", group_col], ascending=[False, True])
    )
    out["positive_rate_pct"] = 100.0 * out["positives"] / out["rows"].clip(lower=1)
    return out


def sample_weight_summary(
    df: pd.DataFrame,
    *,
    label_col: str,
    weights: pd.Series,
) -> dict[str, float]:
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    w = weights.reindex(df.index).astype(float)
    pos = y.eq(1)
    neg = y.eq(0)
    audit = label_uncertain_mask(df).reindex(df.index, fill_value=False)
    total_w = float(w.sum())
    return {
        "rows": float(len(df)),
        "mean": float(w.mean()) if len(w) else 0.0,
        "min": float(w.min()) if len(w) else 0.0,
        "max": float(w.max()) if len(w) else 0.0,
        "positive_rate": float(pos.mean()) if len(pos) else 0.0,
        "weighted_positive_rate": float(w[pos].sum() / total_w) if total_w > 0 else 0.0,
        "positive_weight_mean": float(w[pos].mean()) if pos.any() else 0.0,
        "negative_weight_mean": float(w[neg].mean()) if neg.any() else 0.0,
        "audit_only_rows": float(audit.sum()),
        "audit_only_weight_mean": float(w[audit].mean()) if audit.any() else 0.0,
    }


def format_sample_weight_breakdown(
    df: pd.DataFrame,
    *,
    label_col: str,
    weights: pd.Series,
    group_cols: Iterable[str] = ("label_source", "churn_label_type", "label_rule_reason"),
    max_rows: int = 12,
) -> str:
    sections: list[str] = []
    for col in group_cols:
        summary = _group_summary(df, label_col=label_col, weights=weights, group_col=col)
        if summary.empty:
            continue
        shown = summary.head(max_rows).copy()
        for c in ("weight_mean", "weight_sum", "positive_rate_pct"):
            shown[c] = shown[c].astype(float).round(4)
        sections.append(f"{col}\n{shown.to_string(index=False)}")
    return "\n\n".join(sections)
