from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


SERVICE_CODES = ("c", "e", "m", "p", "r", "u", "l", "q")
VALID_SERVICE_FAMILIES = {
    "postal_traditional",
    "domestic_logistics",
    "international_logistics",
    "value_added",
    "unknown",
}
DEFAULT_SERVICE_FAMILY = {
    "C": "postal_traditional",
    "E": "domestic_logistics",
    "M": "unknown",
    "P": "domestic_logistics",
    "R": "postal_traditional",
    "U": "international_logistics",
    "L": "international_logistics",
    "Q": "unknown",
}


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


def _num(df: pd.DataFrame, *cols: str, default: float = 0.0) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in cols:
        if col in df.columns:
            out = out.combine_first(pd.to_numeric(df[col], errors="coerce"))
    return out.fillna(float(default))


def _normalize_family(value: str | None) -> str:
    family = str(value or "unknown").strip().lower()
    return family if family in VALID_SERVICE_FAMILIES else "unknown"


def _service_family_for_code(code: str | None) -> str:
    svc = str(code or "").strip().upper()
    default = DEFAULT_SERVICE_FAMILY.get(svc, "unknown")
    override = os.getenv(f"CHURN_PL2_SERVICE_FAMILY_{svc}") or os.getenv(f"CHURN_PL1_SERVICE_FAMILY_{svc}")
    return _normalize_family(override or default)


def _dominant_service(df: pd.DataFrame) -> pd.Series:
    valid_codes = {code.upper() for code in SERVICE_CODES}
    if "dominant_service" in df.columns:
        dominant = df["dominant_service"].astype(str).str.strip().str.upper()
        dominant = dominant.where(dominant.isin(valid_codes), "")
    else:
        dominant = pd.Series("", index=df.index, dtype="object")

    unresolved = dominant.eq("")
    service_values = {
        code.upper(): _num(df, f"ser_{code}_sum", f"ser_{code}", default=0.0)
        for code in SERVICE_CODES
    }
    if unresolved.any():
        service_frame = pd.DataFrame(service_values, index=df.index)
        max_service = service_frame.max(axis=1)
        computed = service_frame.idxmax(axis=1).where(max_service.gt(0), "U")
        dominant = dominant.mask(unresolved, computed)
    return dominant.astype(str).str.upper()


def _family_threshold(family: str, suffix: str, default: float) -> float:
    env_key = f"CHURN_PL2_R_{family.upper()}_{suffix}"
    return _env_float(env_key, default)


def _recency_score(recency: pd.Series, family: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    defaults = {
        "postal_traditional": (20.0, 45.0),
        "domestic_logistics": (10.0, 25.0),
        "international_logistics": (20.0, 40.0),
        "value_added": (30.0, 60.0),
        "unknown": (15.0, 60.0),
    }
    low = pd.Series(index=recency.index, dtype="float64")
    high = pd.Series(index=recency.index, dtype="float64")
    for fam, (low_default, high_default) in defaults.items():
        mask = family.eq(fam)
        low.loc[mask] = _family_threshold(fam, "LOW_DAYS", low_default)
        high.loc[mask] = _family_threshold(fam, "HIGH_DAYS", high_default)
    low = low.fillna(defaults["unknown"][0]).clip(lower=0.0)
    high = high.fillna(defaults["unknown"][1]).clip(lower=1.0)
    high = high.where(high.gt(low), low + 1.0)
    score = 1.0 + 4.0 * ((recency - low) / (high - low)).clip(lower=0.0, upper=1.0)
    return score.round(1), low, high


def _frequency_score(frequency: pd.Series) -> pd.Series:
    t1 = _env_float("CHURN_PL2_F_SCORE_1_MIN", 20.0)
    t2 = _env_float("CHURN_PL2_F_SCORE_2_MIN", 10.0)
    t3 = _env_float("CHURN_PL2_F_SCORE_3_MIN", 5.0)
    t4 = _env_float("CHURN_PL2_F_SCORE_4_MIN", 2.0)
    return pd.Series(
        np.select(
            [frequency.ge(t1), frequency.ge(t2), frequency.ge(t3), frequency.ge(t4)],
            [1.0, 2.0, 3.0, 4.0],
            default=5.0,
        ),
        index=frequency.index,
        dtype="float64",
    )


def _monetary_score(ratio: pd.Series) -> pd.Series:
    t1 = _env_float("CHURN_PL2_M_SCORE_1_MIN_RATIO", 1.20)
    t2 = _env_float("CHURN_PL2_M_SCORE_2_MIN_RATIO", 0.90)
    t3 = _env_float("CHURN_PL2_M_SCORE_3_MIN_RATIO", 0.60)
    t4 = _env_float("CHURN_PL2_M_SCORE_4_MIN_RATIO", 0.30)
    return pd.Series(
        np.select(
            [ratio.ge(t1), ratio.ge(t2), ratio.ge(t3), ratio.ge(t4)],
            [1.0, 2.0, 3.0, 4.0],
            default=5.0,
        ),
        index=ratio.index,
        dtype="float64",
    )


def _three_level_score(value: pd.Series, *, low_risk, medium_risk, high_risk) -> pd.Series:
    return pd.Series(
        np.select([low_risk(value), medium_risk(value), high_risk(value)], [1.0, 3.0, 5.0], default=np.nan),
        index=value.index,
        dtype="float64",
    )


def _voc_score(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    components: list[pd.Series] = []
    complaint = _num(df, "complaint_last", "complaint_avg", default=np.nan)
    if complaint.notna().any():
        components.append(
            _three_level_score(
                complaint,
                low_risk=lambda s: s.le(0),
                medium_risk=lambda s: s.gt(0) & s.le(2),
                high_risk=lambda s: s.ge(3),
            )
        )

    pct_complaint = _num(df, "pct_complaint", "pct_complaint_per_item", default=np.nan)
    if pct_complaint.notna().any():
        components.append(
            _three_level_score(
                pct_complaint,
                low_risk=lambda s: s.lt(0.05),
                medium_risk=lambda s: s.ge(0.05) & s.le(0.15),
                high_risk=lambda s: s.gt(0.15),
            )
        )

    satisfaction = _num(df, "satisfaction_last", "satisfaction_avg", "satisfation_last", default=np.nan)
    if satisfaction.notna().any():
        components.append(
            _three_level_score(
                satisfaction,
                low_risk=lambda s: s.ge(4.0),
                medium_risk=lambda s: s.ge(3.0) & s.lt(4.0),
                high_risk=lambda s: s.lt(3.0),
            )
        )

    order_score = _num(df, "order_score_last", "order_score_avg", default=np.nan)
    if order_score.notna().any():
        components.append(
            _three_level_score(
                order_score,
                low_risk=lambda s: s.ge(4.0),
                medium_risk=lambda s: s.ge(3.0) & s.lt(4.0),
                high_risk=lambda s: s.lt(3.0),
            )
        )

    if not components:
        default = _env_float("CHURN_PL2_VOC_DEFAULT_SCORE", 3.0)
        return pd.Series(default, index=df.index, dtype="float64"), pd.Series(0, index=df.index, dtype="int64")

    component_frame = pd.concat(components, axis=1)
    signal_count = component_frame.notna().sum(axis=1).astype(int)
    default = _env_float("CHURN_PL2_VOC_DEFAULT_SCORE", 3.0)
    score = component_frame.mean(axis=1, skipna=True).fillna(default).round(1)
    return score.astype(float), signal_count


def _risk_level(score: pd.Series) -> tuple[pd.Series, pd.Series]:
    level = pd.Series("low", index=score.index, dtype="object")
    code = pd.Series(1, index=score.index, dtype="int64")
    medium = score.ge(2.1) & score.le(3.0)
    high = score.ge(3.1) & score.le(4.0)
    very_high = score.ge(4.1)
    level.loc[medium] = "medium"
    level.loc[high] = "high"
    level.loc[very_high] = "very_high"
    code.loc[medium] = 2
    code.loc[high] = 3
    code.loc[very_high] = 4
    return level, code


@dataclass(frozen=True)
class BusinessChurnWeights:
    r: float = 0.35
    f: float = 0.30
    m: float = 0.25
    voc: float = 0.10

    @classmethod
    def from_env(cls) -> "BusinessChurnWeights":
        return cls(
            r=_env_float("CHURN_PL2_WEIGHT_R", 0.35),
            f=_env_float("CHURN_PL2_WEIGHT_F", 0.30),
            m=_env_float("CHURN_PL2_WEIGHT_M", 0.25),
            voc=_env_float("CHURN_PL2_WEIGHT_VOC", 0.10),
        )


def add_business_churn_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add PL2 business churn score features computed from the current feature window only."""
    out = df.copy()
    if out.empty:
        for col in (
            "business_churn_score",
            "business_r_score",
            "business_f_score",
            "business_m_score",
            "business_voc_score",
            "business_risk_level_code",
            "business_recency_ratio_to_high_threshold",
            "business_monetary_ratio_to_group",
            "business_voc_signal_count",
        ):
            out[col] = pd.Series(dtype="float64")
        out["business_risk_level"] = pd.Series(dtype="object")
        out["business_service_family"] = pd.Series(dtype="object")
        out["business_dominant_service"] = pd.Series(dtype="object")
        return out

    dominant = _dominant_service(out)
    family = dominant.map(_service_family_for_code).astype(str)
    recency = _num(out, "recency", "recency_days", default=0.0).clip(lower=0.0)
    r_score, r_low_days, r_high_days = _recency_score(recency, family)

    frequency = _num(out, "frequency", "item_avg", "item_last", default=0.0).clip(lower=0.0)
    f_score = _frequency_score(frequency)

    monetary = _num(out, "monetary", "revenue_avg", "revenue_last", default=0.0).clip(lower=0.0)
    grouping_cols = []
    monetary_for_group_avg = monetary.where(monetary.gt(0))
    tmp_data = {"monetary": monetary_for_group_avg, "business_service_family": family}
    if "window_end" in out.columns:
        tmp_data["window_end"] = out["window_end"]
        grouping_cols.append("window_end")
    tmp = pd.DataFrame(tmp_data, index=out.index)
    grouping_cols.append("business_service_family")
    group_avg = tmp.groupby(grouping_cols, dropna=False)["monetary"].transform("mean")
    global_avg = float(monetary_for_group_avg.mean()) if monetary_for_group_avg.notna().any() else 0.0
    group_avg = group_avg.where(group_avg.gt(0), global_avg)
    monetary_ratio = (monetary / group_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    m_score = _monetary_score(monetary_ratio)

    voc_score, voc_signal_count = _voc_score(out)

    weights = BusinessChurnWeights.from_env()
    weight_sum = weights.r + weights.f + weights.m + weights.voc
    if _env_bool("CHURN_PL2_NORMALIZE_WEIGHTS", True) and weight_sum > 0:
        wr, wf, wm, wv = weights.r / weight_sum, weights.f / weight_sum, weights.m / weight_sum, weights.voc / weight_sum
    else:
        wr, wf, wm, wv = weights.r, weights.f, weights.m, weights.voc

    score = (wr * r_score + wf * f_score + wm * m_score + wv * voc_score).clip(lower=1.0, upper=5.0).round(2)
    risk_level, risk_code = _risk_level(score)

    out["business_dominant_service"] = dominant
    out["business_service_family"] = family
    out["business_recency_threshold_low_days"] = r_low_days.astype(float)
    out["business_recency_threshold_high_days"] = r_high_days.astype(float)
    out["business_recency_ratio_to_high_threshold"] = (
        recency / r_high_days.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    out["business_monetary_ratio_to_group"] = monetary_ratio.astype(float)
    out["business_voc_signal_count"] = voc_signal_count.astype(int)
    out["business_r_score"] = r_score.astype(float)
    out["business_f_score"] = f_score.astype(float)
    out["business_m_score"] = m_score.astype(float)
    out["business_voc_score"] = voc_score.astype(float)
    out["business_churn_score"] = score.astype(float)
    out["business_risk_level"] = risk_level
    out["business_risk_level_code"] = risk_code.astype(int)
    return out
