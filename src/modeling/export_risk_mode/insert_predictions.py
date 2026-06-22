from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from preprocess.feature_columns import feature_columns
from main_model.xgb_utils import (
    safe_to_category,
    predict_proba_best_iteration,
    date_col_to_ordinal,
    is_date_like_col,
)


def _env_first(*names: str) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return None


def _operating_mode(cfg: dict) -> str:
    raw = (
        _env_first("CHURN_OPERATING_MODE", "MODEL_OPERATING_MODE")
        or cfg.get("operating_mode")
        or "percentile"
    )
    value = str(raw).strip().lower().replace("-", "_")
    if value in {"probability", "probability_threshold", "proba", "proba_threshold"}:
        return "probability"
    if value in {"percentile", "top_percentile", "rank", "top_tail"}:
        return "percentile"
    print(f"WARNING: Invalid operating mode {raw!r}. Falling back to percentile.")
    return "percentile"


def _percentile_cutoff(cfg: dict, risk_threshold: float | None) -> float:
    raw = (
        _env_first("CHURN_OPERATING_RISK_THRESHOLD_PCT", "MODEL_OPERATING_RISK_THRESHOLD_PCT")
        or cfg.get("operating_risk_threshold_pct")
        or risk_threshold
        or 90.0
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        print(f"WARNING: Invalid operating percentile cutoff {raw!r}. Using 90.")
        value = 90.0
    if value <= 1.0:
        value *= 100.0
    return min(max(value, 0.0), 100.0)


def _probability_threshold(cfg: dict, risk_threshold: float | None) -> float:
    raw = (
        _env_first("CHURN_OPERATING_PROBABILITY_THRESHOLD", "MODEL_OPERATING_PROBABILITY_THRESHOLD")
        or cfg.get("operating_probability_threshold")
    )
    if raw is None and risk_threshold is not None and float(risk_threshold) <= 1.0:
        raw = risk_threshold
    if raw is None:
        raw = cfg.get("best_threshold", cfg.get("main_threshold", 0.5))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        print(f"WARNING: Invalid operating probability threshold {raw!r}. Using 0.5.")
        value = 0.5
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    return min(max(value, 0.0), 1.0)


def make_predictions(
    model: Any,
    df_data: pd.DataFrame,
    cfg: dict,
    metadata: dict,
    risk_threshold: float | None = None,
) -> pd.DataFrame:
    """Make predictions using trained XGBoost model."""
    h = int(cfg["horizon"])
    label_col = f"y_churn_t_plus_{h}"

    # Prepare features
    meta_feat_cols = metadata.get("feat_cols")
    if isinstance(meta_feat_cols, list) and meta_feat_cols:
        feat_cols = [c for c in meta_feat_cols if c in df_data.columns]
    else:
        feat_cols = feature_columns(df_data, label_col=label_col)

    X = df_data[feat_cols].copy()

    # Get metadata
    cat_cols = list(metadata.get("cat_cols") or [])
    date_cols = list(metadata.get("date_cols") or [])
    feature_name_map = metadata.get("feature_name_map") or {}

    # Detect date-like cols that were mistakenly put in cat_cols by older model bundles
    # (handles bundles trained before the date_cols fix)
    extra_date_cols = [
        c for c in cat_cols
        if c in X.columns and c not in date_cols and is_date_like_col(X[c])
    ]
    if extra_date_cols:
        print(f"   Auto-detected {len(extra_date_cols)} date col(s) in cat_cols, converting to ordinal: {extra_date_cols}")
        date_cols = date_cols + extra_date_cols
        cat_cols = [c for c in cat_cols if c not in extra_date_cols]

    # Type conversion — date cols (ordinal numeric), then categorical, then numeric
    for c in date_cols:
        if c in X.columns:
            X[c] = date_col_to_ordinal(X[c])

    for c in cat_cols:
        if c in X.columns:
            X[c] = safe_to_category(X[c])

    for c in feat_cols:
        if c not in cat_cols and c not in date_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Predict
    try:
        # Pad missing columns: if model was trained with K=12 but scoring data only has K=11,
        # fill missing *_Nm_ago columns with 0 so XGBoost doesn't crash
        _feat_names_raw = getattr(model, "feature_names_in_", None)
        if _feat_names_raw is not None:
            model_features = list(_feat_names_raw)
        elif feature_name_map:
            model_features = [feature_name_map.get(c, c) for c in metadata.get("feat_cols", [])]
        else:
            model_features = metadata.get("feat_cols") or None

        if model_features is not None:
            X_pred = X.rename(columns=feature_name_map) if feature_name_map else X.copy()
            missing = [c for c in model_features if c not in X_pred.columns]
            if missing:
                print(f"   Padding {len(missing)} missing feature(s): {missing[:5]}...")
                for c in missing:
                    X_pred[c] = 0
            X_pred = X_pred[list(model_features)]
            prob = predict_proba_best_iteration(model, X_pred)[:, 1]
        elif feature_name_map:
            X_renamed = X.rename(columns=feature_name_map)
            prob = predict_proba_best_iteration(model, X_renamed)[:, 1]
        else:
            prob = predict_proba_best_iteration(model, X)[:, 1]
    except Exception as e:
        print(f"Prediction error: {e}")
        print("   Fallback to standard predict_proba")
        prob = model.predict_proba(X)[:, 1]

    # Combine results. `churn_rate` is kept for CRM/output compatibility, but its
    # value is the model churn probability expressed as a 0-100 percentage.
    df_out = df_data.copy()
    df_out["churn_probability"] = prob
    df_out["model_probability_pct"] = (prob * 100).round(6)
    df_out["churn_rate"] = df_out["model_probability_pct"].round(2)

    # Keep both raw probability and population percentile. The operating mode
    # decides which one turns into the CRM risk flag.
    df_out["risk_score"] = df_out["churn_rate"]
    df_out["risk_percentile_pct"] = _score_percentile_pct(df_out["churn_probability"])
    operating_mode = _operating_mode(cfg)
    if operating_mode == "probability":
        proba_threshold = _probability_threshold(cfg, risk_threshold)
        df_out["risk_flag"] = (df_out["churn_probability"] >= proba_threshold).astype(int)
        df_out["operating_decision_mode"] = "probability"
        df_out["operating_threshold_value"] = proba_threshold
    else:
        percentile_cutoff = _percentile_cutoff(cfg, risk_threshold)
        df_out["risk_flag"] = (df_out["risk_percentile_pct"] >= percentile_cutoff).astype(int)
        df_out["operating_decision_mode"] = "percentile"
        df_out["operating_threshold_value"] = percentile_cutoff

    return df_out


def _score_percentile_pct(scores: pd.Series | np.ndarray) -> pd.Series:
    score_series = pd.Series(scores).astype(float)
    return score_series.rank(method="first", pct=True).mul(100.0)


def filter_risk_predictions(
    df_predictions: pd.DataFrame,
    risk_threshold: float,
) -> pd.DataFrame:
    """Keep the top-tail customers by model score percentile.

    The CLI argument is still named risk-threshold-pct for backwards
    compatibility. A value of 95 means "customers at or above the 95th score
    percentile", not "churn_probability >= 95%".
    """
    if df_predictions.empty:
        return df_predictions.copy()

    if "risk_flag" in df_predictions.columns:
        flags = pd.to_numeric(df_predictions["risk_flag"], errors="coerce").fillna(0)
        return df_predictions[flags.astype(int) == 1].copy()

    threshold = float(risk_threshold)
    if "risk_percentile_pct" not in df_predictions.columns:
        if "churn_probability" in df_predictions.columns:
            percentiles = _score_percentile_pct(df_predictions["churn_probability"])
        elif "churn_rate" in df_predictions.columns:
            percentiles = _score_percentile_pct(df_predictions["churn_rate"])
        else:
            raise KeyError("Missing score column for risk percentile filtering")
        df_predictions = df_predictions.copy()
        df_predictions["risk_percentile_pct"] = percentiles

    return df_predictions[df_predictions["risk_percentile_pct"] >= threshold].copy()


# ---------------------------------------------------------------------------
# SHAP-based reason engine
# ---------------------------------------------------------------------------

# Map từ feature name → bucket index (1-8) tương ứng với 8 reason template
# B1=số bưu gửi, B2=khiếu nại, B3=giao muộn, B4=không hoàn thành,
# B5=biến động, B6=giá trị đơn, B7=đa dạng dịch vụ, B8=khách mới
FEATURE_TO_BUCKET: dict[str, int] = {
    # B1 — Số bưu gửi giảm
    "item_last": 1, "item_slope": 1, "item_avg": 1, "item_sum": 1,
    "cv_item": 5,   # cv → B5 (biến động)
    "item_range": 5, "frequency": 1,
    # B2 — Khiếu nại tăng
    "complaint_last": 2, "complaint_avg": 2, "complaint_slope": 2,
    "pct_complaint": 2, "pct_complaint_per_item": 2, "complaint_sum": 2, "complaint_diversity": 2,
    # B3 — Giao muộn tăng
    "delay_last": 3, "pct_delay": 3, "avg_delayday": 3, "delay_sum": 3,
    # B4 — Không hoàn thành
    "nodone_last": 4, "pct_noaccepted": 4, "pct_refund": 4, "pct_lost_order": 4, "nodone_sum": 4,
    # B5 — Biến động đơn hàng cao
    "cv_revenue": 5, "revenue_range": 5, "item_std": 5, "revenue_std": 5,
    # B6 — Giá trị đơn giảm
    "revenue_last": 6, "avg_revenue_per_item": 6, "revenue_slope": 6,
    "revenue_avg": 6, "monetary": 6, "revenue_sum": 6,
    # B7 — Giảm đa dạng dịch vụ
    "service_types_used": 7, "dominant_service_ratio": 7,
    # B8 — Khách hàng mới
    "tenure": 8, "recency": 8, "active_months": 8, "inactive_months": 8,
}

# Prefix matching cho các ratio features tổng hợp (vd: ratio_item_last__lifetime_total_items)
_BUCKET_PREFIX_MAP: list[tuple[str, int]] = [
    ("ratio_item",       1),
    ("ratio_revenue",    6),
    ("ratio_complaint",  2),
    ("ratio_delay",      3),
    ("ratio_nodone",     4),
    ("ratio_satisf",     6),
    ("ratio_order",      6),
]


def _get_bucket(feature_name: str) -> int | None:
    """Trả về bucket id cho một feature, hoặc None nếu không map được."""
    b = FEATURE_TO_BUCKET.get(feature_name)
    if b is not None:
        return b
    for prefix, bid in _BUCKET_PREFIX_MAP:
        if feature_name.startswith(prefix):
            return bid
    return None


REASON_SLOTS = 3


def _num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _avg_prev_3m_active(df: pd.DataFrame, base_col: str) -> pd.Series:
    cols = [f"{base_col}_{i}m_ago" for i in [1, 2, 3]]
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(0.0, index=df.index, dtype="float64")
    mat = df[available].apply(pd.to_numeric, errors="coerce")
    return mat.where(mat > 0).mean(axis=1, skipna=True).fillna(0)


def _safe_ratio_delta_pct(metric: float, baseline: float, *, decrease: bool = False) -> float | None:
    if baseline <= 0:
        return None
    if decrease:
        return float(1 - metric / baseline)
    return float(metric / baseline - 1)


def _candidate(
    *,
    priority: float,
    code: str,
    text: str,
    metric: float,
    baseline: float | None = None,
    delta: float | None = None,
    delta_pct: float | None = None,
    severity: float | None = None,
) -> dict:
    if delta is None and baseline is not None:
        delta = float(metric - baseline)
    if severity is None:
        severity = float(priority + max(delta_pct or 0.0, 0.0))
    return {
        "priority": float(priority),
        "code": code,
        "text": text,
        "metric": None if pd.isna(metric) else float(metric),
        "baseline": None if baseline is None or pd.isna(baseline) else float(baseline),
        "delta": None if delta is None or pd.isna(delta) else float(delta),
        "delta_pct": None if delta_pct is None or pd.isna(delta_pct) else float(delta_pct),
        "severity": None if severity is None or pd.isna(severity) else float(severity),
    }


def _build_reason_candidates(
    df: pd.DataFrame,
    df_static: pd.DataFrame,
) -> tuple[pd.DataFrame, list[list[dict]], pd.Series]:
    d = df.copy()

    if "tenure" not in d.columns and "cms_code_enc" in d.columns and "tenure" in df_static.columns:
        tenure_map = df_static[["cms_code_enc", "tenure"]].drop_duplicates("cms_code_enc")
        d = d.merge(tenure_map, on="cms_code_enc", how="left")

    item_last = _num_series(d, "item_last")
    item_1m_ago = _num_series(d, "item_1m_ago")
    complaint_last = _num_series(d, "complaint_last")
    delay_last = _num_series(d, "delay_last")
    nodone_last = _num_series(d, "nodone_last")
    revenue_last = _num_series(d, "revenue_last")
    cv_item = _num_series(d, "cv_item")
    service_types = _num_series(d, "service_types_used")
    service_types_prev = _num_series(d, "service_types_used_prev") if "service_types_used_prev" in d.columns else service_types
    tenure = _num_series(d, "tenure", 999)

    avg_item_3m = _avg_prev_3m_active(d, "item")
    avg_complaint_3m = _avg_prev_3m_active(d, "complaint")
    avg_delay_3m = _avg_prev_3m_active(d, "delay")
    avg_nodone_3m = _avg_prev_3m_active(d, "nodone")
    avg_revenue_3m = _avg_prev_3m_active(d, "revenue")

    rpi_last = np.where(item_last > 0, revenue_last / item_last, 0)
    rpi_3m = np.where(avg_item_3m > 0, avg_revenue_3m / avg_item_3m, 0)
    active_mask = (item_last > 0) & (item_1m_ago > 0)

    all_candidates: list[list[dict]] = []
    for i in range(len(d)):
        candidates: list[dict] = []
        if not active_mask.iloc[i]:
            all_candidates.append(candidates)
            continue

        metric = float(item_last.iloc[i])
        baseline = float(avg_item_3m.iloc[i])
        if baseline > 0 and metric < 0.6 * baseline:
            delta_pct = _safe_ratio_delta_pct(metric, baseline, decrease=True)
            candidates.append(_candidate(
                priority=10,
                code="item_drop",
                text=f"Số bưu gửi tháng hiện tại thấp hơn {(delta_pct or 0) * 100:.0f}% so với trung bình 3 tháng liền trước",
                metric=metric,
                baseline=baseline,
                delta_pct=delta_pct,
            ))

        metric = float(complaint_last.iloc[i])
        baseline = float(avg_complaint_3m.iloc[i])
        if baseline > 0 and metric > 1.15 * baseline:
            delta_pct = _safe_ratio_delta_pct(metric, baseline)
            candidates.append(_candidate(
                priority=9,
                code="complaint_increase",
                text=f"Số lượng khiếu nại nhận được tăng {(delta_pct or 0) * 100:.0f}% so với trung bình 3 tháng liền trước",
                metric=metric,
                baseline=baseline,
                delta_pct=delta_pct,
            ))

        metric = float(delay_last.iloc[i])
        baseline = float(avg_delay_3m.iloc[i])
        if baseline > 0 and metric > 1.15 * baseline:
            delta_pct = _safe_ratio_delta_pct(metric, baseline)
            candidates.append(_candidate(
                priority=8,
                code="delay_rate_increase",
                text=f"Tỷ lệ số đơn giao muộn tăng {(delta_pct or 0) * 100:.0f}% so với trung bình 3 tháng liền trước",
                metric=metric,
                baseline=baseline,
                delta_pct=delta_pct,
            ))

        metric = float(nodone_last.iloc[i])
        baseline = float(avg_nodone_3m.iloc[i])
        if baseline > 0 and metric > 1.15 * baseline:
            delta_pct = _safe_ratio_delta_pct(metric, baseline)
            candidates.append(_candidate(
                priority=7,
                code="nodone_rate_increase",
                text=f"Tỷ lệ số đơn không hoàn thành tăng {(delta_pct or 0) * 100:.0f}% so với trung bình 3 tháng liền trước",
                metric=metric,
                baseline=baseline,
                delta_pct=delta_pct,
            ))

        metric = float(cv_item.iloc[i])
        if metric > 0.7:
            candidates.append(_candidate(
                priority=6,
                code="volume_volatility",
                text=f"Biến động số lượng bưu gửi cao (CV={metric:.2f})",
                metric=metric,
                baseline=0.7,
                delta_pct=max(metric / 0.7 - 1, 0),
            ))

        metric = float(rpi_last[i])
        baseline = float(rpi_3m[i])
        if baseline > 0 and metric < baseline:
            delta_pct = _safe_ratio_delta_pct(metric, baseline, decrease=True)
            candidates.append(_candidate(
                priority=5,
                code="order_value_drop",
                text=f"Giá trị đơn hàng trung bình giảm {(delta_pct or 0) * 100:.0f}% theo thời gian",
                metric=metric,
                baseline=baseline,
                delta_pct=delta_pct,
            ))

        metric = float(service_types.iloc[i])
        baseline = float(service_types_prev.iloc[i])
        if baseline > 0 and metric < baseline:
            delta_pct = _safe_ratio_delta_pct(metric, baseline, decrease=True)
            candidates.append(_candidate(
                priority=4,
                code="service_diversity_drop",
                text=f"Giảm đa dạng dịch vụ (giảm từ {int(baseline)} còn {int(metric)} loại)",
                metric=metric,
                baseline=baseline,
                delta_pct=delta_pct,
            ))

        metric = float(tenure.iloc[i])
        if metric < 6:
            delta_pct = _safe_ratio_delta_pct(metric, 6, decrease=True)
            candidates.append(_candidate(
                priority=3,
                code="low_tenure",
                text=f"Khách hàng mới, mức độ gắn bó thấp ({int(metric)} tháng)",
                metric=metric,
                baseline=6,
                delta_pct=delta_pct,
            ))

        candidates.sort(key=lambda r: (r["priority"], r["severity"] or 0.0), reverse=True)
        all_candidates.append(candidates)

    return d, all_candidates, active_mask


def _assign_reason_columns(d: pd.DataFrame, ranked_reasons: list[list[dict]]) -> pd.DataFrame:
    out = d.copy()
    for slot in range(1, REASON_SLOTS + 1):
        texts = []
        codes = []
        metrics = []
        baselines = []
        deltas = []
        delta_pcts = []
        severities = []
        for reasons in ranked_reasons:
            reason = reasons[slot - 1] if len(reasons) >= slot else None
            texts.append(reason["text"] if reason else None)
            codes.append(reason["code"] if reason else None)
            metrics.append(reason["metric"] if reason else None)
            baselines.append(reason["baseline"] if reason else None)
            deltas.append(reason["delta"] if reason else None)
            delta_pcts.append(reason["delta_pct"] if reason else None)
            severities.append(reason["severity"] if reason else None)
        out[f"reason_{slot}"] = texts
        out[f"reason_{slot}_code"] = codes
        out[f"reason_{slot}_metric"] = metrics
        out[f"reason_{slot}_baseline"] = baselines
        out[f"reason_{slot}_delta"] = deltas
        out[f"reason_{slot}_delta_pct"] = delta_pcts
        out[f"reason_{slot}_severity"] = severities
    return out


def _rank_reasons_by_buckets(candidates: list[dict], buckets: list[int]) -> list[dict]:
    if not buckets:
        return candidates[:REASON_SLOTS]
    code_by_bucket = {
        1: "item_drop",
        2: "complaint_increase",
        3: "delay_rate_increase",
        4: "nodone_rate_increase",
        5: "volume_volatility",
        6: "order_value_drop",
        7: "service_diversity_drop",
        8: "low_tenure",
    }
    bucket_rank = {
        code_by_bucket[bid]: idx
        for idx, bid in enumerate(buckets)
        if bid in code_by_bucket
    }
    ranked = sorted(
        candidates,
        key=lambda r: (
            1 if r["code"] in bucket_rank else 0,
            -bucket_rank.get(r["code"], 999),
            r["priority"],
            r["severity"] or 0.0,
        ),
        reverse=True,
    )
    return ranked[:REASON_SLOTS]


def compute_shap_reasons(
    model,
    X_scored: pd.DataFrame,
    df_with_raw: pd.DataFrame,
    df_static: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Dùng SHAP TreeExplainer để xác định top-3 features quan trọng nhất
    cho từng khách hàng, map sang 1 trong 8 reason buckets, sau đó render
    text tiếng Việt với số liệu thực tế (giống vỏ bọc compute_simple_reasons).

    - model        : XGBoost model đã train
    - X_scored     : DataFrame features (đã rename/pad, giống lúc predict)
    - df_with_raw  : DataFrame gốc có đủ cột raw (item_last, revenue_last, ...)
    - df_static    : bảng cus_lifetime (để lấy tenure)

    Trả về df_with_raw với cột reason_1/2/3 được điền.
    Nếu shap không import được → fallback sang compute_simple_reasons().
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        import logging as _log
        _log.getLogger(__name__).warning(
            "[SHAP] Thư viện shap chưa được cài. Fallback sang rule-based reasons."
        )
        return compute_simple_reasons(df_with_raw, df_static), None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scored)

        # Build df_shap_raw
        df_shap_raw = pd.DataFrame(shap_values, columns=X_scored.columns, index=X_scored.index)
        if "cms_code_enc" in df_with_raw.columns:
            df_shap_raw.insert(0, "cms_code_enc", df_with_raw["cms_code_enc"].values)

        feat_names = list(X_scored.columns)

        # Build per-customer top-3 bucket list
        top_buckets_per_row: list[list[int]] = []
        for i in range(len(X_scored)):
            row_shap = shap_values[i]
            order = np.argsort(-row_shap)  # desc by actual SHAP value (only positive = pushes churn risk up)
            seen_buckets: list[int] = []
            for idx in order:
                if row_shap[idx] <= 0:
                    break  # no more features that increase churn risk
                if len(seen_buckets) >= 3:
                    break
                fname = feat_names[idx]
                bid = _get_bucket(fname)
                if bid is not None and bid not in seen_buckets:
                    seen_buckets.append(bid)
            top_buckets_per_row.append(seen_buckets)

    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning(
            "[SHAP] Lỗi khi tính SHAP values: %s. Fallback sang rule-based reasons.", exc
        )
        return compute_simple_reasons(df_with_raw, df_static), None

    # ---------- Render reason text giống compute_simple_reasons ----------
    d, candidates, active_mask = _build_reason_candidates(df_with_raw, df_static)
    ranked = [
        _rank_reasons_by_buckets(row_candidates, top_buckets_per_row[i])
        for i, row_candidates in enumerate(candidates)
    ]
    d = _assign_reason_columns(d, ranked)
    d = d[active_mask].copy()
    return d, df_shap_raw


def compute_simple_reasons(df: pd.DataFrame, df_static: pd.DataFrame) -> pd.DataFrame:
    """Build business-rule reasons with CRM text plus structured evidence columns."""
    d, candidates, active_mask = _build_reason_candidates(df, df_static)
    ranked = [row[:REASON_SLOTS] for row in candidates]
    d = _assign_reason_columns(d, ranked)
    return d[active_mask].copy()


def insert_predictions_to_risk_table(
    engine: Engine,
    df_predictions: pd.DataFrame,
    risk_threshold: float = 90.0,
    horizon: int = 1,
) -> int:
    """Insert customers whose score percentile meets the operational threshold."""
    risk_pct = int(risk_threshold)
    table_name = f"cus_risk_{risk_pct}"

    if "churn_rate" not in df_predictions.columns:
        raise KeyError("Missing 'churn_rate' column in predictions")

    df_risk = filter_risk_predictions(df_predictions, risk_threshold)

    if df_risk.empty:
        print(f"??  No customers with risk_percentile_pct >= {risk_threshold}")
        return 0

    # Normalize column names
    if "satisfaction_last" not in df_risk.columns and "satisfation_last" in df_risk.columns:
        df_risk["satisfaction_last"] = df_risk["satisfation_last"]

    # Select columns needed
    cols_needed = [
        "cms_code_enc",
        "predict_period",
        "window_end",
        "item_last",
        "revenue_last",
        "complaint_last",
        "delay_last",
        "nodone_last",
        "order_score_last",
        "satisfaction_last",
        "churn_rate",
        "model_probability_pct",
        "reason_1",
        "reason_2",
        "reason_3",
    ]

    df_insert = df_risk.copy()
    for col in cols_needed:
        if col not in df_insert.columns and col != "predict_period":
            df_insert[col] = None

    df_insert["cms_code_enc"] = df_insert["cms_code_enc"].astype(str)
    df_insert["window_end"] = pd.to_numeric(df_insert["window_end"], errors="coerce").fillna(0).astype("int64")

    # Calculate predict_period: yymm
    y = df_insert["window_end"] // 100
    m = df_insert["window_end"] % 100
    write_horizon = horizon - 1
    m = m + write_horizon
    y = y + (m - 1) // 12
    m = (m - 1) % 12 + 1
    df_insert["predict_period"] = (y * 100 + m).astype("int64")

    df_insert = df_insert[cols_needed].copy()

    # Load SQL templates
    sql_dir = Path(__file__).parent / "sql"
    upsert_sql = (sql_dir / "insert_risk_upsert.sql").read_text()

    # Use temporary table + INSERT ON CONFLICT approach
    temp_table = f"_temp_{table_name}"

    upsert_sql = upsert_sql.replace("{TABLE_NAME}", table_name).replace("{TEMP_TABLE}", temp_table)

    with engine.begin() as conn:
        # Create temp table
        conn.execute(text(f"""
            CREATE TEMP TABLE {temp_table} (LIKE data_static.{table_name} INCLUDING ALL)
        """))

        # Insert to temp table using pandas (use the same connection `conn`)
        df_insert.to_sql(
            temp_table,
            con=conn,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )

        # UPSERT from temp to main table
        conn.execute(text(upsert_sql))
        # Drop temp table
        conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))

    print(f"? Inserted {len(df_insert)} customers to {table_name}")
    return len(df_insert)
