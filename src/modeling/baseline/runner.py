from __future__ import annotations

from typing import Optional, Dict, Tuple
import os

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve,
    f1_score, precision_score, recall_score
)

from preprocess.dataset import build_dataset_for_k, preflight_purged_train_val_for_k
from preprocess.static_features import attach_static
from infra.yymm import shift_yymm
from common.metrics import ranking_metrics_at_n, prefixed_ranking_metrics_at_n
from logging_config import get_logger

logger = get_logger(__name__)


class SparseChurnLabelsError(ValueError):
    """Raised when a K candidate does not have enough positive labels to fit safely."""


def select_feature_cols_mixed(df: pd.DataFrame, label_col: str):
    drop_cols = {
        "cms_code_enc", "window_size", "window_start", "window_end",
        "source_table_t", "source_table_t_plus_h",
        "is_active_now", "is_churned_now", "gate_group",
        "label_source", "label_weight",
        label_col
    }
    num_cols, cat_cols = [], []
    for c in df.columns:
        if c in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        elif df[c].dtype == "object":
            cat_cols.append(c)
        elif "datetime" in str(df[c].dtype).lower():
            # drop datetime columns
            continue
        else:
            # treat others as categorical
            cat_cols.append(c)
    return num_cols, cat_cols

def make_preprocess(num_cols, cat_cols):
    num_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre

def best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # thr length = len(prec)-1
    f1s = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
    if len(f1s) == 0:
        return 0.5
    best_idx = int(np.argmax(f1s))
    # Bug fix: nếu argmax trả về index 0 (predict all positive — degenerate)
    # thì tìm best trong phần còn lại để tránh threshold ≈ 0
    if best_idx == 0 and len(f1s) > 1:
        rest_idx = int(np.argmax(f1s[1:])) + 1
        # chỉ dùng nếu f1 tại rest_idx không kém quá 5%
        if f1s[rest_idx] >= f1s[0] * 0.95:
            best_idx = rest_idx
    # Clamp: không cho threshold xuống dưới 5% để tránh predict all-positive
    return float(max(thr[best_idx], 0.05))

def time_split_train_val_last_month(
    df: pd.DataFrame,
    time_col: str = "window_end",
    *,
    horizon: int = 0,
):
    df2 = df.copy()
    df2[time_col] = df2[time_col].astype(int)
    months = sorted(df2[time_col].unique())
    if len(months) < 2:
        return None, None, None
    val_month = months[-1]
    train_max_month = int(shift_yymm(str(val_month), -int(horizon)))
    historical_train = df2[time_col] <= train_max_month
    future_rule_holdout = pd.Series(False, index=df2.index)
    if "label_source" in df2.columns:
        future_rule_holdout = (
            (df2["label_source"] == "rule_based")
            & (df2[time_col] > val_month)
        )
    df_tr = df2[historical_train].copy()
    df_va = df2[df2[time_col] == val_month].copy()
    logger.info(
        "[PURGED SPLIT] val_month=%s horizon=%s train_origin_max=%s "
        "historical_train_rows=%d future_rule_holdout_rows=%d train_rows=%d val_rows=%d",
        val_month,
        horizon,
        train_max_month,
        int(historical_train.sum()),
        int(future_rule_holdout.sum()),
        len(df_tr),
        len(df_va),
    )
    return df_tr, df_va, val_month


def auxiliary_rule_holdout(
    df: pd.DataFrame,
    val_month: int,
    *,
    time_col: str = "window_end",
) -> pd.DataFrame:
    if "label_source" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    time_values = df[time_col].astype(int)
    mask = (df["label_source"] == "rule_based") & (time_values > int(val_month))
    return df[mask].copy()


def label_source_ranking_metrics(
    df_eval: pd.DataFrame,
    y_prob: np.ndarray,
    *,
    label_col: str,
    ranking_top_n: int,
) -> dict:
    """Compute source-separated ranking metrics for validation reporting."""
    out: dict = {}
    if df_eval.empty:
        return out
    y_prob = np.asarray(y_prob, dtype=float)
    if len(df_eval) != len(y_prob):
        raise ValueError("df_eval and y_prob must have the same length")

    if "label_source" in df_eval.columns:
        source = df_eval["label_source"].astype(str)
        source_specs = [("actual", "actual"), ("rule_based", "rule")]
        for source_value, prefix in source_specs:
            mask = source == source_value
            if not bool(mask.any()):
                continue
            out.update(
                prefixed_ranking_metrics_at_n(
                    df_eval.loc[mask, label_col].astype(int).to_numpy(),
                    y_prob[mask.to_numpy()],
                    n=ranking_top_n,
                    prefix=prefix,
                )
            )

    weights = (
        pd.to_numeric(df_eval["label_weight"], errors="coerce").fillna(1.0).to_numpy()
        if "label_weight" in df_eval.columns
        else None
    )
    out.update(
        prefixed_ranking_metrics_at_n(
            df_eval[label_col].astype(int).to_numpy(),
            y_prob,
            n=ranking_top_n,
            prefix="combined_weighted",
            sample_weight=weights,
        )
    )
    return out

def eval_one_k_train_val(
    engine: Engine,
    k: int,
    horizon: int,
    df_static: Optional[pd.DataFrame] = None,
    use_static: bool = False,
    limit_rows_each: Optional[int] = None,
    df_k: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    label_col = f"y_churn_t_plus_{horizon}"
    if df_k is None:
        preflight_purged_train_val_for_k(engine, k, horizon=horizon)
        df_k = build_dataset_for_k(
            engine,
            k,
            horizon=horizon,
            limit_rows_each=limit_rows_each,
        )
    if df_k.empty or label_col not in df_k.columns:
        return None

    # train churn-risk chỉ trên active_now + có label
    df_k = df_k[df_k["is_active_now"] == 1].dropna(subset=[label_col]).copy()
    if df_k.empty or df_k[label_col].nunique() < 2:
        return None

    if use_static:
        if df_static is None:
            raise ValueError("use_static=True nhưng df_static=None")
        df_k = attach_static(df_k, df_static)

    df_tr, df_va, val_month = time_split_train_val_last_month(
        df_k,
        time_col="window_end",
        horizon=horizon,
    )
    if df_tr is None or df_tr.empty or df_va.empty:
        return None
    val_label_sources = (
        sorted(df_va["label_source"].dropna().astype(str).unique())
        if "label_source" in df_va.columns
        else []
    )
    validation_label_source = (
        "unknown"
        if not val_label_sources
        else val_label_sources[0]
        if len(val_label_sources) == 1
        else "mixed"
    )
    bundle_lifecycle = "PRODUCTION"
    logger.info(
        "[VALIDATION PROVENANCE] val_month=%s source=%s lifecycle=%s policy=mixed_actual_rule",
        val_month,
        validation_label_source,
        bundle_lifecycle,
    )

    num_cols, cat_cols = select_feature_cols_mixed(df_k, label_col=label_col)
    X_tr = df_tr[num_cols + cat_cols]
    y_tr = df_tr[label_col].astype(int)
    X_va = df_va[num_cols + cat_cols]
    y_va = df_va[label_col].astype(int)
    sample_weight = (
        pd.to_numeric(df_tr["label_weight"], errors="coerce").fillna(1.0)
        if "label_weight" in df_tr.columns
        else pd.Series(1.0, index=df_tr.index)
    )

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    weighted_pos = float(sample_weight[y_tr == 1].sum())
    weighted_neg = float(sample_weight[y_tr == 0].sum())
    spw_raw = weighted_neg / max(weighted_pos, 1.0)
    max_positive_class_weight = float(
        os.getenv("BASELINE_MAX_POSITIVE_CLASS_WEIGHT", "100")
    )
    spw = min(spw_raw, max_positive_class_weight)

    # ---------- Guardrail: tự động chọn class_weight ----------
    CHURN_RATIO_THRESHOLD = 0.35
    churn_ratio = n_pos / max(n_pos + n_neg, 1)
    min_positive_rows = int(os.getenv("BASELINE_MIN_POSITIVE_ROWS", "500"))
    min_positive_rate = float(os.getenv("BASELINE_MIN_POSITIVE_RATE", "0.001"))
    if n_pos < min_positive_rows or churn_ratio < min_positive_rate:
        raise SparseChurnLabelsError(
            "Baseline training aborted before fit: implausibly sparse churn labels "
            f"for K={k} (positive_rows={n_pos}, total_rows={n_pos + n_neg}, "
            f"positive_rate={churn_ratio:.6%}). "
            f"Required: rows>={min_positive_rows} and rate>={min_positive_rate:.4%}. "
            "Check label ingestion and label generation before running modeling."
        )

    if churn_ratio > CHURN_RATIO_THRESHOLD:
        class_weight_used = {0: 1.0, 1: 1.0}
        spw_rule = "churn_ratio > 35% → class_weight={1:1.0} (dữ liệu đủ cân bằng)"
    else:
        class_weight_used = {0: 1.0, 1: spw}
        spw_rule = f"churn_ratio <= 35% → class_weight={{1:{spw:.2f}}} (bù mất cân bằng)"

    if churn_ratio <= CHURN_RATIO_THRESHOLD:
        spw_rule = (
            f"churn_ratio <= 35% -> weighted class_weight={{1:{spw:.2f}}} "
            f"(raw={spw_raw:.2f}, cap={max_positive_class_weight:.2f})"
        )

    logger.info(
        "[BASELINE K=%d] Tập huấn luyện: Churn=%d | Active=%d | Total=%d | "
        "Tỷ lệ Churn=%.2f%% | Quyết định: %s",
        k, n_pos, n_neg, n_pos + n_neg, churn_ratio * 100, spw_rule,
    )

    if "label_source" in df_tr.columns:
        provenance = (
            df_tr.assign(_label_weight=sample_weight)
            .groupby("label_source", dropna=False)
            .agg(
                rows=(label_col, "size"),
                positives=(label_col, "sum"),
                effective_weight=("_label_weight", "sum"),
            )
            .reset_index()
        )
        logger.info("[BASELINE K=%d] Label provenance:\n%s", k, provenance.to_string(index=False))

    pre = make_preprocess(num_cols, cat_cols)
    clf = LogisticRegression(
        max_iter=5000,
        solver="saga",   # saga: solver duy nhất hỗ trợ ElasticNet (l1_ratio ∈ (0,1))
        tol=1e-3,        # relaxed tolerance: converge nhanh hơn mà không ảnh hưởng chất lượng ranking
        class_weight=class_weight_used,
        l1_ratio=0.5,    # ElasticNet: 0=L2, 1=L1, 0.5=50/50 mix — penalty string deprecated từ sklearn 1.8
        C=0.1,           # regularization strength (ngược với lambda); 0.1 = mạnh hơn default (1.0)
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    pipe.fit(X_tr, y_tr, clf__sample_weight=sample_weight.to_numpy())

    va_prob = pipe.predict_proba(X_va)[:, 1]
    df_rule_aux = auxiliary_rule_holdout(df_k, int(val_month), time_col="window_end")
    aux_prob = np.array([], dtype=float)
    if not df_rule_aux.empty:
        X_aux = df_rule_aux[num_cols + cat_cols]
        aux_prob = pipe.predict_proba(X_aux)[:, 1]
        logger.info(
            "[AUX RULE HOLDOUT] K=%d use_static=%s rows=%d origins=%s",
            k,
            use_static,
            len(df_rule_aux),
            sorted(df_rule_aux["window_end"].astype(int).unique()),
        )

    pr_auc  = average_precision_score(y_va, va_prob)
    roc_auc = roc_auc_score(y_va, va_prob)
    ranking_top_n = int(os.getenv("MODEL_RANKING_TOP_N", "5000"))
    ranking = ranking_metrics_at_n(y_va.to_numpy(), va_prob, n=ranking_top_n)
    eval_df = pd.concat([df_va, df_rule_aux], axis=0, ignore_index=True)
    eval_prob = np.concatenate([va_prob, aux_prob]) if len(aux_prob) else va_prob
    source_ranking = label_source_ranking_metrics(
        eval_df,
        eval_prob,
        label_col=label_col,
        ranking_top_n=ranking_top_n,
    )
    logger.info(
        "[RANKING METRICS][PRIMARY %s] K=%d use_static=%s top_n=%d effective_n=%d "
        "hits=%d precision=%.4f%% recall=%.2f%% lift=%.2fx prevalence=%.4f%%",
        validation_label_source.upper(),
        k,
        use_static,
        ranking["ranking_top_n"],
        ranking["ranking_effective_n"],
        ranking["hits_at_n"],
        100.0 * ranking["precision_at_n"],
        100.0 * ranking["recall_at_n"],
        ranking["lift_at_n"],
        100.0 * ranking["val_prevalence"],
    )
    if "rule_lift_at_n" in source_ranking:
        logger.info(
            "[RANKING METRICS][AUX RULE] K=%d use_static=%s top_n=%d effective_n=%d "
            "hits=%d precision=%.4f%% recall=%.2f%% lift=%.2fx prevalence=%.4f%%",
            k,
            use_static,
            source_ranking["rule_ranking_top_n"],
            source_ranking["rule_ranking_effective_n"],
            source_ranking["rule_hits_at_n"],
            100.0 * source_ranking["rule_precision_at_n"],
            100.0 * source_ranking["rule_recall_at_n"],
            source_ranking["rule_lift_at_n"],
            100.0 * source_ranking["rule_val_prevalence"],
        )
    logger.info(
        "[RANKING METRICS][WEIGHTED COMBINED] K=%d use_static=%s top_n=%d effective_n=%d "
        "weighted_hits=%.2f precision=%.4f%% recall=%.2f%% lift=%.2fx prevalence=%.4f%%",
        k,
        use_static,
        source_ranking["combined_weighted_ranking_top_n"],
        source_ranking["combined_weighted_ranking_effective_n"],
        source_ranking["combined_weighted_hits_at_n"],
        100.0 * source_ranking["combined_weighted_precision_at_n"],
        100.0 * source_ranking["combined_weighted_recall_at_n"],
        source_ranking["combined_weighted_lift_at_n"],
        100.0 * source_ranking["combined_weighted_val_prevalence"],
    )

    thr = best_threshold_by_f1(y_va.to_numpy(), va_prob)
    yhat = (va_prob >= thr).astype(int)

    f1_val = float(f1_score(y_va, yhat, zero_division=0))

    # Bug #2: đánh dấu degenerate nếu F1 gần bằng predict-all-positive
    prevalence = float(y_va.mean())
    dummy_f1 = 2 * prevalence / (prevalence + 1 + 1e-9)
    is_degenerate = abs(f1_val - dummy_f1) < 0.005
    if is_degenerate:
        logger.warning("K=%d use_static=%s: model degenerate (F1=%.4f ≈ dummy=%.4f, predict-all-positive)", k, use_static, f1_val, dummy_f1)

    return {
        "K": int(k),
        "H": int(horizon),
        "use_static": bool(use_static),
        "val_month": int(val_month),
        "validation_label_source": validation_label_source,
        "bundle_lifecycle": bundle_lifecycle,
        "n_rows": int(len(df_k)),
        "n_months": int(df_k["window_end"].nunique()),
        "spw_used": float(class_weight_used.get(1, 1.0)),
        "spw_raw": float(spw_raw),
        "churn_ratio_train": float(churn_ratio),
        "n_num": int(len(num_cols)),
        "n_cat": int(len(cat_cols)),
        "PR_AUC_val": float(pr_auc),
        "ROC_AUC_val": float(roc_auc),
        **ranking,
        **source_ranking,
        "best_threshold": float(thr),
        "precision": float(precision_score(y_va, yhat, zero_division=0)),
        "recall": float(recall_score(y_va, yhat, zero_division=0)),
        "f1": f1_val,
        "degenerate": is_degenerate,
    }


def train_baseline_model_for_config(
    engine: Engine,
    cfg: dict,
    df_static: Optional[pd.DataFrame] = None,
    limit_rows_each: Optional[int] = None,
) -> Dict:
    """Train the sweep LogisticRegression baseline as a deployable fallback bundle."""
    k = int(cfg["best_k"])
    horizon = int(cfg["horizon"])
    use_static = bool(cfg.get("use_static", False))
    label_col = f"y_churn_t_plus_{horizon}"

    df_k = build_dataset_for_k(
        engine,
        k,
        horizon=horizon,
        limit_rows_each=limit_rows_each,
    )
    if df_k.empty or label_col not in df_k.columns:
        raise ValueError(f"Dataset empty for baseline fallback K={k}, H={horizon}")

    df_k = df_k[df_k["is_active_now"] == 1].dropna(subset=[label_col]).copy()
    if df_k.empty or df_k[label_col].nunique() < 2:
        raise ValueError(f"Not enough labeled data for baseline fallback K={k}, H={horizon}")

    if use_static:
        if df_static is None:
            raise ValueError("use_static=True but df_static=None")
        df_k = attach_static(df_k, df_static)

    df_tr, df_va, val_month = time_split_train_val_last_month(
        df_k,
        time_col="window_end",
        horizon=horizon,
    )
    if df_tr is None or df_tr.empty or df_va.empty:
        raise ValueError(f"Not enough train/val data for baseline fallback K={k}, H={horizon}")

    num_cols, cat_cols = select_feature_cols_mixed(df_k, label_col=label_col)
    X_tr = df_tr[num_cols + cat_cols]
    y_tr = df_tr[label_col].astype(int)
    X_va = df_va[num_cols + cat_cols]
    y_va = df_va[label_col].astype(int)
    sample_weight = (
        pd.to_numeric(df_tr["label_weight"], errors="coerce").fillna(1.0)
        if "label_weight" in df_tr.columns
        else pd.Series(1.0, index=df_tr.index)
    )

    class_weight_used = {0: 1.0, 1: float(cfg.get("best_spw") or 1.0)}
    pre = make_preprocess(num_cols, cat_cols)
    clf = LogisticRegression(
        max_iter=5000,
        solver="saga",
        tol=1e-3,
        class_weight=class_weight_used,
        l1_ratio=0.5,
        C=0.1,
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_tr, y_tr, clf__sample_weight=sample_weight.to_numpy())

    va_prob = pipe.predict_proba(X_va)[:, 1]
    ranking_top_n = int(cfg.get("ranking_top_n") or os.getenv("MODEL_RANKING_TOP_N", "5000"))
    ranking = ranking_metrics_at_n(y_va.to_numpy(), va_prob, n=ranking_top_n)
    source_ranking = label_source_ranking_metrics(
        df_va.reset_index(drop=True),
        va_prob,
        label_col=label_col,
        ranking_top_n=ranking_top_n,
    )
    pr_auc = average_precision_score(y_va, va_prob)
    roc_auc = roc_auc_score(y_va, va_prob)
    thr = best_threshold_by_f1(y_va.to_numpy(), va_prob)
    yhat = (va_prob >= thr).astype(int)

    try:
        from monitoring.drift import compute_feature_profile

        feature_profile = compute_feature_profile(
            df_tr,
            feat_cols=num_cols + cat_cols,
            cat_cols=cat_cols,
        )
    except Exception:
        feature_profile = None

    report = {
        "model_type": "baseline_logistic",
        "K": k,
        "H": horizon,
        "use_static": use_static,
        "val_month": int(val_month),
        "train_rows": int(len(df_tr)),
        "val_rows": int(len(df_va)),
        "AP_val": float(pr_auc),
        "ROC_AUC_val": float(roc_auc),
        **ranking,
        **source_ranking,
        "thr_main_opt": float(thr),
        "precision@main_thr": float(precision_score(y_va, yhat, zero_division=0)),
        "recall@main_thr": float(recall_score(y_va, yhat, zero_division=0)),
        "f1@main_thr": float(f1_score(y_va, yhat, zero_division=0)),
        "guardrail_warning": None,
    }
    logger.warning(
        "[BASELINE FALLBACK BUNDLE] K=%d use_static=%s Lift@%d=%.2fx "
        "Precision@%d=%.4f%% Recall@%d=%.2f%% hits=%d",
        k,
        use_static,
        ranking["ranking_top_n"],
        ranking["lift_at_n"],
        ranking["ranking_top_n"],
        100.0 * ranking["precision_at_n"],
        ranking["ranking_top_n"],
        100.0 * ranking["recall_at_n"],
        ranking["hits_at_n"],
    )
    return {
        "model": pipe,
        "report": report,
        "feat_cols": num_cols + cat_cols,
        "cat_cols": cat_cols,
        "date_cols": [],
        "feature_name_map": None,
        "feature_profile": feature_profile,
    }
