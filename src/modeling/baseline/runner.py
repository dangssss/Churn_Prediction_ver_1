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
from preprocess.eligibility import filter_churn_eligible
from preprocess.feature_columns import split_numeric_categorical_features
from preprocess.static_features import attach_static
from infra.yymm import shift_yymm
from logging_config import get_logger

logger = get_logger(__name__)


class SparseChurnLabelsError(ValueError):
    """Raised when a K candidate does not have enough positive labels to fit safely."""


def select_feature_cols_mixed(df: pd.DataFrame, label_col: str):
    return split_numeric_categorical_features(df, label_col=label_col)

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
    validation_origin_count: int | None = None,
):
    df2 = df.copy()
    df2[time_col] = df2[time_col].astype(int)
    months = sorted(df2[time_col].unique())
    if len(months) < 2:
        return None, None, None
    if validation_origin_count is None:
        validation_origin_count = int(os.getenv("VALIDATION_ORIGIN_COUNT", "2"))
    validation_origin_count = max(1, min(int(validation_origin_count), len(months) - 1))
    validation_months = months[-validation_origin_count:]
    val_month = validation_months[-1]
    train_max_month = int(shift_yymm(str(validation_months[0]), -int(horizon)))
    historical_train = df2[time_col] <= train_max_month
    df_tr = df2[historical_train].copy()
    df_va = df2[df2[time_col].isin(validation_months)].copy()
    logger.info(
        "[PURGED SPLIT] val_months=%s horizon=%s train_origin_max=%s "
        "historical_train_rows=%d train_rows=%d val_rows=%d",
        ",".join(str(m) for m in validation_months),
        horizon,
        train_max_month,
        int(historical_train.sum()),
        len(df_tr),
        len(df_va),
    )
    return df_tr, df_va, val_month


def time_series_purged_splits(
    df: pd.DataFrame,
    time_col: str = "window_end",
    *,
    horizon: int = 0,
    validation_origin_count: int | None = None,
    n_folds: int | None = None,
):
    """Create recent walk-forward folds with a purge gap equal to horizon."""
    df2 = df.copy()
    df2[time_col] = df2[time_col].astype(int)
    months = sorted(df2[time_col].unique())
    if len(months) < 2:
        return []
    if validation_origin_count is None:
        validation_origin_count = int(os.getenv("VALIDATION_ORIGIN_COUNT", "2"))
    validation_origin_count = max(1, min(int(validation_origin_count), len(months) - 1))
    if n_folds is None:
        n_folds = int(os.getenv("MODEL_WALK_FORWARD_FOLDS", "6"))
    n_folds = max(1, int(n_folds))

    folds = []
    latest_start = len(months) - validation_origin_count
    for start_idx in range(latest_start, -1, -1):
        validation_months = months[start_idx : start_idx + validation_origin_count]
        if len(validation_months) < validation_origin_count:
            continue
        val_month = validation_months[-1]
        train_max_month = int(shift_yymm(str(validation_months[0]), -int(horizon)))
        df_tr = df2[df2[time_col] <= train_max_month].copy()
        df_va = df2[df2[time_col].isin(validation_months)].copy()
        if df_tr.empty or df_va.empty:
            continue
        folds.append(
            {
                "train": df_tr,
                "val": df_va,
                "val_month": int(val_month),
                "validation_months": [int(m) for m in validation_months],
                "train_max_month": int(train_max_month),
            }
        )
        if len(folds) >= n_folds:
            break
    folds = list(reversed(folds))
    logger.info(
        "[WALK FORWARD SPLIT] folds=%d requested=%d validation_origin_count=%d details=%s",
        len(folds),
        n_folds,
        validation_origin_count,
        "; ".join(
            f"train<= {f['train_max_month']} val={','.join(str(m) for m in f['validation_months'])}"
            for f in folds
        ),
    )
    return folds


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
    df_k = filter_churn_eligible(df_k, k=k, context=f"baseline_k{k}")
    if df_k.empty or df_k[label_col].nunique() < 2:
        return None

    if use_static:
        if df_static is None:
            raise ValueError("use_static=True nhưng df_static=None")
        df_k = attach_static(df_k, df_static)

    folds = time_series_purged_splits(
        df_k,
        time_col="window_end",
        horizon=horizon,
    )
    if not folds:
        return None

    num_cols, cat_cols = select_feature_cols_mixed(df_k, label_col=label_col)
    fold_reports = []
    bundle_lifecycle = "PRODUCTION"
    last_class_weight_used = {0: 1.0, 1: 1.0}
    last_spw_raw = 1.0
    last_churn_ratio = 0.0
    last_threshold = 0.5

    for fold_idx, fold in enumerate(folds, start=1):
        df_tr = fold["train"]
        df_va = fold["val"]
        val_month = int(fold["val_month"])

        X_tr = df_tr[num_cols + cat_cols]
        y_tr = df_tr[label_col].astype(int)
        X_va = df_va[num_cols + cat_cols]
        y_va = df_va[label_col].astype(int)

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        spw_raw = float(n_neg) / max(float(n_pos), 1.0)
        max_positive_class_weight = float(os.getenv("BASELINE_MAX_POSITIVE_CLASS_WEIGHT", "100"))
        spw = min(spw_raw, max_positive_class_weight)

        churn_ratio = n_pos / max(n_pos + n_neg, 1)
        min_positive_rows = int(os.getenv("BASELINE_MIN_POSITIVE_ROWS", "500"))
        min_positive_rate = float(os.getenv("BASELINE_MIN_POSITIVE_RATE", "0.001"))
        if n_pos < min_positive_rows or churn_ratio < min_positive_rate:
            raise SparseChurnLabelsError(
                "Baseline training aborted before fit: implausibly sparse churn labels "
                f"for K={k} fold={fold_idx} (positive_rows={n_pos}, total_rows={n_pos + n_neg}, "
                f"positive_rate={churn_ratio:.6%}). "
                f"Required: rows>={min_positive_rows} and rate>={min_positive_rate:.4%}. "
                "Check label ingestion and label generation before running modeling."
            )

        class_weight_threshold = float(os.getenv("BASELINE_CLASS_WEIGHT_MAX_RATE", "0.10"))
        if churn_ratio >= class_weight_threshold:
            class_weight_used = {0: 1.0, 1: 1.0}
            class_weight_reason = (
                f"churn_ratio >= {class_weight_threshold:.1%} -> "
                "class_weight={1:1.0} (final labels are sufficiently dense)"
            )
        else:
            class_weight_used = {0: 1.0, 1: spw}
            class_weight_reason = (
                f"churn_ratio < {class_weight_threshold:.1%} -> weighted class_weight={{1:{spw:.2f}}} "
                f"(raw={spw_raw:.2f}, cap={max_positive_class_weight:.2f})"
            )

        logger.info(
            "[BASELINE K=%d FOLD=%d/%d] Train: Churn=%d | Active=%d | Total=%d | "
            "Churn rate=%.2f%% | Decision: %s",
            k,
            fold_idx,
            len(folds),
            n_pos,
            n_neg,
            n_pos + n_neg,
            churn_ratio * 100,
            class_weight_reason,
        )

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
        pipe.fit(X_tr, y_tr)

        va_prob = pipe.predict_proba(X_va)[:, 1]
        pr_auc = average_precision_score(y_va, va_prob)
        roc_auc = roc_auc_score(y_va, va_prob)
        thr = best_threshold_by_f1(y_va.to_numpy(), va_prob)
        yhat = (va_prob >= thr).astype(int)

        f1_val = float(f1_score(y_va, yhat, zero_division=0))
        precision_val = float(precision_score(y_va, yhat, zero_division=0))
        recall_val = float(recall_score(y_va, yhat, zero_division=0))
        prevalence = float(y_va.mean())
        dummy_f1 = 2 * prevalence / (prevalence + 1 + 1e-9)
        is_degenerate = abs(f1_val - dummy_f1) < 0.005
        if is_degenerate:
            logger.warning(
                "K=%d use_static=%s fold=%d: model degenerate (F1=%.4f ~= dummy=%.4f, predict-all-positive)",
                k,
                use_static,
                fold_idx,
                f1_val,
                dummy_f1,
            )

        logger.info(
            "[CLASSIFICATION METRICS] K=%d use_static=%s fold=%d/%d val=%s "
            "F1=%.4f precision=%.4f recall=%.4f PR_AUC=%.4f ROC_AUC=%.4f prevalence=%.4f%%",
            k,
            use_static,
            fold_idx,
            len(folds),
            val_month,
            f1_val,
            precision_val,
            recall_val,
            float(pr_auc),
            float(roc_auc),
            100.0 * prevalence,
        )
        fold_reports.append({
            "val_month": val_month,
            "spw_used": float(class_weight_used.get(1, 1.0)),
            "spw_raw": float(spw_raw),
            "churn_ratio_train": float(churn_ratio),
            "PR_AUC_val": float(pr_auc),
            "ROC_AUC_val": float(roc_auc),
            "val_prevalence": float(prevalence),
            "best_threshold": float(thr),
            "precision": precision_val,
            "recall": recall_val,
            "f1": f1_val,
            "degenerate": is_degenerate,
        })
        last_class_weight_used = class_weight_used
        last_spw_raw = spw_raw
        last_churn_ratio = churn_ratio
        last_threshold = thr

    if not fold_reports:
        return None

    metric_frame = pd.DataFrame(fold_reports)
    latest = fold_reports[-1]
    is_degenerate = bool(metric_frame["degenerate"].all())
    f1_val = float(metric_frame["f1"].mean())
    precision_val = float(metric_frame["precision"].mean())
    recall_val = float(metric_frame["recall"].mean())
    pr_auc = float(metric_frame["PR_AUC_val"].mean())
    roc_auc = float(metric_frame["ROC_AUC_val"].mean())
    prevalence = float(metric_frame["val_prevalence"].mean())
    logger.info(
        "[WALK FORWARD METRICS] K=%d use_static=%s folds=%d F1_mean=%.4f precision_mean=%.4f "
        "recall_mean=%.4f PR_AUC_mean=%.4f ROC_AUC_mean=%.4f latest_val=%s latest_F1=%.4f",
        k,
        use_static,
        len(fold_reports),
        f1_val,
        precision_val,
        recall_val,
        pr_auc,
        roc_auc,
        latest["val_month"],
        latest["f1"],
    )

    return {
        "K": int(k),
        "H": int(horizon),
        "use_static": bool(use_static),
        "val_month": int(latest["val_month"]),
        "bundle_lifecycle": bundle_lifecycle,
        "n_rows": int(len(df_k)),
        "n_months": int(df_k["window_end"].nunique()),
        "n_folds": int(len(fold_reports)),
        "spw_used": float(last_class_weight_used.get(1, 1.0)),
        "spw_raw": float(last_spw_raw),
        "churn_ratio_train": float(last_churn_ratio),
        "n_num": int(len(num_cols)),
        "n_cat": int(len(cat_cols)),
        "PR_AUC_val": pr_auc,
        "ROC_AUC_val": roc_auc,
        "val_prevalence": prevalence,
        "best_threshold": float(last_threshold),
        "precision": precision_val,
        "recall": recall_val,
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
    df_k = filter_churn_eligible(df_k, k=k, context=f"baseline_bundle_k{k}")
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
    pipe.fit(X_tr, y_tr)

    va_prob = pipe.predict_proba(X_va)[:, 1]
    pr_auc = average_precision_score(y_va, va_prob)
    roc_auc = roc_auc_score(y_va, va_prob)
    thr = best_threshold_by_f1(y_va.to_numpy(), va_prob)
    yhat = (va_prob >= thr).astype(int)
    val_prevalence = float(y_va.mean())

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
        "val_prevalence": val_prevalence,
        "thr_main_opt": float(thr),
        "precision@main_thr": float(precision_score(y_va, yhat, zero_division=0)),
        "recall@main_thr": float(recall_score(y_va, yhat, zero_division=0)),
        "f1@main_thr": float(f1_score(y_va, yhat, zero_division=0)),
        "guardrail_warning": None,
    }
    logger.warning(
        "[BASELINE FALLBACK BUNDLE] K=%d use_static=%s F1=%.4f AP=%.4f ROC_AUC=%.4f",
        k,
        use_static,
        report["f1@main_thr"],
        report["AP_val"],
        report["ROC_AUC_val"],
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
