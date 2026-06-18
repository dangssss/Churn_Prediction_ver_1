from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from typing import Any

logger = logging.getLogger(__name__)

from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.metrics import (
    average_precision_np,
    best_threshold_by_f1_np,
    prf1_at_threshold,
)
from preprocess.feature_columns import feature_columns

from monitoring.drift import compute_feature_profile

from .xgb_utils import (
    safe_to_category,
    onehot_align_train_val,
    sanitize_xgb_feature_names,
    fit_xgb_with_early_stopping,
    predict_proba_best_iteration,
    is_date_like_col,
    date_col_to_ordinal,
)


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _score_stats(prob: np.ndarray) -> dict:
    if len(prob) == 0:
        return {}
    q01, q10, q50, q90, q99 = np.quantile(prob, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "score_min": float(np.min(prob)),
        "score_p01": float(q01),
        "score_p10": float(q10),
        "score_p50": float(q50),
        "score_p90": float(q90),
        "score_p99": float(q99),
        "score_max": float(np.max(prob)),
        "score_range": float(np.max(prob) - np.min(prob)),
        "score_unique_rounded_6": int(len(np.unique(np.round(prob, 6)))),
    }


def _top_percentile_metrics(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> dict[str, float]:
    pct = min(max(float(pct), 0.0), 1.0)
    if len(y_prob) == 0 or pct <= 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "predicted_positive_rate": 0.0, "threshold": 1.0}
    cutoff = float(np.quantile(y_prob, max(0.0, 1.0 - pct)))
    precision, recall, f1 = prf1_at_threshold(y_true.astype(int), y_prob, cutoff)
    pred_rate = float(np.mean(y_prob >= cutoff))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive_rate": pred_rate,
        "threshold": cutoff,
    }


def _prepare_bundle_features(
    df: pd.DataFrame,
    *,
    cfg: dict,
    metadata: dict,
    model: Any,
) -> pd.DataFrame:
    h = int(cfg.get("horizon") or metadata.get("cfg", {}).get("horizon") or 2)
    label_col = f"y_churn_t_plus_{h}"
    meta_feat_cols = metadata.get("feat_cols")
    if isinstance(meta_feat_cols, list) and meta_feat_cols:
        feat_cols = [c for c in meta_feat_cols if c in df.columns]
    else:
        feat_cols = feature_columns(df, label_col=label_col)

    X = df[feat_cols].copy()
    cat_cols = list(metadata.get("cat_cols") or [])
    date_cols = list(metadata.get("date_cols") or [])
    feature_name_map = metadata.get("feature_name_map") or {}

    extra_date_cols = [
        c for c in cat_cols
        if c in X.columns and c not in date_cols and is_date_like_col(X[c])
    ]
    if extra_date_cols:
        date_cols = date_cols + extra_date_cols
        cat_cols = [c for c in cat_cols if c not in extra_date_cols]

    for c in date_cols:
        if c in X.columns:
            X[c] = date_col_to_ordinal(X[c])
    for c in cat_cols:
        if c in X.columns:
            X[c] = safe_to_category(X[c])
    for c in feat_cols:
        if c not in cat_cols and c not in date_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    if feature_name_map:
        X = X.rename(columns=feature_name_map)

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        model_features = list(model_features)
    elif feature_name_map and isinstance(meta_feat_cols, list):
        model_features = [feature_name_map.get(c, c) for c in meta_feat_cols]
    else:
        model_features = None

    if model_features is not None:
        for c in model_features:
            if c not in X.columns:
                X[c] = 0
        X = X[list(model_features)]
    return X


def _evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray, cfg: dict) -> dict[str, Any]:
    threshold_cfg = dict(cfg)
    if threshold_cfg.get("main_fixed_threshold") is None:
        threshold_cfg["main_fixed_threshold"] = float(threshold_cfg.get("best_threshold", 0.5))
        threshold_cfg["main_fixed_threshold_source"] = "fixed_from_config"
    threshold_metrics = _main_threshold_metrics(y_true, y_prob, threshold_cfg)
    ap = average_precision_np(y_true, y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None
    top5 = _top_percentile_metrics(y_true, y_prob, 0.05)
    top10 = _top_percentile_metrics(y_true, y_prob, 0.10)
    score_stats = _score_stats(y_prob)
    return {
        "AP_val": float(ap),
        "ROC_AUC_val": roc_auc,
        **score_stats,
        "thr_main_opt": float(threshold_metrics["threshold"]),
        "threshold_source": threshold_metrics["threshold_source"],
        "precision@main_thr": float(threshold_metrics["precision"]),
        "recall@main_thr": float(threshold_metrics["recall"]),
        "f1@main_thr": float(threshold_metrics["f1"]),
        "predicted_positive_rate@main_thr": float(np.mean(y_prob >= float(threshold_metrics["threshold"]))),
        "precision@top_5pct": float(top5["precision"]),
        "recall@top_5pct": float(top5["recall"]),
        "f1@top_5pct": float(top5["f1"]),
        "threshold@top_5pct": float(top5["threshold"]),
        "precision@top_10pct": float(top10["precision"]),
        "recall@top_10pct": float(top10["recall"]),
        "f1@top_10pct": float(top10["f1"]),
        "threshold@top_10pct": float(top10["threshold"]),
        "val_prevalence": float(np.mean(y_true == 1)),
    }


def _xgb_scale_pos_weight(
    y: np.ndarray,
    baseline_spw: float,
) -> tuple[float, float, float]:
    """Add class balance only when labels are sparse."""
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    class_ratio = neg / max(pos, 1e-9)
    max_spw = max(_env_float("MAIN_XGB_MAX_SCALE_POS_WEIGHT", 20.0), 1.0)
    mode = (os.getenv("MAIN_XGB_SCALE_POS_WEIGHT_MODE") or "auto").strip().lower()
    churn_ratio = float(np.mean(y == 1)) if len(y) else 0.0
    balance_max_rate = max(_env_float("MAIN_XGB_CLASS_WEIGHT_MAX_RATE", 0.10), 0.0)

    if mode in {"none", "off", "1", "false"}:
        effective = 1.0
    elif mode == "auto":
        effective = 1.0 if churn_ratio >= balance_max_rate else float(np.sqrt(max(class_ratio, 1.0)))
    elif mode in {"baseline", "best_spw"}:
        effective = baseline_spw
    elif mode in {"weighted", "raw_weighted"}:
        effective = class_ratio
    else:
        effective = float(np.sqrt(max(class_ratio, 1.0)))

    effective = min(max(float(effective), 1.0), max_spw)
    return effective, class_ratio, max_spw


def _xgb_training_params(cfg: dict, spw: float) -> tuple[dict[str, Any], int]:
    """Build XGBoost params from env defaults plus optional tuned overrides."""
    tuned = dict(cfg.get("main_xgb_params") or {})
    es_rounds = int(tuned.pop("early_stopping_rounds", cfg.get("main_es_rounds", _env_int("MAIN_XGB_ES_ROUNDS", 200))))

    params: dict[str, Any] = dict(
        n_estimators=_env_int("MAIN_XGB_N_ESTIMATORS", 5000),
        learning_rate=_env_float("MAIN_XGB_LEARNING_RATE", 0.03),
        max_depth=_env_int("MAIN_XGB_MAX_DEPTH", 6),
        max_leaves=_env_int("MAIN_XGB_MAX_LEAVES", 63),
        subsample=_env_float("MAIN_XGB_SUBSAMPLE", 0.8),
        colsample_bytree=_env_float("MAIN_XGB_COLSAMPLE_BYTREE", 0.8),
        colsample_bylevel=_env_float("MAIN_XGB_COLSAMPLE_BYLEVEL", 0.7),
        reg_lambda=_env_float("MAIN_XGB_REG_LAMBDA", 1.0),
        reg_alpha=_env_float("MAIN_XGB_REG_ALPHA", 0.1),
        min_child_weight=_env_float("MAIN_XGB_MIN_CHILD_WEIGHT", 2.0),
        gamma=_env_float("MAIN_XGB_GAMMA", 0.0),
        tree_method=os.getenv("MAIN_XGB_TREE_METHOD", "hist"),
        random_state=int(cfg.get("seed", 42)),
        scale_pos_weight=float(spw),
        eval_metric=["aucpr", "logloss"],
    )
    n_jobs = _env_int("MAIN_XGB_N_JOBS", 0)
    if n_jobs > 0:
        params["n_jobs"] = int(n_jobs)

    params.update(tuned)
    params["scale_pos_weight"] = float(params.get("scale_pos_weight", spw))
    params["random_state"] = int(params.get("random_state", cfg.get("seed", 42)))
    if "eval_metric" not in params or params["eval_metric"] in (None, ""):
        params["eval_metric"] = ["aucpr", "logloss"]
    return params, max(int(es_rounds), 1)


def _cfg_float(cfg: dict, key: str, env_name: str, default: float) -> float:
    value = cfg.get(key)
    if value is None:
        return _env_float(env_name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid cfg float %s=%r. Falling back to %s.", key, value, env_name)
        return _env_float(env_name, default)


def _main_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, cfg: dict) -> dict[str, Any]:
    min_threshold = _cfg_float(cfg, "main_threshold_min", "MAIN_XGB_THRESHOLD_MIN", 0.005)
    max_pred_pos_rate = _cfg_float(
        cfg,
        "main_max_predicted_positive_rate",
        "MAIN_XGB_MAX_PREDICTED_POSITIVE_RATE",
        0.25,
    )
    min_precision = _cfg_float(
        cfg,
        "main_min_precision",
        "MAIN_XGB_MIN_PRECISION",
        0.0,
    )
    min_recall = _cfg_float(
        cfg,
        "main_min_recall",
        "MAIN_XGB_MIN_RECALL",
        0.0,
    )
    search_thr, search_precision, search_recall, search_f1 = best_threshold_by_f1_np(
        y_true,
        y_prob,
        n_grid=600,
        min_threshold=min_threshold,
        max_predicted_positive_rate=max_pred_pos_rate,
        min_precision=min_precision if min_precision > 0 else None,
        min_recall=min_recall if min_recall > 0 else None,
    )

    fixed_threshold = cfg.get("main_fixed_threshold")
    if fixed_threshold is not None:
        threshold = float(fixed_threshold)
        precision, recall, f1 = prf1_at_threshold(y_true, y_prob, threshold)
        threshold_source = str(cfg.get("main_fixed_threshold_source") or "fixed")
    else:
        threshold = float(search_thr)
        precision = float(search_precision)
        recall = float(search_recall)
        f1 = float(search_f1)
        threshold_source = "optimized_on_validation_fold"

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold_source": threshold_source,
        "search_threshold": float(search_thr),
        "search_precision": float(search_precision),
        "search_recall": float(search_recall),
        "search_f1": float(search_f1),
        "min_threshold": float(min_threshold),
        "max_predicted_positive_rate": float(max_pred_pos_rate),
        "min_precision": float(min_precision),
        "min_recall": float(min_recall),
    }


def select_feature_cols_for_model(df: pd.DataFrame, label_col: str):
    return feature_columns(df, label_col=label_col)


def _guardrail_warnings(rep: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    main_ap = float(rep["main"]["AP"])
    const0_ap = float(rep["dummy_const0"]["AP"])
    random_ap = float(rep["dummy_random_uniform"]["AP"])

    if main_ap + 1e-6 < const0_ap:
        warnings.append(
            "MAIN_AP < dummy_const0_AP "
            f"(main_AP={main_ap:.4f}, const0_AP={const0_ap:.4f}); "
            "check label/pipeline direction."
        )

    if random_ap > main_ap - 0.01:
        warnings.append(
            "Dummy random is close to MAIN "
            f"(main_AP={main_ap:.4f}, random_AP={random_ap:.4f}); "
            "check split/label signal."
        )

    simple2 = rep.get("dummy_simple2feat_lr")
    simple2_margin = _env_float("MAIN_XGB_GUARDRAIL_SIMPLE2_AP_MARGIN", 0.02)
    if simple2:
        simple2_ap = float(simple2["AP"])
        simple2_delta = simple2_ap - main_ap
        if simple2_delta > simple2_margin:
            simple2_features = ",".join(simple2.get("features") or [])
            warnings.append(
                "2-feature LR beats MAIN on AP "
                f"(delta={simple2_delta:.4f}, main_AP={main_ap:.4f}, "
                f"simple2_AP={simple2_ap:.4f}, features={simple2_features}, "
                f"margin={simple2_margin:.4f}); investigate weak trial, feature handling, or noisy features."
            )

    return warnings


def guardrail_sanity(df_tr, df_va, label_col: str, feat_cols: list, main_prob: np.ndarray, seed: int = 42):
    y_va = df_va[label_col].astype(int).to_numpy()
    prev = float(y_va.mean())

    def met(y, p):
        out = {"AP": float(average_precision_score(y, p))}
        out["ROC_AUC"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
        return out

    rep = {"val_prevalence": prev, "main": met(y_va, main_prob)}

    # Dummy A: constant 0
    rep["dummy_const0"] = met(y_va, np.zeros_like(main_prob, dtype=float))

    # Dummy B: random uniform
    rng = np.random.default_rng(seed)
    rep["dummy_random_uniform"] = met(y_va, rng.random(len(main_prob)))

    # Dummy C: simple 2-feature LR (auto pick top 2 numeric by abs corr)
    num_cols = [c for c in feat_cols if np.issubdtype(df_tr[c].dtype, np.number)]
    pick = []
    if len(num_cols) >= 2:
        y_tr = df_tr[label_col].astype(int).to_numpy()
        corrs = []
        for c in num_cols:
            x = pd.to_numeric(df_tr[c], errors="coerce").fillna(0).to_numpy()
            if np.std(x) < 1e-9:
                continue
            cor = np.corrcoef(x, y_tr)[0, 1]
            if np.isnan(cor):
                continue
            corrs.append((c, abs(float(cor))))
        corrs.sort(key=lambda z: z[1], reverse=True)
        pick = [c for c, _ in corrs[:2]]

    if len(pick) == 2:
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(max_iter=2000))
        ])
        pipe.fit(df_tr[pick], df_tr[label_col].astype(int))
        p2 = pipe.predict_proba(df_va[pick])[:, 1]
        rep["dummy_simple2feat_lr"] = {**met(y_va, p2), "features": pick}
    else:
        rep["dummy_simple2feat_lr"] = None

    # ---- RULES / WARNINGS
    warns = []
    main_ap = rep["main"]["AP"]
    base_ap = rep["dummy_const0"]["AP"]
    if main_ap + 1e-6 < base_ap:
        warns.append("MAIN_AP < dummy_const0_AP → pipeline/label có vấn đề hoặc leakage ngược.")
    if rep["dummy_random_uniform"]["AP"] > main_ap - 0.01:
        warns.append("Dummy random gần bằng MAIN → nghi leakage/label sai/split sai.")
    if rep["dummy_simple2feat_lr"] and rep["dummy_simple2feat_lr"]["AP"] > main_ap + 0.01:
        warns.append("Model 2-feature vượt MAIN → MAIN training/feature handling có vấn đề.")
    rep["warnings"] = _guardrail_warnings(rep)

    return rep

def train_main_xgb_option_B(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    cfg: dict,
):
    h = int(cfg["horizon"])
    label_col = f"y_churn_t_plus_{h}"

    feat_cols = select_feature_cols_for_model(
        pd.concat([df_tr, df_va], axis=0),
        label_col=label_col
    )

    X_tr = df_tr[feat_cols].copy()
    X_va = df_va[feat_cols].copy()
    y_tr = df_tr[label_col].astype(int).to_numpy()
    y_va = df_va[label_col].astype(int).to_numpy()

    baseline_spw = float(cfg["best_spw"])
    spw, weighted_spw_raw, spw_cap = _xgb_scale_pos_weight(y_tr, baseline_spw)
    thr_baseline = float(cfg["best_threshold"])
    params, es_rounds = _xgb_training_params(cfg, spw)
    effective_spw = float(params.get("scale_pos_weight", spw))

    n_churn_train = int(y_tr.sum())
    n_active_train = len(y_tr) - n_churn_train
    churn_ratio_train = n_churn_train / max(len(y_tr), 1)

    n_churn_val = int(y_va.sum())
    n_active_val = len(y_va) - n_churn_val
    churn_ratio_val = n_churn_val / max(len(y_va), 1)

    logger.info(
        "[MAIN MODEL][XGB BALANCE] baseline_spw=%.2f | weighted_spw_raw=%.2f | "
        "xgb_spw=%.2f | spw_cap=%.2f | mode=%s | auto_no_balance_rate>=%.2f%%",
        baseline_spw,
        weighted_spw_raw,
        effective_spw,
        spw_cap,
        (os.getenv("MAIN_XGB_SCALE_POS_WEIGHT_MODE") or "auto"),
        _env_float("MAIN_XGB_CLASS_WEIGHT_MAX_RATE", 0.10) * 100.0,
    )

    logger.info(
        "[MAIN MODEL] Train: Churn=%d | Active=%d | Total=%d | Tỷ lệ Churn=%.2f%% | spw=%.2f",
        n_churn_train, n_active_train, len(y_tr), churn_ratio_train * 100, effective_spw,
    )
    logger.info(
        "[MAIN MODEL] Val:   Churn=%d | Active=%d | Total=%d | Tỷ lệ Churn=%.2f%%",
        n_churn_val, n_active_val, len(y_va), churn_ratio_val * 100,
    )

    # detect categorical — exclude date-string columns (YYYY-MM-DD) which change each month
    obj_cols = [c for c in feat_cols if (X_tr[c].dtype == "object" or str(X_tr[c].dtype) == "category")]
    date_cols = [c for c in obj_cols if is_date_like_col(X_tr[c])]
    cat_cols = [c for c in obj_cols if c not in date_cols]

    # convert date-string cols to numeric ordinal
    for c in date_cols:
        X_tr[c] = date_col_to_ordinal(X_tr[c])
        X_va[c] = date_col_to_ordinal(X_va[c])

    # category must NOT be pandas string[python]
    for c in cat_cols:
        X_tr[c] = safe_to_category(X_tr[c])
        X_va[c] = safe_to_category(X_va[c])

    # numeric cols
    for c in feat_cols:
        if c not in cat_cols and c not in date_cols:
            X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce")
            X_va[c] = pd.to_numeric(X_va[c], errors="coerce")

    feature_name_map = None

    # ---- train native categorical if possible
    try:
        X_tr_s, map_native = sanitize_xgb_feature_names(X_tr)
        X_va_s = X_va.rename(columns=map_native)

        model = xgb.XGBClassifier(**params, enable_categorical=True, early_stopping_rounds=es_rounds)
        model = fit_xgb_with_early_stopping(
            model,
            X_tr_s,
            y_tr,
            X_va_s,
            y_va,
            es_rounds=es_rounds,
        )

        used_mode = "native_categorical"
        feature_name_map = map_native

        va_prob = predict_proba_best_iteration(model, X_va_s)[:, 1]

    except Exception:
        # fallback onehot
        X_tr_oh, X_va_oh, map_oh = onehot_align_train_val(X_tr, X_va, cat_cols=cat_cols)

        model = xgb.XGBClassifier(**params, early_stopping_rounds=es_rounds)
        model = fit_xgb_with_early_stopping(
            model,
            X_tr_oh,
            y_tr,
            X_va_oh,
            y_va,
            es_rounds=es_rounds,
        )

        used_mode = "one_hot"
        feature_name_map = map_oh

        va_prob = predict_proba_best_iteration(model, X_va_oh)[:, 1]

    # baseline threshold (tham chiếu)
    score_stats = _score_stats(va_prob)
    logger.info(
        "[MAIN MODEL] Score spread: min=%.6f p01=%.6f p10=%.6f p50=%.6f "
        "p90=%.6f p99=%.6f max=%.6f range=%.6f unique_rounded_6=%d",
        score_stats["score_min"],
        score_stats["score_p01"],
        score_stats["score_p10"],
        score_stats["score_p50"],
        score_stats["score_p90"],
        score_stats["score_p99"],
        score_stats["score_max"],
        score_stats["score_range"],
        score_stats["score_unique_rounded_6"],
    )

    p_b, r_b, f1_b = prf1_at_threshold(y_va, va_prob, thr_baseline)

    threshold_metrics = _main_threshold_metrics(y_va, va_prob, cfg)
    thr_opt = threshold_metrics["threshold"]
    p_opt = threshold_metrics["precision"]
    r_opt = threshold_metrics["recall"]
    f1_opt = threshold_metrics["f1"]
    threshold_source = threshold_metrics["threshold_source"]
    thr_search_opt = threshold_metrics["search_threshold"]
    p_search_opt = threshold_metrics["search_precision"]
    r_search_opt = threshold_metrics["search_recall"]
    f1_search_opt = threshold_metrics["search_f1"]
    min_threshold = threshold_metrics["min_threshold"]
    max_pred_pos_rate = threshold_metrics["max_predicted_positive_rate"]
    min_precision = threshold_metrics["min_precision"]
    min_recall = threshold_metrics["min_recall"]
    top5 = _top_percentile_metrics(y_va, va_prob, 0.05)
    top10 = _top_percentile_metrics(y_va, va_prob, 0.10)
    ap = average_precision_np(y_va, va_prob)
    roc_auc = float(roc_auc_score(y_va, va_prob)) if len(np.unique(y_va)) == 2 else None
    logger.info(
        "[MAIN CLASSIFICATION METRICS] F1=%.4f precision=%.4f recall=%.4f "
        "AP=%.4f ROC_AUC=%s prevalence=%.4f%% threshold=%.6f threshold_source=%s "
        "search_opt_threshold=%.6f threshold_min=%.6f max_pred_pos_rate=%.2f%% "
        "min_precision=%.4f min_recall=%.4f top5_precision=%.4f top10_precision=%.4f",
        f1_opt,
        p_opt,
        r_opt,
        ap,
        f"{roc_auc:.4f}" if roc_auc is not None else "n/a",
        100.0 * float(y_va.mean()),
        thr_opt,
        threshold_source,
        thr_search_opt,
        min_threshold,
        100.0 * max_pred_pos_rate,
        min_precision,
        min_recall,
        float(top5["precision"]),
        float(top10["precision"]),
    )

    # Calculate confusion matrix at optimal threshold
    y_pred_opt = (va_prob >= thr_opt).astype(int)
    pred_pos_rate = float(y_pred_opt.mean())
    cm = confusion_matrix(y_va, y_pred_opt)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    logger.info(
        "[MAIN MODEL] Confusion Matrix @ threshold=%.4f: "
        "TN=%d FP=%d FN=%d TP=%d | "
        "Precision=%.4f Recall=%.4f F1=%.4f PredictedPositiveRate=%.2f%% ActualPrevalence=%.2f%%",
        thr_opt, tn, fp, fn, tp,
        tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9), f1_opt,
        100.0 * pred_pos_rate, 100.0 * float(y_va.mean()),
    )

    # Degenerate guard: predict-all-positive → vô dụng cho production
    if tn + fn == 0:
        raise ValueError(
            f"XGBoost degenerate: TN=0, FN=0 (predict-all-positive). "
            f"K={cfg['best_k']}, threshold={thr_opt:.4f}, "
            f"val_prevalence={y_va.mean():.3f}. Bỏ qua variant này."
        )

    # Score compression guard: model không phân biệt được customers
    min_score_range = _env_float("MAIN_XGB_MIN_SCORE_RANGE", 0.05)
    score_range = score_stats["score_range"]
    if score_range < min_score_range:
        raise ValueError(
            f"XGBoost score range quá hẹp: {score_range:.4f}. "
            f"Scores trong [{va_prob.min():.3f}, {va_prob.max():.3f}]. "
            f"Model không phân biệt được customers. min_required={min_score_range:.4f}."
        )

    # --- early stop meta
    best_it = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)

    # --- sanity guardrail
    sanity = guardrail_sanity(
        df_tr=df_tr,
        df_va=df_va,
        label_col=label_col,
        feat_cols=feat_cols,
        main_prob=va_prob,
        seed=int(cfg.get("seed", 42))
    )

    guardrail_warning = " | ".join(sanity["warnings"]) if sanity["warnings"] else None
    if guardrail_warning:
        logger.warning("[MAIN GUARDRAIL] %s", guardrail_warning)
    diagnostic_warning = None
    dummy_simple2 = sanity["dummy_simple2feat_lr"]
    dummy_simple2_ap = float(dummy_simple2["AP"]) if dummy_simple2 else None
    dummy_simple2_roc = dummy_simple2.get("ROC_AUC") if dummy_simple2 else None
    dummy_simple2_feats = ",".join(dummy_simple2["features"]) if dummy_simple2 else None
    if guardrail_warning:
        logger.warning(
            "[MAIN GUARDRAIL DETAIL] main_AP=%.4f main_ROC=%s const0_AP=%.4f random_AP=%.4f "
            "simple2_AP=%s simple2_ROC=%s simple2_features=%s",
            float(sanity["main"]["AP"]),
            f"{sanity['main']['ROC_AUC']:.4f}" if sanity["main"]["ROC_AUC"] is not None else "n/a",
            float(sanity["dummy_const0"]["AP"]),
            float(sanity["dummy_random_uniform"]["AP"]),
            f"{dummy_simple2_ap:.4f}" if dummy_simple2_ap is not None else "n/a",
            f"{dummy_simple2_roc:.4f}" if dummy_simple2_roc is not None else "n/a",
            dummy_simple2_feats or "n/a",
        )

    report = {
        "K": int(cfg["best_k"]),
        "H": int(cfg["horizon"]),
        "use_static": bool(cfg.get("use_static", False)),
        "val_month": int(df_va["window_end"].astype(int).max()),
        "train_rows": int(len(df_tr)),
        "val_rows": int(len(df_va)),

        "spw_used": float(effective_spw),
        "spw_baseline": float(baseline_spw),
        "spw_weighted_raw": float(weighted_spw_raw),
        "spw_cap": float(spw_cap),
        "used_mode": used_mode,

        "AP_val": float(ap),
        "ROC_AUC_val": roc_auc,
        **score_stats,

        "xgb_es_rounds": int(es_rounds),
        "xgb_best_iteration": int(best_it) if best_it is not None else None,
        "xgb_best_score": float(best_score) if best_score is not None else None,
        "xgb_params": dict(params),

        "val_prevalence": float(sanity["val_prevalence"]),
        "dummy_ap_const0": float(sanity["dummy_const0"]["AP"]),
        "dummy_ap_random": float(sanity["dummy_random_uniform"]["AP"]),
        "dummy_ap_simple2": dummy_simple2_ap,
        "dummy_roc_simple2": float(dummy_simple2_roc) if dummy_simple2_roc is not None else None,
        "dummy_simple2_features": dummy_simple2_feats,
        "guardrail_warning": guardrail_warning,
        "diagnostic_warning": diagnostic_warning,

        "thr_baseline": float(thr_baseline),
        "precision@baseline_thr": float(p_b),
        "recall@baseline_thr": float(r_b),
        "f1@baseline_thr": float(f1_b),

        "thr_main_opt": float(thr_opt),
        "thr_main_search_opt": float(thr_search_opt),
        "threshold_source": threshold_source,
        "thr_main_min": float(min_threshold),
        "max_predicted_positive_rate": float(max_pred_pos_rate),
        "min_precision": float(min_precision),
        "min_recall": float(min_recall),
        "precision@main_thr": float(p_opt),
        "recall@main_thr": float(r_opt),
        "f1@main_thr": float(f1_opt),
        "predicted_positive_rate@main_thr": float(pred_pos_rate),
        "precision@search_thr": float(p_search_opt),
        "recall@search_thr": float(r_search_opt),
        "f1@search_thr": float(f1_search_opt),
        "precision@top_5pct": float(top5["precision"]),
        "recall@top_5pct": float(top5["recall"]),
        "f1@top_5pct": float(top5["f1"]),
        "threshold@top_5pct": float(top5["threshold"]),
        "precision@top_10pct": float(top10["precision"]),
        "recall@top_10pct": float(top10["recall"]),
        "f1@top_10pct": float(top10["f1"]),
        "threshold@top_10pct": float(top10["threshold"]),
    }

    return model, report, feat_cols, cat_cols, feature_name_map, date_cols


def _wide_xgb_tuning_space() -> dict[str, dict[str, Any]]:
    n_min = _env_int("MAIN_XGB_OPTUNA_N_ESTIMATORS_MIN", 800)
    n_max = _env_int("MAIN_XGB_OPTUNA_N_ESTIMATORS_MAX", 5000)
    n_step = max(_env_int("MAIN_XGB_OPTUNA_N_ESTIMATORS_STEP", 200), 1)
    if n_max < n_min:
        n_max = n_min

    pred_min = _env_float("MAIN_XGB_OPTUNA_MAX_PRED_POS_RATE_MIN", 0.35)
    pred_max = _env_float("MAIN_XGB_OPTUNA_MAX_PRED_POS_RATE_MAX", 0.90)
    if pred_max < pred_min:
        pred_max = pred_min

    space: dict[str, dict[str, Any]] = {
        "n_estimators": {"type": "int", "low": n_min, "high": n_max, "step": n_step},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.20, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 10, "step": 1},
        "max_leaves": {"type": "categorical", "choices": [0, 31, 63, 127, 255]},
        "min_child_weight": {"type": "float", "low": 1.0, "high": 30.0, "log": True},
        "gamma": {"type": "float", "low": 0.0, "high": 8.0, "log": False},
        "subsample": {"type": "float", "low": 0.55, "high": 1.0, "log": False},
        "colsample_bytree": {"type": "float", "low": 0.55, "high": 1.0, "log": False},
        "colsample_bylevel": {"type": "float", "low": 0.50, "high": 1.0, "log": False},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 0.05, "high": 50.0, "log": True},
        "early_stopping_rounds": {"type": "int", "low": 50, "high": 350, "step": 50},
        "max_predicted_positive_rate": {"type": "float", "low": pred_min, "high": pred_max, "log": False},
    }
    if _env_bool("MAIN_XGB_OPTUNA_TUNE_SCALE_POS_WEIGHT", True):
        max_spw = max(
            _env_float(
                "MAIN_XGB_OPTUNA_SCALE_POS_WEIGHT_MAX",
                _env_float("MAIN_XGB_MAX_SCALE_POS_WEIGHT", 20.0),
            ),
            0.5,
        )
        space["scale_pos_weight"] = {"type": "float", "low": 0.5, "high": max_spw, "log": True}
    return space


def _suggest_from_space(trial, name: str, spec: dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=max(int(spec.get("step", 1)), 1),
        )
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    raise ValueError(f"Unsupported tuning space type for {name}: {kind}")


def _build_tuning_trial_cfg(base_cfg: dict, trial, space: dict[str, dict[str, Any]]) -> dict:
    sampled = {name: _suggest_from_space(trial, name, spec) for name, spec in space.items()}

    params: dict[str, Any] = {
        "n_estimators": int(sampled["n_estimators"]),
        "learning_rate": float(sampled["learning_rate"]),
        "max_depth": int(sampled["max_depth"]),
        "max_leaves": int(sampled["max_leaves"]),
        "min_child_weight": float(sampled["min_child_weight"]),
        "gamma": float(sampled["gamma"]),
        "subsample": float(sampled["subsample"]),
        "colsample_bytree": float(sampled["colsample_bytree"]),
        "colsample_bylevel": float(sampled["colsample_bylevel"]),
        "reg_alpha": float(sampled["reg_alpha"]),
        "reg_lambda": float(sampled["reg_lambda"]),
        "tree_method": os.getenv("MAIN_XGB_TREE_METHOD", "hist"),
        "random_state": int(base_cfg.get("seed", 42)),
        "eval_metric": ["aucpr", "logloss"],
    }
    n_jobs = _env_int("MAIN_XGB_N_JOBS", 0)
    if n_jobs > 0:
        params["n_jobs"] = int(n_jobs)

    if "scale_pos_weight" in sampled:
        params["scale_pos_weight"] = float(sampled["scale_pos_weight"])

    tuned_cfg = dict(base_cfg)
    tuned_cfg["main_xgb_params"] = params
    tuned_cfg["main_es_rounds"] = int(sampled["early_stopping_rounds"])
    tuned_cfg["main_max_predicted_positive_rate"] = float(sampled["max_predicted_positive_rate"])
    tuned_cfg["main_threshold_min"] = _env_float("MAIN_XGB_THRESHOLD_MIN", 0.005)
    return tuned_cfg


def _complete_trials(study, optuna_module) -> list:
    return [
        t for t in study.trials
        if t.state == optuna_module.trial.TrialState.COMPLETE and t.value is not None
    ]


def _top_trials_for_narrowing(trials: list) -> list:
    if not trials:
        return []
    ranked = sorted(trials, key=lambda t: float(t.value), reverse=True)
    frac = min(max(_env_float("MAIN_XGB_RANDOM_SEARCH_TOP_FRACTION", 0.25), 0.05), 1.0)
    min_top = max(_env_int("MAIN_XGB_RANDOM_SEARCH_TOP_MIN", 5), 1)
    top_n = min(len(ranked), max(min_top, int(np.ceil(len(ranked) * frac))))
    return ranked[:top_n]


def _align_int_to_step(value: float, origin: int, step: int, direction: str) -> int:
    if step <= 1:
        return int(np.floor(value) if direction == "down" else np.ceil(value))
    offset = (float(value) - float(origin)) / float(step)
    k = np.floor(offset) if direction == "down" else np.ceil(offset)
    return int(origin + int(k) * step)


def _narrow_numeric_spec(spec: dict[str, Any], values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return dict(spec)
    out = dict(spec)
    orig_low = float(spec["low"])
    orig_high = float(spec["high"])
    if orig_high <= orig_low:
        return out

    margin_frac = max(_env_float("MAIN_XGB_RANDOM_SEARCH_NARROW_MARGIN_FRAC", 0.20), 0.0)
    min_width_frac = max(_env_float("MAIN_XGB_RANDOM_SEARCH_NARROW_MIN_WIDTH_FRAC", 0.15), 0.0)

    if bool(spec.get("log", False)):
        log_values = [np.log(v) for v in clean if v > 0]
        if not log_values or orig_low <= 0:
            return out
        low_v = float(np.min(log_values))
        high_v = float(np.max(log_values))
        orig_low_log = float(np.log(orig_low))
        orig_high_log = float(np.log(orig_high))
        orig_span = orig_high_log - orig_low_log
        span = high_v - low_v
        margin = max(span * margin_frac, orig_span * min_width_frac)
        narrowed_low = float(np.exp(max(orig_low_log, low_v - margin)))
        narrowed_high = float(np.exp(min(orig_high_log, high_v + margin)))
    else:
        low_v = float(np.min(clean))
        high_v = float(np.max(clean))
        orig_span = orig_high - orig_low
        span = high_v - low_v
        margin = max(span * margin_frac, orig_span * min_width_frac)
        narrowed_low = max(orig_low, low_v - margin)
        narrowed_high = min(orig_high, high_v + margin)

    if spec["type"] == "int":
        step = max(int(spec.get("step", 1)), 1)
        origin = int(spec["low"])
        low_i = _align_int_to_step(narrowed_low, origin, step, "down")
        high_i = _align_int_to_step(narrowed_high, origin, step, "up")
        low_i = max(int(spec["low"]), low_i)
        high_i = min(int(spec["high"]), high_i)
        if high_i < low_i:
            high_i = low_i
        out["low"] = low_i
        out["high"] = high_i
    else:
        if narrowed_high < narrowed_low:
            narrowed_high = narrowed_low
        out["low"] = float(narrowed_low)
        out["high"] = float(narrowed_high)
    return out


def _narrow_search_space(
    wide_space: dict[str, dict[str, Any]],
    random_trials: list,
) -> dict[str, dict[str, Any]]:
    top_trials = _top_trials_for_narrowing(random_trials)
    if not top_trials:
        return {name: dict(spec) for name, spec in wide_space.items()}

    narrowed: dict[str, dict[str, Any]] = {}
    for name, spec in wide_space.items():
        values = [t.params.get(name) for t in top_trials if name in t.params]
        if spec["type"] == "categorical":
            chosen = []
            for value in values:
                if value not in chosen:
                    chosen.append(value)
            narrowed[name] = {"type": "categorical", "choices": chosen or list(spec["choices"])}
        else:
            narrowed[name] = _narrow_numeric_spec(spec, values)
    return narrowed


def _space_for_log(space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    logged: dict[str, Any] = {}
    for name, spec in space.items():
        if spec["type"] == "categorical":
            logged[name] = list(spec["choices"])
        else:
            logged[name] = {
                "low": spec["low"],
                "high": spec["high"],
                **({"step": spec["step"]} if "step" in spec else {}),
                **({"log": spec["log"]} if "log" in spec else {}),
            }
    return logged


def tune_xgb_hyperparams_for_folds(
    cfg: dict,
    folds: list[dict],
    *,
    n_trials: int | None = None,
    timeout_seconds: int | None = None,
) -> tuple[dict, dict]:
    """Tune XGBoost with random exploration, narrowed TPE Optuna, and walk-forward folds."""
    tpe_trials = int(n_trials if n_trials is not None else _env_int("MAIN_XGB_OPTUNA_TRIALS", 50))
    random_trials = _env_int("MAIN_XGB_RANDOM_SEARCH_TRIALS", 20)
    timeout_seconds = int(
        timeout_seconds
        if timeout_seconds is not None
        else _env_int("MAIN_XGB_OPTUNA_TIMEOUT_SECONDS", 0)
    )
    random_timeout_seconds = _env_int("MAIN_XGB_RANDOM_SEARCH_TIMEOUT_SECONDS", 0)
    if random_trials <= 0 and tpe_trials <= 0:
        return dict(cfg), {"enabled": False, "reason": "random_and_tpe_trials<=0"}

    try:
        import optuna
        from optuna.pruners import MedianPruner, NopPruner
        from optuna.samplers import RandomSampler, TPESampler
    except ImportError as exc:
        logger.warning("[OPTUNA] optuna is not installed; using default XGBoost params: %s", exc)
        return dict(cfg), {"enabled": False, "reason": "optuna_not_installed"}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = int(cfg.get("seed", 42))
    pruning_enabled = _env_bool("MAIN_XGB_OPTUNA_PRUNING_ENABLED", True)
    prune_after_seconds = max(_env_int("MAIN_XGB_OPTUNA_PRUNE_AFTER_SECONDS", 60), 0)

    def make_pruner():
        if not pruning_enabled:
            return NopPruner()
        return MedianPruner(
            n_startup_trials=max(_env_int("MAIN_XGB_OPTUNA_PRUNER_STARTUP_TRIALS", 5), 0),
            n_warmup_steps=max(_env_int("MAIN_XGB_OPTUNA_PRUNER_WARMUP_STEPS", 1), 0),
        )

    def objective(trial, *, phase: str, space: dict[str, dict[str, Any]]) -> float:
        trial_cfg = _build_tuning_trial_cfg(cfg, trial, space)
        trial_cfg["hyperparameter_search_phase"] = phase
        fold_f1: list[float] = []
        fold_ap: list[float] = []
        fold_roc: list[float] = []
        started_at = time.monotonic()
        for fold_idx, fold in enumerate(folds):
            try:
                _model, report, *_rest = train_main_xgb_option_B(
                    fold["train"],
                    fold["val"],
                    trial_cfg,
                )
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                trial.set_user_attr("rejected_reason", str(exc))
                trial.set_user_attr("phase", phase)
                trial.set_user_attr("duration_seconds", float(time.monotonic() - started_at))
                return 0.0

            fold_f1.append(float(report["f1@main_thr"]))
            fold_ap.append(float(report["AP_val"]))
            if report.get("ROC_AUC_val") is not None:
                fold_roc.append(float(report["ROC_AUC_val"]))
            score = float(np.mean(fold_f1))
            trial.report(score, step=fold_idx)
            elapsed = float(time.monotonic() - started_at)
            if pruning_enabled and elapsed >= prune_after_seconds and trial.should_prune():
                trial.set_user_attr("phase", phase)
                trial.set_user_attr("duration_seconds", elapsed)
                raise optuna.TrialPruned()

        trial_cfg["optuna_objective"] = "walk_forward_f1_mean"
        trial.set_user_attr("phase", phase)
        trial.set_user_attr("candidate_cfg", trial_cfg)
        trial.set_user_attr("fold_f1", fold_f1)
        trial.set_user_attr("fold_ap", fold_ap)
        trial.set_user_attr("fold_roc", fold_roc)
        trial.set_user_attr("ap_mean", float(np.mean(fold_ap)) if fold_ap else 0.0)
        trial.set_user_attr("roc_auc_mean", float(np.mean(fold_roc)) if fold_roc else 0.0)
        trial.set_user_attr("duration_seconds", float(time.monotonic() - started_at))
        return float(np.mean(fold_f1)) if fold_f1 else 0.0

    n_jobs = max(_env_int("MAIN_XGB_OPTUNA_N_JOBS", 1), 1)
    wide_space = _wide_xgb_tuning_space()
    logger.info(
        "[HYPERPARAM TUNING] Strategy=random_search_then_narrowed_optuna_tpe "
        "K=%s use_static=%s random_trials=%d optuna_trials=%d pruning=%s prune_after=%ss folds=%d",
        cfg.get("best_k"),
        cfg.get("use_static"),
        random_trials,
        tpe_trials,
        pruning_enabled,
        prune_after_seconds,
        len(folds),
    )

    random_study = None
    random_complete: list = []
    if random_trials > 0:
        random_study = optuna.create_study(
            study_name=f"churn_xgb_random_k{cfg.get('best_k')}_static{int(bool(cfg.get('use_static', False)))}",
            direction="maximize",
            sampler=RandomSampler(seed=seed),
            pruner=make_pruner(),
        )
        logger.info(
            "[RANDOM SEARCH] Start XGBoost exploration K=%s use_static=%s trials=%d timeout=%s objective=walk_forward_f1_mean",
            cfg.get("best_k"),
            cfg.get("use_static"),
            random_trials,
            random_timeout_seconds or "none",
        )
        random_study.optimize(
            lambda trial: objective(trial, phase="random_search", space=wide_space),
            n_trials=int(random_trials),
            timeout=int(random_timeout_seconds) if random_timeout_seconds > 0 else None,
            n_jobs=n_jobs,
            show_progress_bar=False,
            gc_after_trial=True,
        )
        random_complete = _complete_trials(random_study, optuna)
        if random_complete:
            logger.info(
                "[RANDOM SEARCH] Completed=%d/%d best_F1=%.4f best_params=%s",
                len(random_complete),
                len(random_study.trials),
                float(max(t.value for t in random_complete)),
                dict(max(random_complete, key=lambda t: float(t.value)).params),
            )
        else:
            logger.warning("[RANDOM SEARCH] No completed trials; Optuna will use the wide space.")

    narrowed_space = _narrow_search_space(wide_space, random_complete)
    logger.info("[SEARCH SPACE NARROWED] %s", _space_for_log(narrowed_space))

    tpe_study = None
    tpe_complete: list = []
    if tpe_trials > 0:
        tpe_study = optuna.create_study(
            study_name=f"churn_xgb_tpe_k{cfg.get('best_k')}_static{int(bool(cfg.get('use_static', False)))}",
            direction="maximize",
            sampler=TPESampler(seed=seed + 1),
            pruner=make_pruner(),
        )
        logger.info(
            "[OPTUNA] Start narrowed TPE tuning K=%s use_static=%s trials=%d timeout=%s objective=walk_forward_f1_mean",
            cfg.get("best_k"),
            cfg.get("use_static"),
            tpe_trials,
            timeout_seconds or "none",
        )
        tpe_study.optimize(
            lambda trial: objective(trial, phase="optuna_tpe", space=narrowed_space),
            n_trials=int(tpe_trials),
            timeout=int(timeout_seconds) if timeout_seconds > 0 else None,
            n_jobs=n_jobs,
            show_progress_bar=False,
            gc_after_trial=True,
        )
        tpe_complete = _complete_trials(tpe_study, optuna)
        if tpe_complete:
            logger.info(
                "[OPTUNA] Completed=%d/%d best_F1=%.4f best_params=%s",
                len(tpe_complete),
                len(tpe_study.trials),
                float(max(t.value for t in tpe_complete)),
                dict(max(tpe_complete, key=lambda t: float(t.value)).params),
            )
        else:
            logger.warning("[OPTUNA] No completed narrowed TPE trials.")

    complete_trials = random_complete + tpe_complete
    if not complete_trials:
        logger.warning("[OPTUNA] No completed trials for K=%s use_static=%s; using defaults.", cfg.get("best_k"), cfg.get("use_static"))
        return dict(cfg), {"enabled": True, "status": "no_completed_trials"}

    best = max(complete_trials, key=lambda t: float(t.value))
    tuned_cfg = dict(best.user_attrs.get("candidate_cfg") or cfg)
    tuning_meta = {
        "enabled": True,
        "status": "completed",
        "strategy": "random_search_then_narrowed_optuna_tpe",
        "objective": "walk_forward_f1_mean",
        "random_search_trials": len(random_study.trials) if random_study is not None else 0,
        "random_search_completed_trials": len(random_complete),
        "optuna_tpe_trials": len(tpe_study.trials) if tpe_study is not None else 0,
        "optuna_tpe_completed_trials": len(tpe_complete),
        "n_trials": (len(random_study.trials) if random_study is not None else 0) + (len(tpe_study.trials) if tpe_study is not None else 0),
        "completed_trials": len(complete_trials),
        "best_phase": best.user_attrs.get("phase"),
        "best_value": float(best.value),
        "best_params": dict(best.params),
        "fold_f1": best.user_attrs.get("fold_f1", []),
        "fold_ap": best.user_attrs.get("fold_ap", []),
        "fold_roc": best.user_attrs.get("fold_roc", []),
        "ap_mean": best.user_attrs.get("ap_mean"),
        "roc_auc_mean": best.user_attrs.get("roc_auc_mean"),
        "pruning_enabled": pruning_enabled,
        "prune_after_seconds": prune_after_seconds,
        "wide_space": _space_for_log(wide_space),
        "narrowed_space": _space_for_log(narrowed_space),
    }
    tuned_cfg["optuna_tuning"] = tuning_meta
    logger.info(
        "[HYPERPARAM TUNING] Best K=%s use_static=%s phase=%s F1_mean=%.4f params=%s",
        cfg.get("best_k"),
        cfg.get("use_static"),
        best.user_attrs.get("phase"),
        float(best.value),
        dict(best.params),
    )
    return tuned_cfg, tuning_meta


def _split_tuning_and_holdout_folds(folds: list[dict]) -> tuple[list[dict], dict | None, int]:
    min_tuning_folds = max(_env_int("MAIN_XGB_MIN_TUNING_FOLDS", 5), 1)
    holdout_enabled = _env_bool("MAIN_XGB_FINAL_HOLDOUT_ENABLED", True)
    if holdout_enabled and len(folds) >= min_tuning_folds + 1:
        final_holdout_fold = folds[-1]
        tuning_folds = folds[:-1]
        logger.info(
            "[FINAL HOLDOUT] Reserving latest fold for final evaluation only: "
            "train<=%s val=%s tuning_folds=%d",
            final_holdout_fold["train_max_month"],
            ",".join(str(m) for m in final_holdout_fold["validation_months"]),
            len(tuning_folds),
        )
        return tuning_folds, final_holdout_fold, min_tuning_folds

    if holdout_enabled:
        logger.warning(
            "[FINAL HOLDOUT] Not enough folds for %d tuning fold(s) + 1 holdout; "
            "using all %d fold(s) for training/evaluation.",
            min_tuning_folds,
            len(folds),
        )
    return folds, None, min_tuning_folds


def _fixed_threshold_from_fold_outputs(fold_outputs: list[dict]) -> tuple[float | None, list[float]]:
    threshold_values = [
        float(out["report"]["thr_main_opt"])
        for out in fold_outputs
        if out["report"].get("threshold_source") != "fixed_from_tuning_folds"
        and out["report"].get("thr_main_opt") is not None
    ]
    if not threshold_values:
        return None, []
    return float(np.median(threshold_values)), threshold_values


def evaluate_existing_bundle_on_current_folds(
    engine,
    cfg: dict,
    df_static: pd.DataFrame,
    model: Any,
    metadata: dict,
) -> dict[str, Any]:
    """Evaluate an already accepted bundle on the same current walk-forward policy."""
    from preprocess.trainval import build_walk_forward_for_main

    bundle_cfg = dict((metadata or {}).get("cfg") or cfg)
    eval_cfg = dict(cfg)
    eval_cfg.update({k: v for k, v in bundle_cfg.items() if v is not None})
    eval_cfg["horizon"] = int(cfg.get("horizon") or eval_cfg.get("horizon", 2))
    eval_cfg["best_k"] = int(eval_cfg.get("best_k") or cfg["best_k"])
    use_static = bool(eval_cfg.get("use_static", False))

    _df_all, folds = build_walk_forward_for_main(
        engine,
        eval_cfg,
        df_static,
        use_static_override=use_static,
    )
    label_col = f"y_churn_t_plus_{int(eval_cfg['horizon'])}"
    fold_reports: list[dict[str, Any]] = []
    for fold_idx, fold in enumerate(folds, start=1):
        df_va = fold["val"]
        if label_col not in df_va.columns or df_va[label_col].nunique() < 2:
            logger.warning(
                "[PREV BUNDLE RE-EVAL] fold=%d/%d skipped: missing or single-class labels",
                fold_idx,
                len(folds),
            )
            continue
        X_va = _prepare_bundle_features(df_va, cfg=eval_cfg, metadata=metadata or {}, model=model)
        y_va = df_va[label_col].astype(int).to_numpy()
        prob = predict_proba_best_iteration(model, X_va)[:, 1]
        report = _evaluate_probabilities(y_va, prob, eval_cfg)
        report["val_month"] = int(df_va["window_end"].astype(int).max())
        report["train_max_month"] = int(fold["train_max_month"])
        report["validation_months"] = list(fold["validation_months"])
        fold_reports.append(report)
        logger.info(
            "[PREV BUNDLE RE-EVAL] fold=%d/%d train<=%s val=%s F1=%.4f "
            "precision=%.4f recall=%.4f AP=%.4f ROC_AUC=%s threshold=%.6f",
            fold_idx,
            len(folds),
            fold["train_max_month"],
            ",".join(str(m) for m in fold["validation_months"]),
            float(report["f1@main_thr"]),
            float(report["precision@main_thr"]),
            float(report["recall@main_thr"]),
            float(report["AP_val"]),
            f"{report.get('ROC_AUC_val'):.4f}" if report.get("ROC_AUC_val") is not None else "n/a",
            float(report["thr_main_opt"]),
        )

    if not fold_reports:
        raise ValueError("Previous bundle re-evaluation produced no valid folds")

    roc_values = [r.get("ROC_AUC_val") for r in fold_reports if r.get("ROC_AUC_val") is not None]
    out = {
        "folds": len(fold_reports),
        "total_folds": len(folds),
        "F1_val": float(np.mean([r["f1@main_thr"] for r in fold_reports])),
        "precision": float(np.mean([r["precision@main_thr"] for r in fold_reports])),
        "recall": float(np.mean([r["recall@main_thr"] for r in fold_reports])),
        "AP_val": float(np.mean([r["AP_val"] for r in fold_reports])),
        "ROC_AUC_val": float(np.mean(roc_values)) if roc_values else 0.0,
        "latest_F1": float(fold_reports[-1]["f1@main_thr"]),
        "latest_AP": float(fold_reports[-1]["AP_val"]),
        "latest_ROC_AUC": fold_reports[-1].get("ROC_AUC_val"),
        "fold_reports": fold_reports,
    }
    logger.info(
        "[PREV BUNDLE RE-EVAL] folds=%d/%d F1=%.4f precision=%.4f recall=%.4f AP=%.4f ROC_AUC=%.4f latest_F1=%.4f",
        int(out["folds"]),
        int(out["total_folds"]),
        float(out["F1_val"]),
        float(out["precision"]),
        float(out["recall"]),
        float(out["AP_val"]),
        float(out["ROC_AUC_val"]),
        float(out["latest_F1"]),
    )
    return out


def run_main_variant(
    engine,
    cfg: dict,
    df_static: pd.DataFrame,
    use_static_flag: bool,
    *,
    tune_hyperparams: bool = False,
    optuna_trials: int | None = None,
    optuna_timeout_seconds: int | None = None,
):
    from preprocess.trainval import build_walk_forward_for_main
    df_all, folds = build_walk_forward_for_main(
        engine,
        cfg,
        df_static,
        use_static_override=use_static_flag,
    )
    cfg_tmp = dict(cfg)
    cfg_tmp["use_static"] = bool(use_static_flag)
    tuning_meta = None
    final_holdout_fold = None
    tuning_folds = folds

    if tune_hyperparams:
        tuning_folds, final_holdout_fold, min_tuning_folds = _split_tuning_and_holdout_folds(folds)
        if len(tuning_folds) < min_tuning_folds:
            logger.warning(
                "[HYPERPARAM TUNING] Skipping tuning because tuning_folds=%d < min_tuning_folds=%d. "
                "Using base XGBoost parameters.",
                len(tuning_folds),
                min_tuning_folds,
            )
            tuning_meta = {
                "enabled": False,
                "reason": "insufficient_tuning_folds",
                "tuning_folds": len(tuning_folds),
                "min_tuning_folds": min_tuning_folds,
            }
        else:
            cfg_tmp, tuning_meta = tune_xgb_hyperparams_for_folds(
                cfg_tmp,
                tuning_folds,
                n_trials=optuna_trials,
                timeout_seconds=optuna_timeout_seconds,
            )
            cfg_tmp["use_static"] = bool(use_static_flag)

    fold_outputs = []
    rejected_folds = []
    for fold_idx, fold in enumerate(folds, start=1):
        fold_cfg = cfg_tmp
        is_final_holdout = final_holdout_fold is not None and fold is final_holdout_fold
        if is_final_holdout:
            fixed_threshold, threshold_values = _fixed_threshold_from_fold_outputs(fold_outputs)
            if fixed_threshold is not None:
                fold_cfg = dict(cfg_tmp)
                fold_cfg["main_fixed_threshold"] = fixed_threshold
                fold_cfg["main_fixed_threshold_source"] = "fixed_from_tuning_folds"
                logger.info(
                    "[FINAL HOLDOUT] Using fixed threshold=%.6f from %d tuning fold threshold(s): %s",
                    fixed_threshold,
                    len(threshold_values),
                    ",".join(f"{x:.6f}" for x in threshold_values),
                )
            else:
                logger.warning(
                    "[FINAL HOLDOUT] No tuning-fold threshold available; holdout will optimize threshold on itself."
                )
        try:
            model, report, feat_cols, cat_cols, fmap, date_cols = train_main_xgb_option_B(
                fold["train"],
                fold["val"],
                fold_cfg,
            )
        except ValueError as e:
            rejected_folds.append({
                "fold_idx": int(fold_idx),
                "train_max_month": int(fold["train_max_month"]),
                "validation_months": list(fold["validation_months"]),
                "reason": str(e),
                "is_latest": bool(fold is folds[-1]),
                "is_final_holdout": bool(is_final_holdout),
            })
            logger.warning(
                "Variant K=%d use_static=%s fold=%d/%d rejected: %s",
                cfg["best_k"],
                use_static_flag,
                fold_idx,
                len(folds),
                e,
            )
            continue
        logger.info(
            "[MAIN WALK FORWARD] K=%d use_static=%s fold=%d/%d train<=%s val=%s "
            "F1=%.4f AP=%.4f ROC_AUC=%s",
            cfg["best_k"],
            use_static_flag,
            fold_idx,
            len(folds),
            fold["train_max_month"],
            ",".join(str(m) for m in fold["validation_months"]),
            float(report["f1@main_thr"]),
            float(report["AP_val"]),
            f"{report.get('ROC_AUC_val'):.4f}" if report.get("ROC_AUC_val") is not None else "n/a",
        )
        fold_outputs.append({
            "model": model,
            "report": report,
            "feat_cols": feat_cols,
            "cat_cols": cat_cols,
            "feature_name_map": fmap,
            "date_cols": date_cols,
            "fold": fold,
        })

    if not fold_outputs:
        return {"use_static": bool(use_static_flag), "guardrail_warning": "all walk-forward folds rejected",
                "F1_val": 0.0, "AP_val": 0.0, "ROC_AUC_val": 0.0,
                "report": {
                    "K": int(cfg["best_k"]),
                    "H": int(cfg["horizon"]),
                    "use_static": bool(use_static_flag),
                    "rejected_folds": rejected_folds,
                    "walk_forward_total_folds_requested": len(folds),
                    "guardrail_warning": "all walk-forward folds rejected",
                }}

    rejection_rate = len(rejected_folds) / max(len(folds), 1)
    max_rejection_rate = max(_env_float("MAIN_XGB_MAX_REJECTED_FOLD_RATE", 0.25), 0.0)
    latest_rejected = any(bool(r.get("is_latest")) for r in rejected_folds)
    require_latest_valid = _env_bool("MAIN_XGB_REQUIRE_LATEST_FOLD_VALID", True)
    if latest_rejected and require_latest_valid:
        warning = "latest walk-forward fold rejected"
        logger.warning(
            "[MAIN WALK FORWARD REJECT] K=%d use_static=%s rejected variant: %s",
            cfg["best_k"],
            use_static_flag,
            warning,
        )
        return {
            "use_static": bool(use_static_flag),
            "guardrail_warning": warning,
            "F1_val": 0.0,
            "AP_val": 0.0,
            "ROC_AUC_val": 0.0,
            "report": {
                "K": int(cfg["best_k"]),
                "H": int(cfg["horizon"]),
                "use_static": bool(use_static_flag),
                "rejected_folds": rejected_folds,
                "walk_forward_total_folds_requested": len(folds),
                "walk_forward_rejected_fold_rate": float(rejection_rate),
                "guardrail_warning": warning,
            },
        }
    if rejection_rate > max_rejection_rate:
        warning = f"too many rejected walk-forward folds ({rejection_rate:.2%} > {max_rejection_rate:.2%})"
        logger.warning(
            "[MAIN WALK FORWARD REJECT] K=%d use_static=%s rejected variant: %s",
            cfg["best_k"],
            use_static_flag,
            warning,
        )
        return {
            "use_static": bool(use_static_flag),
            "guardrail_warning": warning,
            "F1_val": 0.0,
            "AP_val": 0.0,
            "ROC_AUC_val": 0.0,
            "report": {
                "K": int(cfg["best_k"]),
                "H": int(cfg["horizon"]),
                "use_static": bool(use_static_flag),
                "rejected_folds": rejected_folds,
                "walk_forward_total_folds_requested": len(folds),
                "walk_forward_rejected_fold_rate": float(rejection_rate),
                "guardrail_warning": warning,
            },
        }

    latest = fold_outputs[-1]
    holdout_outputs = [
        out for out in fold_outputs
        if final_holdout_fold is not None and out["fold"] is final_holdout_fold
    ]
    selection_outputs = [
        out for out in fold_outputs
        if final_holdout_fold is None or out["fold"] is not final_holdout_fold
    ]
    if not selection_outputs:
        selection_outputs = fold_outputs
    fold_reports = [out["report"] for out in selection_outputs]
    all_fold_reports = [out["report"] for out in fold_outputs]
    f1_mean = float(np.mean([r["f1@main_thr"] for r in fold_reports]))
    ap_mean = float(np.mean([r["AP_val"] for r in fold_reports]))
    roc_values = [r.get("ROC_AUC_val") for r in fold_reports if r.get("ROC_AUC_val") is not None]
    roc_mean = float(np.mean(roc_values)) if roc_values else 0.0
    precision_mean = float(np.mean([r["precision@main_thr"] for r in fold_reports]))
    recall_mean = float(np.mean([r["recall@main_thr"] for r in fold_reports]))
    val_prevalence_mean = float(np.mean([r["val_prevalence"] for r in fold_reports]))

    report = dict(latest["report"])
    report["walk_forward_folds"] = len(selection_outputs)
    report["walk_forward_total_folds"] = len(fold_outputs)
    report["walk_forward_total_folds_requested"] = len(folds)
    report["walk_forward_rejected_folds"] = rejected_folds
    report["walk_forward_rejected_fold_rate"] = float(rejection_rate)
    report["walk_forward_holdout_excluded_from_selection"] = bool(holdout_outputs)
    report["walk_forward_reports"] = fold_reports
    report["walk_forward_all_reports"] = all_fold_reports
    report["f1@main_thr_latest"] = float(latest["report"]["f1@main_thr"])
    report["AP_val_latest"] = float(latest["report"]["AP_val"])
    report["ROC_AUC_val_latest"] = latest["report"].get("ROC_AUC_val")
    report["precision@main_thr_latest"] = float(latest["report"]["precision@main_thr"])
    report["recall@main_thr_latest"] = float(latest["report"]["recall@main_thr"])
    report["val_prevalence_latest"] = float(latest["report"]["val_prevalence"])
    if tuning_meta is not None:
        report["optuna_tuning"] = tuning_meta
    report["f1@main_thr"] = f1_mean
    report["AP_val"] = ap_mean
    report["ROC_AUC_val"] = roc_mean
    report["precision@main_thr"] = precision_mean
    report["recall@main_thr"] = recall_mean
    report["val_prevalence"] = val_prevalence_mean
    report["val_month"] = int(latest["report"]["val_month"])
    if final_holdout_fold is not None:
        holdout_report = holdout_outputs[-1]["report"] if holdout_outputs else None
        report["final_holdout"] = {
            "enabled": True,
            "status": "completed" if holdout_report is not None else "rejected_or_unavailable",
            "used_for_hyperparameter_search": False,
            "used_for_model_selection": False,
            "train_max_month": final_holdout_fold["train_max_month"],
            "validation_months": list(final_holdout_fold["validation_months"]),
        }
        if holdout_report is not None:
            report["final_holdout"].update({
                "f1": float(holdout_report["f1@main_thr"]),
                "precision": float(holdout_report["precision@main_thr"]),
                "recall": float(holdout_report["recall@main_thr"]),
                "ap": float(holdout_report["AP_val"]),
                "roc_auc": holdout_report.get("ROC_AUC_val"),
                "threshold": float(holdout_report["thr_main_opt"]),
                "search_opt_threshold": float(holdout_report.get("thr_main_search_opt", holdout_report["thr_main_opt"])),
                "threshold_source": holdout_report.get("threshold_source"),
                "val_prevalence": float(holdout_report["val_prevalence"]),
            })
            logger.info(
                "[FINAL HOLDOUT] K=%d use_static=%s val=%s F1=%.4f precision=%.4f "
                "recall=%.4f AP=%.4f ROC_AUC=%s threshold=%.6f threshold_source=%s",
                cfg["best_k"],
                use_static_flag,
                ",".join(str(m) for m in final_holdout_fold["validation_months"]),
                float(holdout_report["f1@main_thr"]),
                float(holdout_report["precision@main_thr"]),
                float(holdout_report["recall@main_thr"]),
                float(holdout_report["AP_val"]),
                f"{holdout_report.get('ROC_AUC_val'):.4f}" if holdout_report.get("ROC_AUC_val") is not None else "n/a",
                float(holdout_report["thr_main_opt"]),
                holdout_report.get("threshold_source"),
            )

    logger.info(
        "[MAIN WALK FORWARD METRICS] K=%d use_static=%s selection_folds=%d total_folds=%d "
        "F1_mean=%.4f precision_mean=%.4f recall_mean=%.4f AP_mean=%.4f ROC_AUC_mean=%.4f latest_F1=%.4f",
        cfg["best_k"],
        use_static_flag,
        len(selection_outputs),
        len(fold_outputs),
        f1_mean,
        precision_mean,
        recall_mean,
        ap_mean,
        roc_mean,
        float(latest["report"]["f1@main_thr"]),
    )

    # baseline profile for monitoring drift (built on all historical labeled rows)
    feature_profile = compute_feature_profile(
        df_all,
        feat_cols=latest["feat_cols"],
        cat_cols=latest["cat_cols"],
    )

    out = {
        "use_static": bool(use_static_flag),
        "val_month": int(report["val_month"]),
        "train_rows": int(sum(len(out["fold"]["train"]) for out in fold_outputs)),
        "val_rows": int(sum(len(out["fold"]["val"]) for out in fold_outputs)),
        "AP_val": ap_mean,
        "F1_val": f1_mean,
        "ROC_AUC_val": float(report["ROC_AUC_val"] or 0.0),
        "guardrail_warning": report.get("guardrail_warning"),
        "report": report,
        "model": latest["model"],
        "feat_cols": latest["feat_cols"],
        "cat_cols": latest["cat_cols"],
        "date_cols": latest["date_cols"],
        "feature_name_map": latest["feature_name_map"],
        "feature_profile": feature_profile,
        "cfg": cfg_tmp,
        "optuna_tuning": tuning_meta,
    }
    return out
