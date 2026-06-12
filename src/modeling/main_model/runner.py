from __future__ import annotations

import os
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


def select_feature_cols_for_model(df: pd.DataFrame, label_col: str):
    drop_cols = {
        "cms_code_enc", "window_size", "window_start", "window_end",
        "source_table_t", "source_table_t_plus_h",
        "is_active_now", "is_churned_now", "gate_group",
        "label_source", "label_weight",
        label_col
    }
    return [c for c in df.columns if c not in drop_cols]

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
    rep["warnings"] = warns

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

    # optimize threshold on MAIN val
    min_threshold = _cfg_float(cfg, "main_threshold_min", "MAIN_XGB_THRESHOLD_MIN", 0.005)
    max_pred_pos_rate = _cfg_float(
        cfg,
        "main_max_predicted_positive_rate",
        "MAIN_XGB_MAX_PREDICTED_POSITIVE_RATE",
        0.80,
    )
    thr_opt, p_opt, r_opt, f1_opt = best_threshold_by_f1_np(
        y_va,
        va_prob,
        n_grid=600,
        min_threshold=min_threshold,
        max_predicted_positive_rate=max_pred_pos_rate,
    )
    ap = average_precision_np(y_va, va_prob)
    roc_auc = float(roc_auc_score(y_va, va_prob)) if len(np.unique(y_va)) == 2 else None
    primary_sources = (
        sorted(df_va["label_source"].dropna().astype(str).unique())
        if "label_source" in df_va.columns
        else []
    )
    primary_label_source = (
        "unknown"
        if not primary_sources
        else primary_sources[0]
        if len(primary_sources) == 1
        else "mixed"
    )
    logger.info(
        "[MAIN CLASSIFICATION METRICS][%s] F1=%.4f precision=%.4f recall=%.4f "
        "AP=%.4f ROC_AUC=%s prevalence=%.4f%% threshold=%.6f threshold_min=%.6f max_pred_pos_rate=%.2f%%",
        primary_label_source.upper(),
        f1_opt,
        p_opt,
        r_opt,
        ap,
        f"{roc_auc:.4f}" if roc_auc is not None else "n/a",
        100.0 * float(y_va.mean()),
        thr_opt,
        min_threshold,
        100.0 * max_pred_pos_rate,
    )
    
    # Calculate confusion matrix at optimal threshold
    y_pred_opt = (va_prob >= thr_opt).astype(int)
    cm = confusion_matrix(y_va, y_pred_opt)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    logger.info(
        "[MAIN MODEL] Confusion Matrix @ threshold=%.4f: "
        "TN=%d FP=%d FN=%d TP=%d | "
        "Precision=%.4f Recall=%.4f F1=%.4f",
        thr_opt, tn, fp, fn, tp,
        tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9), f1_opt,
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
    dummy_simple2_feats = ",".join(dummy_simple2["features"]) if dummy_simple2 else None

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
        "dummy_simple2_features": dummy_simple2_feats,
        "guardrail_warning": guardrail_warning,
        "diagnostic_warning": diagnostic_warning,

        "thr_baseline": float(thr_baseline),
        "precision@baseline_thr": float(p_b),
        "recall@baseline_thr": float(r_b),
        "f1@baseline_thr": float(f1_b),

        "thr_main_opt": float(thr_opt),
        "thr_main_min": float(min_threshold),
        "max_predicted_positive_rate": float(max_pred_pos_rate),
        "precision@main_thr": float(p_opt),
        "recall@main_thr": float(r_opt),
        "f1@main_thr": float(f1_opt),
    }

    return model, report, feat_cols, cat_cols, feature_name_map, date_cols


def _build_optuna_trial_cfg(base_cfg: dict, trial) -> dict:
    n_min = _env_int("MAIN_XGB_OPTUNA_N_ESTIMATORS_MIN", 800)
    n_max = _env_int("MAIN_XGB_OPTUNA_N_ESTIMATORS_MAX", 5000)
    n_step = max(_env_int("MAIN_XGB_OPTUNA_N_ESTIMATORS_STEP", 200), 1)
    if n_max < n_min:
        n_max = n_min

    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", n_min, n_max, step=n_step),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "max_leaves": trial.suggest_categorical("max_leaves", [0, 31, 63, 127, 255]),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 8.0),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.50, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 50.0, log=True),
        "tree_method": os.getenv("MAIN_XGB_TREE_METHOD", "hist"),
        "random_state": int(base_cfg.get("seed", 42)),
        "eval_metric": ["aucpr", "logloss"],
    }
    n_jobs = _env_int("MAIN_XGB_N_JOBS", 0)
    if n_jobs > 0:
        params["n_jobs"] = int(n_jobs)

    if _env_bool("MAIN_XGB_OPTUNA_TUNE_SCALE_POS_WEIGHT", True):
        max_spw = max(_env_float("MAIN_XGB_OPTUNA_SCALE_POS_WEIGHT_MAX", _env_float("MAIN_XGB_MAX_SCALE_POS_WEIGHT", 20.0)), 0.5)
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.5, max_spw, log=True)

    tuned_cfg = dict(base_cfg)
    tuned_cfg["main_xgb_params"] = params
    tuned_cfg["main_es_rounds"] = trial.suggest_int("early_stopping_rounds", 50, 350, step=50)
    tuned_cfg["main_max_predicted_positive_rate"] = trial.suggest_float(
        "max_predicted_positive_rate",
        _env_float("MAIN_XGB_OPTUNA_MAX_PRED_POS_RATE_MIN", 0.35),
        _env_float("MAIN_XGB_OPTUNA_MAX_PRED_POS_RATE_MAX", 0.90),
    )
    tuned_cfg["main_threshold_min"] = _env_float("MAIN_XGB_THRESHOLD_MIN", 0.005)
    return tuned_cfg


def tune_xgb_hyperparams_for_folds(
    cfg: dict,
    folds: list[dict],
    *,
    n_trials: int | None = None,
    timeout_seconds: int | None = None,
) -> tuple[dict, dict]:
    """Tune XGBoost params with Optuna on the existing walk-forward folds."""
    n_trials = int(n_trials if n_trials is not None else _env_int("MAIN_XGB_OPTUNA_TRIALS", 25))
    timeout_seconds = int(
        timeout_seconds
        if timeout_seconds is not None
        else _env_int("MAIN_XGB_OPTUNA_TIMEOUT_SECONDS", 0)
    )
    if n_trials <= 0:
        return dict(cfg), {"enabled": False, "reason": "n_trials<=0"}

    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError as exc:
        logger.warning("[OPTUNA] optuna is not installed; using default XGBoost params: %s", exc)
        return dict(cfg), {"enabled": False, "reason": "optuna_not_installed"}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = int(cfg.get("seed", 42))
    study_name = f"churn_xgb_k{cfg.get('best_k')}_static{int(bool(cfg.get('use_static', False)))}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(
            n_startup_trials=max(_env_int("MAIN_XGB_OPTUNA_PRUNER_STARTUP_TRIALS", 5), 0),
            n_warmup_steps=max(_env_int("MAIN_XGB_OPTUNA_PRUNER_WARMUP_STEPS", 1), 0),
        ),
    )

    def objective(trial) -> float:
        trial_cfg = _build_optuna_trial_cfg(cfg, trial)
        fold_f1: list[float] = []
        fold_ap: list[float] = []
        fold_roc: list[float] = []
        for fold_idx, fold in enumerate(folds):
            try:
                _model, report, *_rest = train_main_xgb_option_B(
                    fold["train"],
                    fold["val"],
                    trial_cfg,
                )
            except Exception as exc:
                trial.set_user_attr("rejected_reason", str(exc))
                return 0.0

            fold_f1.append(float(report["f1@main_thr"]))
            fold_ap.append(float(report["AP_val"]))
            if report.get("ROC_AUC_val") is not None:
                fold_roc.append(float(report["ROC_AUC_val"]))
            score = float(np.mean(fold_f1))
            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial_cfg["optuna_objective"] = "walk_forward_f1_mean"
        trial.set_user_attr("candidate_cfg", trial_cfg)
        trial.set_user_attr("fold_f1", fold_f1)
        trial.set_user_attr("fold_ap", fold_ap)
        trial.set_user_attr("fold_roc", fold_roc)
        trial.set_user_attr("ap_mean", float(np.mean(fold_ap)) if fold_ap else 0.0)
        trial.set_user_attr("roc_auc_mean", float(np.mean(fold_roc)) if fold_roc else 0.0)
        return float(np.mean(fold_f1)) if fold_f1 else 0.0

    logger.info(
        "[OPTUNA] Start XGBoost tuning K=%s use_static=%s trials=%d timeout=%s objective=walk_forward_f1_mean",
        cfg.get("best_k"),
        cfg.get("use_static"),
        n_trials,
        timeout_seconds or "none",
    )
    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=int(timeout_seconds) if timeout_seconds > 0 else None,
        n_jobs=max(_env_int("MAIN_XGB_OPTUNA_N_JOBS", 1), 1),
        show_progress_bar=False,
        gc_after_trial=True,
    )

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete_trials:
        logger.warning("[OPTUNA] No completed trials for K=%s use_static=%s; using defaults.", cfg.get("best_k"), cfg.get("use_static"))
        return dict(cfg), {"enabled": True, "status": "no_completed_trials"}

    best = study.best_trial
    tuned_cfg = dict(best.user_attrs.get("candidate_cfg") or cfg)
    tuning_meta = {
        "enabled": True,
        "status": "completed",
        "study_name": study.study_name,
        "objective": "walk_forward_f1_mean",
        "n_trials": len(study.trials),
        "completed_trials": len(complete_trials),
        "best_value": float(best.value),
        "best_params": dict(best.params),
        "fold_f1": best.user_attrs.get("fold_f1", []),
        "fold_ap": best.user_attrs.get("fold_ap", []),
        "fold_roc": best.user_attrs.get("fold_roc", []),
        "ap_mean": best.user_attrs.get("ap_mean"),
        "roc_auc_mean": best.user_attrs.get("roc_auc_mean"),
    }
    tuned_cfg["optuna_tuning"] = tuning_meta
    logger.info(
        "[OPTUNA] Best K=%s use_static=%s F1_mean=%.4f params=%s",
        cfg.get("best_k"),
        cfg.get("use_static"),
        float(best.value),
        dict(best.params),
    )
    return tuned_cfg, tuning_meta


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

    if tune_hyperparams:
        cfg_tmp, tuning_meta = tune_xgb_hyperparams_for_folds(
            cfg_tmp,
            folds,
            n_trials=optuna_trials,
            timeout_seconds=optuna_timeout_seconds,
        )
        cfg_tmp["use_static"] = bool(use_static_flag)

    fold_outputs = []
    for fold_idx, fold in enumerate(folds, start=1):
        try:
            model, report, feat_cols, cat_cols, fmap, date_cols = train_main_xgb_option_B(
                fold["train"],
                fold["val"],
                cfg_tmp,
            )
        except ValueError as e:
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
                "F1_val": 0.0, "AP_val": 0.0, "ROC_AUC_val": 0.0}

    latest = fold_outputs[-1]
    fold_reports = [out["report"] for out in fold_outputs]
    f1_mean = float(np.mean([r["f1@main_thr"] for r in fold_reports]))
    ap_mean = float(np.mean([r["AP_val"] for r in fold_reports]))
    roc_values = [r.get("ROC_AUC_val") for r in fold_reports if r.get("ROC_AUC_val") is not None]
    roc_mean = float(np.mean(roc_values)) if roc_values else 0.0
    precision_mean = float(np.mean([r["precision@main_thr"] for r in fold_reports]))
    recall_mean = float(np.mean([r["recall@main_thr"] for r in fold_reports]))
    val_prevalence_mean = float(np.mean([r["val_prevalence"] for r in fold_reports]))

    report = dict(latest["report"])
    report["walk_forward_folds"] = len(fold_outputs)
    report["walk_forward_reports"] = fold_reports
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

    logger.info(
        "[MAIN WALK FORWARD METRICS] K=%d use_static=%s folds=%d F1_mean=%.4f "
        "precision_mean=%.4f recall_mean=%.4f AP_mean=%.4f ROC_AUC_mean=%.4f latest_F1=%.4f",
        cfg["best_k"],
        use_static_flag,
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
