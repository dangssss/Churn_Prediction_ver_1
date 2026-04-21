from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.metrics import prf1_at_threshold, average_precision_np, best_threshold_by_f1_np

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

def select_feature_cols_for_model(df: pd.DataFrame, label_col: str):
    drop_cols = {
        "cms_code_enc", "window_size", "window_start", "window_end",
        "source_table_t", "source_table_t_plus_h",
        "is_active_now", "is_churned_now", "gate_group",
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

def train_main_xgb_option_B(df_tr: pd.DataFrame, df_va: pd.DataFrame, cfg: dict):
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

    spw = float(cfg["best_spw"])
    thr_baseline = float(cfg["best_threshold"])  # tham chiếu
    es_rounds = int(cfg.get("main_es_rounds", 200))

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

    params = dict(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        scale_pos_weight=spw,
        eval_metric="aucpr",
    )

    feature_name_map = None

    # ---- train native categorical if possible
    try:
        X_tr_s, map_native = sanitize_xgb_feature_names(X_tr)
        X_va_s = X_va.rename(columns=map_native)

        model = xgb.XGBClassifier(**params, enable_categorical=True, early_stopping_rounds=es_rounds)
        model = fit_xgb_with_early_stopping(model, X_tr_s, y_tr, X_va_s, y_va, es_rounds=es_rounds)

        used_mode = "native_categorical"
        feature_name_map = map_native

        va_prob = predict_proba_best_iteration(model, X_va_s)[:, 1]

    except Exception:
        # fallback onehot
        X_tr_oh, X_va_oh, map_oh = onehot_align_train_val(X_tr, X_va, cat_cols=cat_cols)

        model = xgb.XGBClassifier(**params, early_stopping_rounds=es_rounds)
        model = fit_xgb_with_early_stopping(model, X_tr_oh, y_tr, X_va_oh, y_va, es_rounds=es_rounds)

        used_mode = "one_hot"
        feature_name_map = map_oh

        va_prob = predict_proba_best_iteration(model, X_va_oh)[:, 1]

    # baseline threshold (tham chiếu)
    p_b, r_b, f1_b = prf1_at_threshold(y_va, va_prob, thr_baseline)

    # optimize threshold on MAIN val
    thr_opt, p_opt, r_opt, f1_opt = best_threshold_by_f1_np(y_va, va_prob, n_grid=600)
    ap = average_precision_np(y_va, va_prob)

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

        "spw_used": float(spw),
        "used_mode": used_mode,

        "AP_val": float(ap),

        "xgb_es_rounds": int(es_rounds),
        "xgb_best_iteration": int(best_it) if best_it is not None else None,
        "xgb_best_score": float(best_score) if best_score is not None else None,

        "val_prevalence": float(sanity["val_prevalence"]),
        "dummy_ap_const0": float(sanity["dummy_const0"]["AP"]),
        "dummy_ap_random": float(sanity["dummy_random_uniform"]["AP"]),
        "dummy_ap_simple2": dummy_simple2_ap,
        "dummy_simple2_features": dummy_simple2_feats,
        "guardrail_warning": guardrail_warning,

        "thr_baseline": float(thr_baseline),
        "precision@baseline_thr": float(p_b),
        "recall@baseline_thr": float(r_b),
        "f1@baseline_thr": float(f1_b),

        "thr_main_opt": float(thr_opt),
        "precision@main_thr": float(p_opt),
        "recall@main_thr": float(r_opt),
        "f1@main_thr": float(f1_opt),
    }

    return model, report, feat_cols, cat_cols, feature_name_map, date_cols

def run_main_variant(engine, cfg: dict, df_static: pd.DataFrame, use_static_flag: bool):
    from preprocess.trainval import build_train_val_for_main
    df_all, df_tr, df_va, val_month_main = build_train_val_for_main(
        engine, cfg, df_static, use_static_override=use_static_flag
    )
    cfg_tmp = dict(cfg)
    cfg_tmp["use_static"] = bool(use_static_flag)

    model, report, feat_cols, cat_cols, fmap, date_cols = train_main_xgb_option_B(df_tr, df_va, cfg_tmp)

    # baseline profile for monitoring drift (built on train split)
    feature_profile = compute_feature_profile(df_tr, feat_cols=feat_cols, cat_cols=cat_cols)

    out = {
        "use_static": bool(use_static_flag),
        "val_month": int(val_month_main),
        "train_rows": int(len(df_tr)),
        "val_rows": int(len(df_va)),
        "AP_val": float(report["AP_val"]),
        "F1_val": float(report["f1@main_thr"]),
        "guardrail_warning": report.get("guardrail_warning"),
        "report": report,
        "model": model,
        "feat_cols": feat_cols,
        "cat_cols": cat_cols,
        "date_cols": date_cols,
        "feature_name_map": fmap,
        "feature_profile": feature_profile,
    }
    return out
