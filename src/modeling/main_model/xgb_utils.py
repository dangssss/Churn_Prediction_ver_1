from __future__ import annotations

import re
import inspect
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

def is_date_like_col(s: pd.Series, threshold: float = 0.7) -> bool:
    """Return True if most non-null values look like YYYY-MM-DD strings."""
    sample = s.dropna().astype(str).head(50)
    if len(sample) == 0:
        return False
    return sum(bool(_DATE_RE.match(v)) for v in sample) >= threshold * len(sample)

def date_col_to_ordinal(s: pd.Series) -> pd.Series:
    """Convert YYYY-MM-DD strings to integer day ordinal (days since year 1)."""
    return pd.to_datetime(s, errors="coerce").map(
        lambda d: d.toordinal() if pd.notna(d) else np.nan
    )

def sanitize_xgb_feature_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """XGBoost không cho feature name chứa [, ], < và vài ký tự lạ.
    Đồng thời đảm bảo tất cả tên cột là string và unique.
    Returns: (df_sanitized, mapping_old_to_new)
    """
    old_cols = [str(c) for c in df.columns]
    new_cols: List[str] = []
    seen: Dict[str, int] = {}

    for c in old_cols:
        nc = c.replace("[", "_").replace("]", "_").replace("<", "_")
        nc = re.sub(r"[^0-9a-zA-Z_]+", "_", nc).strip("_")
        if nc == "":
            nc = "f"

        if nc in seen:
            seen[nc] += 1
            nc2 = f"{nc}__{seen[nc]}"
        else:
            seen[nc] = 0
            nc2 = nc
        new_cols.append(nc2)

    out = df.copy()
    out.columns = new_cols
    mapping = dict(zip(old_cols, new_cols))
    return out, mapping

def safe_to_category(s: pd.Series) -> pd.Series:
    """Convert về category 'chuẩn' cho XGBoost:
    - không dùng pandas StringDtype
    - fillna MISSING
    """
    s2 = s.astype(object)
    s2 = s2.where(~s2.isna(), "MISSING")
    s2 = s2.map(lambda x: str(x))
    return s2.astype("category")

def onehot_align_train_val(X_tr: pd.DataFrame, X_va: pd.DataFrame, cat_cols: list):
    X_tr_oh = pd.get_dummies(X_tr, columns=cat_cols, dummy_na=True)
    X_va_oh = pd.get_dummies(X_va, columns=cat_cols, dummy_na=True)
    X_va_oh = X_va_oh.reindex(columns=X_tr_oh.columns, fill_value=0)

    X_tr_oh, map_tr = sanitize_xgb_feature_names(X_tr_oh)
    X_va_oh = X_va_oh.rename(columns=map_tr)
    return X_tr_oh, X_va_oh, map_tr

def fit_xgb_with_early_stopping(model, X_tr, y_tr, X_va, y_va, es_rounds: int):
    """Compatible old/new xgboost early stopping usage."""
    fit_sig = inspect.signature(model.fit)
    kwargs = {}

    if "eval_set" in fit_sig.parameters:
        kwargs["eval_set"] = [(X_va, y_va)]
    if "verbose" in fit_sig.parameters:
        kwargs["verbose"] = False

    # Old versions: early_stopping_rounds is a fit kwarg
    if "early_stopping_rounds" in fit_sig.parameters:
        kwargs["early_stopping_rounds"] = int(es_rounds)
        model.fit(X_tr, y_tr, **kwargs)
        return model

    # Newer versions (e.g. xgboost 3.x): set early_stopping_rounds on estimator
    model.set_params(early_stopping_rounds=int(es_rounds))
    model.fit(X_tr, y_tr, **kwargs)
    return model

def predict_proba_best_iteration(model, X):
    """Use best_iteration when available (after early stopping)."""
    sig = inspect.signature(model.predict_proba)
    kwargs = {}

    best_it = getattr(model, "best_iteration", None)
    if best_it is not None:
        if "iteration_range" in sig.parameters:
            kwargs["iteration_range"] = (0, int(best_it) + 1)
        elif "ntree_limit" in sig.parameters:
            kwargs["ntree_limit"] = int(best_it) + 1

    return model.predict_proba(X, **kwargs)
