
from __future__ import annotations

import numpy as np
import pandas as pd

def _safe_probs(counts: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    s = float(np.nansum(counts))
    if s <= 0:
        return np.full_like(counts, 0.0, dtype=float)
    p = counts / s
    return np.clip(p, eps, 1.0)

def psi_from_counts(train_counts: np.ndarray, cur_counts: np.ndarray, eps: float = 1e-8) -> float:
    """
    PSI = sum((p_cur - p_train) * ln(p_cur / p_train))
    """
    p_t = _safe_probs(train_counts, eps=eps)
    p_c = _safe_probs(cur_counts, eps=eps)
    return float(np.sum((p_c - p_t) * np.log(p_c / p_t)))

def discrete_ks_from_counts(train_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """
    Discrete KS approximation on the same bins.
    """
    p_t = _safe_probs(train_counts)
    p_c = _safe_probs(cur_counts)
    c_t = np.cumsum(p_t)
    c_c = np.cumsum(p_c)
    return float(np.max(np.abs(c_t - c_c)))

def make_numeric_profile(x: pd.Series, n_bins: int = 10) -> dict | None:
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return None
    
    # Convert to float explicitly to avoid boolean issues
    x_values = x.values.astype(float)
    
    # use quantile bins, robust to outliers
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x_values, qs))
    if len(edges) < 3:
        # near-constant
        mn = float(x.min()); mx = float(x.max())
        edges = np.array([mn - 1e-6, mx + 1e-6])
    counts, bins = np.histogram(x_values, bins=edges)
    return {"bin_edges": bins.tolist(), "train_counts": counts.tolist(), "n": int(len(x))}

def counts_on_profile(x: pd.Series, profile: dict) -> np.ndarray:
    edges = np.array(profile["bin_edges"], dtype=float)
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.zeros(len(edges) - 1, dtype=float)
    counts, _ = np.histogram(x, bins=edges)
    return counts
