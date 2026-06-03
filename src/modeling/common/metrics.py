from __future__ import annotations

import numpy as np


def ranking_metrics_at_n(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n: int = 5000,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """Compute business ranking metrics for the highest-risk N customers."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if sample_weight is None:
        weight = np.ones(len(y_true), dtype=float)
        weighted = False
    else:
        weight = np.asarray(sample_weight, dtype=float)
        if len(weight) != len(y_true):
            raise ValueError("sample_weight and y_true must have the same length")
        weight = np.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)
        weight = np.maximum(weight, 0.0)
        weighted = True

    requested_n = max(int(n), 1)
    effective_n = min(requested_n, len(y_true))
    positive_weight = weight * (y_true == 1)
    total_positives = float(positive_weight.sum())
    total_weight = float(weight.sum())
    prevalence = total_positives / max(total_weight, 1e-9)
    if effective_n == 0:
        return {
            "ranking_top_n": requested_n,
            "ranking_effective_n": 0,
            "hits_at_n": 0,
            "precision_at_n": 0.0,
            "recall_at_n": 0.0,
            "lift_at_n": 0.0,
            "val_prevalence": float(prevalence),
        }

    order = np.argsort(-y_prob, kind="stable")[:effective_n]
    selected_weight = float(weight[order].sum())
    hits = float(positive_weight[order].sum())
    precision = hits / max(selected_weight, 1e-9)
    recall = hits / max(total_positives, 1e-9)
    lift = precision / prevalence if prevalence > 0 else 0.0
    return {
        "ranking_top_n": requested_n,
        "ranking_effective_n": effective_n,
        "hits_at_n": float(hits) if weighted else int(hits),
        "precision_at_n": float(precision),
        "recall_at_n": float(recall),
        "lift_at_n": float(lift),
        "val_prevalence": float(prevalence),
    }


def prefixed_ranking_metrics_at_n(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n: int,
    prefix: str,
    sample_weight: np.ndarray | None = None,
) -> dict:
    metrics = ranking_metrics_at_n(y_true, y_prob, n=n, sample_weight=sample_weight)
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def prf1_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thr).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return float(prec), float(rec), float(f1)

def average_precision_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision (AP) = area under precision-recall curve (step-wise)."""
    y_true = y_true.astype(int)
    order = np.argsort(-y_score)  # desc
    y_true_sorted = y_true[order]

    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp[-1] + 1e-9)  # tp[-1] = total positives

    ap = 0.0
    prev_rec = 0.0
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            ap += (rec[i] - prev_rec) * prec[i]
            prev_rec = rec[i]
    return float(ap)

def best_threshold_by_f1_np(y_true: np.ndarray, y_prob: np.ndarray, n_grid: int = 400):
    lo = max(float(np.min(y_prob)), 0.05)  # floor: tránh ngưỡng ≈ 0 → predict all-positive
    hi = float(np.max(y_prob))
    if lo >= hi:
        lo = max(hi * 0.1, 0.05)
    grid = np.linspace(lo, hi, n_grid)
    best = (0.5, 0.0, 0.0, 0.0)  # thr, p, r, f1
    for thr in grid:
        p, r, f1 = prf1_at_threshold(y_true, y_prob, thr)
        if f1 > best[3]:
            best = (float(thr), float(p), float(r), float(f1))
    return best
