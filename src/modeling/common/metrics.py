from __future__ import annotations

import numpy as np


def prf1_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thr).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(prec), float(rec), float(f1)


def average_precision_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision (AP) = area under precision-recall curve (step-wise)."""
    y_true = y_true.astype(int)
    order = np.argsort(-y_score)  # desc
    y_true_sorted = y_true[order]

    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp[-1] + 1e-9)  # tp[-1] = total positives

    ap = 0.0
    prev_rec = 0.0
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            ap += (rec[i] - prev_rec) * prec[i]
            prev_rec = rec[i]
    return float(ap)


def best_threshold_by_f1_np(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_grid: int = 400,
    *,
    min_threshold: float = 0.05,
    max_predicted_positive_rate: float | None = None,
):
    lo = max(float(np.min(y_prob)), float(min_threshold))
    hi = float(np.max(y_prob))
    if lo >= hi:
        lo = max(hi * 0.1, float(min_threshold))
    grid = np.linspace(lo, hi, n_grid)

    best = (0.5, 0.0, 0.0, 0.0)  # thr, p, r, f1
    best_unconstrained = best
    for thr in grid:
        p, r, f1 = prf1_at_threshold(y_true, y_prob, thr)
        if f1 > best_unconstrained[3]:
            best_unconstrained = (float(thr), float(p), float(r), float(f1))

        if max_predicted_positive_rate is not None:
            pred_pos_rate = float(np.mean(y_prob >= thr))
            if pred_pos_rate > float(max_predicted_positive_rate):
                continue

        if f1 > best[3]:
            best = (float(thr), float(p), float(r), float(f1))

    return best if best[3] > 0 else best_unconstrained
