"""Probability-quality metrics for imbalanced binary risk models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Quantile-bin ECE: weighted mean |observed rate - mean predicted| over bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    order = np.argsort(y_prob, kind="stable")
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            ece += len(chunk) / len(y_true) * abs(y_true[chunk].mean() - y_prob[chunk].mean())
    return float(ece)


def calibration_points(y_true, y_prob, n_bins: int = 10):
    """Quantile-binned (mean predicted, observed rate, count) for reliability plots."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(y_prob, kind="stable")
    chunks = [c for c in np.array_split(order, n_bins) if len(c)]
    pred = [float(y_prob[c].mean()) for c in chunks]
    obs = [float(y_true[c].mean()) for c in chunks]
    cnt = [int(len(c)) for c in chunks]
    return pred, obs, cnt


def evaluate(y_true, y_prob) -> dict[str, float]:
    """PR-AUC, ROC-AUC, log loss, Brier score and ECE."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "base_rate": float(y_true.mean()),
    }
    if len(np.unique(y_true)) == 2:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
    return out
