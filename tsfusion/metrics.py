"""Regression metrics on the ORIGINAL target scale."""
from __future__ import annotations

from typing import Dict

import numpy as np

NMAD_SCALE = 1.4826  # makes the MAD a consistent estimator of the std for normal errors


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def r2(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nmad(y_true, y_pred, relative: bool = False) -> float:
    """Normalised median absolute deviation of the errors::

        NMAD = 1.4826 * median(|e - median(e)|)

    with ``e = y_pred - y_true`` (default) or ``e = (y_pred - y_true) / y_true``
    (``relative=True``, needs y_true > 0). A robust estimate of the error spread that is
    not dominated by the few extreme targets (unlike RMSE).
    """
    e = y_pred - y_true
    if relative:
        e = e / np.where(y_true != 0, y_true, np.nan)
        e = e[np.isfinite(e)]
    return float(NMAD_SCALE * np.median(np.abs(e - np.median(e))))


def regression_metrics(y_true, y_pred, nmad_relative: bool = False) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    if not np.isfinite(y_pred).all():
        nan = float("nan")
        return {"rmse": nan, "mae": nan, "r2": nan, "nmad": nan}
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nmad": nmad(y_true, y_pred, relative=nmad_relative),
    }
