"""Regression metrics: RMSE, MAE, R2, NMAD.

All functions take ``y_true`` / ``y_pred`` as numpy arrays of shape ``(n,)`` or
``(n, n_targets)`` and return plain floats (multi-target results are averaged;
:func:`compute_metrics` also returns the per-target values).

Always feed these the **original target scale** -- the trainer inverse-transforms
predictions before scoring so RMSE/MAE are in business units.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = ["rmse", "mae", "r2_score", "nmad", "compute_metrics", "METRIC_MODES"]

#: whether a lower or higher value is better -- used for model selection.
METRIC_MODES = {"rmse": "min", "mae": "min", "r2": "max", "nmad": "min", "loss": "min"}

_MAD_TO_SIGMA = 1.4826  # scale factor making MAD a consistent estimator of sigma


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def rmse(y_true, y_pred) -> float:
    yt, yp = _as_2d(y_true), _as_2d(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _as_2d(y_true), _as_2d(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination, averaged over targets.

    ``SS_tot`` uses the mean of *this* split -- the usual convention, and the
    reason R2 on a small or low-variance OOT slice can look poor even when RMSE
    is fine.
    """
    yt, yp = _as_2d(y_true), _as_2d(y_pred)
    ss_res = ((yt - yp) ** 2).sum(axis=0)
    ss_tot = ((yt - yt.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return float(np.nanmean(out))


def nmad(y_true, y_pred, variant: str = "residual", eps: float = 1e-8) -> float:
    """Normalised median absolute deviation of the residuals.

    ``variant="residual"`` (default)
        ``1.4826 * median(|r - median(r)|)`` with ``r = y_true - y_pred``.
        A robust, outlier-insensitive estimate of the residual spread, in the
        same units as the target; directly comparable to RMSE (they coincide
        for clean Gaussian residuals).

    ``variant="relative"``
        ``1.4826 * median(|r| / max(|y_true|, eps))`` -- a scale-free version
        for targets spanning orders of magnitude.  Undefined-ish near
        ``y_true = 0``; only use it when the target is bounded away from zero.

    Pick one and keep it fixed: the two are not comparable.
    """
    yt, yp = _as_2d(y_true), _as_2d(y_pred)
    r = yt - yp
    if variant == "residual":
        med = np.median(r, axis=0, keepdims=True)
        return float(np.mean(_MAD_TO_SIGMA * np.median(np.abs(r - med), axis=0)))
    if variant == "relative":
        denom = np.maximum(np.abs(yt), eps)
        return float(np.mean(_MAD_TO_SIGMA * np.median(np.abs(r) / denom, axis=0)))
    raise ValueError(f"unknown nmad variant {variant!r}")


def compute_metrics(
    y_true,
    y_pred,
    nmad_variant: str = "residual",
    target_names: Optional[Sequence[str]] = None,
    per_target: bool = False,
) -> Dict[str, float]:
    """All four metrics in one dict (plus per-target breakdown if asked)."""
    yt, yp = _as_2d(y_true), _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    out = {
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "r2": r2_score(yt, yp),
        "nmad": nmad(yt, yp, variant=nmad_variant),
    }
    if per_target and yt.shape[1] > 1:
        names = list(target_names) if target_names else [f"t{i}" for i in range(yt.shape[1])]
        for j, name in enumerate(names):
            col_t, col_p = yt[:, [j]], yp[:, [j]]
            out[f"rmse__{name}"] = rmse(col_t, col_p)
            out[f"mae__{name}"] = mae(col_t, col_p)
            out[f"r2__{name}"] = r2_score(col_t, col_p)
            out[f"nmad__{name}"] = nmad(col_t, col_p, variant=nmad_variant)
    return out
