"""Evaluation helpers for a fitted model: metrics, decile (calibration) table, Gini."""

from __future__ import annotations

import numpy as np
import pandas as pd
from distributed import Client
from xgboost import dask as dxgb


def predict(client: Client, booster, X):
    """Distributed prediction on a Dask (cuDF) frame; returns a lazy Dask collection."""
    return dxgb.inplace_predict(client, booster, X)


def to_numpy(collection) -> np.ndarray:
    """Bring a Dask / cuDF / pandas / CuPy column to the driver as a float64 NumPy array.

    Fine for tens of millions of rows; for much more, evaluate on a sample.
    """
    obj = collection.compute() if hasattr(collection, "compute") else collection
    if hasattr(obj, "to_numpy"):  # pandas / cuDF Series
        obj = obj.to_numpy()
    if hasattr(obj, "get"):  # CuPy array
        obj = obj.get()
    return np.asarray(obj, dtype=np.float64).ravel()


def _gini(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p, kind="mergesort")
    cum = np.cumsum(y[order]) / y.sum()
    n = len(y)
    return cum.sum() / n - (n + 1) / (2 * n)


def normalized_gini(y: np.ndarray, p: np.ndarray) -> float:
    """Ranking quality (1 = perfect ordering, 0 = random); insensitive to scale/bias."""
    return _gini(y, p) / _gini(y, y)


def tweedie_deviance(y: np.ndarray, p: np.ndarray, power: float) -> float:
    """Mean Tweedie deviance for 1 < power < 2 (lower is better)."""
    p = np.maximum(p, 1e-12)
    dev = 2 * (
        np.power(y, 2 - power) / ((1 - power) * (2 - power))
        - y * np.power(p, 1 - power) / (1 - power)
        + np.power(p, 2 - power) / (2 - power)
    )
    return float(dev.mean())


def regression_metrics(
    y: np.ndarray,
    p: np.ndarray,
    huber_slope: float = 1.0,
    tweedie_power: float | None = None,
) -> dict[str, float]:
    """rmse, mae, mphe (same definition as XGBoost), R2, normalized Gini, averages."""
    r = p - y
    out = {
        "n": float(len(y)),
        "rmse": float(np.sqrt(np.mean(r**2))),
        "mae": float(np.mean(np.abs(r))),
        "mphe": float(np.mean(huber_slope**2 * (np.sqrt(1 + (r / huber_slope) ** 2) - 1))),
        "r2": float(1 - np.sum(r**2) / np.sum((y - y.mean()) ** 2)),
        "normalized_gini": float(normalized_gini(y, p)),
        "avg_target": float(y.mean()),
        "avg_prediction": float(p.mean()),
        "avg_pred / avg_target": float(p.mean() / y.mean()),
    }
    if tweedie_power is not None:
        out[f"tweedie_deviance@{tweedie_power}"] = tweedie_deviance(y, p, tweedie_power)
    return out


def decile_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Group rows by predicted-value quantile: avg target vs avg prediction per bin.

    For a right-skewed target this shows whether the model separates the high values
    (top bins) and stays calibrated (ratio near 1) across the range.
    """
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    t = df.groupby("bin").agg(
        n=("y", "size"),
        min_pred=("p", "min"),
        max_pred=("p", "max"),
        avg_prediction=("p", "mean"),
        avg_target=("y", "mean"),
        max_target=("y", "max"),
    )
    t["pred / target"] = t["avg_prediction"] / t["avg_target"]
    t["share_of_total_target"] = df.groupby("bin")["y"].sum() / df["y"].sum()
    return t
