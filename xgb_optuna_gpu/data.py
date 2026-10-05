"""Data preparation: Dask (cuDF) dataframes -> DaskQuantileDMatrix."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from distributed import Client, wait
from xgboost import dask as dxgb


def _read(path: str, fmt: str, use_gpu: bool, **kwargs):
    if use_gpu:
        import dask_cudf as dd
    else:
        import dask.dataframe as dd
    if fmt == "parquet":
        return dd.read_parquet(path, **kwargs)
    if fmt == "csv":
        return dd.read_csv(path, **kwargs)
    raise ValueError(f"Unsupported format: {fmt!r} (use 'parquet' or 'csv')")


def load_dask_data(
    train_path: str,
    valid_path: str,
    target: str,
    features: Sequence[str] | None = None,
    fmt: str = "parquet",
    use_gpu: bool = True,
    npartitions: int | None = None,
    client: Client | None = None,
    **read_kwargs,
):
    """Read train/valid files into (X_train, y_train, X_valid, y_valid) Dask collections.

    With ``use_gpu=True`` the frames are ``dask_cudf`` (data lives on the GPUs).
    The collections are persisted so repeated trials do not re-read from disk.

    Tip: store each split as several files / row groups (>= number of GPUs) so the read
    tasks land on different workers. A single small file ends up on one worker and XGBoost
    warns "Worker ... has an empty DMatrix" (that GPU then sits idle).
    """
    tr = _read(train_path, fmt, use_gpu, **read_kwargs)
    va = _read(valid_path, fmt, use_gpu, **read_kwargs)
    if features is None:
        features = [c for c in tr.columns if c != target]
    features = list(features)

    if npartitions is None and client is not None:
        # At least one partition per worker so every GPU gets data.
        npartitions = max(len(client.scheduler_info()["workers"]), 1)
    if npartitions is not None:
        tr = tr.repartition(npartitions=max(npartitions, tr.npartitions))
        va = va.repartition(npartitions=max(npartitions, va.npartitions))

    X_train, y_train = tr[features], tr[target]
    X_valid, y_valid = va[features], va[target]
    if client is not None:
        X_train, y_train, X_valid, y_valid = client.persist([X_train, y_train, X_valid, y_valid])
        wait([X_train, y_train, X_valid, y_valid])
    return X_train, y_train, X_valid, y_valid


def build_dmatrices(
    client: Client,
    X_train,
    y_train,
    X_valid,
    y_valid,
    max_bin: int = 256,
    enable_categorical: bool = False,
):
    """Build train/valid ``DaskQuantileDMatrix``.

    The validation matrix uses ``ref=dtrain`` so it shares the training quantile cuts
    (required for correct evaluation). Build these once and reuse them across trials;
    ``max_bin`` here must match ``max_bin`` in the training params.
    """
    dtrain = dxgb.DaskQuantileDMatrix(
        client, X_train, y_train, max_bin=max_bin, enable_categorical=enable_categorical
    )
    dvalid = dxgb.DaskQuantileDMatrix(
        client, X_valid, y_valid, max_bin=max_bin, ref=dtrain, enable_categorical=enable_categorical
    )
    return dtrain, dvalid


def make_synthetic_data(
    n_train: int = 200_000,
    n_valid: int = 50_000,
    n_features: int = 20,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positive, right-skewed target with a heavy tail (a few extreme values).

    Compound Poisson-Gamma-like mean structure + Pareto shocks, so all three losses
    (MSE / Huber / Tweedie) have something to chew on. Returns pandas train/valid frames
    with feature columns ``f0..f{n-1}`` and target ``y``.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_valid
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    lin = 0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2] * X[:, 3] + 0.2 * np.sin(3 * X[:, 4])
    mu = np.exp(1.0 + lin)
    y = rng.gamma(shape=2.0, scale=mu / 2.0)
    shock = rng.random(n) < 0.01  # 1% extreme values
    y[shock] *= rng.pareto(1.5, shock.sum()) + 5.0
    y = np.maximum(y, 1e-3).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["y"] = y
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)
