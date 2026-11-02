"""Synthetic data that mimics the real data layout, for the demo notebooks and tests.

Produces pandas DataFrames in the "as delivered" format:

* ``ts_df``        long format, index (record_id, time_step), one column per variable.
                   Missing cells are imputed with 0; special cells hold a sentinel
                   (-999 / -998 / -997 ...), exactly what usually arrives from upstream.
* ``missing_df``   same shape, 1 = missing
* ``special_df``   same shape, 0 = not special, k = special code k
* ``snapshot_ple_df`` one row per record, PLE columns ``<feature>_bin<j>``
* ``snapshot_raw_df`` the raw snapshot variables (for reference)
* ``y``            positive, right-skewed target with a heavy tail
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .data import fit_ple_bins, ple_encode


def make_synthetic_dataset(n_records: int = 6000, n_steps: int = 11, n_vars: int = 8,
                           n_special_codes: int = 3, val_frac: float = 0.2,
                           seed: int = 0) -> Dict[str, object]:
    if n_vars < 5 or n_special_codes < 2:
        raise ValueError("the synthetic target uses variables 0-4 and special code 2: "
                         "need n_vars >= 5 and n_special_codes >= 2")
    rng = np.random.default_rng(seed)
    N, T, V, K = n_records, n_steps, n_vars, n_special_codes
    ids = np.arange(100000, 100000 + N)
    t = np.arange(1, T + 1)
    var_names = [f"var_{v:02d}" for v in range(V)]

    # ---- latent drivers ---------------------------------------------------------------
    risk = rng.normal(size=N)
    size = rng.normal(size=N)

    # ---- snapshot (raw) ---------------------------------------------------------------
    snap_raw = pd.DataFrame({
        "age": rng.uniform(20, 75, N),
        "income": np.exp(10.6 + 0.5 * size + 0.3 * rng.normal(size=N)),
        "tenure_m": rng.exponential(48, N),
        "score": 650 - 60 * risk + 40 * rng.normal(size=N),
        "utilization": 1 / (1 + np.exp(-(0.8 * risk + rng.normal(size=N)))),
    }, index=pd.Index(ids, name="record_id"))

    # ---- time series: level + trend + AR(1) noise ---------------------------------------
    level = 0.8 * risk[:, None] * rng.normal(1, 0.3, V) + 0.6 * size[:, None] * rng.normal(1, 0.3, V)
    slope = 0.15 * risk[:, None] + 0.1 * rng.normal(size=(N, V))
    noise = np.zeros((N, T, V))
    eps = rng.normal(scale=0.3, size=(N, T, V))
    for i in range(T):
        noise[:, i] = (0.7 * noise[:, i - 1] if i else 0) + eps[:, i]
    x = level[:, None, :] + slope[:, None, :] * (t[None, :, None] - T / 2) / T * 4 + noise
    scale = rng.uniform(1, 500, V)          # variables live on very different scales
    offset = rng.uniform(0, 1000, V)
    x = x * scale + offset

    # ---- missing: late starts (account opened recently) + informative random gaps ------
    missing = np.zeros((N, T, V), dtype=bool)
    late = rng.random((N, V)) < 0.15
    start = rng.integers(1, T, size=(N, V))
    missing |= late[:, None, :] & (np.arange(T)[None, :, None] < start[:, None, :])
    p_gap = 0.03 + 0.05 / (1 + np.exp(-2 * risk))          # riskier -> more gaps
    missing |= rng.random((N, T, V)) < p_gap[:, None, None]

    # ---- special codes (meaning identical for all variables) ---------------------------
    special = np.zeros((N, T, V), dtype=np.int64)
    sentinels = {-999.0 + i: K - i for i in range(K)}      # -999 -> K, ..., -(999-K+1) -> 1
    code_probs = np.full(K, 0.01)
    for k in range(1, K + 1):
        p = code_probs[k - 1] * (1 + 2 * (k == 2) * (risk > 1))[:, None, None]
        hit = (rng.random((N, T, V)) < p) & ~missing & (special == 0)
        special[hit] = k
    code_to_sentinel = {c: s for s, c in sentinels.items()}

    # ---- target -------------------------------------------------------------------------
    z = (x - x.mean(axis=(0, 1))) / x.std(axis=(0, 1))       # "true" standardised values
    recent = z[:, -3:, 0].mean(1)
    trend = z[:, -1, 1] - z[:, 0, 1]
    vol = z[:, :, 2].std(1)
    frac_code2 = (special == 2).mean(axis=(1, 2)) * 10
    last_missing_v3 = missing[:, -1, 3]
    interaction = 0.6 * last_missing_v3 * (z[:, -2, 4] > 0.5)
    log_mu = (0.3 + 0.5 * recent + 0.35 * trend + 0.3 * vol + 0.3 * frac_code2 + interaction
              + 0.6 * (snap_raw["utilization"].to_numpy() - 0.5)
              + 0.3 * np.log(snap_raw["income"].to_numpy() / 40000)
              - 0.006 * (snap_raw["age"].to_numpy() - 45))
    mu = 1000 * np.exp(log_mu)
    shape = 2.0                                               # Gamma noise -> right skew
    y = mu * rng.gamma(shape, 1 / shape, N)
    tail = rng.random(N) < 0.01                               # a few extreme values
    y[tail] *= 1 + 4 * rng.pareto(2.5, tail.sum())

    # ---- what the pipeline receives: imputed 0 at missing, sentinel at special ----------
    x_obs = x.copy()
    x_obs[missing] = 0.0
    for c, s in code_to_sentinel.items():
        x_obs[special == c] = s

    idx = pd.MultiIndex.from_product([ids, t], names=["record_id", "time_step"])
    to_long = lambda a: pd.DataFrame(a.reshape(N * T, V), index=idx, columns=var_names)  # noqa
    ts_df = to_long(x_obs)
    missing_df = to_long(missing.astype(np.int8))
    special_df = to_long(special.astype(np.int8))

    # ---- train / validation split + PLE fitted on train only --------------------------
    perm = rng.permutation(N)
    n_val = int(round(val_frac * N))
    val_ids, train_ids = np.sort(ids[perm[:n_val]]), np.sort(ids[perm[n_val:]])
    n_bins = {"age": 8, "income": 10, "tenure_m": 6, "score": 10, "utilization": 8}
    ple_parts = []
    for col, nb in n_bins.items():
        edges = fit_ple_bins(snap_raw.loc[train_ids, col].to_numpy(), nb)
        enc = ple_encode(snap_raw[col].to_numpy(), edges)
        ple_parts.append(pd.DataFrame(enc, index=snap_raw.index,
                                      columns=[f"{col}_bin{j}" for j in range(enc.shape[1])]))
    snapshot_ple_df = pd.concat(ple_parts, axis=1)

    return {
        "ts_df": ts_df, "missing_df": missing_df, "special_df": special_df,
        "snapshot_raw_df": snap_raw, "snapshot_ple_df": snapshot_ple_df,
        "y": pd.Series(y, index=pd.Index(ids, name="record_id"), name="y"),
        "train_ids": train_ids, "val_ids": val_ids,
        "var_names": var_names, "time_steps": list(t),
        "special_sentinels": sentinels, "n_special_codes": K,
    }
