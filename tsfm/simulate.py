"""Synthetic data generator used by the demo notebook and the tests.

Mimics a credit-card panel: ``n_records`` accounts, ``n_features`` monthly
behavioural features over ``seq_len`` months, plus static (snapshot) attributes
and a continuous target.  Both kinds of missingness are injected:

* **systematic** -- accounts booked ``tenure`` months ago have no history before
  that (the earliest ``seq_len - tenure`` months do not exist);
* **random** -- a fraction of the *existing* cells is blanked out.

The target depends on the observed history, the tenure and the snapshot
variables, so a model that handles missingness properly should beat one that
does not.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

__all__ = ["make_synthetic_panel"]


def make_synthetic_panel(
    n_records: int = 4000,
    n_features: int = 8,
    seq_len: int = 24,
    n_cat: int = 3,
    n_num: int = 4,
    random_missing_rate: float = 0.05,
    noise: float = 0.35,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Returns ``(ts_long, snapshot, target, splits)``.

    ``ts_long`` has columns ``[account_id, month_idx, f0..f{C-1}]`` with
    ``month_idx = 0`` the **oldest** month.  ``splits`` maps
    ``train``/``holdout``/``oot`` to arrays of ``account_id``; the OOT split is
    carved out by booking vintage (the most recent bookings), as in a real
    out-of-time validation.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(n_records)

    # ---- static attributes ------------------------------------------------ #
    tenure = rng.integers(1, seq_len + 1, size=n_records)          # months on book
    vintage = rng.integers(0, 36, size=n_records)                  # booking month
    cat_cards = [4, 6, 3][:n_cat] or []
    cats = {f"cat{i}": rng.integers(0, card, size=n_records) for i, card in enumerate(cat_cards)}
    nums = {f"num{i}": rng.normal(size=n_records) for i in range(n_num)}

    # ---- time series ------------------------------------------------------ #
    t_axis = np.arange(seq_len)
    level = rng.normal(size=(n_records, n_features))
    slope = rng.normal(scale=0.05, size=(n_records, n_features))
    season = rng.normal(scale=0.4, size=(n_records, n_features))

    x = (
        level[:, :, None]
        + slope[:, :, None] * t_axis[None, None, :]
        + season[:, :, None] * np.sin(2 * np.pi * t_axis[None, None, :] / 12.0)
        + rng.normal(scale=0.3, size=(n_records, n_features, seq_len))
    )
    # a little cross-feature structure so feature-axis attention has work to do
    if n_features > 1:
        x[:, 1] += 0.6 * x[:, 0]

    # ---- systematic missingness (short tenure) ---------------------------- #
    # month_idx 0 is oldest -> the first (seq_len - tenure) months do not exist
    sys_mask_2d = t_axis[None, :] < (seq_len - tenure)[:, None]     # (R, T)
    sys_mask = np.repeat(sys_mask_2d[:, None, :], n_features, axis=1)

    # ---- target ----------------------------------------------------------- #
    obs_w = (~sys_mask_2d).astype(float)
    recent = np.clip(t_axis[None, :] - (seq_len - tenure)[:, None], 0, None)
    weight = obs_w * np.exp(0.08 * recent)                         # recency-weighted
    weight = weight / weight.sum(axis=1, keepdims=True)

    y = (
        1.5 * (x[:, 0] * weight).sum(axis=1)
        - 0.8 * (x[:, min(1, n_features - 1)] * weight).sum(axis=1)
        + 0.5 * (x[:, min(2, n_features - 1)] * weight).sum(axis=1) ** 2 * 0.3
        + 0.04 * tenure                                            # tenure itself matters
        + sum(0.6 * (v - v.mean()) for v in nums.values()) / max(n_num, 1)
        + sum(0.5 * (v - v.mean()) for v in cats.values()) / max(len(cats), 1)
        + rng.normal(scale=noise, size=n_records)
    )
    y = 100.0 + 20.0 * y                                           # arbitrary business units

    # ---- random missingness ------------------------------------------------ #
    x_missing = x.copy()
    x_missing[sys_mask] = np.nan
    rand = (rng.random(x.shape) < random_missing_rate) & ~sys_mask
    x_missing[rand] = np.nan

    # ---- frames ------------------------------------------------------------ #
    feature_names = [f"f{i}" for i in range(n_features)]
    ts_long = pd.DataFrame(
        {
            "account_id": np.repeat(ids, seq_len),
            "month_idx": np.tile(t_axis, n_records),
        }
    )
    for i, name in enumerate(feature_names):
        ts_long[name] = x_missing[:, i, :].reshape(-1)

    snapshot = pd.DataFrame({"account_id": ids, "tenure_months": tenure, "vintage": vintage})
    for k, v in cats.items():
        snapshot[k] = pd.Categorical(v).astype(object)
    for k, v in nums.items():
        snapshot[k] = v
    # a few missing snapshot values, to exercise that path
    if n_num:
        holes = rng.random(n_records) < 0.03
        snapshot.loc[holes, "num0"] = np.nan
    if cats:
        holes = rng.random(n_records) < 0.02
        snapshot.loc[holes, "cat0"] = np.nan

    target = pd.DataFrame({"account_id": ids, "y": y})

    # ---- splits: OOT by vintage, then a random train/holdout cut ----------- #
    oot_mask = vintage >= np.quantile(vintage, 0.8)
    in_time = ids[~oot_mask]
    shuffled = rng.permutation(in_time)
    n_hold = int(0.2 * len(shuffled))
    splits = {
        "train": shuffled[n_hold:],
        "holdout": shuffled[:n_hold],
        "oot": ids[oot_mask],
    }
    return ts_long, snapshot, target, splits
