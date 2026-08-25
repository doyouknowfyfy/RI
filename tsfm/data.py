"""Data processing for the three input tables.

The pipeline expects **three tables sharing one primary key**:

1. ``ts_df``     -- time-series panel (records x features x time steps)
2. ``snap_df``   -- snapshot / static variables (one row per record)
3. ``target_df`` -- target(s) (one row per record)

and produces one :class:`TSDataset` per split (``train`` / ``holdout`` / ``oot``).

Everything that *learns* from the data (feature scalers, categorical
vocabularies, target scaler) is fitted on **train only** and then applied
unchanged to holdout and OOT -- that is the whole point of keeping the OOT
slice honest.

Missing-value policy
--------------------
* **random missing** -> imputed with ``0`` *after* scaling (= the training mean
  for a standard scaler) and flagged in ``obs_mask``;
* **systematic missing** (time step outside the record's lifetime) -> also
  zero-filled, flagged in ``obs_mask`` **and** in ``sys_mask``, which is what
  drives the learned missing embedding in the model.
"""

from __future__ import annotations

import copy
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "SystematicMissingSpec",
    "DataConfig",
    "TSDataset",
    "DataProcessor",
    "prepare_datasets",
    "make_dataloader",
    "wide_to_panel",
    "infer_wide_columns",
]


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
@dataclass
class SystematicMissingSpec:
    """How to tell *systematic* missingness from *random* missingness.

    ``method``
        ``"tenure"``
            Use a snapshot column holding the number of periods the record has
            actually existed (months on book, time since booking, ...).  Steps
            outside that window are systematic.  This is the most reliable
            option -- prefer it when the column exists.
        ``"boundary_nan"``
            Infer it: an unbroken run of all-NaN time steps at the *start*
            (``time_order="oldest_first"``) or *end* (``"newest_first"``) of the
            window is treated as systematic; NaNs elsewhere are random.
        ``"explicit"``
            You supply the mask yourself (``sys_mask_col`` in the long
            time-series frame, or a ``(R, C, T)`` / ``(R, T)`` array passed to
            :meth:`DataProcessor.transform`).
        ``"none"``
            Everything missing is random.

    ``per_feature``
        ``False`` (default) -- systematic missingness is a property of the
        *record* (the card did not exist yet), so the same steps are flagged
        for every feature.  ``True`` -- infer/flag it per feature.

    ``tenure_unit_steps``
        Time steps per unit of ``tenure_col`` (e.g. ``1`` if tenure is in months
        and steps are monthly).
    """

    method: Literal["tenure", "boundary_nan", "explicit", "none"] = "boundary_nan"
    tenure_col: Optional[str] = None
    tenure_unit_steps: float = 1.0
    sys_mask_col: Optional[str] = None
    per_feature: bool = False


@dataclass
class DataConfig:
    """Column names and preprocessing options."""

    key: str
    ts_features: List[str]
    target_cols: List[str]
    time_col: Optional[str] = None                 # required for long-format input
    time_index: Optional[List[Any]] = None         # explicit ordered time steps
    seq_len: Optional[int] = None                  # inferred when None
    time_order: Literal["oldest_first", "newest_first"] = "oldest_first"

    snapshot_cat: List[str] = field(default_factory=list)
    snapshot_num: List[str] = field(default_factory=list)

    systematic: SystematicMissingSpec = field(default_factory=SystematicMissingSpec)

    ts_scaler: Literal["standard", "robust", "none"] = "standard"
    snap_scaler: Literal["standard", "robust", "none"] = "standard"
    target_scaler: Literal["standard", "none"] = "standard"
    clip_sigma: Optional[float] = 10.0             # clip scaled TS values to +-clip_sigma
    min_cat_freq: int = 1                          # rarer levels collapse into "missing" (0)
    dtype: str = "float32"


# --------------------------------------------------------------------------- #
# Wide-format helpers
# --------------------------------------------------------------------------- #
def infer_wide_columns(
    columns: Sequence[str], pattern: str = r"^(?P<feature>.+?)_(?P<step>\d+)$"
) -> Dict[str, List[Tuple[int, str]]]:
    """Group wide columns like ``bal_1 ... bal_24`` into ``{feature: [(step, col)]}``."""
    rx = re.compile(pattern)
    out: Dict[str, List[Tuple[int, str]]] = {}
    for col in columns:
        m = rx.match(str(col))
        if m:
            out.setdefault(m.group("feature"), []).append((int(m.group("step")), col))
    for feat in out:
        out[feat].sort(key=lambda t: t[0])
    return out


def wide_to_panel(
    df: pd.DataFrame,
    key: str,
    feature_columns: Mapping[str, Sequence[str]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Wide frame -> ``(ids, X[R, C, T], feature_names)``.

    ``feature_columns`` maps a feature name to its time-ordered column list;
    every feature must contribute the same number of columns.
    """
    features = list(feature_columns)
    lengths = {len(v) for v in feature_columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"features have differing numbers of time columns: "
                         f"{ {f: len(c) for f, c in feature_columns.items()} }")
    t = lengths.pop()
    ids = df[key].to_numpy()
    x = np.empty((len(df), len(features), t), dtype=np.float32)
    for i, feat in enumerate(features):
        x[:, i, :] = df[list(feature_columns[feat])].to_numpy(dtype=np.float64)
    return ids, x, features


def long_to_panel(
    df: pd.DataFrame,
    key: str,
    time_col: str,
    features: Sequence[str],
    time_index: Optional[Sequence[Any]] = None,
    extra_cols: Sequence[str] = (),
) -> Tuple[np.ndarray, np.ndarray, List[Any], Dict[str, np.ndarray]]:
    """Long frame -> ``(ids, X[R, C, T], time_index, extras)``.

    Missing ``(key, time)`` combinations become ``NaN``; duplicates keep the
    **last** row (and warn).
    """
    if time_index is None:
        time_index = sorted(pd.unique(df[time_col]))
    time_pos = {v: i for i, v in enumerate(time_index)}

    ids = pd.unique(df[key])
    row_pos = {v: i for i, v in enumerate(ids)}

    known_time = df[time_col].map(time_pos)
    keep = known_time.notna()
    if (~keep).any():
        warnings.warn(
            f"{int((~keep).sum())} rows dropped: {time_col} outside the requested time_index"
        )
    sub = df.loc[keep]
    r_idx = sub[key].map(row_pos).to_numpy(dtype=np.int64)
    t_idx = known_time.loc[keep].to_numpy(dtype=np.int64)

    if sub.duplicated([key, time_col]).any():
        warnings.warn("duplicate (key, time) rows found; keeping the last occurrence")

    x = np.full((len(ids), len(features), len(time_index)), np.nan, dtype=np.float32)
    vals = sub[list(features)].to_numpy(dtype=np.float64)          # (n_rows, C)
    x[r_idx, :, t_idx] = vals

    extras: Dict[str, np.ndarray] = {}
    for col in extra_cols:
        e = np.full((len(ids), len(time_index)), np.nan, dtype=np.float32)
        e[r_idx, t_idx] = sub[col].to_numpy(dtype=np.float64)
        extras[col] = e
    return np.asarray(ids), x, list(time_index), extras


# --------------------------------------------------------------------------- #
# Scalers (mask-aware, numpy)
# --------------------------------------------------------------------------- #
class _FeatureScaler:
    """Per-feature scaler fitted on **observed** values only.

    ``kind='standard'`` -> mean/std;  ``'robust'`` -> median/IQR;  ``'none'`` -> identity.
    """

    def __init__(self, kind: str = "standard"):
        self.kind = kind
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, observed: np.ndarray) -> "_FeatureScaler":
        """``x``: ``(R, C, T)`` or ``(R, C)``; ``observed``: same shape, bool."""
        flat = x.reshape(x.shape[0], x.shape[1], -1) if x.ndim == 3 else x[:, :, None]
        obs = observed.reshape(flat.shape)
        c = flat.shape[1]
        center = np.zeros(c)
        scale = np.ones(c)
        if self.kind != "none":
            for j in range(c):
                v = flat[:, j, :][obs[:, j, :]].astype(np.float64, copy=False)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    warnings.warn(f"feature index {j} has no observed training value; not scaled")
                    continue
                if self.kind == "standard":
                    center[j] = v.mean()
                    s = v.std()
                else:
                    center[j] = np.median(v)
                    q75, q25 = np.percentile(v, [75, 25])
                    s = (q75 - q25) / 1.349
                scale[j] = s if s > 1e-12 else 1.0
        self.center_, self.scale_ = center, scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None:
            raise RuntimeError("scaler not fitted")
        # cast the statistics to the panel's dtype so a float32 panel stays float32
        center = self.center_.astype(x.dtype, copy=False)
        scale = self.scale_.astype(x.dtype, copy=False)
        if x.ndim == 3:
            return (x - center[None, :, None]) / scale[None, :, None]
        return (x - center[None, :]) / scale[None, :]


class _TargetScaler:
    def __init__(self, kind: str = "standard"):
        self.kind = kind
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "_TargetScaler":
        y = np.asarray(y, dtype=np.float64)
        if self.kind == "none":
            self.mean_ = np.zeros(y.shape[1])
            self.std_ = np.ones(y.shape[1])
        else:
            self.mean_ = np.nanmean(y, axis=0)
            std = np.nanstd(y, axis=0)
            self.std_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64) - self.mean_) / self.std_

    def inverse_transform(self, y) -> np.ndarray:
        arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        arr = arr.reshape(-1, len(self.mean_)) if arr.ndim == 1 else arr
        return arr * self.std_ + self.mean_


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TSDataset(Dataset):
    """Tensors for one split.

    Stored on CPU; masks are kept as ``bool`` to save memory and cast inside the
    model.  Item -> dict with ``x, obs_mask, sys_mask, cat, num, num_missing, y``.
    """

    def __init__(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        sys_mask: torch.Tensor,
        y: torch.Tensor,
        cat: Optional[torch.Tensor] = None,
        num: Optional[torch.Tensor] = None,
        num_missing: Optional[torch.Tensor] = None,
        ids: Optional[np.ndarray] = None,
        name: str = "",
    ):
        self.x, self.obs_mask, self.sys_mask, self.y = x, obs_mask, sys_mask, y
        self.cat, self.num, self.num_missing = cat, num, num_missing
        self.ids = ids
        self.name = name

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        item = {
            "x": self.x[i],
            "obs_mask": self.obs_mask[i],
            "sys_mask": self.sys_mask[i],
            "y": self.y[i],
        }
        if self.cat is not None:
            item["cat"] = self.cat[i]
        if self.num is not None:
            item["num"] = self.num[i]
            item["num_missing"] = self.num_missing[i]
        return item

    # convenience -------------------------------------------------------- #
    @property
    def n_features(self) -> int:
        return self.x.shape[1]

    @property
    def seq_len(self) -> int:
        return self.x.shape[2]

    def summary(self) -> Dict[str, Any]:
        total = self.obs_mask.numel()
        return {
            "name": self.name,
            "n_records": len(self),
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "pct_missing": 100.0 * (1 - self.obs_mask.float().mean().item()),
            "pct_systematic_missing": 100.0 * self.sys_mask.float().mean().item(),
            "pct_random_missing": 100.0
            * ((~self.obs_mask) & (~self.sys_mask)).float().sum().item()
            / max(total, 1),
            "y_mean": self.y.mean().item(),
            "y_std": self.y.std(unbiased=False).item(),
        }


def make_dataloader(
    ds: TSDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
    )


# --------------------------------------------------------------------------- #
# Prepared bundle
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    """Datasets plus everything needed to build a matching model."""

    datasets: Dict[str, TSDataset]
    feature_names: List[str]
    target_names: List[str]
    cat_cardinalities: List[int]
    cat_names: List[str]
    numeric_names: List[str]
    target_scaler: "_TargetScaler"
    time_index: List[Any]
    processor: "DataProcessor"

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def seq_len(self) -> int:
        return next(iter(self.datasets.values())).seq_len

    def model_config(self, **overrides):
        """Build a :class:`~tsfm.config.ModelConfig` whose input dimensions match
        this bundle.  Any nested config can be overridden by keyword, e.g.
        ``model_config(tokenizer=TokenizerConfig(...), transformer=...)``."""
        from .config import (
            HeadConfig,
            MissingConfig,
            ModelConfig,
            SnapshotConfig,
            TokenizerConfig,
            TransformerConfig,
        )

        tok = copy.deepcopy(overrides.pop("tokenizer", None)) or TokenizerConfig(
            in_channels=self.n_features
        )
        tok.in_channels = self.n_features                       # keep it consistent
        snap = copy.deepcopy(overrides.pop("snapshot", None)) or SnapshotConfig()
        snap.cat_cardinalities = list(self.cat_cardinalities)
        snap.n_numeric = len(self.numeric_names)
        head = copy.deepcopy(overrides.pop("head", None)) or HeadConfig()
        head.n_targets = len(self.target_names)
        return ModelConfig(
            tokenizer=tok,
            seq_len=self.seq_len,
            missing=overrides.pop("missing", None) or MissingConfig(),
            transformer=overrides.pop("transformer", None) or TransformerConfig(),
            snapshot=snap,
            head=head,
            **overrides,
        )

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([ds.summary() for ds in self.datasets.values()]).set_index("name")


# --------------------------------------------------------------------------- #
# Processor
# --------------------------------------------------------------------------- #
class DataProcessor:
    """Turns the three raw tables into model-ready tensors.

    All statistics are fitted on the **train** split only.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.ts_scaler_ = _FeatureScaler(cfg.ts_scaler)
        self.snap_scaler_ = _FeatureScaler(cfg.snap_scaler)
        self.target_scaler_ = _TargetScaler(cfg.target_scaler)
        self.cat_vocab_: Dict[str, Dict[Any, int]] = {}
        self.cat_cardinalities_: List[int] = []
        self.fitted_ = False

    # -- panel construction --------------------------------------------- #
    def _build_panel(self, ts) -> Tuple[np.ndarray, np.ndarray, List[Any], Dict[str, np.ndarray]]:
        cfg = self.cfg
        if isinstance(ts, pd.DataFrame):
            if cfg.time_col is None:
                raise ValueError(
                    "DataConfig.time_col is required for a long-format time-series frame. "
                    "For wide input pass a (ids, X) tuple built with wide_to_panel()."
                )
            extra = [cfg.systematic.sys_mask_col] if cfg.systematic.sys_mask_col else []
            ids, x, tindex, extras = long_to_panel(
                ts, cfg.key, cfg.time_col, cfg.ts_features, cfg.time_index, extra
            )
        elif isinstance(ts, tuple) and len(ts) == 2:
            ids, x = np.asarray(ts[0]), np.asarray(ts[1], dtype=np.float32)
            if x.ndim != 3:
                raise ValueError(f"expected a (R, C, T) array, got shape {x.shape}")
            if x.shape[1] != len(cfg.ts_features):
                raise ValueError(
                    f"array has {x.shape[1]} features but DataConfig.ts_features lists "
                    f"{len(cfg.ts_features)}"
                )
            tindex = list(cfg.time_index or range(x.shape[2]))
            extras = {}
        else:
            raise TypeError("ts must be a long DataFrame or an (ids, X[R, C, T]) tuple")

        if cfg.seq_len is not None and x.shape[2] != cfg.seq_len:
            raise ValueError(f"seq_len={cfg.seq_len} but the panel has {x.shape[2]} time steps")
        return ids, x, tindex, extras

    # -- missing masks ---------------------------------------------------- #
    def _systematic_mask(
        self,
        x: np.ndarray,
        snap: Optional[pd.DataFrame],
        extras: Dict[str, np.ndarray],
        explicit: Optional[np.ndarray],
    ) -> np.ndarray:
        cfg = self.cfg
        spec = cfg.systematic
        r, c, t = x.shape
        is_nan = ~np.isfinite(x)

        if spec.method == "none":
            return np.zeros((r, c, t), dtype=bool)

        if spec.method == "explicit":
            m = explicit
            if m is None and spec.sys_mask_col and spec.sys_mask_col in extras:
                m = extras[spec.sys_mask_col]
            if m is None:
                raise ValueError(
                    "systematic method='explicit' but no mask supplied "
                    "(pass sys_mask=... or set sys_mask_col)"
                )
            m = np.asarray(m)
            if m.ndim == 2:
                m = np.repeat(m[:, None, :], c, axis=1)
            if m.shape != (r, c, t):
                raise ValueError(f"explicit sys mask has shape {m.shape}, expected {(r, c, t)}")
            return np.nan_to_num(m, nan=0.0).astype(bool)

        if spec.method == "tenure":
            if snap is None or spec.tenure_col is None or spec.tenure_col not in snap.columns:
                raise ValueError(
                    "systematic method='tenure' needs SystematicMissingSpec.tenure_col "
                    "to name a column of the snapshot frame"
                )
            tenure = pd.to_numeric(snap[spec.tenure_col], errors="coerce").to_numpy(dtype=np.float64)
            if np.isnan(tenure).any():
                warnings.warn(
                    f"{int(np.isnan(tenure).sum())} records have a missing "
                    f"{spec.tenure_col}; assuming a full window for them"
                )
            steps = np.where(np.isnan(tenure), t, np.ceil(tenure * spec.tenure_unit_steps))
            steps = np.clip(steps, 0, t).astype(np.int64)
            pos = np.arange(t)[None, :]
            if cfg.time_order == "oldest_first":
                m2d = pos < (t - steps)[:, None]        # early steps do not exist yet
            else:
                m2d = pos >= steps[:, None]             # late steps are the old ones
            return np.repeat(m2d[:, None, :], c, axis=1)

        # boundary_nan
        if spec.per_feature:
            base = is_nan
            axis = 2
        else:
            base = is_nan.all(axis=1)                   # (R, T)
            axis = 1
        if cfg.time_order == "oldest_first":
            run = np.logical_and.accumulate(base, axis=axis)
        else:
            run = np.flip(np.logical_and.accumulate(np.flip(base, axis=axis), axis=axis), axis=axis)
        if not spec.per_feature:
            run = np.repeat(run[:, None, :], c, axis=1)
        return run

    # -- snapshot --------------------------------------------------------- #
    def _fit_categoricals(self, snap: pd.DataFrame, train_rows: np.ndarray) -> None:
        cfg = self.cfg
        self.cat_vocab_, self.cat_cardinalities_ = {}, []
        for col in cfg.snapshot_cat:
            s = snap[col].astype(object).iloc[train_rows]
            counts = s.value_counts(dropna=True)
            levels = [v for v, n in counts.items() if n >= cfg.min_cat_freq]
            self.cat_vocab_[col] = {v: i + 1 for i, v in enumerate(levels)}   # 0 = missing/unseen
            self.cat_cardinalities_.append(len(levels) + 1)

    def _encode_categoricals(self, snap: pd.DataFrame) -> np.ndarray:
        cfg = self.cfg
        if not cfg.snapshot_cat:
            return np.zeros((len(snap), 0), dtype=np.int64)
        cols = []
        for col in cfg.snapshot_cat:
            vocab = self.cat_vocab_[col]
            codes = snap[col].astype(object).map(vocab)
            cols.append(codes.fillna(0).to_numpy(dtype=np.int64))
        return np.stack(cols, axis=1)

    # -- main entry point -------------------------------------------------- #
    def fit_transform(
        self,
        ts,
        snapshot: Optional[pd.DataFrame],
        target: pd.DataFrame,
        splits: Union[Mapping[str, Sequence[Any]], str],
        sys_mask: Optional[np.ndarray] = None,
        train_split: str = "train",
    ) -> PreparedData:
        """Build one :class:`TSDataset` per split.

        Parameters
        ----------
        ts : long ``DataFrame`` or ``(ids, X[R, C, T])``
        snapshot : ``DataFrame`` with one row per key (or ``None``)
        target : ``DataFrame`` with the key and ``DataConfig.target_cols``
        splits : mapping ``{"train": keys, "holdout": keys, "oot": keys}`` or the
            name of a column in ``target`` holding the split label
        sys_mask : optional explicit systematic-missing mask, ``(R, T)`` or
            ``(R, C, T)``, aligned with the panel's record order
        train_split : which split fits the scalers/vocabularies
        """
        cfg = self.cfg
        if snapshot is None and (cfg.snapshot_cat or cfg.snapshot_num):
            raise ValueError(
                "DataConfig lists snapshot columns "
                f"({cfg.snapshot_cat + cfg.snapshot_num}) but snapshot=None was passed"
            )
        if snapshot is not None:
            wanted = set(cfg.snapshot_cat) | set(cfg.snapshot_num)
            if cfg.systematic.method == "tenure" and cfg.systematic.tenure_col:
                wanted.add(cfg.systematic.tenure_col)
            missing_cols = sorted(wanted - set(snapshot.columns))
            if missing_cols:
                raise KeyError(f"snapshot frame is missing columns: {missing_cols}")
        missing_targets = sorted(set(cfg.target_cols) - set(target.columns))
        if missing_targets:
            raise KeyError(f"target frame is missing columns: {missing_targets}")

        ids, x, tindex, extras = self._build_panel(ts)

        # ---- align the three tables on the key --------------------------- #
        target = target.drop_duplicates(subset=cfg.key, keep="last").set_index(cfg.key)
        snap = None
        if snapshot is not None and (cfg.snapshot_cat or cfg.snapshot_num or
                                     cfg.systematic.method == "tenure"):
            snap = snapshot.drop_duplicates(subset=cfg.key, keep="last").set_index(cfg.key)

        has_target = np.isin(ids, target.index.to_numpy())
        if not has_target.all():
            warnings.warn(f"{int((~has_target).sum())} records have no target row and are dropped")
        if snap is not None:
            has_snap = np.isin(ids, snap.index.to_numpy())
            if not has_snap.all():
                warnings.warn(
                    f"{int((~has_snap).sum())} records have no snapshot row; "
                    "their snapshot values are treated as missing"
                )

        keep = has_target
        y_raw = target.reindex(ids[keep])[cfg.target_cols].to_numpy(dtype=np.float64)
        finite_y = np.isfinite(y_raw).all(axis=1)
        if not finite_y.all():
            warnings.warn(f"{int((~finite_y).sum())} records have a missing target and are dropped")
            idx_keep = np.where(keep)[0][finite_y]
            keep = np.zeros(len(ids), dtype=bool)
            keep[idx_keep] = True
            y_raw = y_raw[finite_y]

        ids = ids[keep]
        x = x[keep]
        if sys_mask is not None:
            sys_mask = np.asarray(sys_mask)[keep]
        for k in extras:
            extras[k] = extras[k][keep]

        snap_aligned = (
            snap.reindex(ids).reset_index() if snap is not None
            else pd.DataFrame({cfg.key: ids})
        )

        # ---- missing masks ---------------------------------------------- #
        sys = self._systematic_mask(x, snap_aligned, extras, sys_mask)
        is_nan = ~np.isfinite(x)
        observed = ~is_nan & ~sys
        conflict = int((~is_nan & sys).sum())
        if conflict:
            warnings.warn(
                f"{conflict} cells are flagged systematically missing but carry a value; "
                "the value is discarded (the mask wins)"
            )

        # ---- splits ------------------------------------------------------ #
        if isinstance(splits, str):
            labels = target.reindex(ids)[splits]
            if labels.isna().any():
                warnings.warn(
                    f"{int(labels.isna().sum())} records have no '{splits}' label "
                    "and are excluded from every split"
                )
            arr = labels.to_numpy()
            split_map = {
                str(v): np.where(arr == v)[0] for v in pd.unique(labels.dropna())
            }
        else:
            pos = pd.Series(np.arange(len(ids)), index=ids)
            split_map = {}
            for name, keys in splits.items():
                sel = pos.reindex(pd.Index(keys).unique()).dropna().to_numpy(dtype=np.int64)
                missing_keys = len(pd.Index(keys).unique()) - len(sel)
                if missing_keys:
                    warnings.warn(f"split '{name}': {missing_keys} keys are not in the panel")
                split_map[name] = sel
        if train_split not in split_map or len(split_map[train_split]) == 0:
            raise ValueError(f"train split '{train_split}' is empty; got {list(split_map)}")
        seen = np.concatenate(list(split_map.values())) if split_map else np.array([])
        if len(seen) != len(np.unique(seen)):
            warnings.warn("splits overlap: some records appear in more than one split")

        train_rows = split_map[train_split]

        # ---- fit on train, transform everything -------------------------- #
        self.ts_scaler_.fit(x[train_rows], observed[train_rows])
        xs = self.ts_scaler_.transform(x)
        xs[~observed] = 0.0                                   # both kinds of missing -> 0
        if cfg.clip_sigma:
            np.clip(xs, -cfg.clip_sigma, cfg.clip_sigma, out=xs)
        xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        self._fit_categoricals(snap_aligned, train_rows)
        cat = self._encode_categoricals(snap_aligned)

        if cfg.snapshot_num:
            num_raw = snap_aligned[cfg.snapshot_num].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float64)
            num_obs = np.isfinite(num_raw)
            self.snap_scaler_.fit(num_raw[train_rows][:, :, None], num_obs[train_rows][:, :, None])
            num = np.where(num_obs, self.snap_scaler_.transform(num_raw), 0.0)
            num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
            num_missing = (~num_obs).astype(np.float64)
        else:
            num = np.zeros((len(ids), 0))
            num_missing = np.zeros((len(ids), 0))

        self.target_scaler_.fit(y_raw[train_rows])
        y = self.target_scaler_.transform(y_raw)

        # ---- tensors ------------------------------------------------------ #
        dt = getattr(torch, cfg.dtype)
        datasets: Dict[str, TSDataset] = {}
        for name, rows in split_map.items():
            datasets[name] = TSDataset(
                x=torch.as_tensor(xs[rows], dtype=dt),
                obs_mask=torch.as_tensor(observed[rows]),
                sys_mask=torch.as_tensor(sys[rows]),
                y=torch.as_tensor(y[rows], dtype=dt),
                cat=torch.as_tensor(cat[rows], dtype=torch.long) if cat.shape[1] else None,
                num=torch.as_tensor(num[rows], dtype=dt) if num.shape[1] else None,
                num_missing=torch.as_tensor(num_missing[rows], dtype=dt) if num.shape[1] else None,
                ids=ids[rows],
                name=name,
            )

        self.fitted_ = True
        return PreparedData(
            datasets=datasets,
            feature_names=list(cfg.ts_features),
            target_names=list(cfg.target_cols),
            cat_cardinalities=list(self.cat_cardinalities_),
            cat_names=list(cfg.snapshot_cat),
            numeric_names=list(cfg.snapshot_num),
            target_scaler=self.target_scaler_,
            time_index=tindex,
            processor=self,
        )


def prepare_datasets(
    ts,
    snapshot: Optional[pd.DataFrame],
    target: pd.DataFrame,
    cfg: DataConfig,
    splits: Union[Mapping[str, Sequence[Any]], str],
    sys_mask: Optional[np.ndarray] = None,
    train_split: str = "train",
) -> PreparedData:
    """One-call wrapper around :class:`DataProcessor`."""
    return DataProcessor(cfg).fit_transform(
        ts, snapshot, target, splits, sys_mask=sys_mask, train_split=train_split
    )
