"""Data containers, preprocessing and batching.

Conventions
-----------
* Time-series tensors are ``(N records, T time steps, V variables)``.
* Missing / special information comes EITHER as one combined matrix

  * ``state``  : ``(N, T, V)`` uint8 code: 0 = regular, 1 = missing, 1+k = special value k

  OR as two matrices that are merged into that same coding:

  * ``missing``  : ``(N, T, V)`` 0/1 indicator, 1 = value is missing (imputed as 0 upstream).
  * ``special``  : ``(N, T, V)`` integer code, 0 = not special, ``k`` in ``1..K`` = special value k.

  The meaning of each code is the same for every variable. The coding is::

      state = 0          regular (observed) value
      state = 1          missing
      state = 1 + k      special value k   (k = 1..K)

  so a single ``nn.Embedding(2 + K, d)`` learns one vector for "missing" and one per
  special value, shared across all variables and time steps.
* Snapshot variables are passed as one 2-D array ``(N, sum(group_sizes))`` - the PLE
  vectors of all snapshot variables concatenated column-wise - plus ``group_sizes``
  (bins per variable). Internally they are re-packed to ``(N, S, max_bins)`` with zero
  padding so every variable gets its own linear embedding.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

STATE_REGULAR = 0
STATE_MISSING = 1
TARGET_MODES = ("none", "standard", "log1p_standard", "mean_scale")


# =============================================================================
# Helpers to go from pandas DataFrames to arrays
# =============================================================================
def long_df_to_array(
    df: pd.DataFrame,
    value_cols: Optional[Sequence[str]] = None,
    id_col: str = "record_id",
    time_col: str = "time_step",
    record_ids: Optional[Sequence] = None,
    time_steps: Optional[Sequence] = None,
    fill_value: float = np.nan,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a long DataFrame (one row per record x time step, one column per
    variable) into an ``(N, T, V)`` array.

    ``id_col``/``time_col`` may be columns or index levels. Pass the *same*
    ``record_ids`` / ``time_steps`` / ``value_cols`` for the value, missing and special
    DataFrames (and the same record order for snapshot / target) so that everything is
    aligned. Absent (record, time) rows are filled with ``fill_value``.
    """
    if id_col in df.columns or time_col in df.columns:
        df = df.set_index([id_col, time_col])
    if list(df.index.names) != [id_col, time_col]:
        raise ValueError(f"expected index levels [{id_col!r}, {time_col!r}], got {df.index.names}")
    if df.index.duplicated().any():
        raise ValueError("duplicated (record, time) rows in DataFrame")
    if value_cols is None:
        value_cols = list(df.columns)
    if record_ids is None:
        record_ids = df.index.get_level_values(0).unique().sort_values()
    if time_steps is None:
        levels = df.index.get_level_values(1)
        if not (pd.api.types.is_numeric_dtype(levels) or pd.api.types.is_datetime64_any_dtype(levels)):
            raise ValueError("time labels are not numeric/datetime - pass time_steps explicitly in "
                             "chronological order (sorting strings would put 't10' before 't2')")
        time_steps = levels.unique().sort_values()
    full_index = pd.MultiIndex.from_product([list(record_ids), list(time_steps)],
                                            names=[id_col, time_col])
    found = full_index.isin(df.index)
    if not found.any():
        raise ValueError("none of the requested (record, time) pairs exist in the DataFrame - "
                         "check that record_ids / time_steps have the same dtype as the index")
    if not found.all():
        warnings.warn(f"{int((~found).sum())} (record, time) rows are absent and filled with "
                      f"{fill_value}")
    out = df.reindex(full_index)[list(value_cols)]
    if fill_value is not None and not (isinstance(fill_value, float) and np.isnan(fill_value)):
        out = out.fillna(fill_value)
    arr = out.to_numpy(dtype=dtype).reshape(len(record_ids), len(time_steps), len(value_cols))
    return np.ascontiguousarray(arr), np.asarray(record_ids)


def wide_df_to_array(
    df: pd.DataFrame,
    var_names: Sequence[str],
    time_steps: Sequence,
    template: str = "{var}_t{t}",
    dtype=np.float32,
) -> np.ndarray:
    """Convert a wide DataFrame (one row per record, columns like ``bal_t1 .. bal_t11``)
    into an ``(N, T, V)`` array."""
    cols = [template.format(var=v, t=t) for t in time_steps for v in var_names]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"columns not found, e.g. {missing_cols[:5]}")
    return df[cols].to_numpy(dtype=dtype).reshape(len(df), len(time_steps), len(var_names))


def group_columns_by_prefix(columns: Sequence[str], pattern: str = r"^(.*)_bin\d+$"
                            ) -> List[Tuple[str, List[str]]]:
    """Group PLE columns such as ``age_bin0, age_bin1, ..., income_bin0, ...`` by feature.

    Returns ``[(feature_name, [columns...]), ...]`` in order of first appearance.
    Columns that do not match ``pattern`` are treated as a one-column (raw) feature.

    IMPORTANT: build the snapshot matrix with the *returned* column order, e.g.
    ``df[[c for _, cols in groups for c in cols]].to_numpy()``, and pass
    ``[len(cols) for _, cols in groups]`` as ``snapshot_group_sizes``.
    Bins are kept in the order they appear in ``columns``.
    """
    groups: Dict[str, List[str]] = {}
    rx = re.compile(pattern)
    for c in columns:
        m = rx.match(str(c))
        name = m.group(1) if m else str(c)
        groups.setdefault(name, []).append(c)
    ordered = [c for cols in groups.values() for c in cols]
    if ordered != list(columns):
        warnings.warn("PLE columns of different features are interleaved; use the returned "
                      "column order to build the snapshot matrix")
    return list(groups.items())


def fit_ple_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges for piecewise linear encoding (fit on training data only)."""
    v = values[np.isfinite(values)]
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([v.min(), v.min() + 1.0])
    return edges


def ple_encode(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Piecewise linear encoding (Gorishniy et al., 2022) clipped to [0, 1].

    For bin ``j`` with edges ``[b_{j-1}, b_j)``: 1 if x >= b_j, 0 if x < b_{j-1},
    otherwise the fractional position inside the bin.
    """
    left, right = edges[:-1], edges[1:]
    x = values[:, None]
    enc = (x - left) / np.maximum(right - left, 1e-12)
    return np.clip(enc, 0.0, 1.0).astype(np.float32)


def encode_special_values(x_raw: np.ndarray, special_values: Dict[float, int],
                          special_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Build the integer special-code tensor from sentinel values in the raw data.

    ``special_values`` maps sentinel -> code, e.g. ``{-999: 1, -998: 2, -997: 3}``.
    If ``special_mask`` (0/1 "is special" indicator) is given, only flagged cells are coded.
    """
    codes = np.zeros(x_raw.shape, dtype=np.int64)
    for value, code in special_values.items():
        if code < 1:
            raise ValueError("special codes must be >= 1 (0 means 'not special')")
        hit = np.isclose(x_raw, value, rtol=0.0, atol=1e-6)   # exact match, no relative tol
        if special_mask is not None:
            hit &= special_mask.astype(bool)
        codes[hit] = code
    if special_mask is not None:
        unexplained = special_mask.astype(bool) & (codes == 0)
        if unexplained.any():
            raise ValueError(f"{int(unexplained.sum())} cells flagged special but their value "
                             f"is not in special_values")
    return codes


def _state_dtype(max_code: int):
    return np.uint8 if max_code <= 255 else np.int16


def _merge_state(missing: np.ndarray, special: np.ndarray) -> Tuple[np.ndarray, int]:
    """(state, number of cells flagged both missing and special)."""
    missing = np.asarray(missing)
    special = np.asarray(special)
    if missing.shape != special.shape:
        raise ValueError(f"missing {missing.shape} and special {special.shape} shapes differ")
    if special.size and special.min() < 0:
        raise ValueError("special codes must be >= 0")
    smax = int(special.max(initial=0))
    miss = missing.astype(bool, copy=False)
    state = np.zeros(missing.shape, dtype=_state_dtype(smax + 1))   # C-contiguous
    state[miss] = STATE_MISSING
    sp = special > 0
    conflicts = int(np.count_nonzero(miss & sp))
    state[sp] = special[sp] + 1
    return state, conflicts


def build_state(missing: np.ndarray, special: np.ndarray) -> np.ndarray:
    """Merge the missing indicator and the special code into one state matrix
    (0 regular, 1 missing, 1+k special k; uint8).

    A cell flagged as both missing and special is treated as *special* (the more
    specific information); a warning reports how many such cells exist.
    """
    state, conflicts = _merge_state(missing, special)
    if conflicts:
        warnings.warn(f"{conflicts} cells are flagged both missing and special; "
                      "treating them as special.")
    return state


# =============================================================================
# Containers
# =============================================================================
@dataclass
class RawData:
    """Unprocessed inputs for one split (train / validation / test).

    Pass the missing / special information EITHER as one combined matrix ``state``
    (0 = regular, 1 = missing, 1+k = special value k; store it as uint8 = 1 byte per cell)
    OR as the two matrices ``missing`` and ``special``::

        RawData(x, state=state, snapshot=snapshot, y=y)        # combined matrix
        RawData(x, missing, special, snapshot, y)              # separate matrices

    Arrays are used as given (no copies): ``x`` should already be float32. Memory-mapped
    arrays (``np.load(..., mmap_mode="r")``) work - the preprocessor reads them in chunks.
    """
    x: np.ndarray                              # (N, T, V) values; anything at missing / special cells
    missing: Optional[np.ndarray] = None       # (N, T, V) 0/1            (separate form)
    special: Optional[np.ndarray] = None       # (N, T, V) int codes 0..K (separate form)
    snapshot: Optional[np.ndarray] = None      # (N, sum(group_sizes)) PLE vectors (or raw values)
    y: Optional[np.ndarray] = None             # (N,) target
    state: Optional[np.ndarray] = None         # (N, T, V) uint8 0..K+1   (combined form)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)      # no copy if already float32
        if self.x.ndim != 3:
            raise ValueError(f"x must be (N, T, V), got {self.x.shape}")
        if self.state is not None:
            if self.missing is not None or self.special is not None:
                raise ValueError("pass either state (combined) or missing + special, not both")
            self.state = np.asarray(self.state)
            if self.state.shape != self.x.shape:
                raise ValueError(f"state {self.state.shape} and x {self.x.shape} shapes differ")
            if not np.issubdtype(self.state.dtype, np.integer):
                raise TypeError(f"state must be an integer array (got {self.state.dtype}); "
                                "convert with .astype(np.uint8)")
            if self.state.dtype.itemsize > 1:
                warnings.warn(f"state is {self.state.dtype}: store it as uint8 to use 1 byte "
                              f"per cell instead of {self.state.dtype.itemsize}")
        else:
            if self.missing is None or self.special is None:
                raise ValueError("pass state (combined matrix), or both missing and special")
            self.missing = np.asarray(self.missing)
            self.special = np.asarray(self.special)
            if self.missing.shape != self.x.shape or self.special.shape != self.x.shape:
                raise ValueError("x, missing and special must have the same (N, T, V) shape")
        if self.snapshot is not None:
            self.snapshot = np.asarray(self.snapshot, dtype=np.float32)
        if self.y is not None:
            self.y = np.asarray(self.y, dtype=np.float64).reshape(-1)
        n = len(self.x)
        if self.snapshot is not None and (self.snapshot.ndim != 2 or len(self.snapshot) != n):
            raise ValueError(f"snapshot must be (N, F) with N={n}, got {self.snapshot.shape}")
        if self.y is not None and len(self.y) != n:
            raise ValueError(f"y has {len(self.y)} rows, expected {n}")
        self.consumed = False          # set by Preprocessor.transform(..., inplace=True)

    def __len__(self) -> int:
        return len(self.x)

    @property
    def combined(self) -> bool:
        return self.state is not None

    def nbytes(self) -> int:
        arrs = (self.x, self.missing, self.special, self.state, self.snapshot, self.y)
        return sum(a.nbytes for a in arrs if a is not None)

    def state_slice(self, sl: slice) -> Tuple[np.ndarray, int]:
        """Combined state of records ``sl`` and the number of missing+special conflicts.
        For a combined input this is a view (no copy)."""
        if self.state is not None:
            return self.state[sl], 0
        return _merge_state(self.missing[sl], self.special[sl])

    def get_state(self) -> np.ndarray:
        """Full combined state matrix (0 regular, 1 missing, 1+k special k)."""
        return self.state if self.state is not None else build_state(self.missing, self.special)

    def subset(self, idx: np.ndarray) -> "RawData":
        pick = lambda a: None if a is None else a[idx]  # noqa: E731
        return RawData(self.x[idx], pick(self.missing), pick(self.special), pick(self.snapshot),
                       pick(self.y), pick(self.state))


@dataclass
class ProcessedData:
    """Model-ready tensors, on the CPU. The trainer either copies them to the GPU once
    (small data) or streams them batch by batch (``TrainConfig.data_on_device``)."""
    x: torch.Tensor                        # (N, T, V) float32, standardised, 0 at non-regular cells
    state: torch.Tensor                    # (N, T, V) uint8 state codes
    snapshot: Optional[torch.Tensor]       # (N, S, max_bins) float32 or None
    y: Optional[torch.Tensor]              # (N,) float32 target in *transformed* space
    y_raw: Optional[np.ndarray]            # (N,) float64 target on the original scale

    def __len__(self) -> int:
        return self.x.shape[0]

    def nbytes(self) -> int:
        """Memory the tensors need on a device (y_raw stays on the CPU)."""
        return sum(t.numel() * t.element_size()
                   for t in (self.x, self.state, self.snapshot, self.y) if t is not None)

    def to(self, device: torch.device) -> "ProcessedData":
        mv = lambda t: None if t is None else t.to(device)  # noqa: E731
        return ProcessedData(mv(self.x), mv(self.state), mv(self.snapshot), mv(self.y), self.y_raw)

    def batch(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {"x": self.x[idx], "state": self.state[idx]}
        if self.snapshot is not None:
            out["snapshot"] = self.snapshot[idx]
        if self.y is not None:
            out["y"] = self.y[idx]
        return out


# =============================================================================
# Scalers
# =============================================================================
class TargetScaler:
    """Invertible target transformation.

    * ``standard``       : (y - mean) / std             (MSE on the original scale)
    * ``log1p_standard`` : (log1p(y) - m) / s           (MSE on the log scale; predictions are
                                                        biased low for the mean - see notebook)
    * ``mean_scale``     : y / mean(y)                 (Tweedie: keeps y >= 0, log-link friendly)
    * ``none``           : identity
    """

    def __init__(self, mode: str = "standard"):
        if mode not in TARGET_MODES:
            raise ValueError(f"target mode must be one of {TARGET_MODES}")
        self.mode = mode
        self.loc = 0.0
        self.scale = 1.0

    def fit(self, y: np.ndarray) -> "TargetScaler":
        y = np.asarray(y, dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("target contains non-finite values")
        if self.mode == "standard":
            self.loc, self.scale = float(y.mean()), float(y.std() or 1.0)
        elif self.mode == "log1p_standard":
            if (y <= -1).any():
                raise ValueError("log1p transform needs y > -1")
            ly = np.log1p(y)
            self.loc, self.scale = float(ly.mean()), float(ly.std() or 1.0)
        elif self.mode == "mean_scale":
            if (y < 0).any():
                raise ValueError("mean_scale (Tweedie) needs y >= 0")
            self.loc, self.scale = 0.0, float(y.mean() or 1.0)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.mode == "log1p_standard":
            return (np.log1p(y) - self.loc) / self.scale
        return (y - self.loc) / self.scale

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if self.mode == "log1p_standard":
            return np.clip(np.expm1(z * self.scale + self.loc), 0.0, None)
        return z * self.scale + self.loc

    def state_dict(self) -> dict:
        return {"mode": self.mode, "loc": float(self.loc), "scale": float(self.scale)}

    @classmethod
    def from_state_dict(cls, d: dict) -> "TargetScaler":
        s = cls(d["mode"])
        s.loc, s.scale = float(d["loc"]), float(d["scale"])
        return s


def default_target_mode(loss: str) -> str:
    return "mean_scale" if loss == "tweedie" else "standard"


class Preprocessor:
    """Fit on the training split only, then transform every split.

    Steps for the time series
      0. optional ``sign(x) * log1p(|x|)`` for heavy-tailed variables (``ts_log_vars``).
      1. ``state`` = the combined matrix as given (or merge(missing, special)); NaN / inf
         in ``x`` at a regular cell turns that cell into "missing".
      2. per-variable mean / std computed on **regular cells only** (sentinels such as
         -999 and imputed zeros would otherwise corrupt the statistics).
      3. standardise, clip to ``+/- ts_clip`` and set every non-regular cell to exactly 0.
         (After standardisation the upstream "impute 0" is no longer 0, so we re-impute.)
    Snapshot
      * ``snapshot_kind="ple"``: PLE vectors are used as is (already in [0, 1]).
      * ``snapshot_kind="raw"``: each column is standardised; NaN -> 0 (= mean).
    Target
      * see :class:`TargetScaler`.

    Memory: ``fit`` and ``transform`` work through the records in chunks (``chunk_size``
    records; default: about ``max_chunk_cells`` cells per chunk), so the working memory is
    ~10-20 bytes per cell *of one chunk*, independent of the dataset size.
    ``transform(..., inplace=True)`` writes the standardised values into ``data.x`` (and the
    final state codes into ``data.state``) instead of allocating new arrays: the processed
    data then costs no extra memory, but the raw arrays are overwritten.
    """

    max_chunk_cells = 50_000_000      # ~0.4-1 GB of temporary arrays per chunk

    def __init__(self, n_special_codes: int, snapshot_group_sizes: Optional[Sequence[int]] = None,
                 snapshot_kind: str = "ple", target_mode: str = "standard",
                 ts_clip: Optional[float] = 10.0, ts_log_vars: Optional[Sequence[int]] = None):
        if snapshot_kind not in ("ple", "raw"):
            raise ValueError("snapshot_kind must be 'ple' or 'raw'")
        self.n_special_codes = int(n_special_codes)
        self.snapshot_group_sizes = [int(g) for g in (snapshot_group_sizes or [])]
        self.snapshot_kind = snapshot_kind
        self.target_scaler = TargetScaler(target_mode)
        self.ts_clip = ts_clip
        self.ts_log_vars = [int(v) for v in (ts_log_vars or [])]
        self.ts_mean: Optional[np.ndarray] = None
        self.ts_std: Optional[np.ndarray] = None
        self.snap_mean: Optional[np.ndarray] = None
        self.snap_std: Optional[np.ndarray] = None
        self.n_steps: Optional[int] = None
        self.n_vars: Optional[int] = None
        self.fitted = False

    @property
    def n_states(self) -> int:
        """0 regular, 1 missing, 2..K+1 special codes."""
        return 2 + self.n_special_codes

    # ---------------------------------------------------------------- internals
    def _chunks(self, data: RawData, chunk_size: Optional[int]) -> Iterator[slice]:
        n, t, v = data.x.shape
        if chunk_size is None:
            chunk_size = max(1, self.max_chunk_cells // max(1, t * v))
        for start in range(0, n, chunk_size):
            yield slice(start, min(n, start + chunk_size))

    def _check_codes(self, st: np.ndarray) -> None:
        if st.size == 0:
            return
        lo, hi = int(st.min()), int(st.max())
        if lo < 0 or hi > self.n_states - 1:
            raise ValueError(
                f"state code {hi if hi > self.n_states - 1 else lo} found, but with "
                f"n_special_codes={self.n_special_codes} the codes must be 0 (regular), "
                f"1 (missing) or 2..{self.n_states - 1} (special)")

    def _signed_log(self, xc: np.ndarray) -> None:
        if self.ts_log_vars:
            v = self.ts_log_vars
            xc[..., v] = np.sign(xc[..., v]) * np.log1p(np.abs(xc[..., v]))

    def _check_snapshot(self, data: RawData) -> None:
        if not self.snapshot_group_sizes:
            return
        if data.snapshot is None:
            raise ValueError("snapshot_group_sizes given but data.snapshot is None")
        if data.snapshot.shape[1] != sum(self.snapshot_group_sizes):
            raise ValueError(f"snapshot has {data.snapshot.shape[1]} columns but "
                             f"sum(snapshot_group_sizes) = {sum(self.snapshot_group_sizes)}")
        if self.snapshot_kind == "ple" and not np.isfinite(data.snapshot).all():
            raise ValueError("PLE snapshot contains NaN/inf - encode missing snapshot values "
                             "explicitly (e.g. an extra indicator group) before training")

    @staticmethod
    def _warn_conflicts(conflicts: int) -> None:
        if conflicts:
            warnings.warn(f"{conflicts} cells are flagged both missing and special; "
                          "treating them as special.")

    @staticmethod
    def _check_not_consumed(data: RawData) -> None:
        if getattr(data, "consumed", False):
            raise ValueError("this RawData was already transformed in place - its x / state "
                             "now hold processed values")

    # ---------------------------------------------------------------- API
    def fit(self, data: RawData, chunk_size: Optional[int] = None) -> "Preprocessor":
        """Per-variable mean / std over regular cells (streamed over chunks with the
        parallel-variance update, so the result equals a single pass over all data)."""
        self._check_not_consumed(data)
        _, t, v = data.x.shape
        count, mean, m2 = np.zeros(v), np.zeros(v), np.zeros(v)
        conflicts = 0
        for sl in self._chunks(data, chunk_size):
            st, c = data.state_slice(sl)
            conflicts += c
            self._check_codes(st)
            xc = data.x[sl].astype(np.float64)                     # chunk copy
            reg = (st == STATE_REGULAR) & np.isfinite(xc)
            self._signed_log(xc)
            xc[~reg] = 0.0
            cnt = reg.sum(axis=(0, 1)).astype(np.float64)
            mean_b = np.divide(xc.sum(axis=(0, 1)), cnt, out=np.zeros(v), where=cnt > 0)
            xc -= mean_b
            xc[~reg] = 0.0
            m2_b = np.einsum("ntv,ntv->v", xc, xc)
            total = count + cnt
            w = np.divide(cnt, total, out=np.zeros(v), where=total > 0)
            delta = mean_b - mean
            mean = mean + delta * w
            m2 = m2 + m2_b + delta ** 2 * count * w
            count = total
        self._warn_conflicts(conflicts)
        std = np.sqrt(np.divide(m2, count, out=np.zeros(v), where=count > 0))
        empty = count == 0
        if empty.any():
            warnings.warn(f"variables {np.where(empty)[0].tolist()} have no regular values "
                          "in the training data")
        mean = np.where(empty, 0.0, mean)
        std = np.where(empty | ~(std > 1e-12), 1.0, std)
        self.ts_mean, self.ts_std = mean.astype(np.float32), std.astype(np.float32)
        self.n_steps, self.n_vars = t, v

        self._check_snapshot(data)
        if self.snapshot_group_sizes and self.snapshot_kind == "raw":
            snap = data.snapshot.astype(np.float64)
            self.snap_mean = np.nanmean(snap, axis=0).astype(np.float32)
            sd = np.nanstd(snap, axis=0)
            self.snap_std = np.where(sd > 1e-12, sd, 1.0).astype(np.float32)

        if data.y is None:
            raise ValueError("training data needs y")
        self.target_scaler.fit(data.y)
        self.fitted = True
        return self

    def transform(self, data: RawData, chunk_size: Optional[int] = None,
                  inplace: bool = False) -> ProcessedData:
        """Model-ready tensors. ``inplace=True`` reuses ``data.x`` (and a uint8 ``data.state``)
        as the output buffers: no second copy of the data, but ``data`` is consumed."""
        if not self.fitted:
            raise RuntimeError("call fit() first")
        self._check_not_consumed(data)
        if data.x.shape[1:] != (self.n_steps, self.n_vars):
            raise ValueError(f"expected (N, {self.n_steps}, {self.n_vars}), got {data.x.shape}")
        n = len(data)
        sdtype = _state_dtype(self.n_states - 1)
        if inplace:
            if not (data.x.flags.writeable and data.x.flags.c_contiguous):
                raise ValueError("inplace=True needs a writable, C-contiguous float32 x "
                                 "(read-only memory-mapped input: use inplace=False)")
            out_x = data.x
        else:
            out_x = np.empty(data.x.shape, dtype=np.float32)
        reuse_state = (inplace and data.state is not None and data.state.dtype == sdtype
                       and data.state.flags.writeable and data.state.flags.c_contiguous)
        out_s = data.state if reuse_state else np.empty(data.x.shape, dtype=sdtype)
        mean = self.ts_mean.reshape(1, 1, -1)
        std = self.ts_std.reshape(1, 1, -1)

        conflicts = 0
        for sl in self._chunks(data, chunk_size):
            st_src, c = data.state_slice(sl)
            conflicts += c
            self._check_codes(st_src)
            st = st_src if reuse_state else np.array(st_src, dtype=sdtype)
            xc = out_x[sl] if inplace else np.array(data.x[sl], dtype=np.float32)
            bad = ~np.isfinite(xc)
            if bad.any():
                st[bad & (st == STATE_REGULAR)] = STATE_MISSING   # NaN at a regular cell
                xc[bad] = 0.0
            del bad
            self._signed_log(xc)
            xc -= mean
            xc /= std
            if self.ts_clip is not None:
                np.clip(xc, -self.ts_clip, self.ts_clip, out=xc)
            xc[st != STATE_REGULAR] = 0.0
            if not inplace:
                out_x[sl] = xc
            if not reuse_state:
                out_s[sl] = st
        self._warn_conflicts(conflicts)
        if inplace:
            data.consumed = True

        snap = None
        if self.snapshot_group_sizes:
            self._check_snapshot(data)
            s = data.snapshot.astype(np.float32)
            if self.snapshot_kind == "raw":
                s = np.nan_to_num((s - self.snap_mean) / self.snap_std, nan=0.0, posinf=0.0,
                                  neginf=0.0)
            snap = torch.from_numpy(pack_snapshot(s, self.snapshot_group_sizes))

        y_t = y_raw = None
        if data.y is not None:
            y_raw = data.y.astype(np.float64)
            y_t = torch.from_numpy(self.target_scaler.transform(y_raw).astype(np.float32))

        assert len(out_x) == n
        return ProcessedData(x=torch.from_numpy(out_x), state=torch.from_numpy(out_s),
                             snapshot=snap, y=y_t, y_raw=y_raw)

    # ---------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        """Only python scalars / lists / tensors -> loadable with ``torch.load(weights_only=True)``."""
        t = lambda a: None if a is None else torch.as_tensor(a)  # noqa: E731
        return {
            "n_special_codes": self.n_special_codes,
            "snapshot_group_sizes": list(self.snapshot_group_sizes),
            "snapshot_kind": self.snapshot_kind,
            "ts_clip": self.ts_clip, "ts_log_vars": list(self.ts_log_vars),
            "ts_mean": t(self.ts_mean), "ts_std": t(self.ts_std),
            "snap_mean": t(self.snap_mean), "snap_std": t(self.snap_std),
            "n_steps": self.n_steps, "n_vars": self.n_vars,
            "target_scaler": self.target_scaler.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "Preprocessor":
        p = cls(d["n_special_codes"], d["snapshot_group_sizes"], d["snapshot_kind"],
                d["target_scaler"]["mode"], d["ts_clip"], d.get("ts_log_vars"))
        n = lambda a: None if a is None else a.cpu().numpy()  # noqa: E731
        p.ts_mean, p.ts_std = n(d["ts_mean"]), n(d["ts_std"])
        p.snap_mean, p.snap_std = n(d["snap_mean"]), n(d["snap_std"])
        p.n_steps, p.n_vars = d["n_steps"], d["n_vars"]
        p.target_scaler = TargetScaler.from_state_dict(d["target_scaler"])
        p.fitted = True
        return p


def pack_snapshot(snapshot_2d: np.ndarray, group_sizes: Sequence[int]) -> np.ndarray:
    """``(N, sum(group_sizes))`` -> zero-padded ``(N, S, max(group_sizes))``."""
    n = snapshot_2d.shape[0]
    out = np.zeros((n, len(group_sizes), max(group_sizes)), dtype=np.float32)
    start = 0
    for j, g in enumerate(group_sizes):
        out[:, j, :g] = snapshot_2d[:, start:start + g]
        start += g
    return out


# =============================================================================
# Saving / loading prepared splits (for scripts and multi-GPU runs)
# =============================================================================
_RAW_FIELDS = ("x", "missing", "special", "state", "snapshot", "y")


def save_dataset(directory: str, splits: Dict[str, "RawData"], n_special_codes: int,
                 snapshot_group_sizes: Sequence[int], **extra_meta) -> None:
    """Write each split as plain ``.npy`` files (``<split>_x.npy``, ``<split>_state.npy`` or
    ``<split>_missing.npy`` + ``<split>_special.npy``, ...) plus ``meta.json``. Arrays must
    already be aligned (same record order)."""
    import json
    import os
    os.makedirs(directory, exist_ok=True)
    for name, raw in splits.items():
        for field in _RAW_FIELDS:
            arr = getattr(raw, field)
            if arr is not None:
                np.save(os.path.join(directory, f"{name}_{field}.npy"), arr)
    meta = {"splits": list(splits), "n_special_codes": int(n_special_codes),
            "snapshot_group_sizes": [int(g) for g in snapshot_group_sizes], **extra_meta}
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_dataset(directory: str, mmap: bool = False) -> Tuple[Dict[str, "RawData"], dict]:
    """Inverse of :func:`save_dataset`: returns ``({split: RawData}, meta)``.

    ``mmap=True`` memory-maps the arrays (read-only): nothing is read into RAM until the
    preprocessor processes it chunk by chunk."""
    import json
    import os
    with open(os.path.join(directory, "meta.json")) as f:
        meta = json.load(f)
    splits = {}
    for name in meta["splits"]:
        arrs = {}
        for field in _RAW_FIELDS:
            path = os.path.join(directory, f"{name}_{field}.npy")
            arrs[field] = (np.load(path, mmap_mode="r" if mmap else None)
                           if os.path.exists(path) else None)
        splits[name] = RawData(**arrs)
    return splits, meta


# =============================================================================
# Batching
# =============================================================================
class InMemoryLoader:
    """Minimal, fast batch iterator over tensors that already sit in (GPU) memory.

    For tabular-sized data this is much faster than ``torch.utils.data.DataLoader`` with
    per-item ``__getitem__`` + collation: each batch is a single fancy-index per tensor.

    Multi-GPU (``world_size > 1``): every rank draws the SAME permutation (same seed and
    epoch), keeps ``len // world_size * world_size`` records (drops at most
    ``world_size - 1`` random records per epoch) and takes every ``world_size``-th one
    starting at ``rank``. So all ranks run the same number of batches, and global batch k
    is exactly what a single GPU would see with ``batch_size * world_size``.

    ``device``: where the batches should end up. If the data already lives there, batches
    are cut on that device. If the data is on the CPU and ``device`` is a GPU, only one
    batch at a time is copied: it is cut on the CPU, pinned, sent with a non-blocking copy,
    and the next batch is prepared while the GPU works on the current one.
    """

    def __init__(self, data: ProcessedData, batch_size: int, shuffle: bool = False,
                 seed: int = 0, indices: Optional[np.ndarray] = None,
                 rank: int = 0, world_size: int = 1, device=None):
        self.data = data
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.source_device = data.x.device
        self.device = self.source_device if device is None else torch.device(device)
        self.rank, self.world_size = int(rank), int(world_size)
        idx = np.arange(len(data)) if indices is None else np.asarray(indices)
        self.indices = torch.as_tensor(idx, dtype=torch.long)

    def _per_rank(self) -> int:
        n = len(self.indices)
        return n // self.world_size if self.world_size > 1 else n

    def __len__(self) -> int:
        return (self._per_rank() + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        idx = self.indices
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            idx = idx[torch.randperm(len(idx), generator=g)]
        if self.world_size > 1:
            idx = idx[: self._per_rank() * self.world_size][self.rank::self.world_size]
        if _same_device(self.source_device, self.device):
            idx = idx.to(self.source_device)
            for start in range(0, len(idx), self.batch_size):
                yield self.data.batch(idx[start:start + self.batch_size])
        else:
            yield from stream_batches(self.data, idx, self.batch_size, self.device)


def _same_device(a: torch.device, b: torch.device) -> bool:
    if a.type != b.type:
        return False
    if a.type != "cuda" or a.index is None or b.index is None:
        return True
    return a.index == b.index


def stream_batches(data: ProcessedData, idx: torch.Tensor, batch_size: int,
                   device: torch.device) -> Iterator[Dict[str, torch.Tensor]]:
    """Batches of ``data`` (kept in CPU memory) moved to ``device`` one at a time, with a
    one-batch look-ahead: while the caller computes on batch k, batch k+1 is already cut,
    pinned and on its way (non-blocking copy from page-locked memory)."""
    idx = idx.cpu()
    pin = device.type == "cuda"

    def load(start: int) -> Dict[str, torch.Tensor]:
        b = data.batch(idx[start:start + batch_size])
        if pin:
            b = {k: v.pin_memory() for k, v in b.items()}
        return {k: v.to(device, non_blocking=pin) for k, v in b.items()}

    starts = list(range(0, len(idx), batch_size))
    nxt = load(starts[0]) if starts else None
    for i in range(len(starts)):
        cur = nxt
        if i + 1 < len(starts):
            nxt = load(starts[i + 1])
        yield cur
