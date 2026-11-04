"""Shared helpers for the tests (fixtures are in conftest.py)."""
import numpy as np

from tsfusion import ModelConfig, RawData, group_columns_by_prefix, long_df_to_array


def _raw(d, ids):
    x, _ = long_df_to_array(d["ts_df"], record_ids=ids)
    m, _ = long_df_to_array(d["missing_df"], record_ids=ids, fill_value=1)
    s, _ = long_df_to_array(d["special_df"], record_ids=ids, fill_value=0)
    cols = [c for _, cs in group_columns_by_prefix(d["snapshot_ple_df"].columns) for c in cs]
    snap = d["snapshot_ple_df"].loc[ids, cols].to_numpy()
    return RawData(x, m, s.astype(np.int64), snap, d["y"].loc[ids].to_numpy())


def _group_sizes(synth):
    return [len(c) for _, c in group_columns_by_prefix(synth["snapshot_ple_df"].columns)]


def _model_cfg(synth, **kw):
    base = dict(n_vars=len(synth["var_names"]), n_steps=11,
                n_special_codes=synth["n_special_codes"],
                snapshot_group_sizes=_group_sizes(synth), d_model=32, n_heads=4,
                n_ts_layers=1, head_hidden=32)
    base.update(kw)
    return ModelConfig(**base)
