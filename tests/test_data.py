import numpy as np
import pandas as pd
import pytest
import torch

from tsfm.data import (
    DataConfig, SystematicMissingSpec, infer_wide_columns, prepare_datasets, wide_to_panel,
)
from tsfm.simulate import make_synthetic_panel

FEATS = ["f0", "f1", "f2"]


def _tiny_long():
    """4 records x 3 features x 6 steps, month_idx 0 = oldest."""
    rows = []
    for rid in range(4):
        for t in range(6):
            rows.append({"pk": rid, "month_idx": t, "f0": rid + t, "f1": t * 1.0, "f2": rid * 1.0})
    df = pd.DataFrame(rows)
    # record 1 has only 2 months of history (systematic: steps 0..3)
    df.loc[(df.pk == 1) & (df.month_idx < 4), FEATS] = np.nan
    # record 2 has one random hole in the middle
    df.loc[(df.pk == 2) & (df.month_idx == 3), "f0"] = np.nan
    snap = pd.DataFrame({
        "pk": range(4), "tenure": [6, 2, 6, 6],
        "seg": ["a", "b", "a", None], "score": [1.0, 2.0, np.nan, 4.0],
    })
    tgt = pd.DataFrame({"pk": range(4), "y": [1.0, 2.0, 3.0, 4.0]})
    return df, snap, tgt


def _cfg(**kw):
    base = dict(
        key="pk", time_col="month_idx", ts_features=FEATS, target_cols=["y"],
        snapshot_cat=["seg"], snapshot_num=["score"],
    )
    base.update(kw)
    return DataConfig(**base)


def test_long_to_panel_shape_and_nan():
    df, snap, tgt = _tiny_long()
    cfg = _cfg(systematic=SystematicMissingSpec(method="none"))
    prep = prepare_datasets(df, snap, tgt, cfg, {"train": [0, 1], "holdout": [2], "oot": [3]})
    ds = prep.datasets["train"]
    assert ds.x.shape == (2, 3, 6)
    assert prep.seq_len == 6 and prep.n_features == 3


def test_tenure_systematic_mask():
    df, snap, tgt = _tiny_long()
    cfg = _cfg(systematic=SystematicMissingSpec(method="tenure", tenure_col="tenure"))
    prep = prepare_datasets(df, snap, tgt, cfg, {"train": [0, 1, 2], "holdout": [3], "oot": [3]})
    ds = prep.datasets["train"]
    row = list(ds.ids).index(1)
    sys = ds.sys_mask[row]
    assert sys[:, :4].all() and not sys[:, 4:].any()      # oldest 4 steps do not exist
    other = ds.sys_mask[list(ds.ids).index(0)]
    assert not other.any()


def test_boundary_nan_inference_matches_tenure():
    df, snap, tgt = _tiny_long()
    splits = {"train": [0, 1, 2], "holdout": [3], "oot": [3]}
    a = prepare_datasets(df, snap, tgt, _cfg(
        systematic=SystematicMissingSpec(method="tenure", tenure_col="tenure")), splits)
    b = prepare_datasets(df, snap, tgt, _cfg(
        systematic=SystematicMissingSpec(method="boundary_nan")), splits)
    assert torch.equal(a.datasets["train"].sys_mask, b.datasets["train"].sys_mask)


def test_random_hole_is_not_systematic():
    df, snap, tgt = _tiny_long()
    cfg = _cfg(systematic=SystematicMissingSpec(method="tenure", tenure_col="tenure"))
    prep = prepare_datasets(df, snap, tgt, cfg, {"train": [0, 1, 2], "holdout": [3], "oot": [3]})
    ds = prep.datasets["train"]
    row = list(ds.ids).index(2)
    assert not ds.sys_mask[row].any()                     # record 2 has full tenure
    assert not ds.obs_mask[row, 0, 3]                      # but the hole is unobserved
    assert ds.x[row, 0, 3] == 0.0                          # random missing -> zero fill


def test_newest_first_time_order():
    df, snap, tgt = _tiny_long()
    cfg = _cfg(systematic=SystematicMissingSpec(method="tenure", tenure_col="tenure"),
               time_order="newest_first")
    prep = prepare_datasets(df, snap, tgt, cfg, {"train": [0, 1, 2], "holdout": [3], "oot": [3]})
    ds = prep.datasets["train"]
    sys = ds.sys_mask[list(ds.ids).index(1)]
    assert sys[:, 2:].all() and not sys[:, :2].any()       # newest 2 steps exist


def test_explicit_mask():
    df, snap, tgt = _tiny_long()
    cfg = _cfg(systematic=SystematicMissingSpec(method="explicit"))
    mask = np.zeros((4, 6), dtype=bool)
    mask[3, :3] = True
    prep = prepare_datasets(df, snap, tgt, cfg,
                            {"train": [0, 1, 2], "holdout": [3], "oot": [3]}, sys_mask=mask)
    ds = prep.datasets["holdout"]
    assert ds.sys_mask[0][:, :3].all() and not ds.sys_mask[0][:, 3:].any()


def test_scalers_are_fitted_on_train_only():
    ts, snap, tgt, splits = make_synthetic_panel(n_records=300, n_features=3, seq_len=8, seed=3)
    cfg = DataConfig(key="account_id", time_col="month_idx", ts_features=["f0", "f1", "f2"],
                     target_cols=["y"], snapshot_num=["num0"], snapshot_cat=["cat0"],
                     systematic=SystematicMissingSpec(method="tenure", tenure_col="tenure_months"))
    prep = prepare_datasets(ts, snap, tgt, cfg, splits)
    tr = prep.datasets["train"]
    y_tr = prep.target_scaler.transform(
        tgt.set_index("account_id").loc[tr.ids, ["y"]].to_numpy())
    assert abs(y_tr.mean()) < 1e-9 and abs(y_tr.std() - 1) < 1e-6
    # holdout/oot are NOT re-centred
    assert abs(prep.datasets["oot"].y.mean().item()) > 1e-6


def test_target_inverse_transform_roundtrip():
    ts, snap, tgt, splits = make_synthetic_panel(n_records=200, n_features=3, seq_len=8, seed=4)
    cfg = DataConfig(key="account_id", time_col="month_idx", ts_features=["f0", "f1", "f2"],
                     target_cols=["y"])
    prep = prepare_datasets(ts, snap, tgt, cfg, splits)
    ds = prep.datasets["holdout"]
    back = prep.target_scaler.inverse_transform(ds.y.numpy())
    truth = tgt.set_index("account_id").loc[ds.ids, "y"].to_numpy()
    # tensors are stored as float32, so the round-trip is exact to ~1e-6 relative
    np.testing.assert_allclose(back.ravel(), truth, rtol=1e-5, atol=1e-4)


def test_unseen_category_maps_to_zero():
    df, snap, tgt = _tiny_long()
    cfg = _cfg(systematic=SystematicMissingSpec(method="none"))
    prep = prepare_datasets(df, snap, tgt, cfg, {"train": [0, 2], "holdout": [1], "oot": [3]})
    assert prep.cat_cardinalities == [2]                   # {"a"} + missing slot
    assert prep.datasets["holdout"].cat[0, 0].item() == 0   # "b" unseen in train
    assert prep.datasets["oot"].cat[0, 0].item() == 0       # None -> missing


def test_numeric_missing_indicator():
    df, snap, tgt = _tiny_long()
    cfg = _cfg(systematic=SystematicMissingSpec(method="none"))
    prep = prepare_datasets(df, snap, tgt, cfg, {"train": [0, 1], "holdout": [2], "oot": [3]})
    ds = prep.datasets["holdout"]                          # record 2 has score = NaN
    assert ds.num_missing[0, 0].item() == 1.0 and ds.num[0, 0].item() == 0.0


def test_missing_target_records_dropped():
    df, snap, tgt = _tiny_long()
    tgt.loc[tgt.pk == 3, "y"] = np.nan
    cfg = _cfg(systematic=SystematicMissingSpec(method="none"))
    with pytest.warns(UserWarning):
        prep = prepare_datasets(df, snap, tgt, cfg, {"train": [0, 1], "holdout": [2], "oot": [3]})
    assert len(prep.datasets["oot"]) == 0


def test_split_by_column():
    df, snap, tgt = _tiny_long()
    tgt["split"] = ["train", "train", "holdout", "oot"]
    cfg = _cfg(systematic=SystematicMissingSpec(method="none"))
    prep = prepare_datasets(df, snap, tgt, cfg, splits="split")
    assert {k: len(v) for k, v in prep.datasets.items()} == {"train": 2, "holdout": 1, "oot": 1}


def test_wide_input_path():
    n, t = 20, 6
    rng = np.random.default_rng(0)
    cols = {f: [f"{f}_{i}" for i in range(t)] for f in FEATS}
    wide = pd.DataFrame({"pk": range(n)})
    for f in FEATS:
        for c in cols[f]:
            wide[c] = rng.normal(size=n)
    assert set(infer_wide_columns(wide.columns)) == set(FEATS)
    ids, x, names = wide_to_panel(wide, "pk", cols)
    assert x.shape == (n, 3, t) and names == FEATS
    tgt = pd.DataFrame({"pk": range(n), "y": rng.normal(size=n)})
    cfg = DataConfig(key="pk", ts_features=FEATS, target_cols=["y"],
                     systematic=SystematicMissingSpec(method="none"))
    prep = prepare_datasets((ids, x), None, tgt, cfg,
                            {"train": range(12), "holdout": range(12, 16), "oot": range(16, 20)})
    assert prep.datasets["train"].x.shape == (12, 3, t)


def test_duplicate_and_unknown_keys_warn():
    df, snap, tgt = _tiny_long()
    cfg = _cfg(systematic=SystematicMissingSpec(method="none"))
    with pytest.warns(UserWarning):
        prepare_datasets(df, snap, tgt, cfg,
                         {"train": [0, 1], "holdout": [2], "oot": [3, 999]})


def test_model_config_matches_prepared_dims():
    ts, snap, tgt, splits = make_synthetic_panel(n_records=200, n_features=4, seq_len=12, seed=5)
    cfg = DataConfig(key="account_id", time_col="month_idx",
                     ts_features=[f"f{i}" for i in range(4)], target_cols=["y"],
                     snapshot_cat=["cat0", "cat1"], snapshot_num=["num0", "num1"],
                     systematic=SystematicMissingSpec(method="tenure", tenure_col="tenure_months"))
    prep = prepare_datasets(ts, snap, tgt, cfg, splits)
    mc = prep.model_config()
    assert mc.tokenizer.in_channels == 4
    assert mc.seq_len == 12
    assert mc.snapshot.n_numeric == 2 and len(mc.snapshot.cat_cardinalities) == 2
    assert mc.head.n_targets == 1


def test_missing_snapshot_frame_raises():
    df, snap, tgt = _tiny_long()
    with pytest.raises(ValueError, match="snapshot"):
        prepare_datasets(df, None, tgt, _cfg(), {"train": [0, 1], "holdout": [2], "oot": [3]})


def test_missing_snapshot_column_raises():
    df, snap, tgt = _tiny_long()
    with pytest.raises(KeyError, match="nope"):
        prepare_datasets(df, snap, tgt, _cfg(snapshot_num=["nope"]),
                         {"train": [0, 1], "holdout": [2], "oot": [3]})


def test_missing_target_column_raises():
    df, snap, tgt = _tiny_long()
    with pytest.raises(KeyError, match="ynot"):
        prepare_datasets(df, snap, tgt, _cfg(target_cols=["ynot"]),
                         {"train": [0, 1], "holdout": [2], "oot": [3]})


def test_empty_train_split_raises():
    df, snap, tgt = _tiny_long()
    with pytest.raises(ValueError, match="train split"):
        prepare_datasets(df, snap, tgt, _cfg(systematic=SystematicMissingSpec(method="none")),
                         {"holdout": [2], "oot": [3]})


def test_tenure_method_without_column_raises():
    df, snap, tgt = _tiny_long()
    with pytest.raises((ValueError, KeyError)):
        prepare_datasets(df, snap, tgt,
                         _cfg(systematic=SystematicMissingSpec(method="tenure")),
                         {"train": [0, 1], "holdout": [2], "oot": [3]})


def test_pandas_categorical_dtype_is_supported():
    df, snap, tgt = _tiny_long()
    snap["seg"] = snap["seg"].astype("category")
    cfg = _cfg(systematic=SystematicMissingSpec(method="none"))
    prep = prepare_datasets(df, snap, tgt, cfg, {"train": [0, 2], "holdout": [1], "oot": [3]})
    assert prep.datasets["train"].cat.dtype == torch.int64
    assert prep.datasets["oot"].cat[0, 0].item() == 0          # NaN category -> missing slot


def test_nan_split_label_is_excluded():
    df, snap, tgt = _tiny_long()
    tgt["split"] = ["train", "train", "holdout", None]
    cfg = _cfg(systematic=SystematicMissingSpec(method="none"))
    with pytest.warns(UserWarning, match="no 'split' label"):
        prep = prepare_datasets(df, snap, tgt, cfg, splits="split")
    assert set(prep.datasets) == {"train", "holdout"}


def test_panel_is_float32_not_float64():
    ts, snap, tgt, splits = make_synthetic_panel(n_records=100, n_features=3, seq_len=8, seed=9)
    cfg = DataConfig(key="account_id", time_col="month_idx", ts_features=["f0", "f1", "f2"],
                     target_cols=["y"])
    prep = prepare_datasets(ts, snap, tgt, cfg, splits)
    ds = prep.datasets["train"]
    assert ds.x.dtype == torch.float32 and ds.obs_mask.dtype == torch.bool


def test_scaling_is_correct_on_observed_cells():
    """Scaled train values must be ~N(0,1) over the observed cells only."""
    ts, snap, tgt, splits = make_synthetic_panel(n_records=1500, n_features=3, seq_len=12, seed=10)
    cfg = DataConfig(key="account_id", time_col="month_idx", ts_features=["f0", "f1", "f2"],
                     target_cols=["y"], clip_sigma=None,
                     systematic=SystematicMissingSpec(method="tenure", tenure_col="tenure_months"),
                     snapshot_num=["tenure_months"])
    prep = prepare_datasets(ts, snap, tgt, cfg, splits)
    ds = prep.datasets["train"]
    for j in range(3):
        vals = ds.x[:, j][ds.obs_mask[:, j]].numpy()
        assert abs(vals.mean()) < 0.02 and abs(vals.std() - 1.0) < 0.02
