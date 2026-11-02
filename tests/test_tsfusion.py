import math
import os

import numpy as np
import pandas as pd
import pytest
import torch

from tsfusion import (ModelConfig, Preprocessor, RawData, Trainer, TrainConfig, TSFusionTransformer,
                      TweedieLoss, build_state, default_target_mode, encode_special_values,
                      group_columns_by_prefix, load_checkpoint, long_df_to_array, predict_with,
                      regression_metrics)
from tsfusion.model import SlidingPatcher
from tsfusion.synthetic import make_synthetic_dataset


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def synth():
    return make_synthetic_dataset(n_records=400, n_vars=5, seed=1)


def _raw(d, ids):
    x, _ = long_df_to_array(d["ts_df"], record_ids=ids)
    m, _ = long_df_to_array(d["missing_df"], record_ids=ids, fill_value=1)
    s, _ = long_df_to_array(d["special_df"], record_ids=ids, fill_value=0)
    cols = [c for _, cs in group_columns_by_prefix(d["snapshot_ple_df"].columns) for c in cs]
    snap = d["snapshot_ple_df"].loc[ids, cols].to_numpy()
    return RawData(x, m, s.astype(np.int64), snap, d["y"].loc[ids].to_numpy())


@pytest.fixture(scope="module")
def raw_splits(synth):
    return _raw(synth, synth["train_ids"]), _raw(synth, synth["val_ids"])


def _group_sizes(synth):
    return [len(c) for _, c in group_columns_by_prefix(synth["snapshot_ple_df"].columns)]


def _model_cfg(synth, **kw):
    base = dict(n_vars=len(synth["var_names"]), n_steps=11,
                n_special_codes=synth["n_special_codes"],
                snapshot_group_sizes=_group_sizes(synth), d_model=32, n_heads=4,
                n_ts_layers=1, head_hidden=32)
    base.update(kw)
    return ModelConfig(**base)


# ----------------------------------------------------------------------------- data
def test_build_state_codes_and_precedence():
    missing = np.array([[[0, 1, 1, 0]]])
    special = np.array([[[0, 0, 2, 1]]])
    with pytest.warns(UserWarning):
        st = build_state(missing, special)
    assert st.tolist() == [[[0, 1, 3, 2]]]          # special wins over missing


def test_encode_special_values():
    x = np.array([1.0, -999.0, -998.0, 5.0])
    codes = encode_special_values(x, {-999: 1, -998: 2})
    assert codes.tolist() == [0, 1, 2, 0]


def test_long_df_roundtrip(synth):
    ids = synth["train_ids"][:7]
    x, out_ids = long_df_to_array(synth["ts_df"], record_ids=ids)
    assert x.shape == (7, 11, 5)
    np.testing.assert_allclose(x[3, 4], synth["ts_df"].loc[(ids[3], 5)].to_numpy(), rtol=1e-6)
    assert (out_ids == ids).all()


def test_group_columns_by_prefix():
    cols = ["age_bin0", "age_bin1", "income_bin0", "income_bin1", "income_bin2", "flag"]
    g = group_columns_by_prefix(cols)
    assert [(n, len(c)) for n, c in g] == [("age", 2), ("income", 3), ("flag", 1)]


def test_preprocessor_ignores_sentinels_and_zeroes_nonregular(raw_splits):
    train, _ = raw_splits
    prep = Preprocessor(3, [8, 10, 6, 10, 8], target_mode="standard").fit(train)
    state = build_state(train.missing, train.special)
    reg = state == 0
    for v in range(train.x.shape[2]):
        np.testing.assert_allclose(prep.ts_mean[v], train.x[:, :, v][reg[:, :, v]].mean(),
                                   rtol=1e-4)
    p = prep.transform(train)
    assert torch.all(p.x[torch.from_numpy(~reg)] == 0)
    assert p.state.dtype == torch.uint8 and int(p.state.max()) <= 4
    assert p.snapshot.shape == (len(train), 5, 10)
    # round trip of the target scaler
    np.testing.assert_allclose(prep.target_scaler.inverse_transform(p.y.numpy()), train.y,
                               rtol=1e-4)
    # persistence
    prep2 = Preprocessor.from_state_dict(prep.state_dict())
    np.testing.assert_allclose(prep2.transform(train).x.numpy(), p.x.numpy())


@pytest.mark.parametrize("T,P,S,expected", [(11, 3, 1, 9), (11, 4, 2, 5), (11, 11, 1, 1),
                                            (11, 1, 1, 11), (11, 5, 3, 3),
                                            (11, 4, 3, 4)])
def test_sliding_patcher(T, P, S, expected):
    patcher = SlidingPatcher(T, P, S)
    x = torch.arange(T, dtype=torch.float32).view(1, T, 1).repeat(2, 1, 3)
    out = patcher(x)
    assert out.shape == (2, 3, expected, P)
    assert patcher.n_patches == expected
    assert out[0, 0, -1, -1] == T - 1          # most recent step always covered
    assert out[0, 0, 0].tolist() == list(range(P))


# ----------------------------------------------------------------------------- model
@pytest.mark.parametrize("fusion", ["late", "early", "hybrid"])
def test_forward_shapes_and_no_nan(synth, raw_splits, fusion):
    train, _ = raw_splits
    cfg = _model_cfg(synth, missing_fusion=fusion)
    prep = Preprocessor(3, cfg.snapshot_group_sizes).fit(train)
    d = prep.transform(train)
    # add one fully-missing sample and one fully-regular sample
    d.state[0] = 1
    d.x[0] = 0
    d.state[1] = 0
    model = TSFusionTransformer(cfg).eval()
    with torch.no_grad():
        out = model(d.x[:16], d.state[:16], d.snapshot[:16])
        emb = model.embed(d.x[:16], d.state[:16], d.snapshot[:16])
    assert out.shape == (16,)
    assert torch.isfinite(out).all()
    expected = {"late": 3, "early": 2, "hybrid": 3}[fusion]
    assert len(emb) == expected and model.n_cls == expected
    loss = model(d.x[:16], d.state[:16], d.snapshot[:16]).pow(2).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_special_branch_constant_without_missing(synth, raw_splits):
    """With mask_regular_patches, samples without any missing/special value all get the
    same special-branch CLS (it only attends to CLS tokens) - 'absence' is one embedding."""
    train, _ = raw_splits
    cfg = _model_cfg(synth)
    d = Preprocessor(3, cfg.snapshot_group_sizes).fit(train).transform(train)
    state = torch.zeros_like(d.state[:4])
    model = TSFusionTransformer(cfg).eval()
    with torch.no_grad():
        cls_sp = model.embed(d.x[:4], state, d.snapshot[:4])["cls_special"]
    assert torch.allclose(cls_sp, cls_sp[:1].expand_as(cls_sp), atol=1e-5)


def test_late_mode_ignores_values_in_fully_missing_patches(synth, raw_splits):
    train, _ = raw_splits
    cfg = _model_cfg(synth, patch_len=1, patch_stride=1)
    d = Preprocessor(3, cfg.snapshot_group_sizes).fit(train).transform(train)
    x, st = d.x[:4].clone(), d.state[:4].clone()
    st[:, 5, 2] = 1                      # one missing cell -> with P=1 its patch is empty
    model = TSFusionTransformer(cfg).eval()
    with torch.no_grad():
        a = model(x, st, d.snapshot[:4])
        x[:, 5, 2] = 123.0               # garbage in the masked cell must not matter
        b = model(x, st, d.snapshot[:4])
    assert torch.allclose(a, b, atol=1e-5)


# ----------------------------------------------------------------------------- losses
@pytest.mark.parametrize("p", [1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0])
def test_tweedie_minimiser_is_mean(p):
    y = torch.tensor([0.5, 1.0, 2.0, 10.0])
    f = torch.tensor([math.log(y.mean().item())] * 4, requires_grad=True)
    loss = TweedieLoss(p)(f, y)
    loss.backward()
    assert abs(f.grad.sum().item()) < 1e-5     # gradient wrt a shared constant is 0 at mean


def test_tweedie_gradient_continuous_at_special_powers():
    y = torch.tensor([0.3, 2.0, 7.0])
    grads = []
    for p in (2.0 - 1e-4, 2.0, 2.0 + 1e-4, 1.0, 1.0 + 1e-4):
        f = torch.tensor([0.1, -0.2, 0.5], requires_grad=True)
        TweedieLoss(p)(f, y).backward()
        grads.append(f.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-3)
    assert torch.allclose(grads[1], grads[2], atol=1e-3)
    assert torch.allclose(grads[3], grads[4], atol=1e-3)


def test_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1 and m["nmad"] == 0
    m = regression_metrics(y, y + np.array([1, -1, 1, -1]))
    assert m["rmse"] == pytest.approx(1.0) and m["mae"] == pytest.approx(1.0)
    assert m["nmad"] == pytest.approx(1.4826)
    assert math.isnan(regression_metrics(y, np.array([1, np.nan, 1, 1]))["rmse"])


# ----------------------------------------------------------------------------- trainer
@pytest.mark.parametrize("loss", ["mse", "tweedie"])
def test_trainer_end_to_end(tmp_path, synth, raw_splits, loss):
    train, val = raw_splits
    mcfg = _model_cfg(synth)
    tcfg = TrainConfig(loss=loss, tweedie_power=1.7, batch_size=64, max_epochs=3, lr=1e-3,
                       warmup_steps=5, device="cpu", output_dir=str(tmp_path / loss),
                       early_stopping_patience=5, verbose=False)
    prep = Preprocessor(3, mcfg.snapshot_group_sizes, target_mode=default_target_mode(loss))
    prep.fit(train)
    tr, va = prep.transform(train), prep.transform(val)
    trainer = Trainer(mcfg, tcfg, prep)
    hist = trainer.fit(tr, va)
    assert len(hist) == 3
    for col in ["train_rmse", "train_mae", "train_r2", "train_nmad",
                "val_rmse", "val_mae", "val_r2", "val_nmad", "train_loss", "val_loss"]:
        assert col in hist.columns and hist[col].notna().all()
    files = set(os.listdir(tcfg.output_dir))
    assert {"epoch_001.pt", "epoch_002.pt", "epoch_003.pt", "best_model.pt", "metrics.csv",
            "train.log", "config.json"} <= files
    assert len(pd.read_csv(os.path.join(tcfg.output_dir, "metrics.csv"))) == 3

    # the trainer ends with the best weights; a reloaded best checkpoint gives the same preds
    best_epoch = int(hist.loc[hist["val_rmse"].idxmin(), "epoch"])
    model, prep2, tcfg2, ckpt = load_checkpoint(os.path.join(tcfg.output_dir, "best_model.pt"))
    assert ckpt["epoch"] == best_epoch
    p1 = trainer.predict(va)
    p2 = predict_with(model, prep2, tcfg2, prep2.transform(val))
    np.testing.assert_allclose(p1, p2, rtol=1e-4, atol=1e-6)
    assert regression_metrics(val.y, p1)["rmse"] == pytest.approx(hist["val_rmse"].min(),
                                                                   rel=1e-5)


def test_early_stopping_and_resume(tmp_path, synth, raw_splits):
    train, val = raw_splits
    mcfg = _model_cfg(synth)
    out = str(tmp_path / "es")
    tcfg = TrainConfig(batch_size=128, max_epochs=10, device="cpu", output_dir=out,
                       early_stopping_patience=2, early_stopping_min_delta=1e12,
                       warmup_steps=0, verbose=False)
    prep = Preprocessor(3, mcfg.snapshot_group_sizes).fit(train)
    tr, va = prep.transform(train), prep.transform(val)
    hist = Trainer(mcfg, tcfg, prep).fit(tr, va)
    assert len(hist) == 3                    # epoch 1 best, then 2 bad epochs -> stop

    tcfg2 = TrainConfig(**{**tcfg.to_dict(), "max_epochs": 5, "early_stopping_patience": 100,
                           "early_stopping_min_delta": 0.0})
    t2 = Trainer(mcfg, tcfg2, prep)
    start = t2.resume(os.path.join(out, "epoch_002.pt"))
    assert start == 3
    hist2 = t2.fit(tr, va, start_epoch=start)
    assert hist2["epoch"].tolist() == [1, 2, 3, 4, 5]
    assert pd.read_csv(os.path.join(out, "metrics.csv"))["epoch"].tolist() == [1, 2, 3, 4, 5]


def test_can_overfit_small_batch(synth, raw_splits):
    train, _ = raw_splits
    mcfg = _model_cfg(synth, dropout=0.0, head_dropout=0.0)
    prep = Preprocessor(3, mcfg.snapshot_group_sizes).fit(train)
    d = prep.transform(train)
    torch.manual_seed(0)
    model = TSFusionTransformer(mcfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    xb, sb, nb, yb = d.x[:32], d.state[:32], d.snapshot[:32], d.y[:32]
    first = None
    for _ in range(150):
        loss = torch.mean((model(xb, sb, nb) - yb) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1 * first


# ----------------------------------------------------------------------------- review fixes
def test_long_df_requires_chronological_time_labels():
    idx = pd.MultiIndex.from_product([[1, 2], [f"t{i}" for i in range(1, 12)]],
                                     names=["record_id", "time_step"])
    df = pd.DataFrame({"a": np.arange(22.0)}, index=idx)
    with pytest.raises(ValueError, match="time_steps"):
        long_df_to_array(df)
    x, _ = long_df_to_array(df, time_steps=[f"t{i}" for i in range(1, 12)])
    assert x[0, :, 0].tolist() == list(range(11))           # chronological, not t1, t10, t11, t2
    with pytest.raises(ValueError, match="dtype"):
        long_df_to_array(df, record_ids=["1", "2"], time_steps=[f"t{i}" for i in range(1, 12)])


def test_group_columns_interleaved_warns_and_raw_columns_kept():
    with pytest.warns(UserWarning, match="interleaved"):
        g = group_columns_by_prefix(["a_bin0", "b_bin0", "a_bin1", "b_bin1"])
    assert g == [("a", ["a_bin0", "a_bin1"]), ("b", ["b_bin0", "b_bin1"])]
    assert group_columns_by_prefix(["x_2023", "x_2024"]) == [("x_2023", ["x_2023"]),
                                                            ("x_2024", ["x_2024"])]


def test_sentinel_matching_is_exact():
    codes = encode_special_values(np.array([-999.0, -999.005, 99999.5, 99999.0]),
                                  {-999: 1, 99999: 2})
    assert codes.tolist() == [1, 0, 0, 2]


def test_stride_longer_than_patch_rejected():
    with pytest.raises(ValueError, match="skip"):
        ModelConfig(n_vars=3, patch_len=2, patch_stride=3)


def test_head_is_fp32_under_autocast(synth, raw_splits):
    train, _ = raw_splits
    cfg = _model_cfg(synth)
    d = Preprocessor(3, cfg.snapshot_group_sizes).fit(train).transform(train)
    model = TSFusionTransformer(cfg).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(d.x[:8], d.state[:8], d.snapshot[:8])
    assert out.dtype == torch.float32


def test_weight_decay_groups():
    from tsfusion.trainer import _param_groups
    model = TSFusionTransformer(ModelConfig(n_vars=3, n_special_codes=2, snapshot_group_sizes=[4]))
    groups = _param_groups(model, 0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    decay = {names[id(p)] for p in groups[0]["params"]}
    no_decay = {names[id(p)] for p in groups[1]["params"]}
    assert "value_embed.proj.weight" in decay and "state_embed.proj.weight" in decay
    assert "state_embed.state_emb.weight" in no_decay and "ts_encoder.var_emb" in no_decay
    assert "ts_encoder.cls_global" in no_decay and "head.net.1.weight" in decay


def test_cosine_resume_restores_lr_and_fresh_run_ignores_stale_best(tmp_path, synth, raw_splits):
    train, val = raw_splits
    mcfg = _model_cfg(synth)
    out = str(tmp_path / "cos")
    tcfg = TrainConfig(batch_size=128, max_epochs=2, device="cpu", output_dir=out,
                       scheduler="cosine", warmup_steps=2, verbose=False)
    prep = Preprocessor(3, mcfg.snapshot_group_sizes).fit(train)
    tr, va = prep.transform(train), prep.transform(val)
    t1 = Trainer(mcfg, tcfg, prep)
    t1.fit(tr, va)
    lr_end = t1.optimizer.param_groups[0]["lr"]
    t2 = Trainer(mcfg, TrainConfig(**{**tcfg.to_dict(), "max_epochs": 3}), prep)
    start = t2.resume(os.path.join(out, "epoch_002.pt"))
    t2.train_one_epoch = lambda loader, epoch: (                   # inspect lr of the 1st step
        setattr(t2, "_lr_at_resume", t2.optimizer.param_groups[0]["lr"]) or 0.0)
    t2.fit(tr, va, start_epoch=start)
    assert t2._lr_at_resume == pytest.approx(lr_end)

    # a fresh run (different architecture) in the same dir clears the old run's files
    mcfg_small = _model_cfg(synth, d_model=16, n_heads=2)
    t3 = Trainer(mcfg_small, TrainConfig(**{**tcfg.to_dict(), "max_epochs": 1}), prep)
    t3.fit(tr, va)
    assert sorted(f for f in os.listdir(out) if f.endswith(".pt")) == ["best_model.pt",
                                                                      "epoch_001.pt"]
    _, _, _, ck = load_checkpoint(os.path.join(out, "best_model.pt"))
    assert ck["model_config"]["d_model"] == 16
    with pytest.raises(FileExistsError):
        Trainer(mcfg, TrainConfig(**{**tcfg.to_dict(), "overwrite_output_dir": False}),
                prep).fit(tr, va)


def test_preprocessor_log_vars_roundtrip(raw_splits):
    train, _ = raw_splits
    prep = Preprocessor(3, [8, 10, 6, 10, 8], ts_log_vars=[0, 2]).fit(train)
    a = prep.transform(train).x
    b = Preprocessor.from_state_dict(prep.state_dict()).transform(train).x
    assert torch.equal(a, b) and prep.ts_log_vars == [0, 2]
