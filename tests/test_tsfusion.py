import math
import os

import numpy as np
import pandas as pd
import pytest
import torch

from tsfusion import (ModelConfig, Preprocessor, Trainer, TrainConfig, TSFusionTransformer,
                      TweedieLoss, build_state, default_target_mode, encode_special_values,
                      group_columns_by_prefix, load_checkpoint, long_df_to_array, predict_with,
                      regression_metrics)
from tsfusion.model import SlidingPatcher


from helpers import _model_cfg  # noqa: E402  (fixtures synth / raw_splits live in conftest.py)


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


# ----------------------------------------------------------------------------- memory / streaming
def test_stream_batches_match_in_memory_batches(raw_splits):
    from tsfusion import InMemoryLoader
    from tsfusion.data import _same_device, stream_batches
    train, _ = raw_splits
    d = Preprocessor(3, [8, 10, 6, 10, 8]).fit(train).transform(train)
    idx = torch.randperm(len(d), generator=torch.Generator().manual_seed(0))
    direct = [d.batch(idx[s:s + 37]) for s in range(0, len(d), 37)]
    streamed = list(stream_batches(d, idx, 37, torch.device("cpu")))
    assert len(direct) == len(streamed)
    for a, b in zip(direct, streamed):
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert d.nbytes() == sum(t.numel() * t.element_size() for t in (d.x, d.state, d.snapshot, d.y))
    loader = InMemoryLoader(d, 64, device="cpu")
    assert sum(len(b["y"]) for b in loader) == len(d)
    assert _same_device(torch.device("cuda"), torch.device("cuda:1"))
    assert not _same_device(torch.device("cuda:0"), torch.device("cuda:1"))
    assert not _same_device(torch.device("cpu"), torch.device("cuda:0"))


def test_auto_data_placement_rule():
    from tsfusion.trainer import fits_on_gpu
    gib = 2 ** 30
    assert not fits_on_gpu(50 * gib, 39 * gib, 0.3)   # the reported OOM case -> stream batches
    assert fits_on_gpu(5 * gib, 39 * gib, 0.3)
    with pytest.raises(ValueError):
        TrainConfig(data_on_device="yes")
    assert TrainConfig().data_on_device == "auto"


def test_nonfinite_loss_still_raises(tmp_path, synth, raw_splits):
    train, val = raw_splits
    mcfg = _model_cfg(synth)
    prep = Preprocessor(3, mcfg.snapshot_group_sizes).fit(train)
    t = Trainer(mcfg, TrainConfig(batch_size=64, max_epochs=1, device="cpu", verbose=False,
                                  output_dir=str(tmp_path / "nan")), prep)
    with torch.no_grad():
        t.model.head.net[-1].bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        t.fit(prep.transform(train), prep.transform(val))


# ----------------------------------------------------------------------------- combined state matrix
def _combined(raw):
    from tsfusion import RawData
    return RawData(raw.x.copy(), snapshot=raw.snapshot, y=raw.y,
                   state=build_state(raw.missing, raw.special))


def test_combined_state_matches_separate_inputs(raw_splits):
    train, val = raw_splits
    comb_tr, comb_va = _combined(train), _combined(val)
    assert comb_tr.state.dtype == np.uint8 and comb_tr.combined
    p_sep = Preprocessor(3, [8, 10, 6, 10, 8]).fit(train)
    p_cmb = Preprocessor(3, [8, 10, 6, 10, 8]).fit(comb_tr)
    np.testing.assert_allclose(p_sep.ts_mean, p_cmb.ts_mean, rtol=1e-6)
    np.testing.assert_allclose(p_sep.ts_std, p_cmb.ts_std, rtol=1e-6)
    a, b = p_sep.transform(val), p_cmb.transform(comb_va)
    assert torch.equal(a.state, b.state) and torch.allclose(a.x, b.x, atol=1e-6)


@pytest.mark.parametrize("chunk", [1, 7, 50, None])
def test_chunked_fit_and_transform_are_exact(raw_splits, chunk):
    train, _ = raw_splits
    comb = _combined(train)
    st = comb.state
    reg = st == 0
    ref_mean = np.array([train.x[:, :, v][reg[:, :, v]].astype(np.float64).mean() for v in range(5)])
    ref_std = np.array([train.x[:, :, v][reg[:, :, v]].astype(np.float64).std() for v in range(5)])
    p = Preprocessor(3, [8, 10, 6, 10, 8]).fit(comb, chunk_size=chunk)
    np.testing.assert_allclose(p.ts_mean, ref_mean, rtol=1e-5)
    np.testing.assert_allclose(p.ts_std, ref_std, rtol=1e-5)
    full = p.transform(comb, chunk_size=len(comb))
    part = p.transform(comb, chunk_size=chunk)
    assert torch.equal(full.x, part.x) and torch.equal(full.state, part.state)


def test_inplace_transform_reuses_input_buffers(raw_splits):
    train, _ = raw_splits
    p = Preprocessor(3, [8, 10, 6, 10, 8]).fit(_combined(train))
    ref = p.transform(_combined(train))
    comb = _combined(train)
    out = p.transform(comb, inplace=True, chunk_size=13)
    assert np.shares_memory(out.x.numpy(), comb.x) and np.shares_memory(out.state.numpy(), comb.state)
    assert torch.equal(out.x, ref.x) and torch.equal(out.state, ref.state)
    with pytest.raises(ValueError, match="already transformed"):
        p.transform(comb)


def test_nan_becomes_missing_and_code_validation(raw_splits):
    from tsfusion import RawData
    train, _ = raw_splits
    comb = _combined(train)
    reg = np.argwhere(comb.state == 0)[0]
    spec = np.argwhere(comb.state >= 2)[0]
    comb.x[tuple(reg)] = np.nan
    comb.x[tuple(spec)] = np.nan
    p = Preprocessor(3, [8, 10, 6, 10, 8]).fit(comb)
    out = p.transform(comb)
    assert out.state[tuple(reg)] == 1 and out.state[tuple(spec)] == comb.state[tuple(spec)]
    assert out.x[tuple(reg)] == 0
    bad = _combined(train)
    bad.state[0, 0, 0] = 9                                   # > K+1 = 4
    with pytest.raises(ValueError, match="codes must be"):
        Preprocessor(3, [8, 10, 6, 10, 8]).fit(bad)
    with pytest.raises(TypeError, match="integer"):
        RawData(train.x, state=comb.state.astype(np.float32))
    with pytest.raises(ValueError, match="not both"):
        RawData(train.x, train.missing, train.special, state=comb.state)


def test_chunked_inplace_memory_is_small():
    import tracemalloc
    from tsfusion import RawData
    rng = np.random.default_rng(0)
    n, t, v = 4000, 11, 60
    x = rng.normal(size=(n, t, v)).astype(np.float32)
    st = np.where(rng.random((n, t, v)) < 0.1, rng.integers(1, 5, (n, t, v)), 0).astype(np.uint8)
    raw = RawData(x, state=st, y=rng.gamma(2, 1, n))
    p = Preprocessor(3, target_mode="mean_scale").fit(raw, chunk_size=200)
    tracemalloc.start()
    p.transform(RawData(x.copy(), state=st.copy(), y=raw.y), chunk_size=200, inplace=True)
    tracemalloc.stop()      # the two input copies above are made before tracing starts? no:
    tracemalloc.start()
    raw2 = RawData(x.copy(), state=st.copy(), y=raw.y)
    base, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    p.transform(raw2, chunk_size=200, inplace=True)
    _, peak_inplace = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    p.transform(RawData(x, state=st, y=raw.y), chunk_size=200)
    _, peak_copy = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cells = n * t * v
    assert (peak_inplace - base) / cells < 1.0               # only chunk-sized temporaries
    assert (peak_copy - base) / cells < 5.0 + 1.0            # new x (4 B) + state (1 B) + chunk


def test_save_load_combined_with_mmap(tmp_path, raw_splits):
    from tsfusion import load_dataset, save_dataset
    train, val = raw_splits
    save_dataset(str(tmp_path), {"train": _combined(train), "val": _combined(val)}, 3, [8, 10, 6, 10, 8])
    splits, meta = load_dataset(str(tmp_path), mmap=True)
    tr = splits["train"]
    for a in (tr.x, tr.state):          # views onto the files on disk, not copies in RAM
        assert not a.flags.owndata and not a.flags.writeable
    assert tr.combined
    p = Preprocessor(3, meta["snapshot_group_sizes"]).fit(splits["train"])
    out = p.transform(splits["val"])
    ref = Preprocessor(3, [8, 10, 6, 10, 8]).fit(_combined(train)).transform(_combined(val))
    assert torch.allclose(out.x, ref.x, atol=1e-6) and torch.equal(out.state, ref.state)
    with pytest.raises(ValueError, match="writable"):
        p.transform(splits["val"], inplace=True)


def test_training_with_combined_input(tmp_path, synth, raw_splits):
    train, val = raw_splits
    mcfg = _model_cfg(synth, missing_fusion="early")
    prep = Preprocessor(3, mcfg.snapshot_group_sizes).fit(_combined(train))
    hist = Trainer(mcfg, TrainConfig(batch_size=64, max_epochs=2, device="cpu", verbose=False,
                                     output_dir=str(tmp_path / "comb")), prep).fit(
        prep.transform(_combined(train)), prep.transform(_combined(val)))
    assert len(hist) == 2 and hist["val_rmse"].notna().all()


# ----------------------------------------------------------------------------- many sequences
@pytest.mark.parametrize("fusion", ["early", "late"])
def test_attention_split_into_slices_matches_single_call(synth, raw_splits, fusion, monkeypatch):
    """Attention across time runs batch x (V+1) sequences (118k for 256 x 461), more than
    CUDA's 65,535 grid limit; it is split into slices - results must be identical."""
    from tsfusion.model import SelfAttention
    train, _ = raw_splits
    cfg = _model_cfg(synth, missing_fusion=fusion, dropout=0.0, head_dropout=0.0)
    d = Preprocessor(3, cfg.snapshot_group_sizes).fit(train).transform(train)
    torch.manual_seed(0)
    model = TSFusionTransformer(cfg)
    b = slice(0, 12)
    out_full = model(d.x[b], d.state[b], d.snapshot[b])
    g_full = torch.autograd.grad(out_full.sum(), list(model.parameters()), allow_unused=True)
    monkeypatch.setattr(SelfAttention, "max_batch", 7)       # force many slices
    out_split = model(d.x[b], d.state[b], d.snapshot[b])
    g_split = torch.autograd.grad(out_split.sum(), list(model.parameters()), allow_unused=True)
    assert torch.allclose(out_full, out_split, atol=1e-6)
    for a, c in zip(g_full, g_split):
        if a is not None:
            assert torch.allclose(a, c, atol=1e-5)
    from tsfusion.interpret import run_with_attention
    _, _, _, maps = run_with_attention(model, d.x[b], d.state[b], d.snapshot[b])
    assert maps.time[0].shape[0] == 12 and torch.allclose(maps.time[0].sum(-1), torch.ones(()), atol=1e-5)
