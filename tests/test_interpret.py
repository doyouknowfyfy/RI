import numpy as np
import pytest
import torch

from tsfusion import Preprocessor, TrainConfig, TSFusionTransformer, default_target_mode
from tsfusion.interpret import (cell_embedding_map, cls_attention_summary, cls_rollout,
                                collect_interpretation, modality_ablation, patch_state_category,
                                permutation_importance, relevance_by_category,
                                run_with_attention, snapshot_rollout, state_probe, state_similarity)
from tsfusion.model import SelfAttention

from helpers import _model_cfg


@pytest.fixture(scope="module")
def setup(synth, raw_splits):
    train, val = raw_splits
    torch.manual_seed(0)
    cfg = _model_cfg(synth, missing_fusion="early", n_ts_layers=2)
    prep = Preprocessor(3, cfg.snapshot_group_sizes, target_mode=default_target_mode("tweedie")).fit(train)
    model = TSFusionTransformer(cfg).eval()
    return model, prep, prep.transform(val), TrainConfig(loss="tweedie", tweedie_power=1.5)


@pytest.mark.parametrize("fusion", ["early", "late"])
def test_recorded_attention_matches_fast_path(synth, raw_splits, fusion):
    train, _ = raw_splits
    cfg = _model_cfg(synth, missing_fusion=fusion)
    d = Preprocessor(3, cfg.snapshot_group_sizes).fit(train).transform(train)
    model = TSFusionTransformer(cfg).eval()
    with torch.no_grad():
        fast = model(d.x[:16], d.state[:16], d.snapshot[:16])
        out, _, _, maps = run_with_attention(model, d.x[:16], d.state[:16], d.snapshot[:16])
    assert torch.allclose(fast, out.view(-1), atol=1e-5)
    R, C = cfg.n_vars + 1, model.n_patches + 1
    assert maps.var[0].shape == (16, C, cfg.n_heads, R, R)
    assert maps.time[0].shape == (16, R, cfg.n_heads, C, C)
    for a in maps.var + maps.time + maps.snapshot:
        assert torch.allclose(a.sum(-1), torch.ones(()), atol=1e-5)
    assert all(not m.store_attn and m.attn_weights is None
               for m in model.modules() if isinstance(m, SelfAttention))
    if fusion == "late":                                   # masked patches get no attention
        empty = ~(model.patcher(d.state[:16].long()) == 0).any(-1)   # (B, V, Np)
        if empty.any():
            b, v, n = [int(i) for i in torch.nonzero(empty)[0]]
            assert torch.all(maps.time[0][b, v + 1, :, :, n + 1] == 0)
        assert len(maps.special_var) == cfg.n_special_layers


def test_rollout_matches_brute_force(setup):
    model, _, d, _ = setup
    _, _, _, maps = run_with_attention(model, d.x[:3], d.state[:3], d.snapshot[:3])
    roll = cls_rollout(maps.var, maps.time)
    B, R, C = roll.shape
    assert torch.allclose(roll.sum((1, 2)), torch.ones(B), atol=1e-5)
    for b in range(B):
        J = torch.eye(R * C)
        for av, at in zip(maps.var, maps.time):
            av_, at_ = 0.5 * torch.eye(R) + 0.5 * av[b].mean(1), 0.5 * torch.eye(C) + 0.5 * at[b].mean(1)
            Mv, Mt = torch.zeros(R * C, R * C), torch.zeros(R * C, R * C)
            for r in range(R):
                for c in range(C):
                    for r2 in range(R):
                        Mv[r * C + c, r2 * C + c] = av_[c, r, r2]
                    for c2 in range(C):
                        Mt[r * C + c, r * C + c2] = at_[r, c, c2]
            J = Mt @ Mv @ J                       # variables first, then time
        assert torch.allclose(roll[b].reshape(-1), J[0], atol=1e-5)


def test_summaries_and_categories(setup):
    model, _, d, _ = setup
    _, _, det, maps = run_with_attention(model, d.x[:8], d.state[:8], d.snapshot[:8])
    s = cls_attention_summary(maps)
    V, Np = model.cfg.n_vars, model.n_patches
    assert s["global_to_var"].shape == (8, V) and s["timecls_to_var"].shape == (8, V, Np)
    assert torch.all(s["global_to_var"].sum(-1) <= 1 + 1e-5)
    sr = snapshot_rollout(maps)
    assert torch.allclose(sr.sum(-1), torch.ones(8), atol=1e-5)
    sp = torch.tensor([[[[0, 0, 0], [0, 1, 0], [1, 3, 0], [4, 2, 1]]]])
    assert patch_state_category(sp).tolist() == [[[0, 1, 3, 4]]]
    assert det["tokens"].shape == (8, V, Np, model.cfg.d_model)


def test_collect_and_relevance_tables(setup):
    model, _, d, _ = setup
    res = collect_interpretation(model, d, batch_size=32, max_records=50)
    assert res.rollout.shape[0] == 50 and res.patch_relevance.shape[1:] == (model.cfg.n_vars, model.n_patches)
    tab = relevance_by_category(res.patch_relevance, res.patch_category, 3)
    assert np.isclose((tab.share_of_patches * tab.lift).sum(), 1.0)
    assert np.isclose(tab.share_of_relevance.sum(), 1.0)
    assert res.snapshot_tokens.shape == (50, len(model.cfg.snapshot_group_sizes), model.cfg.d_model)


def test_embeddings_and_behaviour(setup):
    model, prep, d, tcfg = setup
    cm = cell_embedding_map(model)
    reg = cm["regular"] - cm["regular"].mean(0)
    assert np.linalg.matrix_rank(reg, tol=1e-5) == 1          # values lie on one line
    sim = state_similarity(model)
    assert np.allclose(np.diag(sim.values), 1, atol=1e-5)
    small = type(d)(d.x[:80], d.state[:80], d.snapshot[:80], d.y[:80], d.y_raw[:80])
    ab = modality_ablation(model, small, prep, tcfg)
    assert set(ab.branch) == {"cls_ts", "cls_snapshot"} and np.isfinite(ab.mean_abs_log_change).all()
    pi = permutation_importance(model, small, prep, tcfg, [f"v{i}" for i in range(model.cfg.n_vars)],
                                [f"s{j}" for j in range(5)], n_repeats=1)
    assert len(pi) == model.cfg.n_vars + 5 and set(pi.kind) == {"time series", "snapshot"}
    sp = state_probe(model, small, prep, tcfg, n_cells=30)
    assert list(sp.state) == ["missing", "special 1", "special 2", "special 3"]
