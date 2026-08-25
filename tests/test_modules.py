import pytest
import torch

from tsfm.config import (
    HeadConfig, MissingConfig, ModelConfig, SnapshotConfig, TokenizerConfig, TransformerConfig,
)
from tsfm.model import TSFormerRegressor
from tsfm.modules.encoder import AxialTransformerEncoder, MaskedPooling
from tsfm.modules.missing import MissingEmbedding
from tsfm.modules.snapshot import SnapshotEncoder


# ----------------------------- missing embedding ---------------------------- #
def test_additive_missing_embedding_is_identity_at_init():
    """Zero-init means the module cannot corrupt the tokenizer signal at step 0."""
    me = MissingEmbedding(MissingConfig(combine="add"), d_model=8, n_features=3)
    tokens = torch.randn(2, 3, 5, 8)
    frac = torch.rand(2, 3, 5)
    assert torch.allclose(me(tokens, frac), tokens)
    assert me.d_missing == 8            # forced to d_model


def test_additive_missing_embedding_learns():
    me = MissingEmbedding(MissingConfig(combine="add"), d_model=8, n_features=3)
    tokens = torch.randn(2, 3, 5, 8)
    frac = torch.ones(2, 3, 5)
    me(tokens, frac).sum().backward()
    assert me.embedding.grad is not None and me.embedding.grad.abs().sum() > 0


def test_concat_missing_embedding_shape_and_effect():
    cfg = MissingConfig(combine="concat", d_missing=4)
    me = MissingEmbedding(cfg, d_model=8, n_features=3)
    tokens = torch.randn(2, 3, 5, 8)
    out_none = me(tokens, torch.zeros(2, 3, 5))
    out_all = me(tokens, torch.ones(2, 3, 5))
    assert out_none.shape == tokens.shape == out_all.shape
    assert not torch.allclose(out_none, out_all)


def test_missing_combine_none_is_passthrough():
    me = MissingEmbedding(MissingConfig(combine="none"), d_model=8, n_features=3)
    tokens = torch.randn(2, 3, 5, 8)
    assert torch.allclose(me(tokens, torch.ones(2, 3, 5)), tokens)


def test_hard_vs_soft_gate():
    hard = MissingEmbedding(MissingConfig(combine="concat", soft=False), 8, 2)
    frac = torch.tensor([[[0.2, 0.8]], [[0.6, 0.4]]]).expand(2, 2, 2)
    w = hard.missing_weight(frac)
    assert set(w.flatten().tolist()) <= {0.0, 1.0}


# --------------------------------- encoder ---------------------------------- #
def test_encoder_shapes_all_modes():
    for mode in ("feature", "time", "factorized", "full"):
        enc = AxialTransformerEncoder(
            TransformerConfig(n_layers=2, n_heads=2, d_ff=16, attn_mode=mode), 8, 4, 6
        )
        tokens = torch.randn(3, 4, 6, 8)
        valid = torch.ones(3, 4, 6, dtype=torch.bool)
        assert enc(tokens, valid).shape == tokens.shape


def test_encoder_handles_fully_masked_rows_without_nan():
    """A record with no valid token anywhere must not produce NaNs."""
    enc = AxialTransformerEncoder(TransformerConfig(n_layers=2, n_heads=2, d_ff=16), 8, 4, 6)
    tokens = torch.randn(2, 4, 6, 8)
    valid = torch.ones(2, 4, 6, dtype=torch.bool)
    valid[0] = False                                    # record 0 fully missing
    out = enc(tokens, valid)
    assert torch.isfinite(out).all()


def test_feature_attention_requires_multiple_streams():
    with pytest.raises(ValueError):
        AxialTransformerEncoder(TransformerConfig(attn_mode="feature", n_heads=2), 8, 1, 6)


def test_masked_pooling_ignores_invalid_tokens():
    for mode in ("mean", "max", "attention"):
        pool = MaskedPooling(mode, 8, 4)
        tokens = torch.randn(2, 4, 6, 8)
        valid = torch.ones(2, 4, 6, dtype=torch.bool)
        valid[:, :, 3:] = False
        emb_a, _ = pool(tokens, valid)
        tokens2 = tokens.clone()
        tokens2[:, :, 3:] = 999.0                       # garbage in the invalid region
        emb_b, _ = pool(tokens2, valid)
        assert torch.allclose(emb_a, emb_b, atol=1e-5), mode


def test_masked_pooling_all_invalid_is_finite():
    for mode in ("mean", "max", "attention", "flatten_feature"):
        pool = MaskedPooling(mode, 8, 4)
        out, _ = pool(torch.randn(2, 4, 6, 8), torch.zeros(2, 4, 6, dtype=torch.bool))
        assert torch.isfinite(out).all(), mode


def test_flatten_feature_pool_dim():
    pool = MaskedPooling("flatten_feature", 8, 4)
    assert pool.out_dim == 32
    out, _ = pool(torch.randn(2, 4, 6, 8), torch.ones(2, 4, 6, dtype=torch.bool))
    assert out.shape == (2, 32)


# -------------------------------- snapshot ---------------------------------- #
def test_snapshot_encoder_and_missing_code_zero():
    cfg = SnapshotConfig(cat_cardinalities=[5, 3], n_numeric=4, d_out=16)
    enc = SnapshotEncoder(cfg)
    out = enc(torch.randint(0, 3, (7, 2)), torch.randn(7, 4), torch.zeros(7, 4))
    assert out.shape == (7, 16)
    # index 0 is the padding/missing slot
    assert torch.allclose(enc.embeddings[0].weight[0], torch.zeros(enc.emb_dims[0]))


def test_snapshot_encoder_empty_is_none():
    enc = SnapshotEncoder(SnapshotConfig())
    assert enc.out_dim == 0 and enc(None, None, None) is None


# ---------------------------------- model ----------------------------------- #
def _model(**kw):
    cfg = ModelConfig(
        tokenizer=TokenizerConfig(in_channels=5, conv_channels=8, kernel_size=5, stride=2, d_model=16),
        seq_len=24,
        missing=kw.pop("missing", MissingConfig()),
        transformer=kw.pop("transformer", TransformerConfig(n_layers=2, n_heads=4, d_ff=32)),
        snapshot=SnapshotConfig(cat_cardinalities=[4, 6], n_numeric=3, d_out=8),
        head=HeadConfig(hidden_dims=(16,)),
    )
    return TSFormerRegressor(cfg)


def _batch(b=6, c=5, t=24):
    sys = torch.zeros(b, c, t, dtype=torch.bool)
    sys[:, :, :7] = True                                # short-tenure records
    obs = ~sys
    obs[:, :, 10] = False                               # a random hole
    return {
        "x": torch.randn(b, c, t) * obs,
        "obs_mask": obs,
        "sys_mask": sys,
        "cat": torch.randint(0, 4, (b, 2)),
        "num": torch.randn(b, 3),
        "num_missing": torch.zeros(b, 3),
        "y": torch.randn(b, 1),
    }


@pytest.mark.parametrize("combine", ["add", "concat", "none"])
def test_model_forward_and_backward(combine):
    m = _model(missing=MissingConfig(combine=combine))
    out = m(_batch())
    assert out.shape == (6, 1) and torch.isfinite(out).all()
    out.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("mode", ["feature", "time", "factorized", "full"])
def test_model_attention_modes(mode):
    m = _model(transformer=TransformerConfig(n_layers=2, n_heads=4, d_ff=32, attn_mode=mode))
    assert m(_batch()).shape == (6, 1)


def test_model_aux_outputs():
    m = _model()
    out, aux = m(_batch(), return_aux=True)
    assert aux["tokens"].shape[:3] == (6, 5, m.n_patches)
    assert aux["pool_attn"].shape == (6, 5, m.n_patches)
    assert aux["sys_frac"].min() >= 0 and aux["sys_frac"].max() <= 1


def test_systematic_missing_changes_the_prediction():
    """The missing embedding must actually carry information."""
    m = _model(missing=MissingConfig(combine="concat", d_missing=8)).eval()
    b = _batch()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2["sys_mask"] = torch.zeros_like(b2["sys_mask"])
    with torch.no_grad():
        assert not torch.allclose(m(b), m(b2))


def test_record_with_no_history_is_finite():
    m = _model().eval()
    b = _batch()
    b["sys_mask"][0] = True
    b["obs_mask"][0] = False
    b["x"][0] = 0
    with torch.no_grad():
        assert torch.isfinite(m(b)).all()
