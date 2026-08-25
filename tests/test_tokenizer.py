import pytest
import torch

from tsfm.config import TokenizerConfig
from tsfm.modules.tokenizer import ConvPatchTokenizer

GEOMETRIES = [(5, 2, 24, 1), (3, 3, 24, 1), (5, 1, 16, 1), (4, 2, 24, 2), (7, 2, 12, 1)]


@pytest.mark.parametrize("k,s,t,layers", GEOMETRIES)
@pytest.mark.parametrize("causal", [False, True])
def test_shapes_and_mask_alignment(k, s, t, layers, causal):
    cfg = TokenizerConfig(
        in_channels=4, conv_channels=6, kernel_size=k, stride=s, n_layers=layers,
        d_model=8, causal=causal,
    )
    tok = ConvPatchTokenizer(cfg, use_obs_channel=True)
    x, m = torch.randn(3, 4, t), torch.ones(3, 4, t)
    out = tok(x, m)
    n = tok.n_patches(t)
    assert out.shape == (3, 4, n, 8)
    # the mask must land on exactly the same patch grid as the tokens
    assert tok.pool_mask(torch.zeros(3, 4, t)).shape == (3, 4, n)


@pytest.mark.parametrize("k,s,t,layers", GEOMETRIES)
def test_last_time_step_is_covered(k, s, t, layers):
    """pad_to_cover must keep the final (usually most recent) step in a window."""
    cfg = TokenizerConfig(in_channels=2, kernel_size=k, stride=s, n_layers=layers,
                          d_model=8, pad_to_cover=True)
    tok = ConvPatchTokenizer(cfg)
    left, _, n = tok.pad_plan(t)[0]
    last_covered = (n - 1) * s + tok._eff_kernel - 1 - left
    assert last_covered >= t - 1


def test_per_feature_conv_does_not_mix_features():
    cfg = TokenizerConfig(in_channels=3, conv_channels=4, kernel_size=3, stride=1,
                          d_model=8, per_feature=True, norm="none")
    tok = ConvPatchTokenizer(cfg).eval()
    x = torch.zeros(1, 3, 12)
    base = tok(x)
    x2 = x.clone()
    x2[0, 0] = 5.0                      # perturb feature 0 only
    out = tok(x2)
    assert not torch.allclose(out[:, 0], base[:, 0])
    assert torch.allclose(out[:, 1:], base[:, 1:], atol=1e-6)


def test_shared_conv_mixes_features():
    cfg = TokenizerConfig(in_channels=3, conv_channels=4, kernel_size=3, stride=1,
                          d_model=8, per_feature=False, norm="none")
    tok = ConvPatchTokenizer(cfg).eval()
    assert tok.n_token_streams == 1
    assert tok(torch.randn(2, 3, 12)).shape[1] == 1


def test_pool_mask_matches_window_fraction():
    """A patch half inside the missing region must report ~0.5."""
    cfg = TokenizerConfig(in_channels=1, kernel_size=4, stride=4, d_model=8,
                          pad_to_cover=False)
    tok = ConvPatchTokenizer(cfg)
    mask = torch.zeros(1, 1, 8)
    mask[0, 0, :2] = 1.0                # first half of the first patch
    frac = tok.pool_mask(mask)
    assert frac.shape == (1, 1, 2)
    assert frac[0, 0, 0].item() == pytest.approx(0.5)
    assert frac[0, 0, 1].item() == pytest.approx(0.0)


def test_causal_token_ignores_the_future():
    cfg = TokenizerConfig(in_channels=1, conv_channels=4, kernel_size=3, stride=1,
                          d_model=8, causal=True, norm="none")
    tok = ConvPatchTokenizer(cfg).eval()
    x = torch.randn(1, 1, 16)
    base = tok(x)
    x2 = x.clone()
    x2[0, 0, -1] += 10.0                # change only the last step
    out = tok(x2)
    assert torch.allclose(out[:, :, :-1], base[:, :, :-1], atol=1e-5)


def test_bad_geometry_raises():
    cfg = TokenizerConfig(in_channels=1, kernel_size=9, stride=4, n_layers=3,
                          d_model=8, pad_to_cover=False)
    with pytest.raises(ValueError):
        ConvPatchTokenizer(cfg).n_patches(10)
