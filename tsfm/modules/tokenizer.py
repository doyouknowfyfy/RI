"""Conv1d patch tokenizer.

``(B, C, T)``  ->  ``(B, C, N, d_model)``

Each input feature is a conv *channel*; the kernel slides along the time axis
and the stride turns adjacent windows into **patch tokens**.  With
``per_feature=True`` the conv is *grouped* (``groups=C``) so feature ``c`` only
ever sees feature ``c`` -- that is what makes the downstream feature-axis
attention meaningful and keeps every token attributable to one input feature.

Padding is applied explicitly with :func:`torch.nn.functional.pad` (the conv
modules themselves use ``padding=0``) so that :meth:`ConvPatchTokenizer.pool_mask`
can reproduce *exactly* the same geometry for the missing masks.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TokenizerConfig

__all__ = ["ConvPatchTokenizer", "conv_out_length"]


def conv_out_length(t_in: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    """Length of a 1-D conv output (PyTorch formula)."""
    return (t_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def _activation(name: str) -> nn.Module:
    return {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}[name]


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a ``(B, Ch, T)`` tensor."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "layer":
        return _ChannelLayerNorm(channels)
    return nn.Identity()


class ConvPatchTokenizer(nn.Module):
    """Strided (grouped) Conv1d local feature extractor / patch tokenizer.

    Forward
    -------
    x : ``(B, C, T)`` -- scaled values, **random**-missing entries already
        replaced by ``0``.
    obs_mask : ``(B, C, T)`` -- 1 where observed, 0 where missing (either
        kind).  Fed as an extra conv channel when ``use_obs_channel`` is set,
        so the local extractor can tell an imputed zero from a true zero.

    Returns ``(B, C_tok, N, d_model)``; ``C_tok == C`` for a per-feature
    tokenizer, ``1`` for a shared (feature-mixing) one.
    """

    def __init__(self, cfg: TokenizerConfig, use_obs_channel: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_obs_channel = use_obs_channel
        self.n_features = cfg.in_channels
        self.n_token_streams = cfg.in_channels if cfg.per_feature else 1

        in_per_feature = 2 if use_obs_channel else 1
        groups = cfg.in_channels if cfg.per_feature else 1

        self.convs = nn.ModuleList()
        self.post = nn.ModuleList()
        in_ch = cfg.in_channels * in_per_feature
        for _ in range(cfg.n_layers):
            out_ch = groups * cfg.conv_channels
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride,
                    padding=0,                       # padding is done explicitly
                    dilation=cfg.dilation,
                    groups=groups,
                    bias=cfg.norm == "none",
                )
            )
            block = [_norm(cfg.norm, out_ch), _activation(cfg.activation)]
            if cfg.dropout > 0:
                block.append(nn.Dropout(cfg.dropout))
            self.post.append(nn.Sequential(*block))
            in_ch = out_ch

        # per-feature 1x1 projection: conv_channels -> d_model
        self.proj = nn.Conv1d(in_ch, groups * cfg.d_model, kernel_size=1, groups=groups)

    # ------------------------------------------------------------------ #
    @property
    def _eff_kernel(self) -> int:
        """Receptive width of one kernel including dilation."""
        return self.cfg.dilation * (self.cfg.kernel_size - 1) + 1

    def _layer_pad(self, t_in: int) -> Tuple[int, int]:
        """``(left, right)`` padding for one conv layer over length ``t_in``."""
        cfg = self.cfg
        k, s = self._eff_kernel, cfg.stride
        left = right = cfg.padding
        t = t_in + 2 * cfg.padding

        if cfg.causal:
            left += k - 1
            t += k - 1
            if cfg.pad_to_cover:
                # extra left pad so the last window's right edge lands on t_in-1
                m = (t_in - 1) % s
                extra = (s - m) % s
                left += extra
            return left, right

        if cfg.pad_to_cover:
            right += (k - t) if t < k else ((s - ((t - k) % s)) % s)
        return left, right

    def pad_plan(self, seq_len: int) -> List[Tuple[int, int, int]]:
        """``[(left, right, out_len), ...]`` -- one entry per conv layer."""
        cfg = self.cfg
        plan: List[Tuple[int, int, int]] = []
        t = seq_len
        for _ in range(cfg.n_layers):
            left, right = self._layer_pad(t)
            out = conv_out_length(t + left + right, cfg.kernel_size, cfg.stride, 0, cfg.dilation)
            if out < 1:
                raise ValueError(
                    f"Tokenizer geometry collapses the sequence: seq_len={seq_len}, "
                    f"kernel_size={cfg.kernel_size}, stride={cfg.stride}, "
                    f"n_layers={cfg.n_layers}. Reduce kernel/stride/layers or add padding."
                )
            plan.append((left, right, out))
            t = out
        return plan

    def n_patches(self, seq_len: int) -> int:
        """Number of patch tokens produced from ``seq_len`` time steps."""
        return self.pad_plan(seq_len)[-1][2]

    def receptive_field(self, seq_len: int) -> int:
        """Time steps each final token can see (useful for sanity-checking geometry)."""
        cfg = self.cfg
        rf, jump = 1, 1
        for _ in range(cfg.n_layers):
            rf = rf + (self._eff_kernel - 1) * jump
            jump *= cfg.stride
        return min(rf, seq_len)

    # ------------------------------------------------------------------ #
    def pool_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Push a per-time-step mask through the conv's exact geometry.

        ``mask``: ``(B, C, T)`` float in ``[0, 1]``.  Returns ``(B, C, N)`` --
        the *fraction* of each patch's receptive window that was flagged, so
        patch tokens and mask values line up cell for cell (including patches
        that straddle the edge of a record's observed span).  Padded positions
        count as flagged (``value=1.0``): they carry no data either.
        """
        cfg = self.cfg
        m = mask
        for left, right, _ in self.pad_plan(mask.shape[-1]):
            if left or right:
                m = F.pad(m, (left, right), value=1.0)
            if cfg.dilation == 1:
                m = F.avg_pool1d(m, kernel_size=cfg.kernel_size, stride=cfg.stride)
            else:  # emulate a dilated window
                m = m.unfold(-1, self._eff_kernel, cfg.stride)[..., :: cfg.dilation].mean(-1)
        return m

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, t = x.shape
        if c != self.cfg.in_channels:
            raise ValueError(f"expected {self.cfg.in_channels} channels, got {c}")

        if self.use_obs_channel:
            if obs_mask is None:
                obs_mask = torch.ones_like(x)
            # interleave [v0, m0, v1, m1, ...] so the grouped conv keeps each
            # feature's value and mask inside the same group
            h = torch.stack([x, obs_mask.to(x.dtype)], dim=2).reshape(b, 2 * c, t)
        else:
            h = x

        for (left, right, _), conv, post in zip(self.pad_plan(t), self.convs, self.post):
            if left or right:
                h = F.pad(h, (left, right))
            h = post(conv(h))

        h = self.proj(h)                       # (B, streams*d_model, N)
        n = h.shape[-1]
        h = h.reshape(b, self.n_token_streams, self.cfg.d_model, n)
        return h.permute(0, 1, 3, 2).contiguous()   # (B, C_tok, N, d_model)
