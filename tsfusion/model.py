"""Transformer for multivariate time series + snapshot variables (regression).

Architecture (``missing_fusion="late"`` = the requested design)::

    x (B,T,V) ──► sliding patches per variable (B,V,Np,P) ──► value patch embedding ─┐
                                                                                    ▼
                              2-D token grid + [CLS] row/column ──► L x [ attention across variables
                                                                         ─► attention across time ─► FFN ]
                                                                    ──► CLS_regular (B,d)
    state (B,T,V) ──► same sliding patches ──► shared missing/special embedding
                              ──► 2-D grid + [CLS] ──► L' x factorized blocks ──► CLS_special (B,d)
    snapshot PLE (B,S,bins) ──► per-variable linear embedding ──► [CLS] + S tokens
                              ──► transformer ──► CLS_snapshot (B,d)

    head( concat[CLS_regular, CLS_special, CLS_snapshot] ) ──► prediction (B,)

The 2-D CLS scheme (BERT-style [CLS] generalised to a grid)::

            col 0          col 1 .. Np
    row 0   [CLS global]   [CLS time_1] ... [CLS time_Np]    <- summaries across variables
    row v   [CLS var_v]    patch(v,1)   ... patch(v,Np)      <- variable v, patch positions
              ^ summaries across time

  * attention across variables runs over each column  -> [CLS time_n] collects time slot n
  * attention across time runs over each row          -> [CLS global] collects all
    [CLS time_n]; [CLS var_v] collects the history of variable v.
  After a single (variable -> time) block the global CLS has seen every patch.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


# =============================================================================
# Building blocks
# =============================================================================
class SlidingPatcher(nn.Module):
    """``(B, T, V)`` -> ``(B, V, Np, P)`` overlapping windows along time, per variable.

    If ``(T - P)`` is not a multiple of the stride, the series is right-padded by
    repeating the last step so that the most recent observations are always covered.
    """

    def __init__(self, n_steps: int, patch_len: int, stride: int, pad_end: bool = True):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        rem = (n_steps - patch_len) % stride
        self.pad = (stride - rem) % stride if pad_end else 0
        self.n_patches = (n_steps + self.pad - patch_len) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)                                   # (B, V, T)
        if self.pad:
            x = torch.cat([x, x[..., -1:].expand(*x.shape[:-1], self.pad)], dim=-1)
        return x.unfold(-1, self.patch_len, self.stride)        # (B, V, Np, P)


class SelfAttention(nn.Module):
    """Multi-head self-attention on ``(B, L, d)`` with an optional key mask.

    ``key_valid``: ``(B, L)`` bool, True = the token can be attended to. Uses
    ``F.scaled_dot_product_attention`` (flash / memory-efficient kernels on GPU).

    The factorized encoder calls this with very many short sequences (attention across
    time runs ``batch x (V+1)`` sequences, e.g. 256 x 461 = 118,016). CUDA limits the 2nd
    and 3rd grid dimensions of a kernel launch to 65,535 and the fused attention kernels
    put the sequence batch there, so larger calls fail with "CUDA error: invalid
    configuration argument". Calls are therefore split into slices of at most
    ``max_batch`` sequences - same result, gradients flow through every slice.
    """

    max_batch = 32_768

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = attn_dropout
        # interpretation: when True, attention is computed explicitly and the weights
        # (B, H, L, L) are kept in `attn_weights` (see tsfusion.interpret.record_attention)
        self.store_attn = False
        self.attn_weights: Optional[torch.Tensor] = None

    def _attend(self, q, k, v, mask):
        if self.store_attn:
            scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.d_head)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            attn = scores.softmax(dim=-1)
            return (attn @ v.float()).to(v.dtype), attn.detach()
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_dropout if self.training else 0.0)
        return out, None

    def forward(self, x: torch.Tensor, key_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        mask = None if key_valid is None else key_valid[:, None, None, :]   # (B,1,1,L)
        step = max(1, int(self.max_batch))
        if B <= step:
            out, weights = self._attend(q, k, v, mask)
        else:
            parts = [self._attend(q[i:i + step], k[i:i + step], v[i:i + step],
                                  None if mask is None else mask[i:i + step])
                     for i in range(0, B, step)]
            out = torch.cat([o for o, _ in parts])
            weights = torch.cat([w for _, w in parts]) if self.store_attn else None
        if self.store_attn:
            self.attn_weights = weights
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ff_mult * d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Standard pre-LN encoder block on a 1-D token sequence."""

    def __init__(self, d_model, n_heads, ff_mult, dropout, attn_dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, attn_dropout)
        self.ffn = FeedForward(d_model, ff_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_valid=None):
        x = x + self.drop(self.attn(self.ln1(x), key_valid))
        return x + self.drop(self.ffn(self.ln2(x)))


class FactorizedBlock(nn.Module):
    """Pre-LN block on a 2-D grid ``(B, R, C, d)`` (rows = variables, cols = patches):
    attention across variables (within each column) -> attention across time (within
    each row) -> feed-forward."""

    def __init__(self, d_model, n_heads, ff_mult, dropout, attn_dropout):
        super().__init__()
        self.ln_var, self.ln_time, self.ln_ffn = (nn.LayerNorm(d_model) for _ in range(3))
        self.var_attn = SelfAttention(d_model, n_heads, attn_dropout)
        self.time_attn = SelfAttention(d_model, n_heads, attn_dropout)
        self.ffn = FeedForward(d_model, ff_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, R, C, D = h.shape
        # 1) across variables: one sequence of length R per (sample, column)
        hv = h.transpose(1, 2).reshape(B * C, R, D)
        mv = None if valid is None else valid.transpose(1, 2).reshape(B * C, R)
        hv = hv + self.drop(self.var_attn(self.ln_var(hv), mv))
        h = hv.view(B, C, R, D).transpose(1, 2)
        # 2) across time: one sequence of length C per (sample, row)
        ht = h.reshape(B * R, C, D)
        mt = None if valid is None else valid.reshape(B * R, C)
        ht = ht + self.drop(self.time_attn(self.ln_time(ht), mt))
        ht = ht + self.drop(self.ffn(self.ln_ffn(ht)))
        return ht.view(B, R, C, D)


class GridEncoder(nn.Module):
    """Adds variable / patch-position embeddings and the 2-D CLS row & column to a token
    grid ``(B, V, Np, d)``, runs factorized blocks and returns the global CLS ``(B, d)``."""

    def __init__(self, n_vars: int, n_patches: int, cfg: ModelConfig, n_layers: int):
        super().__init__()
        d = cfg.d_model
        self.var_emb = nn.Parameter(torch.zeros(n_vars, d))
        self.pos_emb = nn.Parameter(torch.zeros(n_patches, d))
        self.cls_global = nn.Parameter(torch.zeros(d))
        self.cls_var = nn.Parameter(torch.zeros(d))     # column 0, row v: variable v across time
        self.cls_time = nn.Parameter(torch.zeros(d))    # row 0, column n: patch n across variables
        for p in (self.var_emb, self.pos_emb, self.cls_global, self.cls_var, self.cls_time):
            nn.init.trunc_normal_(p, std=0.02)
        self.blocks = nn.ModuleList(FactorizedBlock(d, cfg.n_heads, cfg.ff_mult, cfg.dropout,
                                                    cfg.attn_dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.grad_ckpt = cfg.gradient_checkpointing

    def forward(self, tokens: torch.Tensor, valid: Optional[torch.Tensor] = None,
                return_grid: bool = False):
        B, V, N, D = tokens.shape
        x = tokens + self.var_emb[None, :, None, :] + self.pos_emb[None, None, :, :]
        top = torch.cat([self.cls_global[None], self.cls_time[None] + self.pos_emb], 0)  # (N+1, D)
        left = self.cls_var[None] + self.var_emb                                          # (V, D)
        x = torch.cat([left[None, :, None, :].expand(B, V, 1, D).to(x.dtype), x], dim=2)
        x = torch.cat([top[None, None].expand(B, 1, N + 1, D).to(x.dtype), x], dim=1)
        x = self.emb_drop(x)                                                              # (B,V+1,N+1,D)
        grid_valid = None
        if valid is not None:
            grid_valid = torch.ones(B, V + 1, N + 1, dtype=torch.bool, device=x.device)
            grid_valid[:, 1:, 1:] = valid       # CLS row / column are always attendable,
                                                # so no sequence is ever fully masked
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(blk, x, grid_valid, use_reentrant=False)
            else:
                x = blk(x, grid_valid)
        x = self.norm(x)
        cls = x[:, 0, 0]
        return (cls, x) if return_grid else cls


# =============================================================================
# Tokenizers
# =============================================================================
class ValuePatchEmbedding(nn.Module):
    """Regular values only (late fusion). A patch of P standardised values (non-regular
    cells are 0) -> d. With ``include_mask`` the P "is regular" bits are appended so the
    network can tell an imputed 0 from a real value."""

    def __init__(self, patch_len: int, d_model: int, include_mask: bool):
        super().__init__()
        self.include_mask = include_mask
        self.proj = nn.Linear(patch_len * (2 if include_mask else 1), d_model)

    def forward(self, x_p: torch.Tensor, regular_p: torch.Tensor) -> torch.Tensor:
        if self.include_mask:
            x_p = torch.cat([x_p, regular_p.to(x_p.dtype)], dim=-1)
        return self.proj(x_p)


class StatePatchEmbedding(nn.Module):
    """Missing/special branch. ONE embedding per state (regular, missing, special_1..K),
    shared by all variables and time steps; a patch concatenates its P state embeddings
    and projects them to d (so the *position* of a missing value inside the patch is kept)."""

    def __init__(self, n_states: int, state_dim: int, patch_len: int, d_model: int):
        super().__init__()
        self.state_emb = nn.Embedding(n_states, state_dim)
        self.proj = nn.Linear(patch_len * state_dim, d_model)

    def forward(self, state_p: torch.Tensor) -> torch.Tensor:
        return self.proj(self.state_emb(state_p).flatten(-2))


class ValueStatePatchEmbedding(nn.Module):
    """Early fusion (value-aware tokens). For every cell::

        e(v,t) = 1[regular] * (w * x + b)  +  E_state[state(v,t)]

    i.e. a regular cell is its projected value (+ the learned 'regular' vector E_state[0]);
    a missing / special cell is *replaced* by its learned embedding - like BERT's [MASK].
    The P cell embeddings of a patch are concatenated and projected to d."""

    def __init__(self, n_states: int, state_dim: int, patch_len: int, d_model: int):
        super().__init__()
        self.value_proj = nn.Linear(1, state_dim)
        self.state_emb = nn.Embedding(n_states, state_dim)
        self.proj = nn.Linear(patch_len * state_dim, d_model)

    def forward(self, x_p: torch.Tensor, state_p: torch.Tensor) -> torch.Tensor:
        regular = (state_p == 0).unsqueeze(-1).to(x_p.dtype)
        e = regular * self.value_proj(x_p.unsqueeze(-1)) + self.state_emb(state_p)
        return self.proj(e.flatten(-2))


class SnapshotEncoder(nn.Module):
    """FT-Transformer style encoder for snapshot variables.

    Each snapshot variable j has its own linear embedding of its PLE vector
    (``tok_j = ple_j @ W_j + b_j``; a raw variable is just ``group_size = 1``), a [CLS]
    token is prepended and a small transformer returns the CLS embedding."""

    def __init__(self, group_sizes: List[int], cfg: ModelConfig):
        super().__init__()
        d, S, K = cfg.d_model, len(group_sizes), max(group_sizes)
        self.weight = nn.Parameter(torch.empty(S, K, d))
        self.bias = nn.Parameter(torch.zeros(S, d))
        with torch.no_grad():
            for j, g in enumerate(group_sizes):
                bound = 1.0 / math.sqrt(g)
                self.weight[j].uniform_(-bound, bound)
                self.weight[j, g:].zero_()                 # padding bins (inputs are 0 anyway)
                self.bias[j].uniform_(-bound, bound)
        self.register_buffer("bin_mask", torch.tensor(
            [[1.0] * g + [0.0] * (K - g) for g in group_sizes]), persistent=False)
        self.cls = nn.Parameter(torch.zeros(d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(TransformerBlock(d, cfg.n_heads, cfg.ff_mult, cfg.dropout,
                                                     cfg.attn_dropout)
                                    for _ in range(cfg.n_snapshot_layers))
        self.norm = nn.LayerNorm(d)
        self.emb_drop = nn.Dropout(cfg.dropout)

    def tokenize(self, snap: torch.Tensor) -> torch.Tensor:         # (B, S, K) -> (B, S, d)
        snap = snap * self.bin_mask                                  # guard against junk padding
        return torch.einsum("bsk,skd->bsd", snap, self.weight) + self.bias

    def forward(self, snap: torch.Tensor) -> torch.Tensor:          # (B, S, K)
        tok = self.tokenize(snap)
        x = torch.cat([self.cls.expand(tok.shape[0], 1, -1).to(tok.dtype), tok], dim=1)
        x = self.emb_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


# =============================================================================
# Full model
# =============================================================================
class TSFusionTransformer(nn.Module):
    """See module docstring. ``forward`` returns the raw output on the *link* scale
    (identity for MSE, log(mu) for Tweedie)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patcher = SlidingPatcher(cfg.n_steps, cfg.patch_len, cfg.patch_stride,
                                      cfg.patch_pad_end)
        n_patches = self.patcher.n_patches
        self.fusion = cfg.missing_fusion

        if self.fusion == "late":
            self.value_embed = ValuePatchEmbedding(cfg.patch_len, cfg.d_model,
                                                   cfg.patch_include_mask)
        else:
            self.value_embed = ValueStatePatchEmbedding(cfg.n_states, cfg.state_emb_dim,
                                                        cfg.patch_len, cfg.d_model)
        self.ts_encoder = GridEncoder(cfg.n_vars, n_patches, cfg, cfg.n_ts_layers)

        self.has_special_branch = self.fusion in ("late", "hybrid")
        if self.has_special_branch:
            self.state_embed = StatePatchEmbedding(cfg.n_states, cfg.state_emb_dim,
                                                   cfg.patch_len, cfg.d_model)
            self.special_encoder = GridEncoder(cfg.n_vars, n_patches, cfg, cfg.n_special_layers)

        self.has_snapshot = len(cfg.snapshot_group_sizes) > 0
        if self.has_snapshot:
            self.snapshot_encoder = SnapshotEncoder(cfg.snapshot_group_sizes, cfg)

        self.n_cls = 1 + int(self.has_special_branch) + int(self.has_snapshot)
        self.head = RegressionHead(self.n_cls * cfg.d_model, cfg.head_hidden, cfg.head_dropout)
        self.apply(self._init_weights)
        nn.init.normal_(self.head.net[-1].weight, std=1e-3)   # start close to the base rate
        nn.init.zeros_(self.head.net[-1].bias)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    @property
    def n_patches(self) -> int:
        return self.patcher.n_patches

    def embed(self, x: torch.Tensor, state: torch.Tensor,
              snapshot: Optional[torch.Tensor] = None,
              details: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Return the CLS embeddings of every branch (useful for inspection / reuse).

        If a dict is passed as ``details`` it is filled with intermediate tensors:
        ``state_patches`` (B,V,Np,P), ``tokens`` (patch tokens entering the time-series
        encoder, B,V,Np,d), ``ts_grid`` (encoder output incl. CLS row/column,
        B,V+1,Np+1,d) and ``snapshot_tokens`` (B,S,d)."""
        state = state.long()
        # values at missing / special cells never reach the value pathway, whatever the
        # caller put there (imputed 0, sentinel like -999, NaN ...)
        x = torch.where(state == 0, torch.nan_to_num(x), torch.zeros((), dtype=x.dtype,
                                                                     device=x.device))
        x_p = self.patcher(x)                    # (B, V, Np, P)
        s_p = self.patcher(state)                # (B, V, Np, P)
        regular_p = s_p == 0
        out: Dict[str, torch.Tensor] = {}

        if self.fusion == "late":
            tok = self.value_embed(x_p, regular_p)
            valid = regular_p.any(-1) if self.cfg.mask_empty_patches else None
        else:
            tok = self.value_embed(x_p, s_p)
            valid = None                          # every token carries information
        if details is None:
            out["cls_ts"] = self.ts_encoder(tok, valid)
        else:
            out["cls_ts"], details["ts_grid"] = self.ts_encoder(tok, valid, return_grid=True)
            details["state_patches"], details["tokens"] = s_p, tok

        if self.has_special_branch:
            stok = self.state_embed(s_p)
            svalid = (~regular_p).any(-1) if self.cfg.mask_regular_patches else None
            out["cls_special"] = self.special_encoder(stok, svalid)

        if self.has_snapshot:
            if snapshot is None:
                raise ValueError("model was built with snapshot variables but none were given")
            out["cls_snapshot"] = self.snapshot_encoder(snapshot)
            if details is not None:
                details["snapshot_tokens"] = self.snapshot_encoder.tokenize(snapshot)
        return out

    def forward(self, x: torch.Tensor, state: torch.Tensor,
                snapshot: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embed(x, state, snapshot)
        z = torch.cat(list(emb.values()), dim=-1)
        # the head runs in fp32 even under autocast: bf16 rounding of the output (~0.4%)
        # would add noise to predictions, metrics and model selection
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self.head(z.float())

    def set_output_bias(self, value: float) -> None:
        with torch.no_grad():
            self.head.net[-1].bias.fill_(value)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
