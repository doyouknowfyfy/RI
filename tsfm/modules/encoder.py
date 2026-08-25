"""Transformer encoder over the ``(C, N)`` patch-token grid, plus masked pooling.

The Conv1d tokenizer already models the *local temporal* pattern, so the
default attention axis is **across features** (``attn_mode="feature"``): at
each patch position the ``C`` feature tokens attend to one another.  Other
axes are available for ablation (``time``, ``factorized``, ``full``).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import TransformerConfig

__all__ = ["AxialTransformerEncoder", "MaskedPooling"]


def _safe_key_padding_mask(pad: torch.Tensor) -> torch.Tensor:
    """Avoid all-``True`` rows, which make softmax return NaN.

    ``pad``: ``(rows, L)`` bool, ``True`` == ignore.  A row that is entirely
    ignored (e.g. a record with *no* observed month for a feature) is unmasked;
    its output is discarded later by the pooling mask anyway.
    """
    all_pad = pad.all(dim=-1, keepdim=True)
    return pad & ~all_pad


class AxialTransformerEncoder(nn.Module):
    """Stack of transformer blocks applied along the feature and/or patch axis.

    Forward
    -------
    tokens : ``(B, C, N, d_model)``
    valid : ``(B, C, N)`` bool -- ``True`` where the token carries real signal.

    Returns ``(B, C, N, d_model)``.
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        d_model: int,
        n_features: int,
        n_patches: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_features = n_features
        self.n_patches = n_patches

        if d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={cfg.n_heads}")
        if cfg.attn_mode == "feature" and n_features == 1:
            raise ValueError(
                "attn_mode='feature' needs more than one token stream; use "
                "TokenizerConfig(per_feature=True) or switch attn_mode to 'time'."
            )

        self.feature_emb = (
            nn.Parameter(torch.randn(n_features, d_model) * 0.02) if cfg.feature_embedding else None
        )
        self.patch_emb = (
            nn.Parameter(torch.randn(n_patches, d_model) * 0.02) if cfg.patch_embedding else None
        )

        modes: List[str] = []
        for i in range(cfg.n_layers):
            if cfg.attn_mode == "factorized":
                modes.append("feature" if i % 2 == 0 else "time")
            else:
                modes.append(cfg.attn_mode)
        self.modes = modes

        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                activation=cfg.activation,
                batch_first=True,
                norm_first=cfg.pre_norm,
            )
            for _ in modes
        )
        # `nn.TransformerEncoderLayer` applies self_attn dropout via its own
        # `dropout` arg; expose attn_dropout by overriding it explicitly.
        for layer in self.layers:
            layer.self_attn.dropout = cfg.attn_dropout

        self.final_norm = nn.LayerNorm(d_model) if cfg.pre_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------ #
    def forward(self, tokens: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, n, d = tokens.shape
        if valid is None:
            valid = torch.ones(b, c, n, dtype=torch.bool, device=tokens.device)

        if self.feature_emb is not None:
            tokens = tokens + self.feature_emb.to(tokens.dtype).view(1, c, 1, d)
        if self.patch_emb is not None:
            tokens = tokens + self.patch_emb.to(tokens.dtype).view(1, 1, n, d)
        tokens = self.emb_dropout(tokens)

        pad = ~valid  # True == ignore
        for mode, layer in zip(self.modes, self.layers):
            if mode == "feature":
                x = tokens.permute(0, 2, 1, 3).reshape(b * n, c, d)
                kpm = _safe_key_padding_mask(pad.permute(0, 2, 1).reshape(b * n, c))
                x = layer(x, src_key_padding_mask=kpm)
                tokens = x.reshape(b, n, c, d).permute(0, 2, 1, 3)
            elif mode == "time":
                x = tokens.reshape(b * c, n, d)
                kpm = _safe_key_padding_mask(pad.reshape(b * c, n))
                x = layer(x, src_key_padding_mask=kpm)
                tokens = x.reshape(b, c, n, d)
            elif mode == "full":
                x = tokens.reshape(b, c * n, d)
                kpm = _safe_key_padding_mask(pad.reshape(b, c * n))
                x = layer(x, src_key_padding_mask=kpm)
                tokens = x.reshape(b, c, n, d)
            else:  # pragma: no cover -- guarded by the config Literal
                raise ValueError(f"unknown attn_mode {mode!r}")
            tokens = tokens.contiguous()

        return self.final_norm(tokens)


class MaskedPooling(nn.Module):
    """Collapse ``(B, C, N, d)`` tokens into one time-series embedding.

    ``mode``
        ``attention``       -- learned-query attention pooling over all valid
                               tokens (default).
        ``mean`` / ``max``  -- masked mean / max over all valid tokens.
        ``flatten_feature`` -- masked mean over patches, then concatenate the
                               per-feature vectors (output dim ``C * d``);
                               keeps the embedding feature-attributable.
    """

    def __init__(self, mode: str, d_model: int, n_features: int):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.n_features = n_features
        if mode == "attention":
            self.score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1)
            )
        self.out_dim = d_model * n_features if mode == "flatten_feature" else d_model

    def forward(
        self, tokens: torch.Tensor, valid: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns ``(embedding, attn_weights)``; weights are ``None`` unless
        ``mode == 'attention'`` (shape ``(B, C, N)``, useful for explainability).
        """
        b, c, n, d = tokens.shape
        if valid is None:
            valid = torch.ones(b, c, n, dtype=torch.bool, device=tokens.device)
        m = valid.unsqueeze(-1).to(tokens.dtype)

        if self.mode == "flatten_feature":
            denom = m.sum(dim=2).clamp(min=1.0)                 # (B, C, 1)
            per_feature = (tokens * m).sum(dim=2) / denom       # (B, C, d)
            return per_feature.reshape(b, c * d), None

        if self.mode == "mean":
            denom = m.sum(dim=(1, 2)).clamp(min=1.0)            # (B, 1)
            return (tokens * m).sum(dim=(1, 2)) / denom, None

        if self.mode == "max":
            neg_inf = torch.finfo(tokens.dtype).min
            filled = tokens.masked_fill(~valid.unsqueeze(-1), neg_inf)
            pooled = filled.reshape(b, c * n, d).max(dim=1).values
            # a record with no valid token at all -> zeros instead of -inf
            any_valid = valid.reshape(b, -1).any(dim=1, keepdim=True)
            return torch.where(any_valid, pooled, torch.zeros_like(pooled)), None

        # attention pooling
        flat = tokens.reshape(b, c * n, d)
        logits = self.score(flat).squeeze(-1)                   # (B, C*N)
        flat_valid = valid.reshape(b, c * n)
        logits = logits.masked_fill(~flat_valid, torch.finfo(logits.dtype).min)
        # a fully-invalid row would give a uniform-NaN softmax: fall back to uniform
        all_invalid = ~flat_valid.any(dim=1, keepdim=True)
        logits = torch.where(all_invalid, torch.zeros_like(logits), logits)
        w = torch.softmax(logits.float(), dim=1).to(tokens.dtype)
        pooled = torch.einsum("bl,bld->bd", w, flat)
        return pooled, w.reshape(b, c, n)
