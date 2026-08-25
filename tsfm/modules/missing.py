"""Missing-value embedding for *systematic* missingness.

Two kinds of missingness are treated very differently:

* **Random missing** -- the observation exists but was not recorded.  It is
  imputed with ``0`` (the training mean after standard scaling) and, if
  ``add_observed_channel`` is on, flagged to the tokenizer through an extra
  conv channel.  No embedding.
* **Systematic missing** -- the time step *cannot* exist for this record (a
  5-month-old card inside a 24-month window: 19 months are structurally
  absent).  That is information about the record, not noise, so it gets a
  **learned embedding** which is combined with the patch token either
  additively or by concatenation (``MissingConfig.combine``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import MissingConfig

__all__ = ["MissingEmbedding"]


class MissingEmbedding(nn.Module):
    """Combine a learned systematic-missing embedding with patch tokens.

    Parameters
    ----------
    cfg : MissingConfig
    d_model : int
        Token width.
    n_features : int
        Number of token streams (``C``) -- used when
        ``per_feature_embedding`` is on.

    Forward
    -------
    tokens : ``(B, C, N, d_model)``
    sys_frac : ``(B, C, N)`` -- fraction of each patch that is systematically
        missing, in ``[0, 1]`` (produced by
        :meth:`~tsfm.modules.tokenizer.ConvPatchTokenizer.pool_mask`).

    Returns tokens of the same shape ``(B, C, N, d_model)``.
    """

    def __init__(self, cfg: MissingConfig, d_model: int, n_features: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.enabled = cfg.combine != "none"

        # `add` requires matching widths -- force it rather than fail late.
        self.d_missing = d_model if cfg.combine == "add" else cfg.d_missing

        if not self.enabled:
            return

        n_rows = n_features if cfg.per_feature_embedding else 1
        self.embedding = nn.Parameter(torch.empty(n_rows, self.d_missing))
        if cfg.combine == "add":
            # zero init => the module is a no-op at step 0 and cannot destroy
            # the tokenizer's signal before it has learned anything useful.
            nn.init.zeros_(self.embedding)
        else:
            nn.init.normal_(self.embedding, std=0.02)

        if cfg.combine == "concat":
            self.project = nn.Linear(d_model + self.d_missing, d_model)
            self.norm = nn.LayerNorm(d_model)
        else:
            self.project = None
            self.norm = None

    # ------------------------------------------------------------------ #
    def missing_weight(self, sys_frac: torch.Tensor) -> torch.Tensor:
        """``(B, C, N)`` -> gate in ``[0, 1]`` (soft fraction or hard flag)."""
        if self.cfg.soft:
            return sys_frac.clamp(0.0, 1.0)
        return (sys_frac > self.cfg.hard_threshold).to(sys_frac.dtype)

    def forward(self, tokens: torch.Tensor, sys_frac: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return tokens

        b, c, n, d = tokens.shape
        w = self.missing_weight(sys_frac).unsqueeze(-1)          # (B, C, N, 1)

        emb = self.embedding
        if emb.shape[0] == 1:
            emb = emb.expand(c, -1)
        elif emb.shape[0] != c:
            raise ValueError(
                f"missing embedding has {emb.shape[0]} rows but got {c} token streams"
            )
        m = w * emb.to(tokens.dtype).view(1, c, 1, self.d_missing)  # (B, C, N, d_missing)

        if self.cfg.combine == "add":
            return tokens + m
        out = self.project(torch.cat([tokens, m], dim=-1))
        return self.norm(out)
