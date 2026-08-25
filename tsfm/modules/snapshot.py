"""Snapshot (static) branch: categorical + numeric variables constant in time."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..config import SnapshotConfig
from .mlp import build_mlp

__all__ = ["SnapshotEncoder", "default_emb_dim"]


def default_emb_dim(cardinality: int) -> int:
    """A reasonable embedding width for a categorical with ``cardinality`` levels."""
    return int(min(16, max(2, round(1.6 * cardinality ** 0.56))))


class SnapshotEncoder(nn.Module):
    """Embed static variables into a single ``d_out`` vector.

    Conventions (guaranteed by :func:`tsfm.data.prepare_datasets`):

    * categorical codes are ``int64`` in ``[0, cardinality)`` with **0 reserved
      for missing / unseen**;
    * numeric columns are already scaled, with missing entries set to ``0`` and
      flagged by ``num_missing`` when ``numeric_missing_indicator`` is on.

    Forward
    -------
    cat : ``(B, n_cat)`` int64 or ``None``
    num : ``(B, n_numeric)`` float or ``None``
    num_missing : ``(B, n_numeric)`` float/bool or ``None``
    """

    def __init__(self, cfg: SnapshotConfig):
        super().__init__()
        self.cfg = cfg
        self.n_cat = len(cfg.cat_cardinalities)
        self.n_numeric = cfg.n_numeric

        dims = cfg.cat_emb_dims or [default_emb_dim(c) for c in cfg.cat_cardinalities]
        if len(dims) != self.n_cat:
            raise ValueError("cat_emb_dims must have one entry per categorical column")
        self.emb_dims: List[int] = list(dims)
        self.embeddings = nn.ModuleList(
            nn.Embedding(card, dim, padding_idx=0)
            for card, dim in zip(cfg.cat_cardinalities, self.emb_dims)
        )

        num_in = cfg.n_numeric * (2 if cfg.numeric_missing_indicator else 1)
        in_dim = sum(self.emb_dims) + num_in
        self.in_dim = in_dim
        self.out_dim = cfg.d_out if in_dim > 0 else 0

        if in_dim == 0:          # no snapshot data at all -> pass-through
            self.mlp = None
            return
        self.mlp = build_mlp(
            in_dim,
            cfg.hidden_dims,
            cfg.d_out,
            activation=cfg.activation,
            dropout=cfg.dropout,
            norm=cfg.norm,
            final_activation=True,
        )

    def forward(
        self,
        cat: Optional[torch.Tensor] = None,
        num: Optional[torch.Tensor] = None,
        num_missing: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.mlp is None:
            return None

        parts = []
        if self.n_cat:
            if cat is None:
                raise ValueError("model configured with categorical snapshot inputs but got None")
            for i, emb in enumerate(self.embeddings):
                parts.append(emb(cat[:, i]))
        if self.n_numeric:
            if num is None:
                raise ValueError("model configured with numeric snapshot inputs but got None")
            parts.append(num)
            if self.cfg.numeric_missing_indicator:
                if num_missing is None:
                    num_missing = torch.zeros_like(num)
                parts.append(num_missing.to(num.dtype))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)
