"""Late-fusion regression head."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import HeadConfig
from .mlp import build_mlp

__all__ = ["FusionHead"]


class FusionHead(nn.Module):
    """Concatenate the time-series embedding with the snapshot embedding
    (**late fusion**) and regress.

    Forward
    -------
    ts_emb : ``(B, d_ts)``
    snap_emb : ``(B, d_snap)`` or ``None``
    """

    def __init__(self, cfg: HeadConfig, d_ts: int, d_snapshot: int = 0):
        super().__init__()
        self.cfg = cfg
        self.in_dim = d_ts + d_snapshot
        self.mlp = build_mlp(
            self.in_dim,
            cfg.hidden_dims,
            cfg.n_targets,
            activation=cfg.activation,
            dropout=cfg.dropout,
            norm=cfg.norm,
        )
        self.out_act = {
            "none": nn.Identity(),
            "softplus": nn.Softplus(),
            "sigmoid": nn.Sigmoid(),
        }[cfg.output_activation]

    def forward(self, ts_emb: torch.Tensor, snap_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = ts_emb if snap_emb is None else torch.cat([ts_emb, snap_emb], dim=-1)
        return self.out_act(self.mlp(x))
