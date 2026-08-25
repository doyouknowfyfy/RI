"""End-to-end model.

::

    time series (B, C, T)
        -> Conv1d local feature extractor / patch tokenizer   (B, C, N, d)
        -> systematic-missing embedding (additive | concat)
        -> transformer (attention across features)            (B, C, N, d)
        -> masked pooling -> time-series embedding            (B, d_ts)
        -> late fusion with the snapshot embedding            (B, d_ts + d_snap)
        -> regression head                                    (B, n_targets)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .modules.encoder import AxialTransformerEncoder, MaskedPooling
from .modules.head import FusionHead
from .modules.missing import MissingEmbedding
from .modules.snapshot import SnapshotEncoder
from .modules.tokenizer import ConvPatchTokenizer

__all__ = ["TSFormerRegressor"]


class TSFormerRegressor(nn.Module):
    """Conv1d-tokenised transformer regressor with late snapshot fusion.

    Forward accepts a batch ``dict`` (as produced by
    :class:`tsfm.data.TSDataset`) with keys:

    ==================  ==================  =========================================
    key                 shape               meaning
    ==================  ==================  =========================================
    ``x``               ``(B, C, T)``       scaled values, random-missing -> 0
    ``obs_mask``        ``(B, C, T)``       1 = observed
    ``sys_mask``        ``(B, C, T)``       1 = systematically missing
    ``cat``             ``(B, n_cat)``      snapshot categorical codes (0 = missing)
    ``num``             ``(B, n_num)``      snapshot numerics, scaled, missing -> 0
    ``num_missing``     ``(B, n_num)``      1 = snapshot numeric was missing
    ==================  ==================  =========================================
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = ConvPatchTokenizer(
            cfg.tokenizer, use_obs_channel=cfg.missing.add_observed_channel
        )
        self.n_patches = self.tokenizer.n_patches(cfg.seq_len)
        self.n_streams = self.tokenizer.n_token_streams

        self.missing = MissingEmbedding(cfg.missing, cfg.tokenizer.d_model, self.n_streams)
        self.encoder = AxialTransformerEncoder(
            cfg.transformer, cfg.tokenizer.d_model, self.n_streams, self.n_patches
        )
        self.pool = MaskedPooling(cfg.transformer.pool, cfg.tokenizer.d_model, self.n_streams)
        self.snapshot = SnapshotEncoder(cfg.snapshot)
        self.head = FusionHead(cfg.head, self.pool.out_dim, self.snapshot.out_dim)

        self.d_ts = self.pool.out_dim
        self.d_snapshot = self.snapshot.out_dim

    # ------------------------------------------------------------------ #
    def embed_timeseries(
        self, x: torch.Tensor, obs_mask: Optional[torch.Tensor], sys_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the time-series branch; returns ``(embedding, aux)``."""
        tokens = self.tokenizer(x, obs_mask)                      # (B, S, N, d)
        b, s, n, _ = tokens.shape

        if sys_mask is None:
            sys_frac = torch.zeros(b, s, n, device=x.device, dtype=torch.float32)
        else:
            # always pool the mask in fp32: under bf16 autocast a fraction of
            # 0.998 rounds to exactly 1.0 and would wrongly mark the token dead
            sys_frac = self.tokenizer.pool_mask(sys_mask.float())             # (B, C, N)
            if s == 1 and sys_frac.shape[1] != 1:
                # shared (feature-mixing) tokenizer: collapse the feature axis
                sys_frac = sys_frac.mean(dim=1, keepdim=True)

        tokens = self.missing(tokens, sys_frac.to(tokens.dtype))

        if self.cfg.missing.mask_fully_missing_tokens:
            valid = sys_frac < 1.0 - 1e-6
        else:
            valid = torch.ones(b, s, n, dtype=torch.bool, device=x.device)

        tokens = self.encoder(tokens, valid)
        ts_emb, attn = self.pool(tokens, valid)
        return ts_emb, {"tokens": tokens, "valid": valid, "sys_frac": sys_frac, "pool_attn": attn}

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False):
        ts_emb, aux = self.embed_timeseries(
            batch["x"], batch.get("obs_mask"), batch.get("sys_mask")
        )
        snap_emb = self.snapshot(
            batch.get("cat"), batch.get("num"), batch.get("num_missing")
        )
        out = self.head(ts_emb, snap_emb)
        if return_aux:
            aux.update({"ts_emb": ts_emb, "snapshot_emb": snap_emb})
            return out, aux
        return out

    # ------------------------------------------------------------------ #
    def n_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def describe(self) -> Dict[str, Any]:
        return {
            "seq_len": self.cfg.seq_len,
            "n_features": self.cfg.tokenizer.in_channels,
            "n_patches": self.n_patches,
            "n_token_streams": self.n_streams,
            "d_model": self.cfg.tokenizer.d_model,
            "attn_mode": self.cfg.transformer.attn_mode,
            "missing_combine": self.cfg.missing.combine,
            "d_ts_embedding": self.d_ts,
            "d_snapshot_embedding": self.d_snapshot,
            "n_parameters": self.n_parameters(),
        }
