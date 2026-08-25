"""Small shared MLP builder."""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn

__all__ = ["build_mlp", "get_activation"]


def get_activation(name: str) -> nn.Module:
    return {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}[name]


def _norm(kind: str, dim: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str = "gelu",
    dropout: float = 0.0,
    norm: str = "layer",
    final_activation: bool = False,
) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), _norm(norm, h), get_activation(activation)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    if final_activation:
        layers.append(get_activation(activation))
    return nn.Sequential(*layers)
