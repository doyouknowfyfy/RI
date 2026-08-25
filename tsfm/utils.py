"""Small helpers: seeding, device resolution, parameter counting."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_seed", "resolve_device", "count_parameters", "human_bytes"]


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed python / numpy / torch (all devices)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(device: str = "cuda", verbose: bool = True) -> torch.device:
    """Return a usable device, falling back to CPU with a warning."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        if verbose:
            print("[tsfm] CUDA requested but not available -- falling back to CPU.")
        return torch.device("cpu")
    dev = torch.device(device)
    if verbose and dev.type == "cuda":
        idx = dev.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory
        print(f"[tsfm] using {dev} ({name}, {human_bytes(total)})")
    return dev


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"
