"""Multi-GPU helpers: DistributedDataParallel (DDP), one process per GPU.

Nothing here is active unless the script is launched with ``torchrun`` (which sets
``WORLD_SIZE`` > 1); a plain ``python train.py`` or a notebook stays single-GPU.

    torchrun --standalone --nproc_per_node=8 scripts/train.py ...
"""
from __future__ import annotations

import os
from typing import Any, Tuple

import torch
import torch.distributed as dist


def launched_with_torchrun() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed(use_cuda: bool) -> Tuple[int, int, int]:
    """Join the process group created by torchrun. Returns ``(rank, world_size, local_rank)``;
    ``(0, 1, 0)`` when not launched with torchrun."""
    if not launched_with_torchrun():
        return 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        if use_cuda:
            torch.cuda.set_device(local_rank)        # one GPU per process
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")  # CPU (testing)
    return dist.get_rank(), dist.get_world_size(), local_rank


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_gather_equal(t: torch.Tensor) -> torch.Tensor:
    """Concatenate equally-shaped tensors from all ranks (rank order) on every rank."""
    if not is_distributed():
        return t
    parts = [torch.empty_like(t) for _ in range(get_world_size())]
    dist.all_gather(parts, t.contiguous())
    return torch.cat(parts)


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Make every rank use rank ``src``'s value (keeps control flow identical)."""
    if not is_distributed():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]
