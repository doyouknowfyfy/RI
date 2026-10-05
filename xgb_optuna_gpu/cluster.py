"""Dask cluster helpers."""

from __future__ import annotations

from distributed import Client


def make_client(use_gpu: bool = True, n_workers: int | None = None, **cluster_kwargs) -> Client:
    """Start a local Dask cluster and return a connected client.

    GPU: one ``dask_cuda.LocalCUDACluster`` worker per visible GPU (restrict with
    ``CUDA_VISIBLE_DEVICES`` or ``cluster_kwargs['CUDA_VISIBLE_DEVICES']``).
    CPU: a ``distributed.LocalCluster`` (handy for smoke tests without a GPU).

    For a multi-node cluster, skip this helper and connect directly:
    ``Client("tcp://scheduler:8786")``.
    """
    if use_gpu:
        from dask_cuda import LocalCUDACluster

        if n_workers is not None:
            cluster_kwargs.setdefault("n_workers", n_workers)
        # Tip: pass e.g. rmm_pool_size="20GB" to reduce GPU memory fragmentation across
        # many trainings, and device_memory_limit=... to enable spilling.
        cluster = LocalCUDACluster(**cluster_kwargs)
    else:
        from distributed import LocalCluster

        cluster_kwargs.setdefault("n_workers", n_workers or 2)
        cluster_kwargs.setdefault("threads_per_worker", 1)
        cluster = LocalCluster(**cluster_kwargs)
    return Client(cluster)
