from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader

from scvi.module.base._decorators import _move_data_to_device

logger = logging.getLogger(__name__)


def batch_tensors_on_device(batch: Any, device: torch.device) -> bool:
    """Return True if every tensor leaf in ``batch`` is already on ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.device == device

    if isinstance(batch, Mapping):
        if not batch:
            return True
        return all(batch_tensors_on_device(v, device) for v in batch.values())

    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        if not batch:
            return True
        return all(batch_tensors_on_device(v, device) for v in batch)

    if isinstance(batch, Sequence) and not isinstance(batch, str):
        if not batch:
            return True
        return all(batch_tensors_on_device(v, device) for v in batch)

    return True


def transfer_batch_to_cuda(batch: Any, device: torch.device) -> Any:
    """Move all tensor leaves in ``batch`` to ``device`` with non-blocking copies."""
    return _move_data_to_device(batch, device)


class CUDABatchPrefetcher:
    """Overlap H2D transfer with training using a side CUDA stream.

    Parameters
    ----------
    loader
        Source data loader yielding CPU (preferably pin_memory) batches.
    queue_depth
        Number of GPU batches to keep in flight. Default is 2 (double-buffer).
    device
        Target CUDA device. Resolved lazily on first iteration if ``None``.
    """

    def __init__(
        self,
        loader: DataLoader,
        queue_depth: int = 2,
        device: torch.device | None = None,
    ):
        if queue_depth < 1:
            raise ValueError("queue_depth must be at least 1.")
        self.loader = loader
        self.queue_depth = queue_depth
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        if not torch.cuda.is_available():
            yield from self.loader
            return

        device = self.device or torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.Stream()
        loader_iter = iter(self.loader)

        def prefetch() -> Any | None:
            try:
                batch = next(loader_iter)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                return transfer_batch_to_cuda(batch, device)

        queue: list[Any] = []
        for _ in range(self.queue_depth):
            batch = prefetch()
            if batch is None:
                break
            queue.append(batch)

        while queue:
            torch.cuda.current_stream().wait_stream(stream)
            batch = queue.pop(0)
            next_batch = prefetch()
            if next_batch is not None:
                queue.append(next_batch)
            yield batch


def maybe_wrap_cuda_prefetch(
    loader: DataLoader,
    *,
    prefetch_to_gpu: bool,
    cuda_queue_depth: int,
    pin_memory: bool,
    load_sparse_tensor: bool,
) -> DataLoader | CUDABatchPrefetcher:
    """Return ``loader`` wrapped in :class:`CUDABatchPrefetcher` when enabled."""
    if not prefetch_to_gpu:
        return loader

    if load_sparse_tensor:
        raise ValueError(
            "prefetch_to_gpu=True requires load_sparse_tensor=False so batches are "
            "densified on CPU before H2D transfer."
        )

    if not pin_memory:
        raise ValueError(
            "prefetch_to_gpu=True requires pin_memory=True for effective non-blocking H2D."
        )

    if not torch.cuda.is_available():
        logger.warning(
            "prefetch_to_gpu=True but CUDA is unavailable; using the unwrapped data loader."
        )
        return loader

    return CUDABatchPrefetcher(loader, queue_depth=cuda_queue_depth)
