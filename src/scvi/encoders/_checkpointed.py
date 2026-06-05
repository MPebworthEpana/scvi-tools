"""Gradient-checkpointing helpers for encoders."""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint


def should_checkpoint(enabled: bool) -> bool:
    return bool(enabled) and torch.is_grad_enabled()


def maybe_checkpoint(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, *, enabled: bool
) -> torch.Tensor:
    if should_checkpoint(enabled):
        return checkpoint(fn, x, use_reentrant=False)
    return fn(x)
