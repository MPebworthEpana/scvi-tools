"""Sparse row -> token arrays for ATAC peaks."""

from __future__ import annotations

import numpy as np


def _topk_by_value(indices: np.ndarray, values: np.ndarray, max_tokens: int):
    if len(indices) <= max_tokens:
        return indices, values
    keep = np.argpartition(values, -max_tokens)[-max_tokens:]
    return indices[keep], values[keep]


def tokenize_atac(
    indices: np.ndarray,
    values: np.ndarray,
    max_tokens: int,
    coord_table: np.ndarray,
    genomic: bool = True,
) -> dict:
    """ATAC: non-zero peaks ordered by ``(chrom, start)``."""
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)

    if not genomic or len(indices) > max_tokens:
        indices, values = _topk_by_value(indices, values, max_tokens)

    chrom = coord_table[indices, 0]
    start = coord_table[indices, 1]
    end = coord_table[indices, 2]
    order = np.lexsort((start, chrom))

    return {
        "ids": indices[order],
        "values": values[order],
        "chrom": chrom[order].astype(np.int64),
        "pos": start[order].astype(np.int64),
        "length": (end - start)[order].astype(np.int64),
    }
