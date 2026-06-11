"""Batch CSR row slicing -> padded ATAC token arrays."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from scvi.tokenized._tokenizers import _topk_by_value, tokenize_atac


def _torch_csr_to_scipy(atac_batch: object) -> sparse.csr_matrix:
    import torch

    if not isinstance(atac_batch, torch.Tensor) or atac_batch.layout != torch.sparse_csr:
        raise TypeError("Expected a torch sparse CSR tensor.")
    crow = atac_batch.crow_indices().cpu().numpy()
    col = atac_batch.col_indices().cpu().numpy()
    val = atac_batch.values().cpu().numpy()
    return sparse.csr_matrix((val, col, crow), shape=tuple(atac_batch.shape))


def _as_scipy_csr(x_batch) -> sparse.csr_matrix:
    import torch

    if isinstance(x_batch, torch.Tensor) and x_batch.layout == torch.sparse_csr:
        return _torch_csr_to_scipy(x_batch)
    if isinstance(x_batch, np.ndarray):
        return sparse.csr_matrix(x_batch)
    if sparse.issparse(x_batch):
        return x_batch.tocsr()
    raise TypeError("csr_batch_to_tokens expects a scipy CSR matrix or torch sparse CSR tensor.")


def _sort_order(
    indices: np.ndarray,
    row_ids: np.ndarray,
    coord_table: np.ndarray,
    genomic: bool,
    genomic_rank: np.ndarray | None,
) -> np.ndarray:
    if genomic and genomic_rank is not None:
        return np.lexsort((genomic_rank[indices], row_ids))
    chrom = coord_table[indices, 0]
    start = coord_table[indices, 1]
    return np.lexsort((start, chrom, row_ids))


def _scatter_row_tokens(
    row_tokens: list[np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_len = max((len(t) for t in row_tokens), default=0)
    if max_len == 0:
        max_len = 1
    ids = np.zeros((batch_size, max_len), dtype=np.int64)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    for i, peak_ids in enumerate(row_tokens):
        n = len(peak_ids)
        if n:
            ids[i, :n] = peak_ids
            mask[i, :n] = True
    return ids, mask


def csr_batch_to_tokens(
    x_batch,
    coord_table: np.ndarray,
    max_tokens: int,
    genomic: bool = True,
    genomic_rank: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a CSR batch slice ``(B, n_peaks)`` into padded token ids and masks."""
    x_batch = _as_scipy_csr(x_batch)
    batch_size = x_batch.shape[0]
    indptr = x_batch.indptr
    indices = x_batch.indices.astype(np.int64)
    data = x_batch.data.astype(np.float32)
    row_lens = np.diff(indptr)

    row_tokens: list[np.ndarray] = [np.array([], dtype=np.int64) for _ in range(batch_size)]

    heavy_rows = np.flatnonzero(row_lens > max_tokens)
    for i in heavy_rows:
        start, end = indptr[i], indptr[i + 1]
        tok = tokenize_atac(
            indices[start:end],
            data[start:end],
            max_tokens,
            coord_table,
            genomic=genomic,
            genomic_rank=genomic_rank,
        )
        row_tokens[i] = tok["ids"]

    light_rows = np.flatnonzero(row_lens <= max_tokens)
    if light_rows.size > 0:
        nnz_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        for i in light_rows:
            start, end = indptr[i], indptr[i + 1]
            if start == end:
                continue
            row_idx = indices[start:end]
            row_val = data[start:end]
            if not genomic:
                row_idx, _ = _topk_by_value(row_idx, row_val, max_tokens)
            nnz_parts.append(row_idx)
            row_parts.append(np.full(len(row_idx), i, dtype=np.int64))

        if nnz_parts:
            all_idx = np.concatenate(nnz_parts)
            all_row = np.concatenate(row_parts)
            order = _sort_order(all_idx, all_row, coord_table, genomic, genomic_rank)
            sorted_idx = all_idx[order]
            sorted_row = all_row[order]
            boundaries = np.flatnonzero(np.diff(sorted_row)) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [sorted_row.size]))
            for s, e in zip(starts, ends):
                row_tokens[int(sorted_row[s])] = sorted_idx[s:e]

    return _scatter_row_tokens(row_tokens, batch_size)
