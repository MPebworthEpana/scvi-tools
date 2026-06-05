"""Batch CSR row slicing -> padded ATAC token arrays."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from scvi.encoders._tokenizers import tokenize_atac


def csr_batch_to_tokens(
    x_batch: sparse.spmatrix,
    coord_table: np.ndarray,
    max_tokens: int,
    genomic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a CSR batch slice ``(B, n_peaks)`` into padded token ids and masks."""
    if not sparse.issparse(x_batch):
        raise TypeError("csr_batch_to_tokens expects a sparse matrix batch slice.")
    x_batch = x_batch.tocsr()
    batch_size = x_batch.shape[0]
    row_tokens = []
    max_len = 0
    for i in range(batch_size):
        row = x_batch.getrow(i)
        indices = row.indices.astype(np.int64)
        values = row.data.astype(np.float32)
        if len(indices) == 0:
            tok = {"ids": np.array([], dtype=np.int64)}
        else:
            tok = tokenize_atac(indices, values, max_tokens, coord_table, genomic=genomic)
        row_tokens.append(tok["ids"])
        max_len = max(max_len, len(tok["ids"]))

    ids = np.zeros((batch_size, max_len), dtype=np.int64)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    for i, peak_ids in enumerate(row_tokens):
        n = len(peak_ids)
        if n:
            ids[i, :n] = peak_ids
            mask[i, :n] = True
    if max_len == 0:
        max_len = 1
        ids = np.zeros((batch_size, max_len), dtype=np.int64)
        mask = np.zeros((batch_size, max_len), dtype=bool)
    return ids, mask
