"""Per-cell ATAC non-zero counts for CSR streaming length bucketing."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def atac_row_nnz(atac_x: sparse.spmatrix | np.ndarray, *, chunk_size: int = 8192) -> np.ndarray:
    """Return per-cell ATAC non-zero counts without materializing the full matrix."""
    if isinstance(atac_x, np.ndarray):
        return (atac_x != 0).sum(axis=1).astype(np.int64)
    if sparse.issparse(atac_x):
        csr = atac_x.tocsr()
        if csr.has_sorted_indices:
            return np.diff(csr.indptr).astype(np.int64)
    n_obs = int(atac_x.shape[0])
    out = np.empty(n_obs, dtype=np.int64)
    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        chunk = atac_x[start:end]
        if sparse.issparse(chunk):
            chunk = chunk.tocsr()
            out[start:end] = np.diff(chunk.indptr)
        else:
            chunk = sparse.csr_matrix(chunk)
            out[start:end] = np.diff(chunk.indptr)
    return out
