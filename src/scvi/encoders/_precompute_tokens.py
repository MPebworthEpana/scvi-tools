"""One-time precomputation of per-cell ATAC token sequences."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse

from scvi.data import _constants
from scvi.data.fields._atac_token_field import AtacTokenConfigField
from scvi.encoders._constants import ATAC_TOKEN_CONFIG_KEY
from scvi.encoders._tokenizers import tokenize_atac


def atac_row_nnz(atac_x: sparse.spmatrix | np.ndarray) -> np.ndarray:
    """Return per-cell ATAC non-zero counts from a CSR matrix or dense array."""
    if isinstance(atac_x, np.ndarray):
        return (atac_x != 0).sum(axis=1).astype(np.int64)
    if not sparse.issparse(atac_x):
        atac_x = sparse.csr_matrix(atac_x)
    atac_x = atac_x.tocsr()
    return np.diff(atac_x.indptr).astype(np.int64)


def _tokenize_atac_row(
    row_index: int,
    atac_x: sparse.csr_matrix,
    coord_table: np.ndarray,
    genomic_rank: np.ndarray,
    max_atac_tokens: int,
    genomic: bool,
) -> np.ndarray:
    row = atac_x.getrow(row_index)
    indices = row.indices.astype(np.int64)
    values = row.data.astype(np.float32)
    if len(indices) == 0:
        return np.array([], dtype=np.int64)
    return tokenize_atac(
        indices,
        values,
        max_atac_tokens,
        coord_table,
        genomic=genomic,
        genomic_rank=genomic_rank,
    )["ids"]


def precompute_atac_token_sequences(
    atac_x: sparse.spmatrix,
    coord_table: np.ndarray,
    genomic_rank: np.ndarray,
    max_atac_tokens: int,
    genomic: bool = True,
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute genomically sorted open-peak token ids for every cell.

    Returns padded ``(n_obs, max_len)`` ids and per-cell lengths. Only open peaks
    are stored; missing peaks are excluded from encoder tokens.
    """
    if not sparse.issparse(atac_x):
        atac_x = sparse.csr_matrix(atac_x)
    atac_x = atac_x.tocsr()
    n_obs = atac_x.shape[0]
    if n_jobs is None:
        n_jobs = min(8, os.cpu_count() or 1)
    n_jobs = max(1, int(n_jobs))

    if n_jobs == 1 or n_obs < 256:
        ids_list = [
            _tokenize_atac_row(
                i, atac_x, coord_table, genomic_rank, max_atac_tokens, genomic
            )
            for i in range(n_obs)
        ]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            ids_list = list(
                executor.map(
                    lambda i: _tokenize_atac_row(
                        i, atac_x, coord_table, genomic_rank, max_atac_tokens, genomic
                    ),
                    range(n_obs),
                )
            )

    lengths = np.array([len(peak_ids) for peak_ids in ids_list], dtype=np.int64)
    max_len = int(lengths.max(initial=0))
    if max_len == 0:
        max_len = 1
    precomputed_ids = np.zeros((n_obs, max_len), dtype=np.int64)
    for i, peak_ids in enumerate(ids_list):
        n = len(peak_ids)
        if n:
            precomputed_ids[i, :n] = peak_ids
    return precomputed_ids, lengths


def store_precomputed_atac_tokens(adata_manager, precomputed_ids: np.ndarray, lengths: np.ndarray) -> None:
    """Patch the ATAC token state registry with precomputed sequences."""
    state = adata_manager._registry[_constants._FIELD_REGISTRIES_KEY][ATAC_TOKEN_CONFIG_KEY][
        _constants._STATE_REGISTRY_KEY
    ]
    state[AtacTokenConfigField.PRECOMPUTED_KEY] = True
    state[AtacTokenConfigField.PRECOMPUTED_IDS_KEY] = np.asarray(precomputed_ids, dtype=np.int64)
    state[AtacTokenConfigField.PRECOMPUTED_LENGTHS_KEY] = np.asarray(lengths, dtype=np.int64)
