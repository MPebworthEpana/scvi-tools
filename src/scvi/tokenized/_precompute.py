"""Legacy precompute helpers; prefer :class:`~scvi.tokenized._token_store.AtacTokenStore`."""

from __future__ import annotations

import numpy as np

from scvi.data import _constants
from scvi.tokenized._constants import ATAC_TOKEN_CONFIG_KEY
from scvi.tokenized._field import AtacTokenConfigField
from scvi.tokenized._nnz import atac_row_nnz
from scvi.tokenized._token_store import attach_token_store_to_registry, build_token_store


def precompute_atac_token_sequences(
    atac_x,
    coord_table: np.ndarray,
    genomic_rank: np.ndarray,
    max_atac_tokens: int,
    genomic: bool = True,
    chunk_size: int = 4096,
    return_values: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a token store and return legacy padded arrays for backward compatibility."""
    store = build_token_store(
        atac_x,
        coord_table,
        genomic_rank,
        max_encoder_tokens=max_atac_tokens,
        genomic=genomic,
        chunk_size=chunk_size,
        tier="ram",
        **kwargs,
    )
    lengths = store.lengths
    max_len = int(lengths.max(initial=0))
    if max_len == 0:
        max_len = 1
    precomputed_ids = np.zeros((store.n_obs, max_len), dtype=np.int64)
    precomputed_values = np.zeros((store.n_obs, max_len), dtype=np.float32)
    for i in range(store.n_obs):
        ids, vals = store._row_ids_vals(i)
        n = len(ids)
        if n:
            precomputed_ids[i, :n] = ids
            if vals is not None:
                precomputed_values[i, :n] = vals
    if return_values:
        return precomputed_ids, lengths, precomputed_values
    return precomputed_ids, lengths


def store_precomputed_atac_tokens(
    adata_manager,
    precomputed_ids: np.ndarray,
    lengths: np.ndarray,
    precomputed_values: np.ndarray | None = None,
) -> None:
    """Legacy registry patch; prefer :func:`attach_token_store_to_registry`."""
    state = adata_manager._registry[_constants._FIELD_REGISTRIES_KEY][ATAC_TOKEN_CONFIG_KEY][
        _constants._STATE_REGISTRY_KEY
    ]
    state[AtacTokenConfigField.PRECOMPUTED_KEY] = True
    state[AtacTokenConfigField.PRECOMPUTED_IDS_KEY] = np.asarray(precomputed_ids, dtype=np.int64)
    state[AtacTokenConfigField.PRECOMPUTED_LENGTHS_KEY] = np.asarray(lengths, dtype=np.int64)
    if precomputed_values is not None:
        state[AtacTokenConfigField.PRECOMPUTED_VALUES_KEY] = np.asarray(
            precomputed_values, dtype=np.float32
        )


def build_and_attach_token_store(
    adata_manager,
    atac_x,
    *,
    tier: str = "auto",
    out_dir: str | None = None,
    chunk_size: int = 8192,
) -> None:
    """Build a tiered token store and attach it to the ATAC token registry."""
    state = adata_manager._registry[_constants._FIELD_REGISTRIES_KEY][ATAC_TOKEN_CONFIG_KEY][
        _constants._STATE_REGISTRY_KEY
    ]
    store = build_token_store(
        atac_x,
        state[AtacTokenConfigField.COORD_TABLE_KEY],
        state.get(AtacTokenConfigField.GENOMIC_RANK_KEY),
        max_encoder_tokens=state[AtacTokenConfigField.MAX_TOKENS_KEY],
        genomic=state[AtacTokenConfigField.GENOMIC_KEY],
        tier=tier,
        out_dir=out_dir,
        chunk_size=chunk_size,
    )
    attach_token_store_to_registry(state, store, nnz_lengths=atac_row_nnz(atac_x))
