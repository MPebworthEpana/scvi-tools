"""Dataset that streams ATAC tokens from CSR rows at fetch time."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy import sparse

from scvi import REGISTRY_KEYS
from scvi.data._anntorchdataset import AnnTorchDataset
from scvi.data.fields._atac_token_field import AtacTokenConfigField
from scvi.encoders._constants import ATAC_TOKEN_CONFIG_KEY, ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY
from scvi.encoders._csr_batch_tokenize import csr_batch_to_tokens


class SetAnnTorchDataset(AnnTorchDataset):
    """AnnTorchDataset that tokenizes open peaks from CSR ATAC rows on demand."""

    def _token_config(self) -> dict | None:
        from scvi.data import _constants

        field_registries = self.adata_manager.registry.get(_constants._FIELD_REGISTRIES_KEY, {})
        if ATAC_TOKEN_CONFIG_KEY not in field_registries:
            return None
        return self.adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)

    def _csr_tokens_from_rows(self, row_indexes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        token_cfg = self._token_config()
        if token_cfg is None:
            raise RuntimeError("SETVI requires ATAC token config from setup_mudata.")
        row_indexes = np.asarray(row_indexes, dtype=np.int64)
        atac_x = self.adata_manager.get_from_registry(REGISTRY_KEYS.ATAC_X_KEY)
        if isinstance(atac_x, np.ndarray):
            batch_csr = sparse.csr_matrix(atac_x[row_indexes])
        elif sparse.issparse(atac_x):
            batch_csr = atac_x[row_indexes]
        else:
            batch_csr = sparse.csr_matrix(atac_x[row_indexes])
        return csr_batch_to_tokens(
            batch_csr,
            token_cfg[AtacTokenConfigField.COORD_TABLE_KEY],
            token_cfg[AtacTokenConfigField.MAX_TOKENS_KEY],
            genomic=token_cfg[AtacTokenConfigField.GENOMIC_KEY],
            genomic_rank=token_cfg.get(AtacTokenConfigField.GENOMIC_RANK_KEY),
        )

    @staticmethod
    def _batch_from_precomputed(
        precomputed_ids: np.ndarray,
        lengths: np.ndarray,
        row_indexes: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        row_indexes = np.asarray(row_indexes, dtype=np.int64)
        batch_ids = precomputed_ids[row_indexes]
        if batch_ids.ndim == 1:
            batch_ids = batch_ids[np.newaxis, :]
        batch_lengths = lengths[row_indexes]
        max_len = max(int(batch_lengths.max(initial=0)), 1)
        ids = batch_ids[:, :max_len].copy()
        mask = np.zeros((len(row_indexes), max_len), dtype=bool)
        for i, n in enumerate(batch_lengths):
            n = int(n)
            if n:
                mask[i, :n] = True
        return ids, mask

    def __getitem__(
        self, indexes: int | list[int] | slice
    ) -> dict[str, np.ndarray | torch.Tensor]:
        data_map = super().__getitem__(indexes)
        token_cfg = self._token_config()
        if token_cfg is None:
            return data_map

        if not isinstance(indexes, (list, slice, np.ndarray)):
            row_indexes = [int(indexes)]
        else:
            row_indexes = indexes

        if token_cfg.get(AtacTokenConfigField.PRECOMPUTED_KEY):
            ids, mask = self._batch_from_precomputed(
                token_cfg[AtacTokenConfigField.PRECOMPUTED_IDS_KEY],
                token_cfg[AtacTokenConfigField.PRECOMPUTED_LENGTHS_KEY],
                row_indexes,
            )
        else:
            ids, mask = self._csr_tokens_from_rows(row_indexes)

        data_map[ATAC_TOKEN_IDS_KEY] = ids
        data_map[ATAC_TOKEN_MASK_KEY] = mask
        return data_map
