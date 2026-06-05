"""Dataset that emits padded ATAC token ids from CSR batch slices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy import sparse

from scvi import REGISTRY_KEYS
from scvi.data import _constants
from scvi.data._anntorchdataset import AnnTorchDataset
from scvi.data.fields._atac_token_field import AtacTokenConfigField
from scvi.encoders._constants import ATAC_TOKEN_CONFIG_KEY, ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY
from scvi.encoders._csr_batch_tokenize import csr_batch_to_tokens
from scvi.encoders._tokenizers import tokenize_atac


class MambaAnnTorchDataset(AnnTorchDataset):
    """AnnTorchDataset that adds ATAC token tensors from sparse CSR rows."""

    def _token_config(self) -> dict | None:
        field_registries = self.adata_manager.registry.get(_constants._FIELD_REGISTRIES_KEY, {})
        if ATAC_TOKEN_CONFIG_KEY not in field_registries:
            return None
        return self.adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)

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
            data_map[ATAC_TOKEN_IDS_KEY] = ids
            data_map[ATAC_TOKEN_MASK_KEY] = mask
            return data_map

        atac_key = token_cfg.get("atac_source_key", REGISTRY_KEYS.ATAC_X_KEY)
        atac_data = self.adata_manager.get_from_registry(atac_key)
        sliced = atac_data[row_indexes]
        coord_table = token_cfg["coord_table"]
        max_tokens = token_cfg["max_atac_tokens"]
        genomic = token_cfg["genomic"]
        genomic_rank = token_cfg.get(AtacTokenConfigField.GENOMIC_RANK_KEY)

        if sparse.issparse(sliced):
            ids, mask = csr_batch_to_tokens(
                sliced,
                coord_table,
                max_tokens,
                genomic=genomic,
                genomic_rank=genomic_rank,
            )
        else:
            ids_list = []
            mask_list = []
            for i in range(sliced.shape[0]):
                row = sliced[i]
                indices = np.where(row > 0)[0].astype(np.int64)
                values = row[indices].astype(np.float32)
                if len(indices) == 0:
                    tok_ids = np.array([], dtype=np.int64)
                else:
                    tok_ids = tokenize_atac(
                        indices,
                        values,
                        max_tokens,
                        coord_table,
                        genomic=genomic,
                        genomic_rank=genomic_rank,
                    )["ids"]
                ids_list.append(tok_ids)
            max_len = max((len(x) for x in ids_list), default=0)
            ids = np.zeros((len(ids_list), max_len), dtype=np.int64)
            mask = np.zeros((len(ids_list), max_len), dtype=bool)
            for i, peak_ids in enumerate(ids_list):
                n = len(peak_ids)
                if n:
                    ids[i, :n] = peak_ids
                    mask[i, :n] = True

        data_map[ATAC_TOKEN_IDS_KEY] = ids
        data_map[ATAC_TOKEN_MASK_KEY] = mask
        return data_map
