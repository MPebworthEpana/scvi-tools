"""Dataset that serves ATAC tokens from a tiered token store."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from scvi import REGISTRY_KEYS
from scvi.data._anntorchdataset import AnnTorchDataset
from scvi.tokenized._constants import ATAC_TOKEN_CONFIG_KEY, ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY
from scvi.tokenized._field import AtacTokenConfigField
from scvi.tokenized._token_store import AtacTokenStore


class SetAnnTorchDataset(AnnTorchDataset):
    """AnnTorchDataset that serves ATAC tokens without loading the wide ATAC matrix."""

    def __init__(
        self,
        adata_manager,
        getitem_tensors: list | dict[str, type] | None = None,
        load_sparse_tensor: bool = False,
    ):
        if getitem_tensors is None:
            getitem_tensors = [
                key
                for key in adata_manager.data_registry.keys()
                if key != REGISTRY_KEYS.ATAC_X_KEY
            ]
        elif isinstance(getitem_tensors, list):
            getitem_tensors = [k for k in getitem_tensors if k != REGISTRY_KEYS.ATAC_X_KEY]
        elif isinstance(getitem_tensors, dict):
            getitem_tensors = {
                k: v for k, v in getitem_tensors.items() if k != REGISTRY_KEYS.ATAC_X_KEY
            }
        super().__init__(
            adata_manager,
            getitem_tensors=getitem_tensors,
            load_sparse_tensor=load_sparse_tensor,
        )

    def _token_config(self) -> dict | None:
        from scvi.data import _constants

        field_registries = self.adata_manager.registry.get(_constants._FIELD_REGISTRIES_KEY, {})
        if ATAC_TOKEN_CONFIG_KEY not in field_registries:
            return None
        return self.adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)

    def _token_store(self) -> AtacTokenStore | None:
        token_cfg = self._token_config()
        if token_cfg is None:
            return None
        store = token_cfg.get(AtacTokenConfigField.TOKEN_STORE_KEY)
        if store is not None:
            return store
        handle = token_cfg.get(AtacTokenConfigField.TOKEN_STORE_HANDLE_KEY)
        if handle is not None and handle.get("tier") == "mmap":
            store = AtacTokenStore.from_handle(handle)
            token_cfg[AtacTokenConfigField.TOKEN_STORE_KEY] = store
            return store
        return None

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
            row_indexes = np.asarray([int(indexes)], dtype=np.int64)
        else:
            row_indexes = np.asarray(indexes, dtype=np.int64)

        store = self._token_store()
        if store is not None:
            if store.tier == "gpu":
                return data_map
            ids, mask = store.gather(row_indexes, for_encoder=True)
            data_map[ATAC_TOKEN_IDS_KEY] = ids
            data_map[ATAC_TOKEN_MASK_KEY] = mask
            return data_map

        if token_cfg.get(AtacTokenConfigField.PRECOMPUTED_KEY):
            ids, mask = self._batch_from_precomputed(
                token_cfg[AtacTokenConfigField.PRECOMPUTED_IDS_KEY],
                token_cfg[AtacTokenConfigField.PRECOMPUTED_LENGTHS_KEY],
                row_indexes,
            )
            data_map[ATAC_TOKEN_IDS_KEY] = ids
            data_map[ATAC_TOKEN_MASK_KEY] = mask
            return data_map

        raise RuntimeError(
            "SETVI requires an ATAC token store. Call SETVI.setup_mudata with an ATAC modality."
        )
