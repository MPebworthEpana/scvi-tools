"""Data splitter for Mamba ATAC token batches."""

from __future__ import annotations

import numpy as np
import torch

from scvi import REGISTRY_KEYS, settings
from scvi.data import _constants
from scvi.dataloaders._data_splitting import DataSplitter
from scvi.dataloaders._length_bucket_sampler import LengthBucketedBatchSampler
from scvi.dataloaders._mamba_dataloader import MambaAnnDataLoader
from scvi.data.fields._atac_token_field import AtacTokenConfigField
from scvi.encoders._constants import ATAC_TOKEN_CONFIG_KEY, ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY
from scvi.encoders._precompute_tokens import atac_row_nnz


class MambaDataSplitter(DataSplitter):
    """Uses MambaAnnDataLoader and preserves ATAC token tensors on transfer."""

    data_loader_cls = MambaAnnDataLoader

    def __init__(
        self,
        adata_manager,
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        pin_memory: bool = False,
        external_indexing: list[np.array, np.array, np.array] | None = None,
        atac_length_bucketing: bool = False,
        bucket_mult: int = 50,
        **kwargs,
    ):
        self.atac_length_bucketing = atac_length_bucketing
        self.bucket_mult = bucket_mult
        super().__init__(
            adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            load_sparse_tensor=load_sparse_tensor,
            pin_memory=pin_memory,
            external_indexing=external_indexing,
            **kwargs,
        )

    def _token_config(self) -> dict | None:
        field_registries = self.adata_manager.registry.get(_constants._FIELD_REGISTRIES_KEY, {})
        if ATAC_TOKEN_CONFIG_KEY not in field_registries:
            return None
        return dict(self.adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY))

    def _atac_lengths_for_indices(self, indices: np.ndarray) -> np.ndarray | None:
        token_cfg = self._token_config()
        if token_cfg is None:
            return None
        indices = np.asarray(indices, dtype=np.int64)
        if token_cfg.get(AtacTokenConfigField.PRECOMPUTED_KEY):
            return token_cfg[AtacTokenConfigField.PRECOMPUTED_LENGTHS_KEY][indices]
        atac_key = token_cfg.get("atac_source_key", REGISTRY_KEYS.ATAC_X_KEY)
        atac = self.adata_manager.get_from_registry(atac_key)
        return atac_row_nnz(atac)[indices]

    def _bucketed_dataloader(self, indices, *, shuffle: bool, drop_last: bool):
        lengths = self._atac_lengths_for_indices(indices)
        if lengths is None:
            if shuffle:
                return super().train_dataloader()
            return super().val_dataloader()
        batch_size = self.data_loader_kwargs.get("batch_size") or settings.batch_size
        batch_sampler = LengthBucketedBatchSampler(
            lengths,
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=settings.seed,
            bucket_mult=self.bucket_mult,
        )
        loader_kwargs = {
            k: v
            for k, v in self.data_loader_kwargs.items()
            if k not in {"batch_size", "drop_last", "shuffle"}
        }
        return self.data_loader_cls(
            self.adata_manager,
            indices=indices,
            shuffle=False,
            batch_sampler=batch_sampler,
            load_sparse_tensor=self.load_sparse_tensor,
            pin_memory=self.pin_memory,
            **loader_kwargs,
        )

    def train_dataloader(self):
        if self.atac_length_bucketing and len(self.train_idx) > 0:
            return self._bucketed_dataloader(
                self.train_idx, shuffle=True, drop_last=self.drop_last
            )
        return super().train_dataloader()

    def val_dataloader(self):
        if len(self.val_idx) == 0:
            return super().val_dataloader()
        if self.atac_length_bucketing:
            return self._bucketed_dataloader(self.val_idx, shuffle=False, drop_last=False)
        return super().val_dataloader()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        saved = {}
        for key in (ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY):
            if key in batch:
                saved[key] = batch.pop(key)
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        batch.update(saved)
        return batch
