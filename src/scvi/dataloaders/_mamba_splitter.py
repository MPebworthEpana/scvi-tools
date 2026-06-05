"""Data splitter for Mamba ATAC token batches."""

from __future__ import annotations

import torch

from scvi.dataloaders._data_splitting import DataSplitter
from scvi.dataloaders._mamba_dataloader import MambaAnnDataLoader
from scvi.encoders._constants import ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY


class MambaDataSplitter(DataSplitter):
    """Uses MambaAnnDataLoader and preserves ATAC token tensors on transfer."""

    data_loader_cls = MambaAnnDataLoader

    def on_after_batch_transfer(self, batch, dataloader_idx):
        saved = {}
        for key in (ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY):
            if key in batch:
                saved[key] = batch.pop(key)
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        batch.update(saved)
        return batch
