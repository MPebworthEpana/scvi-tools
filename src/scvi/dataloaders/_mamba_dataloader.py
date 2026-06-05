"""AnnDataLoader that uses MambaAnnTorchDataset."""

from __future__ import annotations

import copy

import numpy as np
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from scvi import settings
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from scvi.dataloaders._samplers import BatchDistributedSampler


class MambaAnnDataLoader(AnnDataLoader):
    """Data loader that emits ATAC token ids/masks alongside standard tensors."""

    def __init__(
        self,
        adata_manager,
        indices=None,
        batch_size: int = 128,
        shuffle: bool = False,
        sampler=None,
        batch_sampler=None,
        drop_last: bool = False,
        drop_dataset_tail: bool = False,
        data_and_attributes=None,
        iter_ndarray: bool = False,
        distributed_sampler: bool = False,
        load_sparse_tensor: bool = False,
        **kwargs,
    ):
        if indices is None:
            indices = np.arange(adata_manager.adata.shape[0])
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)
        self.indices = indices
        self.dataset = adata_manager.create_mamba_torch_dataset(
            indices=indices,
            data_and_attributes=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
        )
        if "num_workers" not in kwargs:
            kwargs["num_workers"] = settings.dl_num_workers
        if "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = settings.dl_persistent_workers

        self.kwargs = copy.deepcopy(kwargs)

        if batch_sampler is not None:
            sampler = batch_sampler

        if sampler is not None and distributed_sampler:
            raise ValueError("Cannot specify both `sampler` and `distributed_sampler`.")
        elif sampler is None:
            if not distributed_sampler:
                sampler_cls = SequentialSampler if not shuffle else RandomSampler
                sampler = BatchSampler(
                    sampler=sampler_cls(self.dataset),
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
            else:
                sampler = BatchDistributedSampler(
                    self.dataset,
                    batch_size=batch_size,
                    drop_last=drop_last,
                    drop_dataset_tail=drop_dataset_tail,
                    shuffle=shuffle,
                )
        self.kwargs.update({"batch_size": None, "shuffle": False, "sampler": sampler})

        if iter_ndarray:
            self.kwargs["collate_fn"] = lambda x: x

        DataLoader.__init__(self, self.dataset, **self.kwargs)
