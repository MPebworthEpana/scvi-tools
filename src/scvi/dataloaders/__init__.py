# ZarrDataset / ZarrMultiVIDataModule exports are EXPERIMENTAL.
# for backwards compatibility, this was moved to scvi.data
from scvi.data import AnnTorchDataset

from ._ann_dataloader import AnnDataLoader
from ._anncollection import CollectionAdapter
from ._concat_dataloader import ConcatDataLoader
from ._custom_dataloaders import MappedCollectionDataModule, TileDBDataModule
from ._zarr_datamodule import ZarrMultiVIDataModule
from ._zarr_dataset import (
    ZarrCSRSource,
    ZarrDataset,
    ZarrMatrixSource,
    csr_sources_from_backed_mudata,
    matrix_sources_from_backed_mudata,
)
from ._cuda_prefetch import CUDABatchPrefetcher
from ._data_splitting import (
    DataSplitter,
    DeviceBackedDataSplitter,
    SemiSupervisedDataSplitter,
)
from ._samplers import BatchDistributedSampler
from ._semi_dataloader import SemiSupervisedDataLoader

__all__ = [
    "AnnDataLoader",
    "AnnTorchDataset",
    "CollectionAdapter",
    "ConcatDataLoader",
    "DeviceBackedDataSplitter",
    "SemiSupervisedDataLoader",
    "DataSplitter",
    "SemiSupervisedDataSplitter",
    "BatchDistributedSampler",
    "CUDABatchPrefetcher",
    "MappedCollectionDataModule",
    "TileDBDataModule",
    "ZarrCSRSource",
    "ZarrDataset",
    "ZarrMatrixSource",
    "ZarrMultiVIDataModule",
    "csr_sources_from_backed_mudata",
    "matrix_sources_from_backed_mudata",
]
