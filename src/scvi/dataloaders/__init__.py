# ZarrDataset / ZarrMultiVIDataModule exports are EXPERIMENTAL.
# for backwards compatibility, this was moved to scvi.data
from scvi.data import AnnTorchDataset

from ._ann_dataloader import AnnDataLoader
from ._anncollection import CollectionAdapter
from ._concat_dataloader import ConcatDataLoader
from ._custom_dataloaders import MappedCollectionDataModule, TileDBDataModule
from ._zarr_datamodule import (
    ZARR_DATAMODULE_KWARGS,
    ZarrAnnDataModule,
    ZarrMultiVIDataModule,
    try_auto_zarr_anndata_datamodule,
    try_zarr_inference_dataloader_from_mudata,
)
from ._zarr_dataset import (
    DEFAULT_PREFETCH_QUEUE_DEPTH,
    ZarrCSRSource,
    ZarrDataset,
    ZarrMatrixSource,
    csr_sources_from_backed_mudata,
    matrix_sources_from_backed_anndata,
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
    "DEFAULT_PREFETCH_QUEUE_DEPTH",
    "MappedCollectionDataModule",
    "TileDBDataModule",
    "ZARR_DATAMODULE_KWARGS",
    "ZarrAnnDataModule",
    "ZarrCSRSource",
    "ZarrDataset",
    "ZarrMatrixSource",
    "ZarrMultiVIDataModule",
    "csr_sources_from_backed_mudata",
    "matrix_sources_from_backed_anndata",
    "matrix_sources_from_backed_mudata",
    "try_auto_zarr_anndata_datamodule",
    "try_zarr_inference_dataloader_from_mudata",
]
