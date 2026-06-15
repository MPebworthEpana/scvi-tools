"""Lightning DataModule for MultiVI streaming from zarr-backed CSR stores."""

from __future__ import annotations

import logging
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from mudata import MuData
from torch.utils.data import DataLoader

from scvi import REGISTRY_KEYS, settings
from scvi.data._manager import AnnDataManager
from scvi.data._utils import get_anndata_attribute
from scvi.dataloaders._data_splitting import validate_data_split
from scvi.dataloaders._zarr_dataset import (
    ZarrCSRSource,
    ZarrDataset,
    ZarrMatrixSource,
    _identity_collate,
    csr_sources_from_backed_mudata,
    matrix_sources_from_backed_mudata,
)

logger = logging.getLogger(__name__)

_UNSUPPORTED_COVARIATE_MSG = (
    "ZarrMultiVIDataModule does not yet support categorical or continuous "
    "covariates. Pass models without covariate keys, or use the default "
    "AnnData DataSplitter instead."
)


class ZarrMultiVIDataModule(pl.LightningDataModule):
    """EXPERIMENTAL: Stream paired RNA/ATAC batches from zarr-backed CSR or dense stores into MultiVI.

    Use :meth:`from_backed_mudata` to construct this datamodule. The caller must
    supply a MuData whose modality matrices are zarr-backed (CSR ``CSRDataset`` or
    dense zarr arrays, e.g. from ``mudata.write_zarr`` with backed reopen).
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        *,
        matrix_sources: dict[str, ZarrMatrixSource] | None = None,
        csr_sources: dict[str, ZarrMatrixSource] | None = None,
        n_obs: int,
        store_dir: Path | str | None = None,
        train_size: float = 0.9,
        validation_size: float | None = None,
        batch_size: int = 128,
        block_size: int = 4096,
        shuffle_buffer_blocks: int = 16,
        num_workers: int | None = None,
        pin_memory: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        persistent_workers: bool | None = None,
    ) -> None:
        super().__init__()
        if matrix_sources is None:
            matrix_sources = csr_sources
        if matrix_sources is None:
            raise ValueError("matrix_sources (or csr_sources) is required.")

        self.adata_manager = adata_manager
        self.matrix_sources = matrix_sources
        self.store_dir = Path(store_dir) if store_dir is not None else None
        self.n_obs = int(n_obs)
        self.train_size = train_size
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle_buffer_blocks = shuffle_buffer_blocks
        self.num_workers = settings.dl_num_workers if num_workers is None else num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.drop_last = drop_last
        self._val_persistent_workers = (
            settings.dl_persistent_workers if persistent_workers is None else persistent_workers
        )

        self._train_dataset: ZarrDataset | None = None
        self._val_dataset: ZarrDataset | None = None

        self._reject_unsupported_covariates()
        self.obs_tensors = self._build_obs_tensors()
        self._split_indices()

    @classmethod
    def from_backed_mudata(
        cls,
        mdata: MuData,
        adata_manager: AnnDataManager,
        *,
        registry_map: dict[str, str] | None = None,
        matrix_layout: str = "csr",
        x_suffix: str = "",
        store_dir: Path | str | None = None,
        **kwargs,
    ) -> ZarrMultiVIDataModule:
        """EXPERIMENTAL: Create a datamodule that streams from an already-backed MuData.

        Matrix paths are extracted once in the main process; workers reopen zarr
        handles independently.
        """
        if registry_map is None:
            registry_map = {}
            for key in (REGISTRY_KEYS.X_KEY, REGISTRY_KEYS.ATAC_X_KEY):
                if key not in adata_manager.data_registry:
                    continue
                data_loc = adata_manager.data_registry[key]
                mod_key = getattr(data_loc, "mod_key", None)
                if mod_key is None:
                    raise ValueError(
                        f"Registry key {key!r} has no mod_key; pass registry_map explicitly."
                    )
                registry_map[key] = mod_key
            if not registry_map:
                raise ValueError(
                    "Could not infer registry_map from adata_manager. "
                    "Pass registry_map={REGISTRY_KEYS.X_KEY: 'RNA', ...} explicitly."
                )

        layout = "dense" if matrix_layout == "dense" else "csr"

        if store_dir is None and layout == "dense":
            candidate = getattr(mdata, "filename", None)
            if isinstance(candidate, (str, Path)):
                store_dir = candidate

        source_kwargs: dict = {
            "n_obs": mdata.n_obs,
            "layout": layout,
        }
        if layout == "dense":
            if not x_suffix:
                x_suffix = "_dense"
            source_kwargs["x_suffix"] = x_suffix
            if store_dir is None:
                raise ValueError(
                    "store_dir is required for dense matrix_layout (path to the zarr store)."
                )
            source_kwargs["store_dir"] = store_dir

        sources = matrix_sources_from_backed_mudata(mdata, registry_map, **source_kwargs)
        resolved_store_dir = Path(store_dir) if store_dir is not None else None
        return cls(
            adata_manager,
            matrix_sources=sources,
            n_obs=mdata.n_obs,
            store_dir=resolved_store_dir,
            **kwargs,
        )

    def _reject_unsupported_covariates(self) -> None:
        registry = self.adata_manager.data_registry
        if REGISTRY_KEYS.CAT_COVS_KEY in registry or REGISTRY_KEYS.CONT_COVS_KEY in registry:
            raise NotImplementedError(_UNSUPPORTED_COVARIATE_MSG)
        if REGISTRY_KEYS.SIZE_FACTOR_KEY in registry:
            data_loc = registry[REGISTRY_KEYS.SIZE_FACTOR_KEY]
            values = get_anndata_attribute(
                self.adata_manager.adata,
                data_loc.attr_name,
                data_loc.attr_key,
                mod_key=getattr(data_loc, "mod_key", None),
            )
            arr = np.asarray(values)
            if arr.ndim > 1 and arr.shape[1] > 1:
                raise NotImplementedError(
                    "ZarrMultiVIDataModule does not yet support multi-column size factors."
                )

    def _registry_array(self, registry_key: str, *, dtype) -> np.ndarray:
        data_loc = self.adata_manager.data_registry[registry_key]
        mod_key = getattr(data_loc, "mod_key", None)
        values = get_anndata_attribute(
            self.adata_manager.adata,
            data_loc.attr_name,
            data_loc.attr_key,
            mod_key=mod_key,
        )
        arr = np.asarray(values, dtype=dtype)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise NotImplementedError(
                f"Registry key {registry_key!r} must be 1-D per observation; got shape {arr.shape}."
            )
        return arr

    def _build_obs_tensors(self) -> dict[str, np.ndarray]:
        obs_tensors: dict[str, np.ndarray] = {}

        if REGISTRY_KEYS.INDICES_KEY in self.adata_manager.data_registry:
            obs_tensors[REGISTRY_KEYS.INDICES_KEY] = self._registry_array(
                REGISTRY_KEYS.INDICES_KEY, dtype=np.int64
            )
        else:
            obs_tensors[REGISTRY_KEYS.INDICES_KEY] = np.arange(self.n_obs, dtype=np.int64)

        obs_tensors[REGISTRY_KEYS.BATCH_KEY] = self._registry_array(
            REGISTRY_KEYS.BATCH_KEY, dtype=np.int64
        )

        if REGISTRY_KEYS.LABELS_KEY in self.adata_manager.data_registry:
            obs_tensors[REGISTRY_KEYS.LABELS_KEY] = self._registry_array(
                REGISTRY_KEYS.LABELS_KEY, dtype=np.int64
            )
        else:
            obs_tensors[REGISTRY_KEYS.LABELS_KEY] = np.zeros(self.n_obs, dtype=np.int64)

        if REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry:
            obs_tensors[REGISTRY_KEYS.SIZE_FACTOR_KEY] = self._registry_array(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, dtype=np.float32
            )

        return obs_tensors

    def _split_indices(self) -> None:
        n_train, n_val = validate_data_split(
            self.n_obs,
            self.train_size,
            self.validation_size,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            train_size_is_none=False,
        )
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(self.n_obs)
        self.val_idx = perm[:n_val]
        self.train_idx = perm[n_val : n_val + n_train]
        self.test_idx = perm[n_val + n_train :]
        self.n_train = n_train
        self.n_val = n_val

    def _current_epoch(self) -> int:
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            return int(trainer.current_epoch)
        return 0

    def _make_dataset(self, indices: np.ndarray, *, shuffle: bool, epoch: int) -> ZarrDataset:
        return ZarrDataset(
            obs_tensors=self.obs_tensors,
            indices=indices,
            matrix_sources=self.matrix_sources,
            store_dir=self.store_dir,
            batch_size=self.batch_size,
            block_size=self.block_size,
            shuffle=shuffle,
            shuffle_buffer_blocks=self.shuffle_buffer_blocks,
            seed=self.seed,
            epoch=epoch,
            drop_last=self.drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        epoch = self._current_epoch()
        self._train_dataset = self._make_dataset(self.train_idx, shuffle=True, epoch=epoch)
        # persistent_workers must be False so reload_dataloaders_every_n_epochs
        # recreates loaders and picks up the new epoch for reshuffling.
        return DataLoader(
            self._train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            collate_fn=_identity_collate,
        )

    def val_dataloader(self) -> DataLoader:
        if self.n_val == 0:
            return None  # type: ignore[return-value]
        self._val_dataset = self._make_dataset(
            self.val_idx, shuffle=False, epoch=self._current_epoch()
        )
        return DataLoader(
            self._val_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self._val_persistent_workers and self.num_workers > 0,
            collate_fn=_identity_collate,
        )
