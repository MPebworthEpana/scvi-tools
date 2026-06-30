"""Lightning DataModules for zarr-backed streaming into scvi-tools models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import numpy as np
from anndata import AnnData
from mudata import MuData
from torch.utils.data import DataLoader

from scvi import REGISTRY_KEYS, settings
from scvi.data._manager import AnnDataManager
from scvi.data._utils import get_anndata_attribute, registry_key_to_default_dtype
from scvi.dataloaders._cuda_prefetch import maybe_wrap_cuda_prefetch
from scvi.dataloaders._data_splitting import validate_data_split
from scvi.dataloaders._zarr_dataset import (
    ZarrCSRSource,
    ZarrDataset,
    ZarrMatrixSource,
    _identity_collate,
    csr_sources_from_backed_mudata,
    matrix_sources_from_backed_anndata,
    matrix_sources_from_backed_mudata,
)

logger = logging.getLogger(__name__)

EmitMode = Literal["rolling", "flush"]

DATASPLITTER_ONLY_KWARGS = frozenset({
    "distributed_sampler",
    "shuffle_set_split",
    "load_sparse_tensor",
    "external_indexing",
})

ZARR_DATAMODULE_KWARGS = frozenset({
    "block_size",
    "shuffle_buffer_blocks",
    "emit_mode",
    "prefetch_queue_depth",
    "block_prefetch_depth",
    "prefetch_factor",
    "num_workers",
    "pin_memory",
    "prefetch_to_gpu",
    "cuda_queue_depth",
    "seed",
    "drop_last",
    "persistent_workers",
})

_SCALAR_OBS_KEYS = frozenset({
    REGISTRY_KEYS.BATCH_KEY,
    REGISTRY_KEYS.LABELS_KEY,
    REGISTRY_KEYS.INDICES_KEY,
    REGISTRY_KEYS.SIZE_FACTOR_KEY,
})

_JOINT_OBS_KEYS = frozenset({
    REGISTRY_KEYS.CAT_COVS_KEY,
    REGISTRY_KEYS.CONT_COVS_KEY,
})


class _BaseZarrDataModule(pl.LightningDataModule):
    """Shared zarr streaming datamodule logic for IterableDataset-backed training."""

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
        emit_mode: EmitMode = "rolling",
        prefetch_queue_depth: int = 0,
        block_prefetch_depth: int = 0,
        prefetch_factor: int | None = None,
        num_workers: int | None = None,
        pin_memory: bool = False,
        prefetch_to_gpu: bool = False,
        cuda_queue_depth: int = 2,
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
        self.emit_mode = emit_mode
        self.prefetch_queue_depth = prefetch_queue_depth
        self.block_prefetch_depth = block_prefetch_depth
        self.prefetch_factor = prefetch_factor
        self.num_workers = settings.dl_num_workers if num_workers is None else num_workers
        self.pin_memory = pin_memory
        self.prefetch_to_gpu = prefetch_to_gpu
        self.cuda_queue_depth = cuda_queue_depth
        self.seed = seed
        self.drop_last = drop_last
        self._val_persistent_workers = (
            settings.dl_persistent_workers if persistent_workers is None else persistent_workers
        )

        self._train_dataset: ZarrDataset | None = None
        self._val_dataset: ZarrDataset | None = None

        self._validate_obs_fields()
        self.obs_tensors = self._build_obs_tensors()
        self._split_indices()

    @property
    def n_vars(self) -> int:
        return int(self.adata_manager.summary_stats.n_vars)

    @property
    def n_batch(self) -> int:
        return int(self.adata_manager.summary_stats.n_batch)

    @property
    def n_labels(self) -> int:
        return int(self.adata_manager.summary_stats.n_labels)

    @property
    def n_continuous_cov(self) -> int:
        return int(self.adata_manager.summary_stats.get("n_extra_continuous_covs", 0))

    @property
    def n_cats_per_cov(self) -> tuple[int, ...] | None:
        if REGISTRY_KEYS.CAT_COVS_KEY not in self.adata_manager.data_registry:
            return None
        return self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key

    def _validate_obs_fields(self) -> None:
        registry = self.adata_manager.data_registry
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
                    "Zarr datamodules do not yet support multi-column size factors."
                )

    def _registry_obs_array(self, registry_key: str) -> np.ndarray:
        data_loc = self.adata_manager.data_registry[registry_key]
        mod_key = getattr(data_loc, "mod_key", None)
        values = get_anndata_attribute(
            self.adata_manager.adata,
            data_loc.attr_name,
            data_loc.attr_key,
            mod_key=mod_key,
        )
        dtype = registry_key_to_default_dtype(registry_key)
        arr = np.asarray(values, dtype=dtype)

        if registry_key in _SCALAR_OBS_KEYS:
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            if arr.ndim != 1:
                raise NotImplementedError(
                    f"Registry key {registry_key!r} must be 1-D per observation; "
                    f"got shape {arr.shape}."
                )
            return arr

        if registry_key in _JOINT_OBS_KEYS:
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim != 2:
                raise NotImplementedError(
                    f"Registry key {registry_key!r} must be 2-D per observation; "
                    f"got shape {arr.shape}."
                )
            return arr

        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise NotImplementedError(
                f"Registry key {registry_key!r} must be 1-D or 2-D per observation; "
                f"got shape {arr.shape}."
            )
        return arr

    def _build_obs_tensors(self) -> dict[str, np.ndarray]:
        obs_tensors: dict[str, np.ndarray] = {}

        if REGISTRY_KEYS.INDICES_KEY in self.adata_manager.data_registry:
            obs_tensors[REGISTRY_KEYS.INDICES_KEY] = self._registry_obs_array(
                REGISTRY_KEYS.INDICES_KEY
            )
        else:
            obs_tensors[REGISTRY_KEYS.INDICES_KEY] = np.arange(self.n_obs, dtype=np.int64)

        obs_tensors[REGISTRY_KEYS.BATCH_KEY] = self._registry_obs_array(REGISTRY_KEYS.BATCH_KEY)

        if REGISTRY_KEYS.LABELS_KEY in self.adata_manager.data_registry:
            obs_tensors[REGISTRY_KEYS.LABELS_KEY] = self._registry_obs_array(
                REGISTRY_KEYS.LABELS_KEY
            )
        else:
            obs_tensors[REGISTRY_KEYS.LABELS_KEY] = np.zeros(self.n_obs, dtype=np.int64)

        if REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry:
            obs_tensors[REGISTRY_KEYS.SIZE_FACTOR_KEY] = self._registry_obs_array(
                REGISTRY_KEYS.SIZE_FACTOR_KEY
            )

        for key in (REGISTRY_KEYS.CAT_COVS_KEY, REGISTRY_KEYS.CONT_COVS_KEY):
            if key in self.adata_manager.data_registry:
                obs_tensors[key] = self._registry_obs_array(key)

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
            emit_mode=self.emit_mode,
            prefetch_queue_depth=self.prefetch_queue_depth,
            block_prefetch_depth=self.block_prefetch_depth,
            seed=self.seed,
            epoch=epoch,
            drop_last=self.drop_last,
        )

    def _dataloader_kwargs(self) -> dict:
        kwargs: dict = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 2 if self.prefetch_factor is None else self.prefetch_factor
        return kwargs

    def _make_dataloader(self, dataset: ZarrDataset, *, persistent_workers: bool) -> DataLoader:
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers and self.num_workers > 0,
            collate_fn=_identity_collate,
            **self._dataloader_kwargs(),
        )
        return maybe_wrap_cuda_prefetch(
            loader,
            prefetch_to_gpu=self.prefetch_to_gpu,
            cuda_queue_depth=self.cuda_queue_depth,
            pin_memory=self.pin_memory,
            load_sparse_tensor=False,
        )

    def train_dataloader(self) -> DataLoader:
        epoch = self._current_epoch()
        self._train_dataset = self._make_dataset(self.train_idx, shuffle=True, epoch=epoch)
        return self._make_dataloader(self._train_dataset, persistent_workers=False)

    def val_dataloader(self) -> DataLoader:
        if self.n_val == 0:
            return None  # type: ignore[return-value]
        self._val_dataset = self._make_dataset(
            self.val_idx, shuffle=False, epoch=self._current_epoch()
        )
        return self._make_dataloader(
            self._val_dataset,
            persistent_workers=self._val_persistent_workers,
        )


class ZarrAnnDataModule(_BaseZarrDataModule):
    """EXPERIMENTAL: Stream single-modality batches from zarr-backed AnnData into scVI or PeakVI.

    Use :meth:`from_backed_anndata` to construct this datamodule. The caller must
    supply an AnnData whose registered count matrix is zarr-backed (CSR ``CSRDataset``
    or dense ``zarr.Array``, e.g. from ``adata.write_zarr`` with backed reopen).
    """

    @classmethod
    def from_backed_anndata(
        cls,
        adata: AnnData,
        adata_manager: AnnDataManager,
        *,
        registry_key: str = REGISTRY_KEYS.X_KEY,
        matrix_layout: str = "csr",
        x_suffix: str = "",
        store_dir: Path | str | None = None,
        **kwargs,
    ) -> ZarrAnnDataModule:
        """EXPERIMENTAL: Create a datamodule that streams from an already-backed AnnData."""
        if matrix_layout == "dense":
            layout: str = "dense"
        elif matrix_layout == "auto":
            layout = "auto"
        else:
            layout = "csr"

        if store_dir is None and layout in ("dense", "auto"):
            candidate = getattr(adata, "filename", None)
            if isinstance(candidate, (str, Path)):
                store_dir = candidate

        source_kwargs: dict = {
            "n_obs": adata.n_obs,
            "layout": layout,
            "registry_key": registry_key,
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
        elif layout == "auto":
            if x_suffix:
                if store_dir is None:
                    raise ValueError(
                        "store_dir is required when using x_suffix with matrix_layout='auto'."
                    )
                source_kwargs["x_suffix"] = x_suffix
            if store_dir is not None:
                source_kwargs["store_dir"] = store_dir

        sources = matrix_sources_from_backed_anndata(adata, adata_manager, **source_kwargs)
        resolved_store_dir = Path(store_dir) if store_dir is not None else None
        return cls(
            adata_manager,
            matrix_sources=sources,
            n_obs=adata.n_obs,
            store_dir=resolved_store_dir,
            **kwargs,
        )


class ZarrMultiVIDataModule(_BaseZarrDataModule):
    """EXPERIMENTAL: Stream paired modality batches from zarr-backed CSR or dense stores into MultiVI.

    Use :meth:`from_backed_mudata` to construct this datamodule. The caller must
    supply a MuData whose modality matrices are zarr-backed (CSR ``CSRDataset`` or
    dense ``zarr.Array``, e.g. from ``mudata.write_zarr`` with backed reopen).
    Mixed per-modality layouts (e.g. RNA CSR + ADT dense) are supported when
    ``matrix_layout='auto'``.
    """

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
        """EXPERIMENTAL: Create a datamodule that streams from an already-backed MuData."""
        if registry_map is None:
            registry_map = {}
            for key in (
                REGISTRY_KEYS.X_KEY,
                REGISTRY_KEYS.ATAC_X_KEY,
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
            ):
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

        if matrix_layout == "dense":
            layout: str = "dense"
        elif matrix_layout == "auto":
            layout = "auto"
        else:
            layout = "csr"

        if store_dir is None and layout in ("dense", "auto"):
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
        elif layout == "auto":
            if x_suffix:
                if store_dir is None:
                    raise ValueError(
                        "store_dir is required when using x_suffix with matrix_layout='auto'."
                    )
                source_kwargs["x_suffix"] = x_suffix
            if store_dir is not None:
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


def try_auto_zarr_anndata_datamodule(
    adata: AnnData,
    adata_manager: AnnDataManager,
    *,
    train_size: float | None,
    validation_size: float | None,
    batch_size: int,
    datasplitter_kwargs: dict | None,
) -> ZarrAnnDataModule | None:
    """Return a zarr streaming datamodule when ``adata`` has a zarr-backed count matrix."""
    datasplitter_kwargs = datasplitter_kwargs or {}
    splitter_only = DATASPLITTER_ONLY_KWARGS & datasplitter_kwargs.keys()
    if splitter_only:
        logger.debug(
            "Skipping zarr datamodule auto-selection due to DataSplitter-only kwargs: %s",
            sorted(splitter_only),
        )
        return None

    unsupported = set(datasplitter_kwargs) - ZARR_DATAMODULE_KWARGS
    if unsupported:
        logger.debug(
            "Skipping zarr datamodule auto-selection due to unsupported kwargs: %s",
            sorted(unsupported),
        )
        return None

    zarr_kwargs = {
        key: datasplitter_kwargs[key]
        for key in ZARR_DATAMODULE_KWARGS & datasplitter_kwargs.keys()
    }
    resolved_train_size = 0.9 if train_size is None else train_size

    try:
        datamodule = ZarrAnnDataModule.from_backed_anndata(
            adata,
            adata_manager,
            matrix_layout="auto",
            train_size=resolved_train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            **zarr_kwargs,
        )
    except Exception as exc:
        logger.debug("Zarr datamodule auto-selection failed: %s", exc)
        return None

    logger.info(
        "Detected zarr-backed count matrix; using ZarrAnnDataModule for training."
    )
    return datamodule
