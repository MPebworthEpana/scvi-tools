from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mudata as md
import numpy as np
import pytest
import scipy.sparse as sp
import torch
import zarr
from anndata.io import sparse_dataset
from mudata import MuData
from torch.utils.data import DataLoader

from scvi import REGISTRY_KEYS
from scvi.dataloaders import (
    ZarrAnnDataModule,
    ZarrDataset,
    ZarrMultiVIDataModule,
    csr_sources_from_backed_mudata,
    matrix_sources_from_backed_anndata,
    matrix_sources_from_backed_mudata,
)
from scvi.dataloaders._zarr_dataset import _identity_collate
from scvi.data import synthetic_iid
from scvi.model import MULTIVI, PEAKVI, SCVI


def _write_and_open_backed_anndata(adata, store_path: Path):
    """Test helper: write AnnData to zarr and reopen with zarr-backed CSRDataset on .X."""
    import anndata

    previous = anndata.settings.allow_write_nullable_strings
    anndata.settings.allow_write_nullable_strings = True
    try:
        adata.write_zarr(store_path)
    finally:
        anndata.settings.allow_write_nullable_strings = previous
    backed = anndata.read_zarr(store_path)
    f = zarr.open(str(store_path), mode="r")
    x_group = f["X"]
    enc = x_group.attrs.get("encoding-type", "")
    if enc in ("csr_matrix", "csc_matrix"):
        backed.X = sparse_dataset(x_group)
    return backed


def _make_synthetic_anndata_zarr_store(tmp_path, *, batch_size: int = 32, n_genes: int = 20):
    adata = synthetic_iid(batch_size=batch_size, n_genes=n_genes, n_batches=2)
    adata.X = sp.random(adata.n_obs, n_genes, density=0.2, format="csr", dtype=np.float32)
    store_path = tmp_path / "adata.zarr"
    adata_backed = _write_and_open_backed_anndata(adata, store_path)
    return adata, adata_backed


def _write_and_open_backed_mudata(mdata: MuData, store_path: Path) -> MuData:
    """Test helper: write MuData to zarr and reopen with zarr-backed CSRDatasets on .X."""
    import anndata

    previous = anndata.settings.allow_write_nullable_strings
    anndata.settings.allow_write_nullable_strings = True
    try:
        mdata.write_zarr(store_path)
    finally:
        anndata.settings.allow_write_nullable_strings = previous
    backed = md.read_zarr(store_path)
    f = zarr.open(str(store_path), mode="r")
    for mod in backed.mod:
        x_group = f["mod"][mod]["X"]
        enc = x_group.attrs.get("encoding-type", "")
        if enc in ("csr_matrix", "csc_matrix"):
            backed.mod[mod].X = sparse_dataset(x_group)
    return backed


def _csr_datasets_from_backed(mdata_backed: MuData) -> dict:
    return {
        REGISTRY_KEYS.X_KEY: mdata_backed.mod["RNA"].X,
        REGISTRY_KEYS.ATAC_X_KEY: mdata_backed.mod["ATAC"].X,
    }


def _make_synthetic_zarr_store(
    tmp_path,
    batch_size: int = 32,
    n_genes: int = 20,
    n_regions: int = 15,
):
    mdata = synthetic_iid(
        return_mudata=True,
        batch_size=batch_size,
        n_genes=n_genes,
        n_regions=n_regions,
        n_proteins=0,
        n_batches=2,
    )
    n_obs = mdata.n_obs
    rna = mdata.mod["rna"]
    atac = mdata.mod["accessibility"]
    rna.X = sp.random(n_obs, n_genes, density=0.2, format="csr", dtype=np.float32)
    atac.X = sp.random(n_obs, n_regions, density=0.15, format="csr", dtype=np.float32)
    mdata.mod["RNA"] = mdata.mod.pop("rna")
    mdata.mod["ATAC"] = mdata.mod.pop("accessibility")
    mdata.update()
    store_path = tmp_path / "mdata.zarr"
    mdata_backed = _write_and_open_backed_mudata(mdata, store_path)
    return mdata, mdata_backed


def _obs_tensors(n_obs: int) -> dict[str, np.ndarray]:
    indices = np.arange(n_obs, dtype=np.int64)
    return {
        REGISTRY_KEYS.BATCH_KEY: np.zeros(n_obs, dtype=np.int64),
        REGISTRY_KEYS.LABELS_KEY: np.zeros(n_obs, dtype=np.int64),
        REGISTRY_KEYS.INDICES_KEY: indices,
    }


@pytest.fixture
def zarr_store(tmp_path):
    return _make_synthetic_zarr_store(tmp_path)


def test_mudata_write_zarr_backed_roundtrip(zarr_store):
    _, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    assert REGISTRY_KEYS.X_KEY in csr_datasets
    assert REGISTRY_KEYS.ATAC_X_KEY in csr_datasets
    assert csr_datasets[REGISTRY_KEYS.X_KEY].backend == "zarr"
    assert mdata_backed.n_obs == csr_datasets[REGISTRY_KEYS.X_KEY].shape[0]


def test_zarr_dataset_pairing(zarr_store):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    indices = np.arange(mdata.n_obs, dtype=np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=8,
        block_size=16,
        shuffle=False,
        shuffle_buffer_blocks=1,
        seed=0,
    )
    batch = next(iter(ds))
    assert batch[REGISTRY_KEYS.X_KEY].shape[1] == csr_datasets[REGISTRY_KEYS.X_KEY].shape[1]
    assert batch[REGISTRY_KEYS.ATAC_X_KEY].shape[1] == csr_datasets[REGISTRY_KEYS.ATAC_X_KEY].shape[1]
    assert batch[REGISTRY_KEYS.X_KEY].shape[0] == batch[REGISTRY_KEYS.ATAC_X_KEY].shape[0]
    assert batch[REGISTRY_KEYS.INDICES_KEY].dtype == torch.int64


def test_zarr_dataset_pairing_shuffled_indices(zarr_store):
    """Sort-within-block must preserve row pairing under a random index permutation."""
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    rng = np.random.default_rng(99)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)

    ref_rna = csr_datasets[REGISTRY_KEYS.X_KEY][:].toarray()
    ref_atac = csr_datasets[REGISTRY_KEYS.ATAC_X_KEY][:].toarray()

    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=5,
        block_size=7,
        shuffle=False,
        shuffle_buffer_blocks=1,
        seed=0,
    )
    for batch in ds:
        ind_x = batch[REGISTRY_KEYS.INDICES_KEY].view(-1).numpy()
        rna = batch[REGISTRY_KEYS.X_KEY].numpy()
        atac = batch[REGISTRY_KEYS.ATAC_X_KEY].numpy()
        for i, row_idx in enumerate(ind_x):
            np.testing.assert_allclose(rna[i], ref_rna[row_idx], rtol=1e-5)
            np.testing.assert_allclose(atac[i], ref_atac[row_idx], rtol=1e-5)


def test_zarr_dataset_coverage(zarr_store):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    rng = np.random.default_rng(42)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=7,
        block_size=11,
        shuffle=True,
        shuffle_buffer_blocks=2,
        seed=42,
        drop_last=False,
    )
    seen = set()
    for batch in ds:
        seen.update(batch[REGISTRY_KEYS.INDICES_KEY].view(-1).tolist())
    assert seen == set(indices.tolist())


def test_zarr_dataset_validation_streaming(zarr_store):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    rng = np.random.default_rng(7)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=9,
        block_size=13,
        shuffle=False,
        shuffle_buffer_blocks=1,
        seed=0,
        drop_last=False,
    )
    seen = set()
    for batch in ds:
        assert batch[REGISTRY_KEYS.X_KEY].dtype == torch.float32
        assert batch[REGISTRY_KEYS.ATAC_X_KEY].dtype == torch.float32
        assert batch[REGISTRY_KEYS.INDICES_KEY].dtype == torch.int64
        seen.update(batch[REGISTRY_KEYS.INDICES_KEY].view(-1).tolist())
    assert seen == set(indices.tolist())


def test_zarr_dataset_determinism(zarr_store):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    rng = np.random.default_rng(11)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)

    def collect(epoch: int):
        ds = ZarrDataset(
            obs_tensors=_obs_tensors(mdata.n_obs),
            indices=indices,
            csr_datasets=csr_datasets,
            batch_size=8,
            block_size=16,
            shuffle=True,
            shuffle_buffer_blocks=2,
            seed=7,
            epoch=epoch,
        )
        return [batch[REGISTRY_KEYS.INDICES_KEY].view(-1).tolist() for batch in ds]

    assert collect(0) == collect(0)
    assert collect(0) != collect(1)


def test_zarr_dataset_worker_partition(zarr_store):
    mdata, mdata_backed = zarr_store
    registry_map = {
        REGISTRY_KEYS.X_KEY: "RNA",
        REGISTRY_KEYS.ATAC_X_KEY: "ATAC",
    }
    sources = csr_sources_from_backed_mudata(mdata_backed, registry_map)
    indices = np.arange(mdata.n_obs, dtype=np.int64)

    class _WorkerInfo:
        def __init__(self, id_: int, num_workers: int):
            self.id = id_
            self.num_workers = num_workers

    seen_per_worker = []
    for worker_id in range(3):
        ds = ZarrDataset(
            obs_tensors=_obs_tensors(mdata.n_obs),
            indices=indices,
            csr_sources=sources,
            batch_size=5,
            block_size=10,
            shuffle=False,
            shuffle_buffer_blocks=1,
            seed=0,
        )
        import scvi.dataloaders._zarr_dataset as zds

        original = zds.get_worker_info
        zds.get_worker_info = lambda: _WorkerInfo(worker_id, 3)
        try:
            seen = set()
            for batch in ds:
                seen.update(batch[REGISTRY_KEYS.INDICES_KEY].view(-1).tolist())
            seen_per_worker.append(seen)
        finally:
            zds.get_worker_info = original

    union = set().union(*seen_per_worker)
    assert union == set(indices.tolist())
    for i, s_i in enumerate(seen_per_worker):
        for j, s_j in enumerate(seen_per_worker):
            if i != j:
                assert s_i.isdisjoint(s_j)


def test_zarr_dataset_ddp_equal_batches(zarr_store):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    indices = np.arange(mdata.n_obs, dtype=np.int64)

    def _collect_for_rank(rank: int, world_size: int = 2):
        ds = ZarrDataset(
            obs_tensors=_obs_tensors(mdata.n_obs),
            indices=indices,
            csr_datasets=csr_datasets,
            batch_size=8,
            block_size=16,
            shuffle=True,
            shuffle_buffer_blocks=2,
            seed=3,
            epoch=0,
        )
        import scvi.dataloaders._zarr_dataset as zds

        original = zds._get_rank_worker_ids
        zds._get_rank_worker_ids = lambda: (rank, world_size, 0, 1)
        try:
            batches = list(ds)
            seen = set()
            for batch in batches:
                seen.update(batch[REGISTRY_KEYS.INDICES_KEY].view(-1).tolist())
            return len(batches), seen
        finally:
            zds._get_rank_worker_ids = original

    n_batches_0, seen_0 = _collect_for_rank(0)
    n_batches_1, seen_1 = _collect_for_rank(1)
    assert n_batches_0 == n_batches_1
    assert seen_0.isdisjoint(seen_1)
    assert seen_0 | seen_1 == set(indices.tolist())


def test_zarr_datamodule_epoch_reshuffle(zarr_store):
    _, mdata_backed = zarr_store
    MULTIVI.setup_mudata(
        mdata_backed,
        batch_key="batch",
        modalities={"rna_layer": "RNA", "atac_layer": "ATAC"},
    )
    model = MULTIVI(mdata_backed)
    dm = ZarrMultiVIDataModule.from_backed_mudata(
        mdata_backed,
        model.adata_manager,
        train_size=0.8,
        batch_size=8,
        block_size=8,
        shuffle_buffer_blocks=1,
        num_workers=0,
        seed=0,
    )

    def _collect_epoch(epoch: int):
        dm.trainer = SimpleNamespace(current_epoch=epoch)
        loader = dm.train_dataloader()
        return [
            batch[REGISTRY_KEYS.INDICES_KEY].view(-1).tolist()
            for batch in loader
        ]

    epoch0_a = _collect_epoch(0)
    epoch0_b = _collect_epoch(0)
    epoch1 = _collect_epoch(1)
    assert epoch0_a == epoch0_b
    assert epoch0_a != epoch1


def test_zarr_datamodule_supports_covariates(zarr_store):
    _, mdata_backed = zarr_store
    mdata_backed.obs["extra_cat"] = np.random.randint(0, 3, size=mdata_backed.n_obs)
    MULTIVI.setup_mudata(
        mdata_backed,
        batch_key="batch",
        categorical_covariate_keys=["extra_cat"],
        modalities={"rna_layer": "RNA", "atac_layer": "ATAC"},
    )
    model = MULTIVI(mdata_backed)
    dm = ZarrMultiVIDataModule.from_backed_mudata(
        mdata_backed,
        model.adata_manager,
        num_workers=0,
    )
    assert REGISTRY_KEYS.CAT_COVS_KEY in dm.obs_tensors
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch[REGISTRY_KEYS.CAT_COVS_KEY].ndim == 2


def _collect_indices_from_loader(loader) -> set[int]:
    seen: set[int] = set()
    for batch in loader:
        seen.update(batch[REGISTRY_KEYS.INDICES_KEY].view(-1).tolist())
    return seen


def test_csr_sources_from_backed_mudata(zarr_store):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    registry_map = {
        REGISTRY_KEYS.X_KEY: "RNA",
        REGISTRY_KEYS.ATAC_X_KEY: "ATAC",
    }
    sources = csr_sources_from_backed_mudata(mdata_backed, registry_map)
    assert set(sources) == {REGISTRY_KEYS.X_KEY, REGISTRY_KEYS.ATAC_X_KEY}
    for key in sources:
        assert Path(sources[key].x_relpath).is_absolute()
    opened = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=np.arange(mdata.n_obs, dtype=np.int64),
        csr_sources=sources,
        batch_size=8,
        block_size=16,
        shuffle=False,
    )._get_csr_datasets()
    assert opened[REGISTRY_KEYS.X_KEY].shape == csr_datasets[REGISTRY_KEYS.X_KEY].shape


def test_zarr_dataset_multiworker_backed_mudata(zarr_store):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    registry_map = {
        REGISTRY_KEYS.X_KEY: "RNA",
        REGISTRY_KEYS.ATAC_X_KEY: "ATAC",
    }
    sources = csr_sources_from_backed_mudata(mdata_backed, registry_map)
    rng = np.random.default_rng(17)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)

    ref_rna = csr_datasets[REGISTRY_KEYS.X_KEY][:].toarray()
    ref_atac = csr_datasets[REGISTRY_KEYS.ATAC_X_KEY][:].toarray()

    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_sources=sources,
        batch_size=7,
        block_size=11,
        shuffle=False,
        shuffle_buffer_blocks=1,
        seed=0,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=2,
        collate_fn=_identity_collate,
        persistent_workers=False,
    )
    seen: set[int] = set()
    for batch in loader:
        ind_x = batch[REGISTRY_KEYS.INDICES_KEY].view(-1).numpy()
        rna = batch[REGISTRY_KEYS.X_KEY].numpy()
        atac = batch[REGISTRY_KEYS.ATAC_X_KEY].numpy()
        seen.update(ind_x.tolist())
        for i, row_idx in enumerate(ind_x):
            np.testing.assert_allclose(rna[i], ref_rna[row_idx], rtol=1e-5)
            np.testing.assert_allclose(atac[i], ref_atac[row_idx], rtol=1e-5)
    assert seen == set(indices.tolist())


def test_zarr_datamodule_from_backed_mudata_multiworker(zarr_store):
    _, mdata_backed = zarr_store
    MULTIVI.setup_mudata(
        mdata_backed,
        batch_key="batch",
        modalities={"rna_layer": "RNA", "atac_layer": "ATAC"},
    )
    model = MULTIVI(mdata_backed)
    dm = ZarrMultiVIDataModule.from_backed_mudata(
        mdata_backed,
        model.adata_manager,
        train_size=0.8,
        batch_size=8,
        block_size=8,
        shuffle_buffer_blocks=1,
        num_workers=2,
        seed=0,
    )
    loader = dm.train_dataloader()
    seen = _collect_indices_from_loader(loader)
    assert seen == set(dm.train_idx.tolist())


def test_multivi_zarr_datamodule_smoke(zarr_store):
    _, mdata_backed = zarr_store
    MULTIVI.setup_mudata(
        mdata_backed,
        batch_key="batch",
        modalities={"rna_layer": "RNA", "atac_layer": "ATAC"},
    )
    model = MULTIVI(mdata_backed)
    dm = ZarrMultiVIDataModule.from_backed_mudata(
        mdata_backed,
        model.adata_manager,
        train_size=0.8,
        batch_size=16,
        block_size=16,
        shuffle_buffer_blocks=2,
        num_workers=0,
        seed=0,
    )
    model.train(
        max_epochs=2,
        datamodule=dm,
        early_stopping=False,
        check_val_every_n_epoch=1,
    )
    latent = model.get_latent_representation()
    assert latent.shape[0] == mdata_backed.n_obs
    assert latent.shape[1] == model.module.n_latent


def _write_dense_layers_from_csr(store_path: Path, mdata_backed: MuData, *, block_size: int = 64) -> None:
    """Test helper: write mod/*/X_dense from backed CSR .X."""
    for mod_key in ("RNA", "ATAC"):
        csr = mdata_backed.mod[mod_key].X
        n_obs, n_vars = csr.shape
        dest = store_path / "mod" / mod_key / "X_dense"
        arr = zarr.open_array(
            str(dest),
            mode="w",
            shape=(n_obs, n_vars),
            chunks=(min(block_size, n_obs), n_vars),
            dtype="float32",
        )
        for start in range(0, n_obs, block_size):
            stop = min(start + block_size, n_obs)
            arr[start:stop] = csr[start:stop].toarray().astype(np.float32, copy=False)


def _dense_datasets_from_store(store_path: Path, mdata_backed: MuData) -> dict:
    opened = {}
    for mod_key, registry_key in (("RNA", REGISTRY_KEYS.X_KEY), ("ATAC", REGISTRY_KEYS.ATAC_X_KEY)):
        path = store_path / "mod" / mod_key / "X_dense"
        opened[registry_key] = zarr.open_array(str(path), mode="r")
    return opened


def _make_dense_zarr_store(
    tmp_path,
    batch_size: int = 32,
    n_genes: int = 50,
    n_regions: int = 30,
):
    mdata, mdata_backed = _make_synthetic_zarr_store(
        tmp_path,
        batch_size=batch_size,
        n_genes=n_genes,
        n_regions=n_regions,
    )
    store_path = tmp_path / "mdata.zarr"
    _write_dense_layers_from_csr(store_path, mdata_backed)
    dense_datasets = _dense_datasets_from_store(store_path, mdata_backed)
    return mdata, mdata_backed, dense_datasets, store_path


@pytest.fixture
def dense_zarr_store(tmp_path):
    return _make_dense_zarr_store(tmp_path)


def test_zarr_dataset_dense_pairing(dense_zarr_store):
    mdata, _, dense_datasets, _ = dense_zarr_store
    indices = np.arange(mdata.n_obs, dtype=np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        matrix_datasets=dense_datasets,
        batch_size=8,
        block_size=16,
        shuffle=False,
        shuffle_buffer_blocks=1,
        seed=0,
    )
    batch = next(iter(ds))
    assert batch[REGISTRY_KEYS.X_KEY].shape[1] == dense_datasets[REGISTRY_KEYS.X_KEY].shape[1]
    assert batch[REGISTRY_KEYS.ATAC_X_KEY].shape[1] == dense_datasets[REGISTRY_KEYS.ATAC_X_KEY].shape[1]
    assert batch[REGISTRY_KEYS.X_KEY].shape[0] == batch[REGISTRY_KEYS.ATAC_X_KEY].shape[0]


def test_zarr_dataset_dense_pairing_shuffled_indices(dense_zarr_store):
    mdata, mdata_backed, dense_datasets, _ = dense_zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    rng = np.random.default_rng(99)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)

    ref_rna = csr_datasets[REGISTRY_KEYS.X_KEY][:].toarray()
    ref_atac = csr_datasets[REGISTRY_KEYS.ATAC_X_KEY][:].toarray()

    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        matrix_datasets=dense_datasets,
        batch_size=5,
        block_size=7,
        shuffle=False,
        shuffle_buffer_blocks=1,
        seed=0,
    )
    for batch in ds:
        ind_x = batch[REGISTRY_KEYS.INDICES_KEY].view(-1).numpy()
        rna = batch[REGISTRY_KEYS.X_KEY].numpy()
        atac = batch[REGISTRY_KEYS.ATAC_X_KEY].numpy()
        for i, row_idx in enumerate(ind_x):
            np.testing.assert_allclose(rna[i], ref_rna[row_idx], rtol=1e-5)
            np.testing.assert_allclose(atac[i], ref_atac[row_idx], rtol=1e-5)


def test_zarr_dataset_dense_coverage(dense_zarr_store):
    mdata, _, dense_datasets, _ = dense_zarr_store
    rng = np.random.default_rng(42)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        matrix_datasets=dense_datasets,
        batch_size=7,
        block_size=11,
        shuffle=True,
        shuffle_buffer_blocks=2,
        seed=42,
        drop_last=False,
    )
    seen = set()
    for batch in ds:
        seen.update(batch[REGISTRY_KEYS.INDICES_KEY].view(-1).tolist())
    assert seen == set(indices.tolist())


def test_zarr_dataset_dense_worker_partition(dense_zarr_store):
    mdata, mdata_backed, _, store_path = dense_zarr_store
    registry_map = {
        REGISTRY_KEYS.X_KEY: "RNA",
        REGISTRY_KEYS.ATAC_X_KEY: "ATAC",
    }
    sources = matrix_sources_from_backed_mudata(
        mdata_backed,
        registry_map,
        layout="dense",
        x_suffix="_dense",
        store_dir=store_path,
    )
    indices = np.arange(mdata.n_obs, dtype=np.int64)

    class _WorkerInfo:
        def __init__(self, id_: int, num_workers: int):
            self.id = id_
            self.num_workers = num_workers

    seen_per_worker = []
    for worker_id in range(3):
        ds = ZarrDataset(
            obs_tensors=_obs_tensors(mdata.n_obs),
            indices=indices,
            matrix_sources=sources,
            batch_size=5,
            block_size=10,
            shuffle=False,
            shuffle_buffer_blocks=1,
            seed=0,
        )
        import scvi.dataloaders._zarr_dataset as zds

        original = zds.get_worker_info
        zds.get_worker_info = lambda: _WorkerInfo(worker_id, 3)
        try:
            seen = set()
            for batch in ds:
                seen.update(batch[REGISTRY_KEYS.INDICES_KEY].view(-1).tolist())
            seen_per_worker.append(seen)
        finally:
            zds.get_worker_info = original

    union = set().union(*seen_per_worker)
    assert union == set(indices.tolist())
    for i, s_i in enumerate(seen_per_worker):
        for j, s_j in enumerate(seen_per_worker):
            if i != j:
                assert s_i.isdisjoint(s_j)


def test_zarr_dataset_dense_multiworker_backed_mudata(dense_zarr_store):
    mdata, mdata_backed, _, store_path = dense_zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    registry_map = {
        REGISTRY_KEYS.X_KEY: "RNA",
        REGISTRY_KEYS.ATAC_X_KEY: "ATAC",
    }
    sources = matrix_sources_from_backed_mudata(
        mdata_backed,
        registry_map,
        layout="dense",
        x_suffix="_dense",
        store_dir=store_path,
    )
    rng = np.random.default_rng(17)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)

    ref_rna = csr_datasets[REGISTRY_KEYS.X_KEY][:].toarray()
    ref_atac = csr_datasets[REGISTRY_KEYS.ATAC_X_KEY][:].toarray()

    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        matrix_sources=sources,
        batch_size=7,
        block_size=11,
        shuffle=False,
        shuffle_buffer_blocks=1,
        seed=0,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=2,
        collate_fn=_identity_collate,
        persistent_workers=False,
    )
    seen: set[int] = set()
    for batch in loader:
        ind_x = batch[REGISTRY_KEYS.INDICES_KEY].view(-1).numpy()
        rna = batch[REGISTRY_KEYS.X_KEY].numpy()
        atac = batch[REGISTRY_KEYS.ATAC_X_KEY].numpy()
        seen.update(ind_x.tolist())
        for i, row_idx in enumerate(ind_x):
            np.testing.assert_allclose(rna[i], ref_rna[row_idx], rtol=1e-5)
            np.testing.assert_allclose(atac[i], ref_atac[row_idx], rtol=1e-5)
    assert seen == set(indices.tolist())


def test_zarr_dataset_mixed_layout_pairing(dense_zarr_store):
    """RNA CSR + ATAC dense in one ZarrDataset preserves row pairing."""
    mdata, mdata_backed, dense_datasets, _ = dense_zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    mixed = {
        REGISTRY_KEYS.X_KEY: csr_datasets[REGISTRY_KEYS.X_KEY],
        REGISTRY_KEYS.ATAC_X_KEY: dense_datasets[REGISTRY_KEYS.ATAC_X_KEY],
    }
    rng = np.random.default_rng(99)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)

    ref_rna = csr_datasets[REGISTRY_KEYS.X_KEY][:].toarray()
    ref_atac = dense_datasets[REGISTRY_KEYS.ATAC_X_KEY][:]

    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        matrix_datasets=mixed,
        batch_size=5,
        block_size=7,
        shuffle=False,
        shuffle_buffer_blocks=1,
        seed=0,
    )
    assert ds._matrix_layouts[REGISTRY_KEYS.X_KEY] == "csr"
    assert ds._matrix_layouts[REGISTRY_KEYS.ATAC_X_KEY] == "dense"

    seen: set[int] = set()
    for batch in ds:
        ind_x = batch[REGISTRY_KEYS.INDICES_KEY].view(-1).numpy()
        rna = batch[REGISTRY_KEYS.X_KEY].numpy()
        atac = batch[REGISTRY_KEYS.ATAC_X_KEY].numpy()
        seen.update(ind_x.tolist())
        for i, row_idx in enumerate(ind_x):
            np.testing.assert_allclose(rna[i], ref_rna[row_idx], rtol=1e-5)
            np.testing.assert_allclose(atac[i], ref_atac[row_idx], rtol=1e-5)
    assert seen == set(indices.tolist())


def test_zarr_dataset_mixed_layout_multiworker(dense_zarr_store):
    """Mixed matrix_sources reopen correctly under num_workers > 0."""
    mdata, mdata_backed, dense_datasets, store_path = dense_zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    mdata_backed.mod["ATAC"].X = dense_datasets[REGISTRY_KEYS.ATAC_X_KEY]
    registry_map = {
        REGISTRY_KEYS.X_KEY: "RNA",
        REGISTRY_KEYS.ATAC_X_KEY: "ATAC",
    }
    sources = matrix_sources_from_backed_mudata(
        mdata_backed,
        registry_map,
        layout="auto",
        store_dir=store_path,
    )
    assert sources[REGISTRY_KEYS.X_KEY].layout == "csr"
    assert sources[REGISTRY_KEYS.ATAC_X_KEY].layout == "dense"

    rng = np.random.default_rng(17)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)
    ref_rna = csr_datasets[REGISTRY_KEYS.X_KEY][:].toarray()
    ref_atac = dense_datasets[REGISTRY_KEYS.ATAC_X_KEY][:]

    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        matrix_sources=sources,
        batch_size=7,
        block_size=11,
        shuffle=False,
        shuffle_buffer_blocks=1,
        seed=0,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=2,
        collate_fn=_identity_collate,
        persistent_workers=False,
    )
    seen: set[int] = set()
    for batch in loader:
        ind_x = batch[REGISTRY_KEYS.INDICES_KEY].view(-1).numpy()
        rna = batch[REGISTRY_KEYS.X_KEY].numpy()
        atac = batch[REGISTRY_KEYS.ATAC_X_KEY].numpy()
        seen.update(ind_x.tolist())
        for i, row_idx in enumerate(ind_x):
            np.testing.assert_allclose(rna[i], ref_rna[row_idx], rtol=1e-5)
            np.testing.assert_allclose(atac[i], ref_atac[row_idx], rtol=1e-5)
    assert seen == set(indices.tolist())


def _make_rna_protein_mixed_zarr_store(
    tmp_path,
    batch_size: int = 32,
    n_genes: int = 50,
    n_proteins: int = 20,
):
    """Zarr store with RNA CSR and protein dense ``zarr.Array`` on ``.X``."""
    mdata_raw = synthetic_iid(
        return_mudata=True,
        batch_size=batch_size,
        n_genes=n_genes,
        n_proteins=n_proteins,
        n_regions=0,
        n_batches=2,
    )
    # Preserve canonical MULTIVI modality order (RNA before protein) without relying on
    # mdata._mod reorder, which is unavailable on some mudata versions.
    mdata = md.MuData(
        {
            "RNA": mdata_raw.mod["rna"],
            "protein_expression": mdata_raw.mod["protein_expression"],
        }
    )
    mdata.obs = mdata_raw.obs.copy()
    mdata.mod["RNA"].X = sp.random(
        mdata.n_obs, mdata.mod["RNA"].n_vars, density=0.2, format="csr", dtype=np.float32
    )

    store_path = tmp_path / "rna_protein.zarr"
    import anndata

    previous = anndata.settings.allow_write_nullable_strings
    anndata.settings.allow_write_nullable_strings = True
    try:
        mdata.write_zarr(store_path)
    finally:
        anndata.settings.allow_write_nullable_strings = previous
    backed = md.read_zarr(store_path)
    f = zarr.open(str(store_path), mode="r")
    x_group = f["mod"]["RNA"]["X"]
    enc = x_group.attrs.get("encoding-type", "")
    if enc in ("csr_matrix", "csc_matrix"):
        backed.mod["RNA"].X = sparse_dataset(x_group)

    protein_mod = "protein_expression"
    protein_x = mdata.mod[protein_mod].X
    protein_dense = protein_x.toarray() if sp.issparse(protein_x) else np.asarray(protein_x)
    dest = store_path / "mod" / protein_mod / "X_dense"
    arr = zarr.open_array(
        str(dest),
        mode="w",
        shape=protein_dense.shape,
        chunks=(min(64, mdata.n_obs), protein_dense.shape[1]),
        dtype="float32",
    )
    arr[:] = protein_dense.astype(np.float32, copy=False)
    backed.mod[protein_mod].X = zarr.open_array(str(dest), mode="r")
    return mdata, backed, store_path


def _attach_protein_dense_zarr(backed: MuData, store_path: Path, mod_key: str = "protein_expression") -> zarr.Array:
    """Attach dense zarr.Array to protein modality ``.X`` for streaming tests."""
    dest = store_path / "mod" / mod_key / "X_dense"
    arr = zarr.open_array(str(dest), mode="r")
    backed.mod[mod_key].X = arr
    return arr


def test_zarr_datamodule_mixed_rna_protein_auto_layout(tmp_path):
    """from_backed_mudata infers RNA CSR + protein dense with matrix_layout='auto'."""
    mdata, mdata_backed, store_path = _make_rna_protein_mixed_zarr_store(tmp_path)
    MULTIVI.setup_mudata(
        mdata,
        batch_key="batch",
        modalities={"rna_layer": "RNA", "protein_layer": "protein_expression"},
    )
    _attach_protein_dense_zarr(mdata_backed, store_path)
    model = MULTIVI(mdata)
    dm = ZarrMultiVIDataModule.from_backed_mudata(
        mdata_backed,
        model.adata_manager,
        train_size=0.8,
        batch_size=8,
        block_size=8,
        shuffle_buffer_blocks=1,
        num_workers=0,
        seed=0,
        matrix_layout="auto",
        store_dir=store_path,
    )
    assert REGISTRY_KEYS.X_KEY in dm.matrix_sources
    assert REGISTRY_KEYS.PROTEIN_EXP_KEY in dm.matrix_sources
    assert dm.matrix_sources[REGISTRY_KEYS.X_KEY].layout == "csr"
    assert dm.matrix_sources[REGISTRY_KEYS.PROTEIN_EXP_KEY].layout == "dense"

    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch[REGISTRY_KEYS.X_KEY].shape[0] == batch[REGISTRY_KEYS.PROTEIN_EXP_KEY].shape[0]
    assert batch[REGISTRY_KEYS.X_KEY].shape[1] == mdata.mod["RNA"].n_vars
    assert batch[REGISTRY_KEYS.PROTEIN_EXP_KEY].shape[1] == mdata.mod["protein_expression"].n_vars


@pytest.mark.parametrize("num_workers", [0, 2])
def test_multivi_mixed_rna_protein_datamodule_smoke(tmp_path, num_workers):
    """MULTIVI trains with RNA CSR + protein dense zarr streaming."""
    mdata, mdata_backed, store_path = _make_rna_protein_mixed_zarr_store(tmp_path)
    MULTIVI.setup_mudata(
        mdata,
        batch_key="batch",
        modalities={"rna_layer": "RNA", "protein_layer": "protein_expression"},
    )
    _attach_protein_dense_zarr(mdata_backed, store_path)
    model = MULTIVI(mdata)
    dm = ZarrMultiVIDataModule.from_backed_mudata(
        mdata_backed,
        model.adata_manager,
        train_size=0.8,
        batch_size=16,
        block_size=16,
        shuffle_buffer_blocks=2,
        num_workers=num_workers,
        seed=0,
        matrix_layout="auto",
        store_dir=store_path,
    )
    model.train(
        max_epochs=1,
        datamodule=dm,
        early_stopping=False,
        check_val_every_n_epoch=1,
    )


def _collect_indices(ds: ZarrDataset) -> set[int]:
    seen: set[int] = set()
    for batch in ds:
        seen.update(batch[REGISTRY_KEYS.INDICES_KEY].view(-1).tolist())
    return seen


@pytest.mark.parametrize("emit_mode", ["rolling", "flush"])
def test_zarr_dataset_emit_mode_coverage(zarr_store, emit_mode):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    rng = np.random.default_rng(42)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=7,
        block_size=11,
        shuffle=True,
        shuffle_buffer_blocks=2,
        emit_mode=emit_mode,
        seed=42,
        drop_last=False,
    )
    assert _collect_indices(ds) == set(indices.tolist())


def test_zarr_dataset_rolling_emit_pairing(zarr_store):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    rng = np.random.default_rng(99)
    indices = rng.permutation(mdata.n_obs).astype(np.int64)
    ref_rna = csr_datasets[REGISTRY_KEYS.X_KEY][:].toarray()
    ref_atac = csr_datasets[REGISTRY_KEYS.ATAC_X_KEY][:].toarray()
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=5,
        block_size=7,
        shuffle=True,
        shuffle_buffer_blocks=1,
        emit_mode="rolling",
        seed=0,
    )
    for batch in ds:
        ind_x = batch[REGISTRY_KEYS.INDICES_KEY].view(-1).numpy()
        rna = batch[REGISTRY_KEYS.X_KEY].numpy()
        atac = batch[REGISTRY_KEYS.ATAC_X_KEY].numpy()
        for i, row_idx in enumerate(ind_x):
            np.testing.assert_allclose(rna[i], ref_rna[row_idx], rtol=1e-5)
            np.testing.assert_allclose(atac[i], ref_atac[row_idx], rtol=1e-5)


def test_zarr_dataset_rolling_emit_yields_before_all_blocks_read(zarr_store):
    """Rolling emit must not wait for all blocks when shuffle_buffer_blocks exceeds block count."""
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    indices = np.arange(mdata.n_obs, dtype=np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=4,
        block_size=8,
        shuffle=True,
        shuffle_buffer_blocks=1,
        emit_mode="rolling",
        seed=0,
    )
    read_count = 0
    original_read = ds._read_block

    def counting_read(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        return original_read(*args, **kwargs)

    ds._read_block = counting_read  # type: ignore[method-assign]
    next(iter(ds))
    total_blocks = len(list(range(0, mdata.n_obs, 8)))
    assert read_count < total_blocks


def test_zarr_dataset_min_shuffle_pool_rows():
    from scvi.dataloaders._zarr_dataset import _min_shuffle_pool_rows

    assert _min_shuffle_pool_rows(16, 4096, 128) == 65536
    assert _min_shuffle_pool_rows(1, 8, 32) == 32


@pytest.mark.parametrize("prefetch_queue_depth", [0, 2])
def test_zarr_dataset_prefetch_queue_coverage(zarr_store, prefetch_queue_depth):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    indices = np.arange(mdata.n_obs, dtype=np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=6,
        block_size=10,
        shuffle=True,
        shuffle_buffer_blocks=1,
        emit_mode="rolling",
        prefetch_queue_depth=prefetch_queue_depth,
        seed=1,
    )
    assert _collect_indices(ds) == set(indices.tolist())


def test_zarr_dataset_prefetch_queue_multiworker(zarr_store):
    _, mdata_backed = zarr_store
    registry_map = {
        REGISTRY_KEYS.X_KEY: "RNA",
        REGISTRY_KEYS.ATAC_X_KEY: "ATAC",
    }
    sources = csr_sources_from_backed_mudata(mdata_backed, registry_map)
    indices = np.arange(mdata_backed.n_obs, dtype=np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata_backed.n_obs),
        indices=indices,
        csr_sources=sources,
        batch_size=5,
        block_size=10,
        shuffle=True,
        shuffle_buffer_blocks=1,
        emit_mode="rolling",
        prefetch_queue_depth=2,
        seed=0,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=2,
        collate_fn=_identity_collate,
        prefetch_factor=2,
    )
    seen = _collect_indices_from_loader(loader)
    assert seen == set(indices.tolist())


def test_zarr_dataset_prefetch_producer_error(zarr_store):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    indices = np.arange(mdata.n_obs, dtype=np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=4,
        block_size=8,
        shuffle=True,
        shuffle_buffer_blocks=1,
        emit_mode="rolling",
        prefetch_queue_depth=2,
        seed=0,
    )
    calls = 0
    original_read = ds._read_block

    def failing_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated read failure")
        return original_read(*args, **kwargs)

    ds._read_block = failing_read  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated read failure"):
        list(ds)


@pytest.mark.parametrize("block_prefetch_depth", [0, 1])
def test_zarr_dataset_block_prefetch_coverage(zarr_store, block_prefetch_depth):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    indices = np.arange(mdata.n_obs, dtype=np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=6,
        block_size=10,
        shuffle=True,
        shuffle_buffer_blocks=1,
        emit_mode="rolling",
        block_prefetch_depth=block_prefetch_depth,
        seed=2,
    )
    assert _collect_indices(ds) == set(indices.tolist())


def test_zarr_dataset_block_prefetch_serial_reads(zarr_store):
    mdata, mdata_backed = zarr_store
    csr_datasets = _csr_datasets_from_backed(mdata_backed)
    indices = np.arange(mdata.n_obs, dtype=np.int64)
    ds = ZarrDataset(
        obs_tensors=_obs_tensors(mdata.n_obs),
        indices=indices,
        csr_datasets=csr_datasets,
        batch_size=4,
        block_size=8,
        shuffle=True,
        shuffle_buffer_blocks=1,
        emit_mode="rolling",
        block_prefetch_depth=1,
        seed=0,
    )
    active_reads = 0
    max_active = 0
    original_read = ds._read_block

    def tracked_read(*args, **kwargs):
        nonlocal active_reads, max_active
        active_reads += 1
        max_active = max(max_active, active_reads)
        try:
            return original_read(*args, **kwargs)
        finally:
            active_reads -= 1

    ds._read_block = tracked_read  # type: ignore[method-assign]
    list(ds)
    assert max_active == 1


def test_zarr_datamodule_prefetch_to_gpu_requires_pin_memory(zarr_store):
    mdata, mdata_backed = zarr_store
    MULTIVI.setup_mudata(
        mdata,
        batch_key="batch",
        modalities={"rna_layer": "RNA", "atac_layer": "ATAC"},
    )
    model = MULTIVI(mdata)
    dm = ZarrMultiVIDataModule.from_backed_mudata(
        mdata_backed,
        model.adata_manager,
        num_workers=0,
        pin_memory=False,
        prefetch_to_gpu=True,
    )
    with pytest.raises(ValueError, match="pin_memory=True"):
        dm.train_dataloader()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zarr_datamodule_prefetch_to_gpu_smoke(zarr_store):
    mdata, mdata_backed = zarr_store
    MULTIVI.setup_mudata(
        mdata,
        batch_key="batch",
        modalities={"rna_layer": "RNA", "atac_layer": "ATAC"},
    )
    model = MULTIVI(mdata)
    dm = ZarrMultiVIDataModule.from_backed_mudata(
        mdata_backed,
        model.adata_manager,
        train_size=0.8,
        batch_size=8,
        block_size=8,
        shuffle_buffer_blocks=1,
        emit_mode="rolling",
        prefetch_queue_depth=2,
        num_workers=0,
        pin_memory=True,
        prefetch_to_gpu=True,
        cuda_queue_depth=2,
        seed=0,
    )
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch[REGISTRY_KEYS.X_KEY].is_cuda
    assert batch[REGISTRY_KEYS.ATAC_X_KEY].is_cuda


@pytest.mark.parametrize("num_workers", [0, 2])
def test_multivi_dense_zarr_datamodule_smoke(dense_zarr_store, num_workers):
    _, mdata_backed, _, store_path = dense_zarr_store
    MULTIVI.setup_mudata(
        mdata_backed,
        batch_key="batch",
        modalities={"rna_layer": "RNA", "atac_layer": "ATAC"},
    )
    model = MULTIVI(mdata_backed)
    dm = ZarrMultiVIDataModule.from_backed_mudata(
        mdata_backed,
        model.adata_manager,
        train_size=0.8,
        batch_size=16,
        block_size=16,
        shuffle_buffer_blocks=2,
        num_workers=num_workers,
        seed=0,
        matrix_layout="dense",
        store_dir=store_path,
    )
    model.train(
        max_epochs=1,
        datamodule=dm,
        early_stopping=False,
        check_val_every_n_epoch=1,
    )


@pytest.fixture
def anndata_zarr_store(tmp_path):
    return _make_synthetic_anndata_zarr_store(tmp_path)


def test_matrix_sources_from_backed_anndata(anndata_zarr_store):
    _, adata_backed = anndata_zarr_store
    SCVI.setup_anndata(adata_backed, batch_key="batch")
    model = SCVI(adata_backed)
    sources = matrix_sources_from_backed_anndata(adata_backed, model.adata_manager)
    assert set(sources) == {REGISTRY_KEYS.X_KEY}
    assert sources[REGISTRY_KEYS.X_KEY].layout == "csr"
    assert Path(sources[REGISTRY_KEYS.X_KEY].x_relpath).is_absolute()


def test_zarr_ann_datamodule_smoke(anndata_zarr_store):
    _, adata_backed = anndata_zarr_store
    SCVI.setup_anndata(adata_backed, batch_key="batch")
    model = SCVI(adata_backed)
    dm = ZarrAnnDataModule.from_backed_anndata(
        adata_backed,
        model.adata_manager,
        matrix_layout="auto",
        num_workers=0,
        batch_size=16,
    )
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch[REGISTRY_KEYS.X_KEY].shape[0] == 16
    assert batch[REGISTRY_KEYS.BATCH_KEY].shape[0] == 16


def test_zarr_ann_datamodule_with_covariates(anndata_zarr_store):
    _, adata_backed = anndata_zarr_store
    adata_backed.obs["extra_cat"] = np.random.randint(0, 3, size=adata_backed.n_obs)
    adata_backed.obs["extra_cont"] = np.random.randn(adata_backed.n_obs)
    SCVI.setup_anndata(
        adata_backed,
        batch_key="batch",
        categorical_covariate_keys=["extra_cat"],
        continuous_covariate_keys=["extra_cont"],
    )
    model = SCVI(adata_backed)
    dm = ZarrAnnDataModule.from_backed_anndata(
        adata_backed,
        model.adata_manager,
        matrix_layout="auto",
        num_workers=0,
        batch_size=16,
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch[REGISTRY_KEYS.CAT_COVS_KEY].shape[0] == 16
    assert batch[REGISTRY_KEYS.CONT_COVS_KEY].shape[0] == 16


def test_scvi_train_with_zarr_ann_datamodule(anndata_zarr_store):
    _, adata_backed = anndata_zarr_store
    SCVI.setup_anndata(adata_backed, batch_key="batch")
    model = SCVI(adata_backed)
    dm = ZarrAnnDataModule.from_backed_anndata(
        adata_backed,
        model.adata_manager,
        matrix_layout="auto",
        num_workers=0,
        batch_size=16,
    )
    model.train(
        max_epochs=1,
        datamodule=dm,
        early_stopping=False,
        check_val_every_n_epoch=1,
    )


def test_peakvi_train_with_zarr_ann_datamodule(anndata_zarr_store):
    _, adata_backed = anndata_zarr_store
    PEAKVI.setup_anndata(adata_backed, batch_key="batch")
    model = PEAKVI(adata_backed)
    dm = ZarrAnnDataModule.from_backed_anndata(
        adata_backed,
        model.adata_manager,
        matrix_layout="auto",
        num_workers=0,
        batch_size=16,
    )
    model.train(
        max_epochs=1,
        datamodule=dm,
        early_stopping=False,
        check_val_every_n_epoch=1,
    )
