from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import zarr
from anndata.io import sparse_dataset

from scvi.data import synthetic_iid
from scvi import REGISTRY_KEYS
from scvi.dataloaders import DataSplitter, ZarrAnnDataModule, ZarrMatrixSource
from scvi.model import PEAKVI, SCVI


def _write_and_open_backed_anndata(adata, store_path: Path):
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


class _CaptureTrainRunner:
    last_call: dict | None = None

    def __init__(self, model, training_plan, data_splitter, **kwargs):
        type(self).last_call = {
            "data_splitter": data_splitter,
            "trainer_kwargs": kwargs,
        }

    def __call__(self):
        return None


@pytest.fixture
def backed_adata(tmp_path):
    adata = synthetic_iid(batch_size=32, n_genes=20, n_batches=2)
    adata.X = sp.random(adata.n_obs, 20, density=0.2, format="csr", dtype=np.float32)
    store_path = tmp_path / "adata.zarr"
    return _write_and_open_backed_anndata(adata, store_path)


@pytest.mark.parametrize("model_cls", [SCVI, PEAKVI])
def test_train_auto_selects_zarr_datamodule(model_cls, backed_adata):
    model_cls.setup_anndata(backed_adata, batch_key="batch")
    model = model_cls(backed_adata)
    zarr_dm = ZarrAnnDataModule(
        model.adata_manager,
        matrix_sources={
            REGISTRY_KEYS.X_KEY: ZarrMatrixSource(REGISTRY_KEYS.X_KEY, "/fake", "csr")
        },
        n_obs=backed_adata.n_obs,
    )
    model._try_auto_zarr_datamodule = lambda **kwargs: zarr_dm
    model._train_runner_cls = _CaptureTrainRunner

    model.train(max_epochs=1, early_stopping=False)

    assert _CaptureTrainRunner.last_call["data_splitter"] is zarr_dm
    assert (
        _CaptureTrainRunner.last_call["trainer_kwargs"]["reload_dataloaders_every_n_epochs"]
        == 1
    )


@pytest.mark.parametrize("model_cls", [SCVI, PEAKVI])
def test_train_falls_back_to_data_splitter_for_in_memory(model_cls):
    adata = synthetic_iid(batch_size=32, n_genes=20, n_batches=2)
    model_cls.setup_anndata(adata, batch_key="batch")
    model = model_cls(adata)
    model._train_runner_cls = _CaptureTrainRunner

    model.train(max_epochs=1, early_stopping=False)

    assert isinstance(_CaptureTrainRunner.last_call["data_splitter"], DataSplitter)
    assert (
        "reload_dataloaders_every_n_epochs"
        not in _CaptureTrainRunner.last_call["trainer_kwargs"]
    )


@pytest.mark.parametrize("model_cls", [SCVI, PEAKVI])
def test_train_manual_non_zarr_datamodule_does_not_force_reload(model_cls):
    adata = synthetic_iid(batch_size=32, n_genes=20, n_batches=2)
    model_cls.setup_anndata(adata, batch_key="batch")
    model = model_cls(adata)
    model._train_runner_cls = _CaptureTrainRunner
    explicit_dm = object()

    model.train(max_epochs=1, datamodule=explicit_dm, early_stopping=False)

    assert _CaptureTrainRunner.last_call["data_splitter"] is explicit_dm
    assert (
        "reload_dataloaders_every_n_epochs"
        not in _CaptureTrainRunner.last_call["trainer_kwargs"]
    )


@pytest.mark.parametrize("model_cls", [SCVI, PEAKVI])
def test_train_manual_zarr_datamodule_sets_reload(model_cls, backed_adata):
    model_cls.setup_anndata(backed_adata, batch_key="batch")
    model = model_cls(backed_adata)
    model._train_runner_cls = _CaptureTrainRunner
    zarr_dm = ZarrAnnDataModule(
        model.adata_manager,
        matrix_sources={
            REGISTRY_KEYS.X_KEY: ZarrMatrixSource(REGISTRY_KEYS.X_KEY, "/fake", "csr")
        },
        n_obs=backed_adata.n_obs,
    )

    model.train(max_epochs=1, datamodule=zarr_dm, early_stopping=False)

    assert _CaptureTrainRunner.last_call["data_splitter"] is zarr_dm
    assert (
        _CaptureTrainRunner.last_call["trainer_kwargs"]["reload_dataloaders_every_n_epochs"]
        == 1
    )


def test_scvi_try_auto_zarr_datamodule_with_backed_anndata(backed_adata):
    SCVI.setup_anndata(backed_adata, batch_key="batch")
    model = SCVI(backed_adata)
    datamodule = model._try_auto_zarr_datamodule(
        train_size=0.9,
        validation_size=None,
        batch_size=16,
        datasplitter_kwargs={},
    )
    assert isinstance(datamodule, ZarrAnnDataModule)
