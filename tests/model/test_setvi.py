"""SETVI smoke tests for ATAC-only and multimodal training."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.drift


def _setup_setvi_multimodal():
    from scvi.data import synthetic_iid
    from scvi.model import SETVI

    mdata = synthetic_iid(return_mudata=True, batch_size=32, n_batches=2)
    atac = mdata.mod["accessibility"]
    atac.var_names = [f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(atac.n_vars)]
    SETVI.setup_mudata(
        mdata,
        batch_key="batch",
        modalities={
            "rna_layer": "rna",
            "atac_layer": "accessibility",
        },
        max_atac_tokens=32,
    )
    return SETVI(
        mdata,
        max_atac_tokens=32,
        st_d_model=32,
        st_n_layers=1,
        st_n_inducing=8,
        st_n_heads=4,
    )


def _setup_setvi_atac_only():
    from scvi.data import synthetic_iid
    from scvi.model import SETVI

    mdata = synthetic_iid(return_mudata=True, batch_size=32, n_batches=2)
    atac = mdata.mod["accessibility"]
    atac.var_names = [f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(atac.n_vars)]
    # ATAC-only: omit RNA/protein modalities
    SETVI.setup_mudata(
        mdata,
        batch_key="batch",
        modalities={"atac_layer": "accessibility"},
        max_atac_tokens=32,
    )
    return SETVI(
        mdata,
        n_genes=0,
        max_atac_tokens=32,
        st_d_model=32,
        st_n_layers=1,
        st_n_inducing=8,
        st_n_heads=4,
    )


def test_setvi_multimodal_smoke_train():
    model = _setup_setvi_multimodal()
    model.train(
        max_epochs=1,
        batch_size=32,
        train_size=1.0,
        validation_size=0,
        accelerator="cpu",
        enable_progress_bar=False,
        early_stopping=False,
    )
    assert model.is_trained
    latent = model.get_latent_representation()
    assert latent.shape[0] == model.adata_manager.adata.n_obs
    assert latent.shape[1] == model.module.n_latent
    assert np.isfinite(latent).all()


def test_setvi_atac_only_smoke_train():
    model = _setup_setvi_atac_only()
    model.train(
        max_epochs=1,
        batch_size=32,
        train_size=1.0,
        validation_size=0,
        accelerator="cpu",
        enable_progress_bar=False,
        early_stopping=False,
    )
    assert model.is_trained
    latent = model.get_latent_representation()
    assert latent.shape[0] == model.adata_manager.adata.n_obs
    assert latent.shape[1] == model.module.n_latent
    assert np.isfinite(latent).all()
