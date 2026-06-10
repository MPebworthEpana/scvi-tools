"""Tests for Set Transformer ATAC encoder components."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from scvi.encoders import (
    build_coord_table,
    build_genomic_rank,
    csr_batch_to_tokens,
    tokenize_atac,
)
from scvi.encoders._set_transformer import CardinalityFiLM, PoolingByMultiheadAttention
from scvi.encoders._set_transformer_atac_variational import SetTransformerAtacVariationalEncoder


def test_pma_permutation_invariance():
    d_model = 32
    pool = PoolingByMultiheadAttention(d_model, n_seeds=1, n_heads=4)
    x = torch.randn(2, 8, d_model)
    mask = torch.ones(2, 8, dtype=torch.bool)
    out1 = pool(x, key_mask=mask)
    perm = torch.randperm(8)
    out2 = pool(x[:, perm], key_mask=mask[:, perm])
    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-4)


def test_cardinality_film_changes_output_with_n():
    film = CardinalityFiLM(32)
    z = torch.randn(4, 32)
    n_low = torch.full((4, 1), 100.0)
    n_high = torch.full((4, 1), 5000.0)
    n_total = torch.full((4, 1), 10000.0)
    out_low = film(z, n_low, n_total)
    out_high = film(z, n_high, n_total)
    assert not torch.allclose(out_low, out_high)


def test_set_transformer_variational_encoder_shapes():
    coord = build_coord_table([f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(20)])
    coord_tensor = torch.as_tensor(coord, dtype=torch.long)
    enc = SetTransformerAtacVariationalEncoder(
        n_latent=8,
        coord_table=coord_tensor,
        n_regions=20,
        d_model=32,
        n_layers=1,
        n_inducing=8,
        n_heads=4,
    )
    token_ids = torch.tensor([[0, 2, 5, 0, 0], [1, 3, 0, 0, 0]], dtype=torch.long)
    token_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, False]],
        dtype=torch.bool,
    )
    batch_index = torch.zeros(2, dtype=torch.long)
    q_m, q_v, z = enc(token_ids, token_mask, batch_index)
    assert q_m.shape == (2, 8)
    assert q_v.shape == (2, 8)
    assert z.shape == (2, 8)


def test_csr_batch_matches_tokenize_row():
    x = sp.csr_matrix(
        [
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.float32,
    )
    coord = np.array(
        [[1, 100, 200], [1, 300, 400], [2, 100, 200], [2, 500, 600]],
        dtype=np.int64,
    )
    rank = build_genomic_rank(coord)
    ids_batch, mask_batch = csr_batch_to_tokens(x, coord, 10, genomic=True, genomic_rank=rank)
    row = x.getrow(0)
    tok = tokenize_atac(
        row.indices.astype(np.int64),
        row.data.astype(np.float32),
        10,
        coord,
        genomic=True,
        genomic_rank=rank,
    )
    assert list(ids_batch[0, : mask_batch[0].sum()]) == list(tok["ids"])


def test_set_dataset_csr_streaming_emits_tokens():
    from scvi import REGISTRY_KEYS
    from scvi.data import synthetic_iid
    from scvi.data.fields._atac_token_field import AtacTokenConfigField
    from scvi.encoders._constants import ATAC_TOKEN_CONFIG_KEY, ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY
    from scvi.model import SETVI

    mdata = synthetic_iid(return_mudata=True, batch_size=8)
    atac = mdata.mod["accessibility"]
    atac.var_names = [f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(atac.n_vars)]
    SETVI.setup_mudata(
        mdata,
        modalities={
            "rna_layer": "rna",
            "protein_layer": "protein_expression",
            "atac_layer": "accessibility",
        },
        max_atac_tokens=32,
    )
    model = SETVI(mdata, max_atac_tokens=32, st_d_model=32, st_n_layers=1, st_n_inducing=8)
    token_cfg = model.adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)
    assert not token_cfg[AtacTokenConfigField.PRECOMPUTED_KEY]
    assert AtacTokenConfigField.NN_LENGTHS_KEY in token_cfg

    loader = model._data_splitter_cls(
        model.adata_manager,
        batch_size=4,
        train_data_and_attributes=model._balanced_subsample_train_keys(),
    )
    loader.setup()
    batch = next(iter(loader.train_dataloader()))
    assert ATAC_TOKEN_IDS_KEY in batch
    assert ATAC_TOKEN_MASK_KEY in batch
    assert REGISTRY_KEYS.ATAC_X_KEY not in batch
