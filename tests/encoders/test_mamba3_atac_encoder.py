"""Tests for Mamba3 ATAC encoder components."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from scvi.encoders import build_coord_table, csr_batch_to_tokens, tokenize_atac


def test_build_coord_table_parses_peak_names():
    names = [f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(5)]
    table = build_coord_table(names)
    assert table.shape == (5, 3)
    assert table[0, 0] == 1
    assert table[0, 1] == 1000


def test_tokenize_atac_genomic_order():
    coord = build_coord_table(
        ["chr2:2000-2500", "chr1:1000-1500", "chr1:3000-3500"]
    )
    tok = tokenize_atac(
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        max_tokens=10,
        coord_table=coord,
        genomic=True,
    )
    assert list(tok["ids"]) == [1, 2, 0]


def test_csr_batch_to_tokens_padding():
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
    ids, mask = csr_batch_to_tokens(x, coord, max_tokens=10, genomic=False)
    assert ids.shape == mask.shape
    assert ids.shape[0] == 3
    assert mask[1].sum() == 0
    assert mask[0].sum() == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba3")
def test_bidirectional_mamba3_encoder_forward():
    pytest.importorskip("mamba_ssm")
    from mamba_ssm import Mamba3  # noqa: F401

    from scvi.encoders._mamba3_encoder import BidirectionalMamba3Encoder

    enc = BidirectionalMamba3Encoder(d_model=32, n_layers=1, headdim=16, chunk_size=8)
    enc = enc.cuda()
    tokens = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, 16, dtype=torch.bool, device="cuda")
    chrom = torch.ones(2, 16, dtype=torch.long, device="cuda")
    out = enc(tokens, mask, chrom=chrom)
    assert out.shape == (2, 32)


def test_mamba_dataset_emits_tokens():
    pytest.importorskip("mamba_ssm")
    from mudata import MuData

    import scvi
    from scvi.data import synthetic_iid
    from scvi.dataloaders._mamba_dataset import MambaAnnTorchDataset
    from scvi.encoders._constants import ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY
    from scvi.model import MAMBAVI

    mdata = synthetic_iid(return_mudata=True)
    atac = mdata.mod["accessibility"]
    atac.var_names = [f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(atac.n_vars)]
    MAMBAVI.setup_mudata(
        mdata,
        modalities={
            "rna_layer": "rna",
            "protein_layer": "protein_expression",
            "atac_layer": "accessibility",
        },
        max_atac_tokens=32,
    )
    model = MAMBAVI(mdata, max_atac_tokens=32)
    ds = MambaAnnTorchDataset(model.adata_manager)
    batch = ds[[0, 1, 2]]
    assert ATAC_TOKEN_IDS_KEY in batch
    assert ATAC_TOKEN_MASK_KEY in batch
    assert batch[ATAC_TOKEN_IDS_KEY].shape[0] == 3
