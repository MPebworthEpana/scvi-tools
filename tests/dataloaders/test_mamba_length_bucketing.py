"""Tests for MAMBAVI length-bucketed batching and train split indexing."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scvi.dataloaders._length_bucket_sampler import LengthBucketedBatchSampler
from scvi.dataloaders._mamba_splitter import MambaDataSplitter


def test_length_bucket_batches_have_uniform_lengths():
    lengths = np.array([10] * 8 + [100] * 8, dtype=np.int64)
    sampler = LengthBucketedBatchSampler(
        lengths, batch_size=4, shuffle=False, seed=0, bucket_mult=4
    )
    for batch in sampler:
        batch_lengths = lengths[batch]
        assert batch_lengths.max() - batch_lengths.min() == 0


def test_length_bucket_shuffles_batch_order():
    lengths = np.arange(100, dtype=np.int64)
    sampler = LengthBucketedBatchSampler(
        lengths, batch_size=4, shuffle=True, seed=42, bucket_mult=10
    )
    batch_means = [lengths[batch].mean() for batch in sampler]
    diffs = np.diff(batch_means)
    assert not np.all(diffs <= 0) and not np.all(diffs >= 0)


def test_mamba_splitter_train_loader_respects_train_idx():
    pytest.importorskip("mamba_ssm")
    from scvi.data import synthetic_iid
    from scvi.model import MAMBAVI

    from scvi import REGISTRY_KEYS

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
        precompute_atac_tokens=True,
    )
    model = MAMBAVI(mdata, max_atac_tokens=32)
    splitter = MambaDataSplitter(
        model.adata_manager,
        train_size=0.5,
        validation_size=0.2,
        batch_size=16,
        atac_length_bucketing=True,
        load_sparse_tensor=True,
    )
    splitter.setup()
    train_idx_set = set(splitter.train_idx.tolist())
    loader = splitter.train_dataloader()
    seen = set()
    for batch in loader:
        indices = np.asarray(batch[REGISTRY_KEYS.INDICES_KEY]).reshape(-1).tolist()
        for idx in indices:
            seen.add(int(idx))
            assert int(idx) in train_idx_set
    assert seen.issubset(train_idx_set)
    assert len(seen) > 0
