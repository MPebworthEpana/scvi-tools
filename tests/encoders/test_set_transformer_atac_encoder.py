"""Tests for Set Transformer ATAC encoder components."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from scvi.module import SETVAE
from scvi.tokenized import (
    CardinalityFiLM,
    PoolingByMultiheadAttention,
    SetTransformerAtacVariationalEncoder,
    build_chrom_vocab,
    build_coord_table,
    build_genomic_rank,
    build_token_store,
    csr_batch_to_tokens,
    parse_peak_name,
    tokenize_atac,
)
from scvi.tokenized._field import AtacTokenConfigField
from scvi.tokenized._token_store import AtacTokenStore, rebuild_token_store


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


def test_accessibility_target_from_tokens_matches_sparse():
    coord = build_coord_table([f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(8)])
    x = sp.csr_matrix(
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    rank = build_genomic_rank(coord)
    store = build_token_store(
        x,
        coord,
        rank,
        max_encoder_tokens=8,
        genomic=True,
        tier="ram",
    )
    ids, mask = store.gather([0, 1, 2], for_encoder=False)
    target = SETVAE._accessibility_target_from_tokens(
        torch.as_tensor(ids, dtype=torch.long),
        torch.as_tensor(mask, dtype=torch.bool),
        8,
    )
    expected = torch.as_tensor((x.toarray() > 0).astype(np.float32))
    torch.testing.assert_close(target, expected)


@pytest.mark.parametrize("tier", ["ram", "mmap"])
def test_token_store_tier_equivalence(tier):
    coord = build_coord_table([f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(12)])
    x = sp.random(16, 12, density=0.2, format="csr", dtype=np.float32, random_state=0)
    rank = build_genomic_rank(coord)
    kwargs = {"tier": tier}
    if tier == "mmap":
        tmp = tempfile.mkdtemp()
        kwargs["out_dir"] = tmp
    ram_store = build_token_store(
        x, coord, rank, max_encoder_tokens=12, genomic=True, tier="ram"
    )
    other_store = build_token_store(
        x, coord, rank, max_encoder_tokens=12, genomic=True, **kwargs
    )
    idx = np.array([0, 3, 7, 15], dtype=np.int64)
    ids_a, mask_a = ram_store.gather(idx, for_encoder=False)
    ids_b, mask_b = other_store.gather(idx, for_encoder=False)
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_array_equal(mask_a, mask_b)


def test_set_dataset_token_native_emits_tokens_without_atac_x():
    from scvi import REGISTRY_KEYS
    from scvi.data import synthetic_iid
    from scvi.model import SETVI
    from scvi.tokenized import (
        ATAC_TOKEN_CONFIG_KEY,
        ATAC_TOKEN_IDS_KEY,
        ATAC_TOKEN_MASK_KEY,
        ATAC_TOKEN_VALUES_KEY,
        AtacTokenConfigField,
    )

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
        atac_token_store="ram",
    )
    model = SETVI(mdata, max_atac_tokens=32, st_d_model=32, st_n_layers=1, st_n_inducing=8)
    token_cfg = model.adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)
    assert AtacTokenConfigField.TOKEN_STORE_KEY in token_cfg
    assert AtacTokenConfigField.TOKEN_STORE_HANDLE_KEY in token_cfg
    assert AtacTokenConfigField.PRECOMPUTED_IDS_KEY not in token_cfg

    loader = model._data_splitter_cls(
        model.adata_manager,
        batch_size=4,
        load_sparse_tensor=False,
    )
    loader.setup()
    batch = next(iter(loader.train_dataloader()))
    assert ATAC_TOKEN_IDS_KEY in batch
    assert ATAC_TOKEN_MASK_KEY in batch
    assert ATAC_TOKEN_VALUES_KEY in batch
    assert REGISTRY_KEYS.ATAC_X_KEY not in batch


def test_token_store_handle_roundtrip_mmap():
    coord = build_coord_table([f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(6)])
    x = sp.random(10, 6, density=0.3, format="csr", dtype=np.float32, random_state=1)
    rank = build_genomic_rank(coord)
    with tempfile.TemporaryDirectory() as tmp:
        store = build_token_store(
            x,
            coord,
            rank,
            max_encoder_tokens=6,
            genomic=True,
            tier="mmap",
            out_dir=tmp,
        )
        handle = store.to_handle()
        reopened = AtacTokenStore.from_handle(handle)
        ids_a, mask_a = store.gather([1, 4], for_encoder=False)
        ids_b, mask_b = reopened.gather([1, 4], for_encoder=False)
        np.testing.assert_array_equal(ids_a, ids_b)
        np.testing.assert_array_equal(mask_a, mask_b)
        assert Path(handle["mmap_dir"]).exists()


def test_token_store_late_chunk_truncation_alignment():
    """Regression: truncation discovered in a late chunk must not misalign ids/values."""
    n_regions = 20
    coord = build_coord_table([f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(n_regions)])
    rank = build_genomic_rank(coord)
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(4):
        r = np.zeros(n_regions)
        idx = rng.choice(n_regions, 3, replace=False)
        r[idx] = rng.integers(1, 9, 3)
        rows.append(r)
    for k in range(4):
        r = np.zeros(n_regions)
        if k == 0:
            r[:] = rng.integers(1, 9, n_regions)
        else:
            idx = rng.choice(n_regions, 2, replace=False)
            r[idx] = rng.integers(1, 9, 2)
        rows.append(r)
    x = sp.csr_matrix(np.vstack(rows), dtype=np.float32)
    max_tokens = 5
    store = build_token_store(
        x,
        coord,
        rank,
        max_encoder_tokens=max_tokens,
        genomic=True,
        tier="ram",
        chunk_size=4,
    )
    assert store.truncation_possible
    assert store.values is not None
    assert len(store.values) == len(store.ids)
    for row in range(x.shape[0]):
        row_csr = x.getrow(row)
        expected = tokenize_atac(
            row_csr.indices.astype(np.int64),
            row_csr.data.astype(np.float32),
            max_tokens,
            coord,
            genomic=True,
            genomic_rank=rank,
        )
        enc_ids, enc_mask = store.gather([row], for_encoder=True)
        got = enc_ids[0, enc_mask[0]]
        assert list(got) == list(expected["ids"])
        full_ids, full_mask = store.gather([row], for_encoder=False)
        assert int(full_mask[0].sum()) == int(row_csr.nnz)


@pytest.mark.parametrize("tier", ["ram", "gpu", "mmap"])
def test_vectorized_gather_matches_reference(tier):
    n_regions = 16
    coord = build_coord_table([f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(n_regions)])
    rank = build_genomic_rank(coord)
    rng = np.random.default_rng(1)
    rows = []
    for i in range(12):
        r = np.zeros(n_regions)
        nnz = n_regions if i == 0 else rng.integers(1, 8)
        idx = rng.choice(n_regions, nnz, replace=False)
        r[idx] = rng.integers(1, 9, nnz)
        rows.append(r)
    x = sp.csr_matrix(np.vstack(rows), dtype=np.float32)
    kwargs = {"tier": tier}
    if tier == "mmap":
        kwargs["out_dir"] = tempfile.mkdtemp()
    store = build_token_store(
        x,
        coord,
        rank,
        max_encoder_tokens=5,
        genomic=True,
        **kwargs,
    )
    idx = np.array([0, 2, 5, 11], dtype=np.int64)
    for for_encoder in (True, False):
        ids_v, mask_v = store.gather(idx, for_encoder=for_encoder)
        ids_r, mask_r = store.gather_reference(idx, for_encoder=for_encoder)
        np.testing.assert_array_equal(ids_v, ids_r)
        np.testing.assert_array_equal(mask_v, mask_r)
        if tier == "gpu" and torch.cuda.is_available():
            dev = torch.device("cuda")
            t_ids, t_mask = store.gather_torch(idx, dev, for_encoder=for_encoder)
            np.testing.assert_array_equal(t_ids.cpu().numpy(), ids_r)
            np.testing.assert_array_equal(t_mask.cpu().numpy(), mask_r)


def test_token_store_always_retains_values_without_truncation():
    """Counts are stored even when no row exceeds max_encoder_tokens."""
    n_regions = 8
    coord = build_coord_table([f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(n_regions)])
    rank = build_genomic_rank(coord)
    x = sp.csr_matrix(
        [
            [1, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    store = build_token_store(
        x,
        coord,
        rank,
        max_encoder_tokens=8,
        genomic=True,
        tier="ram",
    )
    assert not store.truncation_possible
    assert store.values is not None
    assert len(store.values) == len(store.ids)
    for row in range(x.shape[0]):
        row_csr = x.getrow(row)
        _, vals = store._row_ids_vals(row)
        if row_csr.nnz:
            np.testing.assert_array_equal(vals, row_csr.data.astype(np.float32))


@pytest.mark.parametrize("tier", ["ram", "gpu", "mmap"])
def test_gather_return_values_matches_reference(tier):
    n_regions = 16
    coord = build_coord_table([f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(n_regions)])
    rank = build_genomic_rank(coord)
    rng = np.random.default_rng(2)
    rows = []
    for i in range(12):
        r = np.zeros(n_regions)
        nnz = n_regions if i == 0 else rng.integers(1, 8)
        idx = rng.choice(n_regions, nnz, replace=False)
        r[idx] = rng.integers(1, 9, nnz)
        rows.append(r)
    x = sp.csr_matrix(np.vstack(rows), dtype=np.float32)
    kwargs = {"tier": tier}
    if tier == "mmap":
        kwargs["out_dir"] = tempfile.mkdtemp()
    store = build_token_store(
        x,
        coord,
        rank,
        max_encoder_tokens=5,
        genomic=True,
        **kwargs,
    )
    assert store.values is not None
    idx = np.array([0, 2, 5, 11], dtype=np.int64)
    for for_encoder in (True, False):
        ids_v, mask_v, vals_v = store.gather(
            idx, for_encoder=for_encoder, return_values=True
        )
        ids_r, mask_r, vals_r = store.gather_reference(
            idx, for_encoder=for_encoder, return_values=True
        )
        np.testing.assert_array_equal(ids_v, ids_r)
        np.testing.assert_array_equal(mask_v, mask_r)
        np.testing.assert_allclose(vals_v, vals_r)
        assert np.all(vals_v[~mask_v] == 0.0)
        if tier == "gpu" and torch.cuda.is_available():
            dev = torch.device("cuda")
            t_ids, t_mask, t_vals = store.gather_torch(
                idx, dev, for_encoder=for_encoder, return_values=True
            )
            np.testing.assert_array_equal(t_ids.cpu().numpy(), ids_r)
            np.testing.assert_array_equal(t_mask.cpu().numpy(), mask_r)
            np.testing.assert_allclose(t_vals.cpu().numpy(), vals_r)
        ids_only, mask_only = store.gather(idx, for_encoder=for_encoder, return_values=False)
        np.testing.assert_array_equal(ids_only, ids_r)
        np.testing.assert_array_equal(mask_only, mask_r)


def test_encoder_forward_with_token_values():
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
        use_counts_in_encoder=True,
    )
    token_ids = torch.tensor([[0, 2, 5, 0, 0], [1, 3, 0, 0, 0]], dtype=torch.long)
    token_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, False]],
        dtype=torch.bool,
    )
    token_values = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.0], [4.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    batch_index = torch.zeros(2, dtype=torch.long)
    q_m, q_v, z = enc(
        token_ids, token_mask, batch_index, token_values=token_values
    )
    assert q_m.shape == (2, 8)
    assert q_v.shape == (2, 8)
    assert z.shape == (2, 8)
    assert torch.isfinite(q_m).all()
    assert torch.isfinite(q_v).all()
    assert torch.isfinite(z).all()


def test_setvi_save_load_roundtrip_latents_and_checkpoint_size():
    from scvi.data import synthetic_iid
    from scvi.model import SETVI

    sizes = {}
    for n_obs in (256, 2048):
        mdata = synthetic_iid(return_mudata=True, batch_size=n_obs)
        atac = mdata.mod["accessibility"]
        peaks = [f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(atac.n_vars)]
        atac.var_names = peaks
        SETVI.setup_mudata(
            mdata,
            modalities={
                "rna_layer": "rna",
                "protein_layer": "protein_expression",
                "atac_layer": "accessibility",
            },
            max_atac_tokens=32,
            atac_token_store="ram",
        )
        model = SETVI(
            mdata,
            max_atac_tokens=32,
            st_d_model=32,
            st_n_layers=1,
            st_n_inducing=8,
        )
        model.train(max_epochs=1, batch_size=64, accelerator="cpu", early_stopping=False)
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "model"
            model.save(save_path, overwrite=True, save_anndata=False)
            sizes[n_obs] = (save_path / "model.pt").stat().st_size
            payload = torch.load(save_path / "model.pt", map_location="cpu", weights_only=False)
            reg = payload["attr_dict"]["registry_"]
            state = reg["field_registries"]["atac_token_config"]["state_registry"]
            assert AtacTokenConfigField.TOKEN_STORE_KEY not in state
            assert AtacTokenConfigField.NN_LENGTHS_KEY not in state
            loaded = SETVI.load(save_path, adata=mdata)
            latents = loaded.get_latent_representation()
            assert latents.shape[0] == mdata.n_obs
            sizes[mdata.n_obs] = sizes.pop(n_obs)
    obs_keys = sorted(sizes)
    assert abs(sizes[obs_keys[1]] - sizes[obs_keys[0]]) < 50_000


def test_chrom_vocab_deterministic_and_collision_free():
    peaks_a = ["chr1:1000-1400", "chr2:2000-2400", "chrX:3000-3400", "chrY:4000-4400"]
    peaks_b = list(reversed(peaks_a))
    vocab_a = build_chrom_vocab(peaks_a)
    vocab_b = build_chrom_vocab(peaks_b)
    assert vocab_a == vocab_b
    coord_a = build_coord_table(peaks_a, vocab=vocab_a)
    for peak in peaks_a:
        np.testing.assert_array_equal(
            parse_peak_name(peak, vocab=vocab_a),
            parse_peak_name(peak, vocab=vocab_b),
        )
    assert vocab_a["1"] == 1
    assert vocab_a["2"] == 2
    assert vocab_a["X"] == 22
    enc = SetTransformerAtacVariationalEncoder(
        n_latent=4,
        coord_table=torch.as_tensor(coord_a, dtype=torch.long),
        n_regions=len(peaks_a),
        d_model=16,
        n_layers=1,
        n_inducing=4,
        n_heads=2,
    )
    assert enc.embedding.chrom_emb.num_embeddings >= int(coord_a[:, 0].max()) + 1


def test_mmap_store_reopens_without_rebuild():
    from scvi.data import synthetic_iid
    from scvi.model import SETVI
    from scvi.tokenized import ATAC_TOKEN_CONFIG_KEY

    mdata = synthetic_iid(return_mudata=True, batch_size=12)
    atac = mdata.mod["accessibility"]
    n_obs, n_vars = atac.shape
    peaks = [f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(n_vars)]
    atac.var_names = peaks
    coord = build_coord_table(peaks)
    x = sp.random(n_obs, n_vars, density=0.25, format="csr", dtype=np.float32, random_state=2)
    atac.X = x
    with tempfile.TemporaryDirectory() as tmp:
        SETVI.setup_mudata(
            mdata,
            modalities={
                "rna_layer": "rna",
                "protein_layer": "protein_expression",
                "atac_layer": "accessibility",
            },
            max_atac_tokens=8,
            atac_token_store="mmap",
            atac_token_store_dir=tmp,
        )
        from scvi.data import _constants

        manager = SETVI._get_most_recent_anndata_manager(mdata, required=True)
        state_registry = manager._registry[_constants._FIELD_REGISTRIES_KEY][
            ATAC_TOKEN_CONFIG_KEY
        ][_constants._STATE_REGISTRY_KEY]
        store = manager.get_state_registry("atac_token_config")[
            AtacTokenConfigField.TOKEN_STORE_KEY
        ]
        mtimes = {p.name: p.stat().st_mtime for p in Path(tmp).iterdir() if p.is_file()}
        state_registry.pop(AtacTokenConfigField.TOKEN_STORE_KEY, None)
        reopened = rebuild_token_store(manager, tier="mmap", out_dir=tmp)
        for name, mtime in mtimes.items():
            assert Path(tmp, name).stat().st_mtime == mtime
        ids_a, mask_a = store.gather([1, 4], for_encoder=False)
        ids_b, mask_b = reopened.gather([1, 4], for_encoder=False)
        np.testing.assert_array_equal(ids_a, ids_b)
        np.testing.assert_array_equal(mask_a, mask_b)


@pytest.mark.parametrize("tier", ["ram", "gpu", "mmap"])
def test_setvi_query_latent_differs_from_training_store(tier):
    from scvi.data import synthetic_iid
    from scvi.model import SETVI
    from scvi.tokenized import ATAC_TOKEN_CONFIG_KEY

    train = synthetic_iid(return_mudata=True, batch_size=12)
    peaks = [
        f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(train.mod["accessibility"].n_vars)
    ]
    train.mod["accessibility"].var_names = peaks
    query = synthetic_iid(return_mudata=True, batch_size=8)
    query.mod["accessibility"].var_names = peaks
    store_dir = tempfile.mkdtemp() if tier == "mmap" else None
    SETVI.setup_mudata(
        train,
        modalities={
            "rna_layer": "rna",
            "protein_layer": "protein_expression",
            "atac_layer": "accessibility",
        },
        max_atac_tokens=16,
        atac_token_store=tier,
        atac_token_store_dir=store_dir,
    )
    SETVI.setup_mudata(
        query,
        modalities={
            "rna_layer": "rna",
            "protein_layer": "protein_expression",
            "atac_layer": "accessibility",
        },
        max_atac_tokens=16,
        atac_token_store="ram",
    )
    model = SETVI(train, max_atac_tokens=16, st_d_model=32, st_n_layers=1, st_n_inducing=8)
    model.train(max_epochs=1, batch_size=6, accelerator="cpu", early_stopping=False)
    latents_query = model.get_latent_representation(adata=query)
    assert latents_query.shape == (query.n_obs, model.module.n_latent)
    wrong_store = model.adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)[
        AtacTokenConfigField.TOKEN_STORE_KEY
    ]
    from scvi.data import _constants

    query = model._validate_anndata(query)
    query_manager = model.get_anndata_manager(query)
    query_state = query_manager._registry[_constants._FIELD_REGISTRIES_KEY][
        ATAC_TOKEN_CONFIG_KEY
    ][_constants._STATE_REGISTRY_KEY]
    query_state.pop(AtacTokenConfigField.TOKEN_STORE_KEY, None)
    ref_store = rebuild_token_store(query_manager, tier="ram")
    ref_ids, _ = ref_store.gather(np.arange(query.n_obs), for_encoder=True)
    wrong_ids, _ = wrong_store.gather(np.arange(query.n_obs), for_encoder=True)
    assert not np.array_equal(ref_ids, wrong_ids)


def test_inference_skips_dense_target_build(monkeypatch):
    from scvi.data import synthetic_iid
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
        atac_token_store="ram",
    )
    model = SETVI(mdata, max_atac_tokens=32, st_d_model=32, st_n_layers=1, st_n_inducing=8)
    model.train(max_epochs=1, batch_size=4, accelerator="cpu", early_stopping=False)

    calls = {"n": 0}
    original = SETVAE._accessibility_target_from_tokens

    def counted(ids, mask, n_regions):
        calls["n"] += 1
        return original(ids, mask, n_regions)

    monkeypatch.setattr(SETVAE, "_accessibility_target_from_tokens", staticmethod(counted))
    model.get_latent_representation()
    assert calls["n"] == 0


def test_setvi_elbo_parity_no_truncation():
    from scvi.data import synthetic_iid
    from scvi.model import SETVI

    mdata = synthetic_iid(return_mudata=True, batch_size=12)
    atac = mdata.mod["accessibility"]
    atac.var_names = [f"chr1:{1000 + i * 500}-{1400 + i * 500}" for i in range(atac.n_vars)]
    SETVI.setup_mudata(
        mdata,
        modalities={
            "rna_layer": "rna",
            "protein_layer": "protein_expression",
            "atac_layer": "accessibility",
        },
        max_atac_tokens=64,
        atac_token_store="ram",
    )
    model = SETVI(mdata, max_atac_tokens=64, st_d_model=32, st_n_layers=1, st_n_inducing=8)
    model.train(max_epochs=1, batch_size=6, accelerator="cpu", early_stopping=False)
    recon = model.get_reconstruction_error()
    assert all(np.isfinite(v) for v in recon.values())
