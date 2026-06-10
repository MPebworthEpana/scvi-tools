"""Tests for balanced subsampled ATAC training (PeakVI decoder)."""

from __future__ import annotations

import pytest
import torch

from scvi import REGISTRY_KEYS
from scvi.encoders._constants import ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY
from scvi.module._mambavae import MAMBAVAE, sample_balanced_atac_loss_indices
from scvi.module._peakvae import Decoder as DecoderPeakVI


def test_sample_balanced_atac_loss_indices_positives_from_tokens():
    token_ids = torch.tensor([[1, 2, 0], [4, 0, 0]], dtype=torch.long)
    token_mask = torch.tensor([[True, True, False], [True, False, False]])
    samples = sample_balanced_atac_loss_indices(
        token_ids, token_mask, n_regions=10, negatives_per_positive=1
    )
    pos0 = token_ids[0][token_mask[0]].tolist()
    pos1 = token_ids[1][token_mask[1]].tolist()
    active0 = samples["loss_peak_ids"][0][samples["loss_sample_mask"][0]].tolist()
    active1 = samples["loss_peak_ids"][1][samples["loss_sample_mask"][1]].tolist()
    assert active0[: len(pos0)] == pos0
    assert active1[: len(pos1)] == pos1
    assert samples["loss_targets"][0, : len(pos0)].tolist() == [1.0] * len(pos0)
    assert samples["loss_targets"][1, : len(pos1)].tolist() == [1.0] * len(pos1)


def test_sample_balanced_atac_loss_indices_negatives_disjoint():
    torch.manual_seed(0)
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    token_mask = torch.tensor([[True, True, True]])
    samples = sample_balanced_atac_loss_indices(
        token_ids, token_mask, n_regions=20, negatives_per_positive=1
    )
    active = samples["loss_peak_ids"][0][samples["loss_sample_mask"][0]].tolist()
    positives = {1, 2, 3}
    negatives = set(active[3:])
    assert positives.isdisjoint(negatives)
    assert samples["loss_targets"][0, 3:6].tolist() == [0.0, 0.0, 0.0]


def test_sample_balanced_atac_loss_indices_caps_negatives_when_sparse():
    """When most peaks are open, negatives are capped to available closed peaks."""
    token_ids = torch.tensor([list(range(8))], dtype=torch.long)
    token_mask = torch.tensor([[True] * 8])
    samples = sample_balanced_atac_loss_indices(
        token_ids, token_mask, n_regions=10, negatives_per_positive=1
    )
    active = samples["loss_peak_ids"][0][samples["loss_sample_mask"][0]].tolist()
    positives = set(range(8))
    negatives = set(active[8:])
    assert positives.isdisjoint(negatives)
    assert len(negatives) == 2  # only peaks 8 and 9 remain closed


def test_sample_balanced_atac_loss_indices_deterministic_with_seed():
    token_ids = torch.tensor([[5, 6, 0]], dtype=torch.long)
    token_mask = torch.tensor([[True, True, False]])
    gen_a = torch.Generator().manual_seed(123)
    gen_b = torch.Generator().manual_seed(123)
    a = sample_balanced_atac_loss_indices(
        token_ids, token_mask, n_regions=50, generator=gen_a
    )
    b = sample_balanced_atac_loss_indices(
        token_ids, token_mask, n_regions=50, generator=gen_b
    )
    torch.testing.assert_close(a["loss_peak_ids"], b["loss_peak_ids"])
    torch.testing.assert_close(a["loss_targets"], b["loss_targets"])
    torch.testing.assert_close(a["loss_sample_mask"], b["loss_sample_mask"])


def test_sample_balanced_atac_loss_indices_padded_shape_from_token_width():
    """Output width is derived from token tensor shape, not a per-batch reduction."""
    token_ids = torch.tensor([[1, 2, 0], [4, 0, 0]], dtype=torch.long)
    token_mask = torch.tensor([[True, True, False], [True, False, False]])
    samples = sample_balanced_atac_loss_indices(
        token_ids, token_mask, n_regions=10, negatives_per_positive=1
    )
    max_tokens = token_ids.shape[1]
    expected_width = max_tokens * (1 + 1)
    assert samples["loss_peak_ids"].shape == (2, expected_width)
    assert samples["loss_targets"].shape == (2, expected_width)
    assert samples["loss_sample_mask"].shape == (2, expected_width)


def test_sample_balanced_atac_loss_indices_randomized_disjointness():
    """Negatives never overlap positives across a randomized mixed batch."""
    torch.manual_seed(42)
    batch_size, max_tokens, n_regions = 16, 8, 200
    token_ids = torch.randint(0, n_regions, (batch_size, max_tokens))
    token_mask = torch.rand(batch_size, max_tokens) > 0.5
    samples = sample_balanced_atac_loss_indices(
        token_ids, token_mask, n_regions=n_regions, negatives_per_positive=2
    )
    for b in range(batch_size):
        active = samples["loss_peak_ids"][b][samples["loss_sample_mask"][b]]
        n_pos = int(token_mask[b].sum().item())
        if n_pos == 0:
            assert active.numel() == 0
            continue
        positives = set(active[:n_pos].tolist())
        negatives = set(active[n_pos:].tolist())
        assert positives.isdisjoint(negatives)
        assert len(negatives) == len(active[n_pos:].tolist())
        assert samples["loss_targets"][b, n_pos : n_pos + len(negatives)].tolist() == [
            0.0
        ] * len(negatives)


def test_decoder_forward_peaks_matches_full_forward_subset():
    decoder = DecoderPeakVI(n_input=4, n_output=20, n_hidden=8, n_layers=1)
    z = torch.randn(2, 4)
    peak_ids = torch.tensor([[1, 5, 9], [2, 7, 11]], dtype=torch.long)
    p_sub = decoder.forward_peaks(z, peak_ids)
    p_full = decoder(z)
    gathered = torch.stack([p_full[b, peak_ids[b]] for b in range(2)], dim=0)
    torch.testing.assert_close(p_sub, gathered, rtol=1e-5, atol=1e-5)


def test_balanced_subsample_guardrail_requires_peakvi_decoder():
    pytest.importorskip("mamba_ssm")
    with pytest.raises(ValueError, match="balanced_subsample"):
        MAMBAVAE(
            n_input_genes=10,
            n_input_regions=20,
            n_input_proteins=0,
            n_batch=1,
            n_obs=5,
            coord_table=[[0, 0, 100]] * 20,
            atac_decoder_module="mamba",
            atac_loss_mode="balanced_subsample",
        )


def _setup_mambavi(mdata, **model_kwargs):
    pytest.importorskip("mamba_ssm")
    from scvi.model import MAMBAVI

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
    defaults = {
        "max_atac_tokens": 32,
        "mamba_d_model": 64,
        "mamba_n_layers": 1,
        "mamba3_kwargs": {"is_mimo": False, "headdim": 16, "chunk_size": 16},
        "atac_decoder_module": "peakvi",
        "atac_loss_mode": "balanced_subsample",
    }
    defaults.update(model_kwargs)
    return MAMBAVI(mdata, **defaults)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba3 forward")
def test_balanced_subsample_forward_and_loss_finite():
    from scvi.data import synthetic_iid

    model = _setup_mambavi(synthetic_iid(return_mudata=True))
    module = model.module.cuda()
    module.train()

    splitter = model._data_splitter_cls(
        model.adata_manager,
        batch_size=4,
        train_data_and_attributes=model._balanced_subsample_train_keys(),
        load_sparse_tensor=True,
    )
    splitter.setup()
    batch = next(iter(splitter.train_dataloader()))
    batch = {
        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    assert REGISTRY_KEYS.ATAC_X_KEY not in batch
    assert ATAC_TOKEN_IDS_KEY in batch
    assert ATAC_TOKEN_MASK_KEY in batch

    inference_outputs, generative_outputs, loss_out = module.forward(batch)
    assert generative_outputs["p"] is None
    assert torch.isfinite(loss_out.loss)
    assert loss_out.reconstruction_loss["reconstruction_loss_accessibility"].sum() >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba3 forward")
def test_balanced_subsample_pretrain_uses_token_loss():
    from scvi.data import synthetic_iid

    model = _setup_mambavi(synthetic_iid(return_mudata=True))
    module = model.module.cuda()
    module.train()
    module.unimodal_pretrain_active = True
    module.unimodal_pretrain_modality = "atac"

    splitter = model._data_splitter_cls(
        model.adata_manager,
        batch_size=4,
        train_data_and_attributes=model._balanced_subsample_train_keys(),
        load_sparse_tensor=True,
    )
    splitter.setup()
    batch = next(iter(splitter.train_dataloader()))
    batch = {
        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    inference_outputs, generative_outputs, loss_out = module.forward(batch)
    assert torch.isfinite(loss_out.loss)
    assert (
        loss_out.reconstruction_loss["reconstruction_loss_accessibility"].sum().item() > 0
    )
