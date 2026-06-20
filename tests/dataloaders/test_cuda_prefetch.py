import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scvi.dataloaders._cuda_prefetch import (
    CUDABatchPrefetcher,
    batch_tensors_on_device,
    maybe_wrap_cuda_prefetch,
    transfer_batch_to_cuda,
)


def _make_loader(batch_size: int = 4, n_samples: int = 16, pin_memory: bool = True):
    x = torch.randn(n_samples, 8)
    y = torch.randint(0, 3, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory)


def test_maybe_wrap_noop_when_disabled():
    loader = _make_loader()
    wrapped = maybe_wrap_cuda_prefetch(
        loader,
        prefetch_to_gpu=False,
        cuda_queue_depth=2,
        pin_memory=True,
        load_sparse_tensor=False,
    )
    assert wrapped is loader


def test_raises_if_sparse_tensor_path():
    loader = _make_loader()
    with pytest.raises(ValueError, match="load_sparse_tensor=False"):
        maybe_wrap_cuda_prefetch(
            loader,
            prefetch_to_gpu=True,
            cuda_queue_depth=2,
            pin_memory=True,
            load_sparse_tensor=True,
        )


def test_raises_if_no_pin_memory():
    loader = _make_loader(pin_memory=False)
    with pytest.raises(ValueError, match="pin_memory=True"):
        maybe_wrap_cuda_prefetch(
            loader,
            prefetch_to_gpu=True,
            cuda_queue_depth=2,
            pin_memory=False,
            load_sparse_tensor=False,
        )


def test_len_preserved():
    loader = _make_loader()
    prefetcher = CUDABatchPrefetcher(loader, queue_depth=2)
    assert len(prefetcher) == len(loader)


def test_batch_tensors_on_device():
    device = torch.device("cpu")
    batch = {"a": torch.zeros(2), "b": (torch.ones(2), torch.full((2,), 2))}
    assert batch_tensors_on_device(batch, device)

    if torch.cuda.is_available():
        cuda_device = torch.device("cuda", torch.cuda.current_device())
        cuda_batch = transfer_batch_to_cuda(batch, cuda_device)
        assert batch_tensors_on_device(cuda_batch, cuda_device)
        assert not batch_tensors_on_device(batch, cuda_device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefetcher_yields_same_values():
    loader = _make_loader(batch_size=4, n_samples=12)
    prefetcher = CUDABatchPrefetcher(loader, queue_depth=2)

    ref_batches = list(loader)
    pref_batches = list(prefetcher)

    assert len(ref_batches) == len(pref_batches)
    for ref, pref in zip(ref_batches, pref_batches, strict=True):
        ref_x, ref_y = ref
        pref_x, pref_y = pref
        assert torch.equal(ref_x.cpu(), pref_x.cpu())
        assert torch.equal(ref_y.cpu(), pref_y.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefetcher_tensors_on_cuda():
    device = torch.device("cuda", torch.cuda.current_device())
    loader = _make_loader(batch_size=4, n_samples=8)
    prefetcher = CUDABatchPrefetcher(loader, queue_depth=2, device=device)

    for batch in prefetcher:
        assert batch_tensors_on_device(batch, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefetcher_dict_batch():
    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(8, 4)
    loader = DataLoader(
        [{"x": row, "y": torch.tensor(i)} for i, row in enumerate(x)],
        batch_size=2,
        pin_memory=True,
        collate_fn=lambda samples: {
            "x": torch.stack([s["x"] for s in samples]),
            "y": torch.stack([s["y"] for s in samples]),
        },
    )
    prefetcher = CUDABatchPrefetcher(loader, queue_depth=2, device=device)
    for batch in prefetcher:
        assert batch_tensors_on_device(batch, device)


def test_cpu_fallback_when_cuda_unavailable(monkeypatch):
    loader = _make_loader()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    wrapped = maybe_wrap_cuda_prefetch(
        loader,
        prefetch_to_gpu=True,
        cuda_queue_depth=2,
        pin_memory=True,
        load_sparse_tensor=False,
    )
    assert wrapped is loader

    ref_batches = list(loader)
    pref_batches = list(CUDABatchPrefetcher(loader))
    assert len(ref_batches) == len(pref_batches)
    for ref, pref in zip(ref_batches, pref_batches, strict=True):
        assert torch.equal(ref[0], pref[0])
        assert torch.equal(ref[1], pref[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_datasplitter_cuda_prefetch_integration():
    import scvi
    from scvi.dataloaders import DataSplitter

    adata = scvi.data.synthetic_iid(batch_size=128, n_batches=2)
    scvi.model.SCVI.setup_anndata(adata)
    adata_manager = scvi.model.SCVI(adata).adata_manager
    splitter = DataSplitter(
        adata_manager,
        batch_size=32,
        pin_memory=True,
        prefetch_to_gpu=True,
        cuda_queue_depth=2,
    )
    splitter.setup()
    train_dl = splitter.train_dataloader()
    assert isinstance(train_dl, CUDABatchPrefetcher)

    device = torch.device("cuda", torch.cuda.current_device())
    batch = next(iter(train_dl))
    assert batch_tensors_on_device(batch, device)
