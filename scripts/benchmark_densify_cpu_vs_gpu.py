"""Micro-benchmark: CPU vs GPU sparse-to-dense for MultiVI-like batches.

Times the same cell batches through:
  - CPU path (AnnTorchDataset default): sparse slice -> .toarray() -> optional dense H2D
  - GPU path (load_sparse_tensor=True): sparse slice -> torch sparse -> H2D -> .to_dense()

Usage (WSL + CUDA):
  wsl bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate scatlas_rapids2410 && \
    cd "/mnt/c/Users/MP Pebworth/Documents/GitHub/scvi-tools-2" && \
    python scripts/benchmark_densify_cpu_vs_gpu.py'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import scvi
from scvi.data import synthetic_iid
from scvi.data._utils import scipy_to_torch_sparse
from scvi.model import MULTIVI


def _make_mudata(cells_per_batch: int, n_batches: int, n_genes: int, n_regions: int, seed: int):
    scvi.settings.seed = seed
    mdata = synthetic_iid(
        batch_size=cells_per_batch,
        n_batches=n_batches,
        n_genes=n_genes,
        n_regions=n_regions,
        n_proteins=100,
        sparse_format="csr_matrix",
        return_mudata=True,
    )
    MULTIVI.setup_mudata(
        mdata,
        batch_key="batch",
        modalities={
            "rna_layer": "rna",
            "atac_layer": "accessibility",
            "protein_layer": "protein_expression",
        },
    )
    return mdata


def _time_cpu_densify(rna, atac, indices: np.ndarray, dtype=np.float32) -> float:
    t0 = time.perf_counter()
    rna[indices].astype(dtype, copy=False).toarray()
    atac[indices].astype(dtype, copy=False).toarray()
    return time.perf_counter() - t0


def _time_cpu_densify_plus_h2d(
    rna, atac, indices: np.ndarray, device: torch.device, dtype=np.float32
) -> float:
    t0 = time.perf_counter()
    rna_dense = rna[indices].astype(dtype, copy=False).toarray()
    atac_dense = atac[indices].astype(dtype, copy=False).toarray()
    torch.as_tensor(rna_dense, device=device)
    torch.as_tensor(atac_dense, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _time_gpu_densify(rna, atac, indices: np.ndarray, device: torch.device, dtype=np.float32) -> float:
    t0 = time.perf_counter()
    rna_s = scipy_to_torch_sparse(rna[indices].astype(dtype, copy=False))
    atac_s = scipy_to_torch_sparse(atac[indices].astype(dtype, copy=False))
    rna_s = rna_s.to(device)
    atac_s = atac_s.to(device)
    rna_s.to_dense()
    atac_s.to_dense()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _bench_fn(fn, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = [fn() for _ in range(repeats)]
    return float(np.mean(times)), float(np.std(times))


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU vs GPU densify micro-benchmark")
    parser.add_argument("--cells-per-batch", type=int, default=2400)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--n-regions", type=int, default=2000)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    n_obs = args.cells_per_batch * args.n_batches

    print("=" * 72)
    print("CPU vs GPU densify micro-benchmark (RNA + ATAC per batch)")
    print("=" * 72)
    print(f"CUDA available: {cuda}")
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Data: {n_obs} cells, {args.n_genes} genes, {args.n_regions} regions (CSR sparse)"
    )
    print(f"repeats={args.repeats}, warmup={args.warmup}\n")

    mdata = _make_mudata(
        args.cells_per_batch, args.n_batches, args.n_genes, args.n_regions, args.seed
    )
    rna = mdata.mod["rna"].X
    atac = mdata.mod["accessibility"].X

    rng = np.random.default_rng(args.seed)
    batch_indices = {
        bs: rng.integers(0, n_obs, size=bs) for bs in args.batch_sizes
    }

    for bs in args.batch_sizes:
        idx = batch_indices[bs]
        print(f"--- batch_size={bs} ---")

        cpu_only_mean, cpu_only_std = _bench_fn(
            lambda i=idx: _time_cpu_densify(rna, atac, i),
            args.warmup,
            args.repeats,
        )
        print(f"  CPU densify only:              {cpu_only_mean*1000:7.2f} ± {cpu_only_std*1000:.2f} ms")

        cpu_h2d_mean, cpu_h2d_std = _bench_fn(
            lambda i=idx: _time_cpu_densify_plus_h2d(rna, atac, i, device),
            args.warmup,
            args.repeats,
        )
        print(f"  CPU densify + dense H2D:       {cpu_h2d_mean*1000:7.2f} ± {cpu_h2d_std*1000:.2f} ms")

        gpu_mean, gpu_std = _bench_fn(
            lambda i=idx: _time_gpu_densify(rna, atac, i, device),
            args.warmup,
            args.repeats,
        )
        print(f"  GPU sparse H2D + densify:      {gpu_mean*1000:7.2f} ± {gpu_std*1000:.2f} ms")

        if gpu_mean > 0:
            ratio = cpu_h2d_mean / gpu_mean
            winner = "GPU" if ratio > 1 else "CPU+dense H2D"
            print(f"  -> {winner} faster for full path ({ratio:.2f}x vs GPU path)")
        print()


if __name__ == "__main__":
    main()
