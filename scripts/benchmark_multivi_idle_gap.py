"""Quick benchmark: is MultiVI training waiting on the dataloader?

Runs three short training profiles on synthetic sparse trimodal MuData:
  1. baseline (num_workers=0, no pin_memory, CPU densify + H2D)
  2. prefetch (workers + pin_memory + CPU densify + dense H2D)
  3. prefetch + larger batch_size (same data path as 2)

Prints Lightning simple-profiler totals and throughput (batches/s, cells/s).
When CUDA is available, also reports GPU name and suggests running nvidia-smi dmon
alongside for utilization valleys.

Usage (from repo root):

  # Recommended: Ubuntu WSL + RAPIDS conda env (CUDA GPU)
  wsl bash scripts/run_benchmark_wsl.sh
  wsl bash scripts/run_benchmark_wsl.sh --limit-train-batches 200

  # Direct (Linux / WSL after `conda activate scatlas_rapids2410`)
  python scripts/benchmark_multivi_idle_gap.py

  # Windows native (CPU-only unless CUDA PyTorch installed; num_workers forced to 0)
  python scripts/benchmark_multivi_idle_gap.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Windows OpenMP duplicate-lib workaround (common in conda + torch stacks)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from lightning.pytorch.profilers import SimpleProfiler

# Allow running without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import scvi
from scvi.data import synthetic_iid
from scvi.model import MULTIVI


@dataclass
class BenchmarkConfig:
    name: str
    batch_size: int
    datasplitter_kwargs: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    name: str
    accelerator: str
    batch_size: int
    limit_train_batches: int
    n_obs: int
    wall_seconds: float
    batches_per_sec: float
    cells_per_sec: float
    profiler_summary: dict[str, float]
    loader_related_sec: float
    transfer_sec: float
    data_path_sec: float
    train_step_sec: float
    epoch_sec: float
    loader_fraction: float


def _make_synthetic_mudata(
    *,
    cells_per_batch: int,
    n_batches: int,
    n_genes: int,
    n_regions: int,
    n_proteins: int,
    seed: int,
) -> tuple:
    """Return (mdata, n_obs) with sparse trimodal matrices."""
    scvi.settings.seed = seed
    mdata = synthetic_iid(
        batch_size=cells_per_batch,
        n_batches=n_batches,
        n_genes=n_genes,
        n_regions=n_regions,
        n_proteins=n_proteins,
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
    return mdata, mdata.n_obs


def _parse_profiler(profiler: SimpleProfiler) -> dict[str, float]:
    """Aggregate SimpleProfiler recorded durations by action name."""
    summary: dict[str, float] = {}
    for action, durations in profiler.recorded_durations.items():
        summary[action] = float(sum(durations))
    return summary


def _loader_related_seconds(summary: dict[str, float]) -> float:
    """Time spent waiting for the next training batch (Lightning dataloader-next hook)."""
    for action, sec in summary.items():
        if "train_dataloader_next" in action:
            return sec
    return 0.0


def _train_step_seconds(summary: dict[str, float]) -> float:
    for action, sec in summary.items():
        if "training_step" in action.lower():
            return sec
    return 0.0


def _transfer_seconds(summary: dict[str, float]) -> float:
    """H2D time in Lightning's device-transfer hooks."""
    for action, sec in summary.items():
        if "transfer_batch_to_device" in action:
            return sec
    for action, sec in summary.items():
        if "batch_to_device" in action:
            return sec
    return 0.0


def _epoch_seconds(summary: dict[str, float]) -> float:
    for action, sec in summary.items():
        if action == "run_training_epoch":
            return sec
    return sum(summary.values())


def run_one(
    mdata,
    cfg: BenchmarkConfig,
    *,
    limit_train_batches: int,
    max_epochs: int,
    seed: int,
    accelerator: str,
    devices: str | int,
) -> BenchmarkResult:
    scvi.settings.seed = seed
    profiler = SimpleProfiler()

    model = MULTIVI(mdata, n_latent=10)
    n_obs = mdata.n_obs

    t0 = time.perf_counter()
    model.train(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        batch_size=cfg.batch_size,
        train_size=1.0,
        early_stopping=False,
        datasplitter_kwargs=dict(cfg.datasplitter_kwargs),
        enable_progress_bar=False,
        logger=False,
        limit_train_batches=limit_train_batches,
        num_sanity_val_steps=0,
        profiler=profiler,
    )
    wall = time.perf_counter() - t0

    summary = _parse_profiler(profiler)
    loader_sec = _loader_related_seconds(summary)
    transfer_sec = _transfer_seconds(summary)
    data_path_sec = loader_sec + transfer_sec
    train_sec = _train_step_seconds(summary)
    epoch_sec = _epoch_seconds(summary)
    loader_frac = loader_sec / epoch_sec if epoch_sec > 0 else 0.0

    batches_per_sec = limit_train_batches / wall
    cells_per_sec = (limit_train_batches * cfg.batch_size) / wall

    return BenchmarkResult(
        name=cfg.name,
        accelerator=accelerator,
        batch_size=cfg.batch_size,
        limit_train_batches=limit_train_batches,
        n_obs=n_obs,
        wall_seconds=wall,
        batches_per_sec=batches_per_sec,
        cells_per_sec=cells_per_sec,
        profiler_summary=summary,
        loader_related_sec=loader_sec,
        transfer_sec=transfer_sec,
        data_path_sec=data_path_sec,
        train_step_sec=train_sec,
        epoch_sec=epoch_sec,
        loader_fraction=loader_frac,
    )


def _windows_multiprocessing_ok() -> bool:
    """DataLoader workers often fail on Windows when dataset objects are not picklable."""
    if sys.platform != "win32":
        return True
    # Python 3.14+ forkserver/spawn is especially brittle with AnnTorchDataset
    return False


def _default_configs(base_batch: int, num_workers: int | None = None) -> list[BenchmarkConfig]:
    if num_workers is None:
        worker_count = 4 if _windows_multiprocessing_ok() else 0
    else:
        worker_count = num_workers
    worker_note = "" if worker_count else " (num_workers=0 on Windows)"

    # Prefetch configs use CPU densify (default AnnTorchDataset path) + pin_memory dense H2D.
    # load_sparse_tensor=False avoids GPU sparse H2D + on_after_batch_transfer to_dense().
    prefetch_kwargs = {
        "num_workers": worker_count,
        "persistent_workers": worker_count > 0,
        "pin_memory": True,
        "load_sparse_tensor": False,
        **({"prefetch_factor": 2} if worker_count > 0 else {}),
    }

    return [
        BenchmarkConfig(
            name="baseline",
            batch_size=base_batch,
            datasplitter_kwargs={
                "num_workers": 0,
                "pin_memory": False,
                "load_sparse_tensor": False,
            },
        ),
        BenchmarkConfig(
            name=f"prefetch_cpu_densify{worker_note}",
            batch_size=base_batch,
            datasplitter_kwargs=dict(prefetch_kwargs),
        ),
        BenchmarkConfig(
            name=f"prefetch_cpu_densify_cuda_stream{worker_note}",
            batch_size=base_batch,
            datasplitter_kwargs={
                **prefetch_kwargs,
                "prefetch_to_gpu": True,
                "cuda_queue_depth": 2,
            },
        ),
        BenchmarkConfig(
            name=f"prefetch_cpu_densify_large_batch{worker_note}",
            batch_size=base_batch * 4,
            datasplitter_kwargs=dict(prefetch_kwargs),
        ),
    ]


def _pick_accelerator() -> tuple[str, str | int, str | None]:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "gpu", 1, name
    return "cpu", "auto", None


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiVI dataloader idle-gap micro-benchmark")
    parser.add_argument("--cells-per-batch", type=int, default=2400, help="Cells per synthetic batch")
    parser.add_argument("--n-batches", type=int, default=8, help="Number of synthetic batches")
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--n-regions", type=int, default=2000)
    parser.add_argument("--n-proteins", type=int, default=100)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--limit-train-batches", type=int, default=150)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers for prefetch configs (default: 4 on Linux, 0 on Windows)")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to write JSON results")
    args = parser.parse_args()

    accelerator, devices, gpu_name = _pick_accelerator()

    print("=" * 72)
    print("MultiVI GPU idle-gap micro-benchmark")
    print("=" * 72)
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if gpu_name:
        print(f"GPU: {gpu_name}")
    else:
        print("GPU: none (timing still shows loader vs train-step split on CPU)")
        print("Tip: install CUDA-enabled PyTorch and re-run; use `nvidia-smi dmon -s u` in parallel.")
    print(
        f"Synthetic shape: {args.cells_per_batch * args.n_batches} cells, "
        f"{args.n_genes} genes, {args.n_regions} regions, {args.n_proteins} proteins (CSR sparse)"
    )
    print(f"Accelerator: {accelerator}, limit_train_batches={args.limit_train_batches}")
    print()

    print("Building synthetic MuData (once, shared across runs)...")
    mdata, n_obs = _make_synthetic_mudata(
        cells_per_batch=args.cells_per_batch,
        n_batches=args.n_batches,
        n_genes=args.n_genes,
        n_regions=args.n_regions,
        n_proteins=args.n_proteins,
        seed=args.seed,
    )
    print(f"  n_obs={n_obs}\n")

    configs = _default_configs(args.base_batch_size, num_workers=args.num_workers)
    results: list[BenchmarkResult] = []

    for cfg in configs:
        print(f"--- Run: {cfg.name} (batch_size={cfg.batch_size}) ---")
        try:
            res = run_one(
                mdata,
                cfg,
                limit_train_batches=args.limit_train_batches,
                max_epochs=args.max_epochs,
                seed=args.seed,
                accelerator=accelerator,
                devices=devices,
            )
            results.append(res)
            cells_processed = args.limit_train_batches * cfg.batch_size
            print(f"  wall (total model.train): {res.wall_seconds:.2f}s")
            print(f"  profiler run_training_epoch: {res.epoch_sec:.2f}s")
            print(
                f"  work: {args.max_epochs} epoch(s), {args.limit_train_batches} batches, "
                f"{cells_processed} cells"
            )
            print(f"  throughput: {res.batches_per_sec:.2f} batches/s, {res.cells_per_sec:.0f} cells/s")
            print(
                f"  data path (loader + transfer): {res.data_path_sec:.2f}s "
                f"(loader {res.loader_related_sec:.2f}s + transfer {res.transfer_sec:.2f}s)"
            )
            print(f"  training_step: {res.train_step_sec:.2f}s")
            if res.profiler_summary:
                top = sorted(res.profiler_summary.items(), key=lambda x: -x[1])[:6]
                print("  profiler top actions:")
                for action, sec in top:
                    print(f"    {sec:8.3f}s  {action}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
        print()

    if len(results) >= 2:
        base = results[0]
        best = max(results, key=lambda r: r.cells_per_sec)
        print("=" * 72)
        print("Summary")
        print("=" * 72)
        print(
            f"Same-work comparison (batch_size={base.batch_size}, "
            f"{args.limit_train_batches} batches, {args.max_epochs} epoch(s)):"
        )
        same_bs = [r for r in results if r.batch_size == base.batch_size]
        ref_wall = base.wall_seconds
        for r in same_bs:
            speedup = ref_wall / r.wall_seconds if r.wall_seconds > 0 else float("nan")
            print(
                f"  {r.name:40s}  wall={r.wall_seconds:6.2f}s  "
                f"data_path={r.data_path_sec:5.2f}s  train_step={r.train_step_sec:6.2f}s  "
                f"speedup_vs_baseline={speedup:.2f}x"
            )
        print()
        print(f"Baseline throughput: {base.cells_per_sec:.0f} cells/s")
        print(f"Best throughput ({best.name}): {best.cells_per_sec:.0f} cells/s")
        if base.cells_per_sec > 0 and best.batch_size == base.batch_size:
            print(f"Speedup vs baseline (same batch size): {best.cells_per_sec / base.cells_per_sec:.2f}x")
        elif base.cells_per_sec > 0:
            print(
                f"Note: best config uses batch_size={best.batch_size} "
                f"(processes {best.batch_size // base.batch_size}x cells per batch)."
            )
        if base.loader_fraction > 0.15:
            print("Interpretation: likely DATA-PIPELINE bound (GPU/CPU waiting on next batch).")
        elif base.loader_fraction > 0.05:
            print("Interpretation: moderate loader wait; prefetch / batch_size tuning may help.")
        else:
            print("Interpretation: likely COMPUTE bound at this batch size (loader wait is small).")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(r) for r in results]
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
