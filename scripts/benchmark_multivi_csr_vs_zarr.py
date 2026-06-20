"""Benchmark MultiVI DataSplitter (CSR/AnnDataLoader) vs ZarrMultiVIDataModule (ZarrDataset).

Compares two loading paths on the same backed RNA+ATAC MuData:
  - Arm A (csr_standard): DataSplitter + AnnDataLoader on zarr-backed CSRDataset .X
  - Arm B (zarr_streaming): ZarrMultiVIDataModule + ZarrDataset block streaming

Reports dataloader-only throughput (first-batch latency, steady-state batches/s) and
short training-loop timings with Lightning SimpleProfiler splits.

Usage (from repo root):

  python scripts/benchmark_multivi_csr_vs_zarr.py
  python scripts/benchmark_multivi_csr_vs_zarr.py --cells-per-batch 64 --n-batches 4 \\
      --limit-train-batches 20 --dataloader-batches 20 --output-json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

# Windows OpenMP duplicate-lib workaround (common in conda + torch stacks)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import anndata
import mudata as md
import numpy as np
import scipy.sparse as sp
import torch
import zarr
from anndata.io import sparse_dataset
from lightning.pytorch.profilers import SimpleProfiler
from mudata import MuData

# Allow running without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import scvi
from scvi.dataloaders import DataSplitter, ZarrMultiVIDataModule
from scvi.data import synthetic_iid
from scvi.model import MULTIVI


@dataclass
class CLIConfig:
    """Reproducible benchmark configuration parsed from CLI."""

    cells_per_batch: int = 2400
    n_batches: int = 8
    n_genes: int = 2000
    n_regions: int = 2000
    rna_density: float = 0.2
    atac_density: float = 0.15
    batch_size: int = 128
    train_size: float = 0.9
    num_workers: int | None = None
    pin_memory: bool = False
    block_size: int = 4096
    shuffle_buffer_blocks: int = 16
    emit_mode: str = "rolling"
    prefetch_queue_depth: int = 0
    block_prefetch_depth: int = 0
    prefetch_factor: int | None = None
    prefetch_to_gpu: bool = False
    cuda_queue_depth: int = 2
    limit_train_batches: int = 100
    max_epochs: int = 1
    dataloader_batches: int = 100
    warmup_batches: int = 5
    repeats: int = 3
    seed: int = 0
    zarr_store: str = ""
    output_json: str = ""
    n_latent: int = 10


@dataclass
class ArmConfig:
    """Per-arm loader configuration."""

    name: str
    arm_type: str  # "csr_standard" | "zarr_streaming"
    batch_size: int
    train_size: float
    num_workers: int
    pin_memory: bool
    block_size: int = 4096
    shuffle_buffer_blocks: int = 16
    emit_mode: str = "rolling"
    prefetch_queue_depth: int = 0
    block_prefetch_depth: int = 0
    prefetch_factor: int | None = None
    prefetch_to_gpu: bool = False
    cuda_queue_depth: int = 2
    seed: int = 0
    load_sparse_tensor: bool = False


@dataclass
class DataloaderMetrics:
    first_batch_sec: float
    steady_state_wall_sec: float
    steady_state_batches_per_sec: float
    steady_state_cells_per_sec: float
    n_batches_timed: int
    warmup_batches: int
    repeats: int


@dataclass
class TrainMetrics:
    wall_seconds: float
    batches_per_sec: float
    cells_per_sec: float
    loader_related_sec: float
    transfer_sec: float
    data_path_sec: float
    train_step_sec: float
    epoch_sec: float
    loader_fraction: float
    profiler_summary: dict[str, float] = field(default_factory=dict)


@dataclass
class ArmResult:
    name: str
    arm_type: str
    n_train: int
    n_val: int
    dataloader: DataloaderMetrics
    training: TrainMetrics


@dataclass
class BenchmarkReport:
    metadata: dict
    config: dict
    arms: list[ArmResult]
    speedup: dict[str, float | None]


def _windows_multiprocessing_ok() -> bool:
    """DataLoader workers often fail on Windows when dataset objects are not picklable."""
    if sys.platform != "win32":
        return True
    return False


def _default_num_workers(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return 4 if _windows_multiprocessing_ok() else 0


def _pick_accelerator() -> tuple[str, str | int, str | None]:
    if torch.cuda.is_available():
        return "gpu", 1, torch.cuda.get_device_name(0)
    return "cpu", "auto", None


def _write_and_open_backed_mudata(mdata: MuData, store_path: Path) -> MuData:
    """Write MuData to zarr and reopen with zarr-backed CSRDatasets on modality .X."""
    previous = anndata.settings.allow_write_nullable_strings
    anndata.settings.allow_write_nullable_strings = True
    try:
        mdata.write_zarr(store_path)
    finally:
        anndata.settings.allow_write_nullable_strings = previous

    backed = md.read_zarr(store_path)
    f = zarr.open(str(store_path), mode="r")
    for mod in backed.mod:
        x_group = f["mod"][mod]["X"]
        enc = x_group.attrs.get("encoding-type", "")
        if enc in ("csr_matrix", "csc_matrix"):
            backed.mod[mod].X = sparse_dataset(x_group)
    return backed


def build_backed_rna_atac_mudata(
    cfg: CLIConfig,
    store_path: Path,
) -> tuple[MuData, MuData, Path]:
    """Build synthetic RNA+ATAC MuData, write to zarr, and reopen backed."""
    scvi.settings.seed = cfg.seed
    mdata = synthetic_iid(
        batch_size=cfg.cells_per_batch,
        n_batches=cfg.n_batches,
        n_genes=cfg.n_genes,
        n_regions=cfg.n_regions,
        n_proteins=0,
        sparse_format="csr_matrix",
        return_mudata=True,
    )
    n_obs = mdata.n_obs
    mdata.mod["RNA"] = mdata.mod.pop("rna")
    mdata.mod["ATAC"] = mdata.mod.pop("accessibility")
    mdata.update()

    mdata.mod["RNA"].X = sp.random(
        n_obs,
        cfg.n_genes,
        density=cfg.rna_density,
        format="csr",
        dtype=np.float32,
        random_state=cfg.seed,
    )
    mdata.mod["ATAC"].X = sp.random(
        n_obs,
        cfg.n_regions,
        density=cfg.atac_density,
        format="csr",
        dtype=np.float32,
        random_state=cfg.seed + 1,
    )

    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    mdata_backed = _write_and_open_backed_mudata(mdata, store_path)

    for mod in ("RNA", "ATAC"):
        x = mdata_backed.mod[mod].X
        backend = getattr(x, "backend", None)
        if backend != "zarr":
            raise RuntimeError(
                f"Expected zarr-backed CSRDataset for {mod}.X after reopen; got backend={backend!r}"
            )

    MULTIVI.setup_mudata(
        mdata_backed,
        batch_key="batch",
        modalities={"rna_layer": "RNA", "atac_layer": "ATAC"},
    )
    return mdata, mdata_backed, store_path


def setup_model(mdata_backed: MuData, cfg: CLIConfig) -> MULTIVI:
    scvi.settings.seed = cfg.seed
    return MULTIVI(mdata_backed, n_latent=cfg.n_latent)


def build_arm_datamodule(
    arm: ArmConfig,
    *,
    model: MULTIVI,
    mdata_backed: MuData,
):
    """Construct explicit DataSplitter or ZarrMultiVIDataModule for the benchmark arm."""
    loader_kwargs: dict = {}
    if arm.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    if arm.arm_type == "csr_standard":
        return DataSplitter(
            model.adata_manager,
            train_size=arm.train_size,
            batch_size=arm.batch_size,
            num_workers=arm.num_workers,
            pin_memory=arm.pin_memory,
            load_sparse_tensor=arm.load_sparse_tensor,
            **loader_kwargs,
        )

    if arm.arm_type == "zarr_streaming":
        zarr_kwargs = {
            "matrix_layout": "auto",
            "train_size": arm.train_size,
            "batch_size": arm.batch_size,
            "block_size": arm.block_size,
            "shuffle_buffer_blocks": arm.shuffle_buffer_blocks,
            "emit_mode": arm.emit_mode,
            "prefetch_queue_depth": arm.prefetch_queue_depth,
            "block_prefetch_depth": arm.block_prefetch_depth,
            "num_workers": arm.num_workers,
            "pin_memory": arm.pin_memory,
            "prefetch_to_gpu": arm.prefetch_to_gpu,
            "cuda_queue_depth": arm.cuda_queue_depth,
            "seed": arm.seed,
        }
        if arm.prefetch_factor is not None:
            zarr_kwargs["prefetch_factor"] = arm.prefetch_factor
        return ZarrMultiVIDataModule.from_backed_mudata(
            mdata_backed,
            model.adata_manager,
            **zarr_kwargs,
        )

    raise ValueError(f"Unknown arm_type: {arm.arm_type!r}")


def _max_train_batches(n_train: int, batch_size: int, *, drop_last: bool = False) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if drop_last:
        return n_train // batch_size
    return math.ceil(n_train / batch_size)


def _count_loader_batches(datamodule) -> int:
    loader = _prepare_train_dataloader(datamodule)
    return sum(1 for _ in loader)


def _cap_timing_batches(
    requested_batches: int,
    requested_warmup: int,
    *,
    n_train: int,
    batch_size: int,
    observed_batches: int | None = None,
) -> tuple[int, int]:
    """Cap warmup/timing batch counts to what the train loader can yield."""
    max_batches = observed_batches
    if max_batches is None:
        max_batches = _max_train_batches(n_train, batch_size)
    max_batches = max(1, max_batches)
    n_batches = min(requested_batches, max_batches)
    warmup = min(requested_warmup, max(0, max_batches - n_batches))
    return n_batches, warmup


def _prepare_train_dataloader(datamodule, *, epoch: int = 0):
    if isinstance(datamodule, DataSplitter):
        datamodule.setup()
    else:
        datamodule.trainer = SimpleNamespace(current_epoch=epoch)
    return datamodule.train_dataloader()


def _parse_profiler(profiler: SimpleProfiler) -> dict[str, float]:
    summary: dict[str, float] = {}
    for action, durations in profiler.recorded_durations.items():
        summary[action] = float(sum(durations))
    return summary


def _loader_related_seconds(summary: dict[str, float]) -> float:
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


def time_dataloader_only(
    datamodule,
    *,
    batch_size: int,
    n_batches: int,
    warmup_batches: int,
    repeats: int,
) -> DataloaderMetrics:
    """Time first-batch latency and steady-state dataloader throughput (no training step)."""
    loader = _prepare_train_dataloader(datamodule)

    t0 = time.perf_counter()
    next(iter(loader))
    first_batch_sec = time.perf_counter() - t0

    repeat_times: list[float] = []
    for _ in range(repeats):
        loader = _prepare_train_dataloader(datamodule)
        it = iter(loader)
        for _ in range(warmup_batches):
            next(it)
        t_start = time.perf_counter()
        for _ in range(n_batches):
            next(it)
        repeat_times.append(time.perf_counter() - t_start)

    steady_wall = float(np.mean(repeat_times))
    steady_batches_per_sec = n_batches / steady_wall if steady_wall > 0 else float("nan")
    steady_cells_per_sec = (n_batches * batch_size) / steady_wall if steady_wall > 0 else float("nan")

    return DataloaderMetrics(
        first_batch_sec=first_batch_sec,
        steady_state_wall_sec=steady_wall,
        steady_state_batches_per_sec=steady_batches_per_sec,
        steady_state_cells_per_sec=steady_cells_per_sec,
        n_batches_timed=n_batches,
        warmup_batches=warmup_batches,
        repeats=repeats,
    )


def time_training_loop(
    mdata_backed: MuData,
    datamodule,
    arm: ArmConfig,
    *,
    limit_train_batches: int,
    max_epochs: int,
    seed: int,
    accelerator: str,
    devices: str | int,
) -> TrainMetrics:
    """Run a short model.train() with SimpleProfiler for loader vs train-step splits."""
    scvi.settings.seed = seed
    profiler = SimpleProfiler()
    model = MULTIVI(mdata_backed, n_latent=10)

    trainer_kwargs: dict = {
        "enable_progress_bar": False,
        "logger": False,
        "num_sanity_val_steps": 0,
        "profiler": profiler,
    }
    if isinstance(datamodule, ZarrMultiVIDataModule):
        trainer_kwargs["reload_dataloaders_every_n_epochs"] = 1

    t0 = time.perf_counter()
    model.train(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        batch_size=arm.batch_size,
        train_size=arm.train_size,
        early_stopping=False,
        limit_train_batches=limit_train_batches,
        datamodule=datamodule,
        **trainer_kwargs,
    )
    wall = time.perf_counter() - t0

    summary = _parse_profiler(profiler)
    loader_sec = _loader_related_seconds(summary)
    transfer_sec = _transfer_seconds(summary)
    data_path_sec = loader_sec + transfer_sec
    train_sec = _train_step_seconds(summary)
    epoch_sec = _epoch_seconds(summary)
    loader_frac = loader_sec / epoch_sec if epoch_sec > 0 else 0.0

    batches_per_sec = limit_train_batches / wall if wall > 0 else float("nan")
    cells_per_sec = (limit_train_batches * arm.batch_size) / wall if wall > 0 else float("nan")

    return TrainMetrics(
        wall_seconds=wall,
        batches_per_sec=batches_per_sec,
        cells_per_sec=cells_per_sec,
        loader_related_sec=loader_sec,
        transfer_sec=transfer_sec,
        data_path_sec=data_path_sec,
        train_step_sec=train_sec,
        epoch_sec=epoch_sec,
        loader_fraction=loader_frac,
        profiler_summary=summary,
    )


def _arm_split_sizes(datamodule) -> tuple[int, int]:
    if isinstance(datamodule, DataSplitter):
        datamodule.setup()
        return len(datamodule.train_idx), len(datamodule.val_idx)
    return int(datamodule.n_train), int(datamodule.n_val)


def run_arm(
    arm: ArmConfig,
    *,
    mdata_backed: MuData,
    model: MULTIVI,
    cfg: CLIConfig,
    accelerator: str,
    devices: str | int,
) -> ArmResult:
    datamodule = build_arm_datamodule(arm, model=model, mdata_backed=mdata_backed)
    n_train, n_val = _arm_split_sizes(datamodule)
    observed_batches = _count_loader_batches(datamodule)
    n_batches, warmup_batches = _cap_timing_batches(
        cfg.dataloader_batches,
        cfg.warmup_batches,
        n_train=n_train,
        batch_size=arm.batch_size,
        observed_batches=observed_batches,
    )
    if n_batches < cfg.dataloader_batches or warmup_batches < cfg.warmup_batches:
        print(
            f"  [{arm.name}] capping dataloader timing to "
            f"{n_batches} batches (warmup={warmup_batches}, loader yields {observed_batches})"
        )

    dataloader_metrics = time_dataloader_only(
        datamodule,
        batch_size=arm.batch_size,
        n_batches=n_batches,
        warmup_batches=warmup_batches,
        repeats=cfg.repeats,
    )

    limit_train_batches = min(cfg.limit_train_batches, observed_batches)
    train_datamodule = build_arm_datamodule(arm, model=model, mdata_backed=mdata_backed)
    training_metrics = time_training_loop(
        mdata_backed,
        train_datamodule,
        arm,
        limit_train_batches=limit_train_batches,
        max_epochs=cfg.max_epochs,
        seed=cfg.seed,
        accelerator=accelerator,
        devices=devices,
    )

    return ArmResult(
        name=arm.name,
        arm_type=arm.arm_type,
        n_train=n_train,
        n_val=n_val,
        dataloader=dataloader_metrics,
        training=training_metrics,
    )


def _compute_speedup(csr: ArmResult, zarr: ArmResult) -> dict[str, float | None]:
    def ratio(numerator: float, denominator: float) -> float | None:
        if denominator <= 0 or np.isnan(denominator) or np.isnan(numerator):
            return None
        return numerator / denominator

    return {
        "dataloader_first_batch_csr_over_zarr": ratio(
            csr.dataloader.first_batch_sec, zarr.dataloader.first_batch_sec
        ),
        "dataloader_batches_per_sec_zarr_over_csr": ratio(
            zarr.dataloader.steady_state_batches_per_sec,
            csr.dataloader.steady_state_batches_per_sec,
        ),
        "dataloader_cells_per_sec_zarr_over_csr": ratio(
            zarr.dataloader.steady_state_cells_per_sec,
            csr.dataloader.steady_state_cells_per_sec,
        ),
        "training_wall_csr_over_zarr": ratio(
            csr.training.wall_seconds, zarr.training.wall_seconds
        ),
        "training_batches_per_sec_zarr_over_csr": ratio(
            zarr.training.batches_per_sec, csr.training.batches_per_sec
        ),
        "training_cells_per_sec_zarr_over_csr": ratio(
            zarr.training.cells_per_sec, csr.training.cells_per_sec
        ),
    }


def _sanity_checks(csr: ArmResult, zarr: ArmResult) -> list[str]:
    issues: list[str] = []
    if csr.n_train != zarr.n_train:
        issues.append(f"n_train mismatch: csr={csr.n_train}, zarr={zarr.n_train}")
    if csr.n_val != zarr.n_val:
        issues.append(f"n_val mismatch: csr={csr.n_val}, zarr={zarr.n_val}")

    for arm in (csr, zarr):
        for label, value in (
            ("first_batch_sec", arm.dataloader.first_batch_sec),
            ("steady_state_wall_sec", arm.dataloader.steady_state_wall_sec),
            ("training_wall_seconds", arm.training.wall_seconds),
        ):
            if np.isnan(value) or value < 0:
                issues.append(f"{arm.name}: invalid timing {label}={value}")
    return issues


def _default_arms(cfg: CLIConfig) -> list[ArmConfig]:
    num_workers = _default_num_workers(cfg.num_workers)
    shared = {
        "batch_size": cfg.batch_size,
        "train_size": cfg.train_size,
        "num_workers": num_workers,
        "pin_memory": cfg.pin_memory,
        "seed": cfg.seed,
    }
    return [
        ArmConfig(
            name="csr_standard",
            arm_type="csr_standard",
            load_sparse_tensor=False,
            **shared,
        ),
        ArmConfig(
            name="zarr_streaming",
            arm_type="zarr_streaming",
            block_size=cfg.block_size,
            shuffle_buffer_blocks=cfg.shuffle_buffer_blocks,
            emit_mode=cfg.emit_mode,
            prefetch_queue_depth=cfg.prefetch_queue_depth,
            block_prefetch_depth=cfg.block_prefetch_depth,
            prefetch_factor=cfg.prefetch_factor,
            prefetch_to_gpu=cfg.prefetch_to_gpu,
            cuda_queue_depth=cfg.cuda_queue_depth,
            **shared,
        ),
    ]


def _parse_cli() -> CLIConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark MultiVI DataSplitter vs ZarrMultiVIDataModule on backed RNA+ATAC MuData"
    )
    parser.add_argument("--cells-per-batch", type=int, default=2400)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--n-regions", type=int, default=2000)
    parser.add_argument("--rna-density", type=float, default=0.2)
    parser.add_argument("--atac-density", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=float, default=0.9)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: 4 on Linux, 0 on Windows)",
    )
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--shuffle-buffer-blocks", type=int, default=16)
    parser.add_argument(
        "--emit-mode",
        choices=("rolling", "flush"),
        default="rolling",
        help="Zarr shuffle emit mode",
    )
    parser.add_argument("--prefetch-queue-depth", type=int, default=0)
    parser.add_argument("--block-prefetch-depth", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--prefetch-to-gpu", action="store_true", default=False)
    parser.add_argument("--cuda-queue-depth", type=int, default=2)
    parser.add_argument("--limit-train-batches", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--dataloader-batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--zarr-store",
        type=str,
        default="",
        help="Optional existing zarr store path; if omitted, a temp directory is used",
    )
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()
    return CLIConfig(
        cells_per_batch=args.cells_per_batch,
        n_batches=args.n_batches,
        n_genes=args.n_genes,
        n_regions=args.n_regions,
        rna_density=args.rna_density,
        atac_density=args.atac_density,
        batch_size=args.batch_size,
        train_size=args.train_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        block_size=args.block_size,
        shuffle_buffer_blocks=args.shuffle_buffer_blocks,
        emit_mode=args.emit_mode,
        prefetch_queue_depth=args.prefetch_queue_depth,
        block_prefetch_depth=args.block_prefetch_depth,
        prefetch_factor=args.prefetch_factor,
        prefetch_to_gpu=args.prefetch_to_gpu,
        cuda_queue_depth=args.cuda_queue_depth,
        limit_train_batches=args.limit_train_batches,
        max_epochs=args.max_epochs,
        dataloader_batches=args.dataloader_batches,
        warmup_batches=args.warmup_batches,
        repeats=args.repeats,
        seed=args.seed,
        zarr_store=args.zarr_store,
        output_json=args.output_json,
    )


def _print_arm_result(arm: ArmResult) -> None:
    print(f"--- Arm: {arm.name} ({arm.arm_type}) ---")
    print(f"  split: n_train={arm.n_train}, n_val={arm.n_val}")
    print(f"  dataloader first batch: {arm.dataloader.first_batch_sec:.3f}s")
    print(
        f"  dataloader steady ({arm.dataloader.n_batches_timed} batches, "
        f"warmup={arm.dataloader.warmup_batches}, repeats={arm.dataloader.repeats}): "
        f"{arm.dataloader.steady_state_wall_sec:.2f}s wall, "
        f"{arm.dataloader.steady_state_batches_per_sec:.2f} batches/s, "
        f"{arm.dataloader.steady_state_cells_per_sec:.0f} cells/s"
    )
    print(
        f"  training ({arm.training.epoch_sec:.2f}s epoch profiler): "
        f"wall={arm.training.wall_seconds:.2f}s, "
        f"{arm.training.batches_per_sec:.2f} batches/s, "
        f"{arm.training.cells_per_sec:.0f} cells/s"
    )
    print(
        f"  data path: loader={arm.training.loader_related_sec:.2f}s + "
        f"transfer={arm.training.transfer_sec:.2f}s "
        f"(fraction={arm.training.loader_fraction:.2%})"
    )
    print(f"  training_step: {arm.training.train_step_sec:.2f}s")
    if arm.training.profiler_summary:
        top = sorted(arm.training.profiler_summary.items(), key=lambda x: -x[1])[:6]
        print("  profiler top actions:")
        for action, sec in top:
            print(f"    {sec:8.3f}s  {action}")
    print()


def _print_summary(csr: ArmResult, zarr: ArmResult, speedup: dict[str, float | None]) -> None:
    print("=" * 72)
    print("Speedup summary (>1 means zarr is faster)")
    print("=" * 72)
    labels = {
        "dataloader_first_batch_csr_over_zarr": "dataloader first batch",
        "dataloader_batches_per_sec_zarr_over_csr": "dataloader batches/s",
        "dataloader_cells_per_sec_zarr_over_csr": "dataloader cells/s",
        "training_wall_csr_over_zarr": "training wall time",
        "training_batches_per_sec_zarr_over_csr": "training batches/s",
        "training_cells_per_sec_zarr_over_csr": "training cells/s",
    }
    for key, label in labels.items():
        value = speedup.get(key)
        if value is None:
            print(f"  {label:45s}  n/a")
        else:
            print(f"  {label:45s}  {value:.2f}x")


def _report_to_dict(report: BenchmarkReport) -> dict:
    return {
        "metadata": report.metadata,
        "config": report.config,
        "arms": [asdict(arm) for arm in report.arms],
        "speedup": report.speedup,
    }


def _run_benchmark(
    cfg: CLIConfig,
    *,
    mdata_backed: MuData,
    store_path: Path,
    accelerator: str,
    devices: str | int,
    gpu_name: str | None,
) -> BenchmarkReport:
    model = setup_model(mdata_backed, cfg)
    arms_cfg = _default_arms(cfg)
    results: list[ArmResult] = []

    for arm_cfg in arms_cfg:
        result = run_arm(
            arm_cfg,
            mdata_backed=mdata_backed,
            model=model,
            cfg=cfg,
            accelerator=accelerator,
            devices=devices,
        )
        results.append(result)
        _print_arm_result(result)

    if len(results) != 2:
        raise RuntimeError("Expected exactly two benchmark arms.")

    csr_result, zarr_result = results[0], results[1]
    speedup = _compute_speedup(csr_result, zarr_result)
    issues = _sanity_checks(csr_result, zarr_result)
    if issues:
        print("Sanity check warnings:")
        for issue in issues:
            print(f"  - {issue}")
        print()
    _print_summary(csr_result, zarr_result, speedup)

    return BenchmarkReport(
        metadata={
            "seed": cfg.seed,
            "accelerator": accelerator,
            "devices": devices,
            "gpu_name": gpu_name,
            "pytorch_cuda_available": torch.cuda.is_available(),
            "n_obs": mdata_backed.n_obs,
            "n_genes": cfg.n_genes,
            "n_regions": cfg.n_regions,
            "zarr_store": str(store_path),
            "sanity_issues": issues,
        },
        config=asdict(cfg),
        arms=results,
        speedup=speedup,
    )


def main() -> None:
    cfg = _parse_cli()
    accelerator, devices, gpu_name = _pick_accelerator()
    num_workers = _default_num_workers(cfg.num_workers)

    print("=" * 72)
    print("MultiVI CSR (DataSplitter) vs Zarr (ZarrMultiVIDataModule) benchmark")
    print("=" * 72)
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if gpu_name:
        print(f"GPU: {gpu_name}")
    else:
        print("GPU: none (CPU timing)")
    n_obs = cfg.cells_per_batch * cfg.n_batches
    print(
        f"Shape: {n_obs} cells, {cfg.n_genes} genes, {cfg.n_regions} regions "
        f"(RNA density={cfg.rna_density}, ATAC density={cfg.atac_density})"
    )
    print(
        f"Loader: batch_size={cfg.batch_size}, train_size={cfg.train_size}, "
        f"num_workers={num_workers}, pin_memory={cfg.pin_memory}"
    )
    print(
        f"Zarr knobs: block_size={cfg.block_size}, "
        f"shuffle_buffer_blocks={cfg.shuffle_buffer_blocks}, emit_mode={cfg.emit_mode}, "
        f"prefetch_queue_depth={cfg.prefetch_queue_depth}, "
        f"block_prefetch_depth={cfg.block_prefetch_depth}, "
        f"prefetch_to_gpu={cfg.prefetch_to_gpu}"
    )
    print(
        f"Timing: dataloader_batches={cfg.dataloader_batches}, "
        f"limit_train_batches={cfg.limit_train_batches}, seed={cfg.seed}"
    )
    print(f"Accelerator: {accelerator}")
    print()

    def _build_and_run(store_path: Path) -> BenchmarkReport:
        print(f"Building backed RNA+ATAC MuData at {store_path}...")
        _, mdata_backed, resolved_store = build_backed_rna_atac_mudata(cfg, store_path)
        print(f"  store={resolved_store}, n_obs={mdata_backed.n_obs}\n")
        return _run_benchmark(
            cfg,
            mdata_backed=mdata_backed,
            store_path=resolved_store,
            accelerator=accelerator,
            devices=devices,
            gpu_name=gpu_name,
        )

    if cfg.zarr_store:
        report = _build_and_run(Path(cfg.zarr_store))
    else:
        with tempfile.TemporaryDirectory(prefix="multivi_csr_zarr_bench_") as tmp:
            report = _build_and_run(Path(tmp) / "mdata.zarr")

    if cfg.output_json:
        out_path = Path(cfg.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_report_to_dict(report), indent=2), encoding="utf-8")
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
