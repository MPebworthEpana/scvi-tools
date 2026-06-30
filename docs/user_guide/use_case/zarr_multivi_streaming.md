# Zarr-backed MultiVI streaming

:::{note}
`ZarrDataset` and `ZarrMultiVIDataModule` are **EXPERIMENTAL** and supported for
{class}`~scvi.model.MULTIVI` training on zarr-backed {class}`~mudata.MuData` objects.
For single-modality {class}`~anndata.AnnData` streaming with scVI or PeakVI, see
{doc}`zarr_anndata_streaming`.
:::

This page describes how to train MultiVI on large datasets stored on disk in zarr format without
loading full count matrices into RAM. The design follows the same bounded-memory, high-throughput
streaming pattern used by {class}`~scvi.dataloaders.TileDBDataModule` (CZI / cellxgene census) and
other custom dataloaders documented in {doc}`custom_dataloaders`: read contiguous row blocks from
backed storage, shuffle in a compact buffer, and emit paired modality minibatches to the trainer.

## When to use zarr streaming

Use {class}`~scvi.dataloaders.ZarrMultiVIDataModule` when:

- Modality matrices live in a zarr-backed MuData store (for example after `mdata.write_zarr(...)`
  and reopening with backed `.X`).
- The dataset is too large to hold all modalities in memory during training.
- You need **mixed layouts** per modality—for example sparse RNA (CSR) plus dense ADT/protein
  counts that do not work well as zarr CSR.

For in-memory MuData, the default {class}`~scvi.dataloaders.DataSplitter` path is usually simpler.

## Matrix layouts (`matrix_layout`)

Each registered modality can be streamed as zarr-backed CSR (`CSRDataset`) or as a dense 2-D
`zarr.Array`. {meth}`~scvi.dataloaders.ZarrMultiVIDataModule.from_backed_mudata` accepts:

| Value | Behavior |
|-------|----------|
| `"csr"` | Treat every modality as a zarr CSR group (default). |
| `"dense"` | Treat every modality as a dense `zarr.Array` under `mod/{mod_key}/X_dense` (or `X{x_suffix}`). Requires `store_dir`. |
| `"auto"` | Infer layout **per modality** (CSR vs dense). Required for RNA CSR + ADT/protein dense in one datamodule. |

### Default caveat

- **Direct construction:** `from_backed_mudata(..., matrix_layout="csr")` is the default. Mixed
  CSR + dense stores will fail unless you pass `matrix_layout="auto"` (or `"dense"` when all
  modalities are dense).
- **Automatic selection:** {meth}`~scvi.model.MULTIVI.train` calls
  `from_backed_mudata(..., matrix_layout="auto")` when it detects fully zarr-backed modality
  matrices, so mixed layouts work without an explicit datamodule.

```python
# Explicit (recommended when RNA is CSR and ADT/protein is dense):
dm = ZarrMultiVIDataModule.from_backed_mudata(
    mdata_backed,
    model.adata_manager,
    matrix_layout="auto",
    store_dir=store_path,
    ...
)

# Or rely on auto-selection when all registered .X are zarr-backed:
model.train(max_epochs=100, datasplitter_kwargs={"num_workers": 4, "block_size": 4096})
```

### `store_dir` expectations

- **`matrix_layout="dense"`:** `store_dir` is required (path to the zarr store root).
- **`matrix_layout="auto"`:** Resolved from `mdata.filename` when set; otherwise pass `store_dir`
  explicitly. Required if you use `x_suffix` or if dense paths cannot be inferred from backed
  `.X` objects.
- **CSR paths:** Usually inferred as absolute paths from backed `CSRDataset` objects; relative
  paths in {class}`~scvi.dataloaders.ZarrMatrixSource` require `store_dir`.

Dense arrays are expected at `store_dir/mod/{modality_key}/X_dense` unless you set `x_suffix`
(for example `x_suffix="_dense"`).

## Backed ADT / protein (dense) with RNA (CSR)

ADT and protein expression are typically **dense** count matrices. Storing them as zarr CSR is
often a poor fit; the streaming loader supports attaching a backed dense `zarr.Array` on
`mdata.mod["protein_expression"].X` (or your ADT modality key) while RNA remains zarr CSR.

**Requirements for mixed RNA + ADT/protein:**

1. Write MuData to zarr and reopen with backed sparse RNA `.X` (`sparse_dataset` on the CSR group).
2. Write ADT/protein counts to a dense zarr array (for example `mod/protein_expression/X_dense`)
   and set `mdata_backed.mod["protein_expression"].X` to that `zarr.Array`.
3. Call {meth}`~scvi.model.MULTIVI.setup_mudata` with the correct modality mapping.
4. Build the datamodule with **`matrix_layout="auto"`** and **`store_dir`** pointing at the store.

### Minimal example

```python
from pathlib import Path

import mudata as md
import numpy as np
import scipy.sparse as sp
import zarr
from anndata.io import sparse_dataset

import scvi
from scvi.dataloaders import ZarrMultiVIDataModule
from scvi.model import MULTIVI

store_path = Path("mdata.zarr")

# 1. Write MuData and reopen RNA as zarr-backed CSR
mdata.write_zarr(store_path)
mdata_backed = md.read_zarr(store_path)
f = zarr.open(str(store_path), mode="r")
rna_x = f["mod"]["RNA"]["X"]
if rna_x.attrs.get("encoding-type", "") in ("csr_matrix", "csc_matrix"):
    mdata_backed.mod["RNA"].X = sparse_dataset(rna_x)

# 2. Store ADT/protein as dense zarr (example: protein_expression modality)
protein_mod = "protein_expression"
protein_dense = np.asarray(mdata.mod[protein_mod].X, dtype=np.float32)
dest = store_path / "mod" / protein_mod / "X_dense"
arr = zarr.open_array(
    str(dest),
    mode="w",
    shape=protein_dense.shape,
    chunks=(min(64, mdata.n_obs), protein_dense.shape[1]),
    dtype="float32",
)
arr[:] = protein_dense
mdata_backed.mod[protein_mod].X = zarr.open_array(str(dest), mode="r")

# 3. Setup MultiVI registry
MULTIVI.setup_mudata(
    mdata_backed,
    batch_key="batch",
    modalities={"rna_layer": "RNA", "protein_layer": protein_mod},
)
model = MULTIVI(mdata_backed)

# 4. Stream with mixed layouts (auto) — not the default "csr" alone
dm = ZarrMultiVIDataModule.from_backed_mudata(
    mdata_backed,
    model.adata_manager,
    matrix_layout="auto",
    store_dir=store_path,
    batch_size=128,
    block_size=4096,
    shuffle_buffer_blocks=16,
    num_workers=4,
    pin_memory=True,
)

model.train(max_epochs=100, datamodule=dm)
```

At batch time, CSR rows are densified to `float32` tensors on CPU before transfer to the device,
matching the standard MultiVI training path.

## How efficient loading works

{class}`~scvi.dataloaders.ZarrDataset` is an `IterableDataset` that streams rows from zarr-backed
matrices without materializing the full dataset. The pipeline is similar in spirit to TileDB-SOMA /
Census streaming:

**Pipeline:** backed MuData → `setup_mudata` → `from_backed_mudata` (`matrix_layout="auto"`) →
`ZarrDataset` → sorted block reads → shuffle buffer → worker/rank partition → `MultiVI.train`.

1. **Picklable sources:** {func}`~scvi.dataloaders.matrix_sources_from_backed_mudata` records
   per-modality zarr paths and layouts once in the main process. Each DataLoader worker reopens
   handles independently (safe for `num_workers > 0`).
2. **Contiguous blocks:** Training indices are processed in blocks of `block_size` rows. Within each
   block, row indices are **sorted** before I/O so zarr reads are sequential rather than random-seek.
3. **Shuffle buffer:** When `shuffle=True`, blocks accumulate in a compact buffer
   (`shuffle_buffer_blocks`). With the default ``emit_mode='rolling'``, one minibatch is emitted
   at a time once the shuffle pool reaches ``max(batch_size, shuffle_buffer_blocks × block_size)``
   rows (capped at the worker's index count). Set ``emit_mode='flush'`` to restore the legacy
   flush-all-then-clear behavior.
4. **Optional prefetch:** ``prefetch_queue_depth`` queues densified CPU batches per worker;
   ``block_prefetch_depth`` reads raw zarr blocks ahead of densification; ``prefetch_to_gpu`` wraps
   the DataLoader with :class:`~scvi.dataloaders.CUDABatchPrefetcher` for H2D overlap.
5. **Worker / rank partitioning:** Blocks are split across DataLoader workers; under DDP, each rank
   takes a disjoint slice of indices so all ranks emit the same number of batches.
6. **Obs metadata in memory:** Batch labels and indices are held as small NumPy arrays; only
   modality matrices are streamed from zarr.

Validation uses the same block reads without shuffling (streaming windows from the buffer front).

## Tuning parameters

Pass these through `from_backed_mudata(...)` or via `MULTIVI.train(datasplitter_kwargs={...})` when
auto-selection is active. Only the kwargs in the table below are forwarded to the zarr datamodule;
DataSplitter-only options (`shuffle_set_split`, `distributed_sampler`, `load_sparse_tensor`,
`external_indexing`) disable auto zarr selection.

| Parameter | Default | Role | When to increase | When to decrease |
|-----------|---------|------|------------------|------------------|
| `block_size` | `4096` | Rows per contiguous zarr read | Larger stores, row-chunked zarr aligned with chunks; fewer seeks | Limited RAM per worker |
| `shuffle_buffer_blocks` | `16` | Minimum shuffle pool in rows: ``max(batch_size, shuffle_buffer_blocks × block_size)`` (rolling emit) | Better shuffle quality | Memory pressure; use `1–2` when `n_train` is small |
| `emit_mode` | `'rolling'` | `'rolling'` emits one batch at a time; `'flush'` legacy burst emit | Match old behavior | Smoother loader queue (default) |
| `prefetch_queue_depth` | `0` | CPU densified batches queued per worker (0 = off) | Hide read+densify stalls | Host RAM (~batch bytes × depth × workers) |
| `prefetch_factor` | `2` | PyTorch DataLoader batches prefetched per worker | Smoother main-process queue | Host RAM |
| `block_prefetch_depth` | `0` | Raw zarr blocks read ahead of densify (0 = off) | Overlap zarr I/O with densify | Host RAM for sparse blocks |
| `prefetch_to_gpu` | `False` | Wrap loader with `CUDABatchPrefetcher` | CUDA training with `pin_memory=True` | CPU-only training |
| `cuda_queue_depth` | `2` | GPU batches in flight when `prefetch_to_gpu=True` | Overlap H2D with training step | GPU memory |
| `batch_size` | `128` | Training minibatch size | GPU utilization | OOM on GPU or host |
| `num_workers` | `scvi.settings.dl_num_workers` (`0`) | Parallel I/O workers | Slow disk / network store; CPU headroom | Windows spawn issues; debugging |
| `pin_memory` | `False` | Pinned host memory for faster H2D | CUDA training | CPU-only training |
| `seed` | `0` | Train/val split and per-epoch block order | Reproducibility across runs | — |
| `drop_last` | `False` | Drop final partial batch | DDP training (also forced when shuffling under DDP) | Maximize data use |

**Practical tips:**

- Align `block_size` with zarr **row chunk size** when you control how arrays are written.
- Set `shuffle_buffer_blocks` so `shuffle_buffer_blocks × block_size` is not much larger than
  `n_train` (rolling emit caps at `n_train`, but smaller pools yield smoother batch delivery).
- For smooth training on CUDA: `num_workers=4`, `pin_memory=True`, `prefetch_queue_depth=2`,
  `prefetch_factor=4`, optional `prefetch_to_gpu=True`.
- Start with `num_workers=4`, `block_size=4096`, `shuffle_buffer_blocks=16` (benchmark defaults in
  `scripts/benchmark_multivi_csr_vs_zarr.py`), then profile.
- Training dataloaders use `persistent_workers=False` so epoch-wise reshuffle works with
  `reload_dataloaders_every_n_epochs=1` (set automatically when using a zarr datamodule).

## Comparison with TileDB-SOMA and other loaders

| Aspect | Zarr MultiVI | TileDB-SOMA / Census | LamindDB / AnnCollection |
|--------|--------------|------------------------|---------------------------|
| Primary models | MultiVI (MuData) | SCVI, SCANVI, … | SCVI-family via collection |
| Storage | Local/cloud zarr MuData | TileDB SOMA experiment | lamindb / disk-backed AnnData |
| Memory model | Stream blocks from zarr | Query slices without full load | Concatenate backed AnnData |
| Mixed sparse + dense | Per-modality `matrix_layout="auto"` | SOMA sparse matrices | Sparse CSR expected |
| Covariates | Batch labels only (no cat/cont covs yet) | TileDB obs columns | Depends on setup |

All three approaches target the same goal: **train on data larger than RAM** with efficient,
batched I/O rather than loading the full matrix up front.

## Benchmarking

Repository scripts under `scripts/` help compare loading paths and tune host/GPU overlap:

- `scripts/benchmark_multivi_csr_vs_zarr.py` — CSR `DataSplitter` vs `ZarrMultiVIDataModule`
  (first-batch latency, steady-state batches/s).
- `scripts/benchmark_multivi_idle_gap.py` — Lightning profiler splits for loader wait vs train step
  (`num_workers`, `pin_memory`, optional `prefetch_to_gpu` on `DataSplitter`).
- `scripts/benchmark_densify_cpu_vs_gpu.py` — CPU densify + H2D vs sparse-on-GPU densify.

Example:

```bash
python scripts/benchmark_multivi_csr_vs_zarr.py --num-workers 4 --pin-memory
```

On Linux/WSL with CUDA, use `scripts/run_benchmark_csr_vs_zarr_wsl.sh` for the CSR vs Zarr
comparison or `scripts/run_benchmark_wsl.sh` for the idle-gap benchmark.

## Limitations

- Multi-column size factors are not supported.
- Experimental API; MultiVI uses `ZarrMultiVIDataModule`, scVI/PeakVI use `ZarrAnnDataModule`.
- CSR modalities are converted to dense `float32` batches before the forward pass (dense ADT avoids
  CSR overhead for protein/ADT but still streams from disk).

## See also

- {doc}`custom_dataloaders` — TileDB, LamindDB, and AnnCollection loaders
- {doc}`/user_guide/models/multivi` — MultiVI model reference
- {doc}`/tutorials/notebooks/multimodal/MultiVI_tutorial` — standard in-memory MultiVI workflow
