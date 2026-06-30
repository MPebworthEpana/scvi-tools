# Zarr-backed scVI and PeakVI streaming

:::{note}
`ZarrDataset` and `ZarrAnnDataModule` are **EXPERIMENTAL** and supported for
{class}`~scvi.model.SCVI` and {class}`~scvi.model.PEAKVI` training on zarr-backed
{class}`~anndata.AnnData` objects.
:::

This page describes how to train scVI or PeakVI on large single-modality datasets stored
on disk in zarr format without loading the full count matrix into RAM. The streaming
mechanics are the same as {doc}`zarr_multivi_streaming`: sorted block reads, a compact
shuffle buffer, and minibatch emission to the trainer.

## When to use zarr streaming

Use {class}`~scvi.dataloaders.ZarrAnnDataModule` when:

- The count matrix lives in a zarr-backed AnnData store (for example after
  `adata.write_zarr(...)` and reopening with a backed `.X`).
- The dataset is too large to hold in memory during training.
- You are training {class}`~scvi.model.SCVI` or {class}`~scvi.model.PEAKVI`.

For in-memory AnnData, the default {class}`~scvi.dataloaders.DataSplitter` path is
usually simpler. For multi-modality MuData, see {doc}`zarr_multivi_streaming`.

## Matrix layouts (`matrix_layout`)

{class}`~scvi.dataloaders.ZarrAnnDataModule.from_backed_anndata` accepts:

| Value | Behavior |
|-------|----------|
| `"csr"` | Treat the registered count matrix as a zarr CSR group (default). |
| `"dense"` | Treat the matrix as a dense `zarr.Array`. Requires `store_dir`. |
| `"auto"` | Infer CSR vs dense from the backed matrix. Used by auto-selection in `train()`. |

## Manual usage

```python
import scvi

adata = ...  # zarr-backed AnnData with backed .X
scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
model = scvi.model.SCVI(adata)

from scvi.dataloaders import ZarrAnnDataModule

dm = ZarrAnnDataModule.from_backed_anndata(
    adata,
    model.adata_manager,
    matrix_layout="auto",
    batch_size=1024,
    block_size=4096,
    num_workers=4,
    pin_memory=True,
)
model.train(datamodule=dm, reload_dataloaders_every_n_epochs=1)
```

PeakVI follows the same pattern after `PEAKVI.setup_anndata(...)`.

## Auto-selection in `train()`

When `SCVI.train()` or `PEAKVI.train()` is called without an explicit `datamodule`,
scvi-tools attempts to construct a {class}`~scvi.dataloaders.ZarrAnnDataModule` if the
registered count matrix is zarr-backed. If construction fails, training falls back to
{class}`~scvi.dataloaders.DataSplitter`.

Zarr tuning kwargs (`block_size`, `shuffle_buffer_blocks`, `num_workers`, `pin_memory`,
and related options) can be passed via `datasplitter_kwargs` when auto-selection is active.

## Covariates

`ZarrAnnDataModule` supports registered batch, label, size-factor, categorical, and
continuous covariates by materializing per-observation arrays once and slicing them
during block reads. Multi-column size factors are not supported.

## Limitations

- Experimental API; currently focused on scVI and PeakVI.
- CSR matrices are converted to dense `float32` minibatches before the forward pass.
- Windows often requires `num_workers=0`.
- See {doc}`zarr_multivi_streaming` for shared tuning guidance (`block_size`,
  `shuffle_buffer_blocks`, `prefetch_to_gpu`, etc.).

## See also

- {doc}`zarr_multivi_streaming` — MultiVI / MuData streaming
- {doc}`custom_dataloaders` — TileDB, LamindDB, and other custom loaders
