"""Streaming IterableDataset over zarr-backed CSR or dense matrices."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse as sp
import torch
import zarr
from anndata.io import sparse_dataset
from torch.utils.data import IterableDataset, get_worker_info

from scvi import REGISTRY_KEYS

if TYPE_CHECKING:
    from anndata.abc import CSRDataset
    from mudata import MuData

logger = logging.getLogger(__name__)

MatrixLayout = Literal["csr", "dense"]

_INT64_OBS_KEYS = frozenset({
    REGISTRY_KEYS.BATCH_KEY,
    REGISTRY_KEYS.LABELS_KEY,
    REGISTRY_KEYS.INDICES_KEY,
})


@dataclass(frozen=True)
class ZarrMatrixSource:
    """EXPERIMENTAL: Descriptor for reopening a zarr-backed matrix inside a worker.

    ``x_relpath`` may be absolute or relative to ``store_dir``. For CSR it must
    point at the CSR zarr group (e.g. ``"mod/RNA/X"``). For dense it must point
    at the zarr array group (e.g. ``"mod/RNA/X_dense"``).
    """

    registry_key: str
    x_relpath: str
    layout: MatrixLayout = "csr"


# Backward-compatible alias (layout defaults to csr).
ZarrCSRSource = ZarrMatrixSource


@dataclass
class _MatrixBuffer:
    """Compact shuffle buffer for one streaming epoch (sparse CSR or dense rows)."""

    matrix_blocks: dict[str, list[sp.csr_matrix | np.ndarray]] = field(default_factory=dict)
    matrix_layouts: dict[str, MatrixLayout] = field(default_factory=dict)
    obs_blocks: dict[str, list[np.ndarray]] = field(default_factory=dict)
    n_rows: int = 0

    def append(self, block: dict, *, layouts: dict[str, MatrixLayout]) -> None:
        for key, value in block.items():
            if key in layouts:
                self.matrix_blocks.setdefault(key, []).append(value)
                self.matrix_layouts[key] = layouts[key]
            else:
                self.obs_blocks.setdefault(key, []).append(value)
        self.n_rows += next(iter(block.values())).shape[0]

    def clear(self) -> None:
        self.matrix_blocks.clear()
        self.matrix_layouts.clear()
        self.obs_blocks.clear()
        self.n_rows = 0

    def materialize(self) -> None:
        for key, blocks in self.matrix_blocks.items():
            if len(blocks) <= 1:
                continue
            layout = self.matrix_layouts.get(key, "csr")
            if layout == "dense":
                self.matrix_blocks[key] = [np.concatenate(blocks, axis=0)]
            else:
                self.matrix_blocks[key] = [sp.vstack(blocks, format="csr")]

        for key, blocks in self.obs_blocks.items():
            if len(blocks) > 1:
                self.obs_blocks[key] = [np.concatenate(blocks, axis=0)]

    def stacked_matrix(self, key: str) -> sp.csr_matrix | np.ndarray:
        blocks = self.matrix_blocks[key]
        if len(blocks) == 1:
            return blocks[0]
        layout = self.matrix_layouts.get(key, "csr")
        if layout == "dense":
            return np.concatenate(blocks, axis=0)
        return sp.vstack(blocks, format="csr")

    def stacked_obs(self, key: str) -> np.ndarray:
        blocks = self.obs_blocks[key]
        if len(blocks) == 1:
            return blocks[0]
        return np.concatenate(blocks, axis=0)


# Backward-compatible alias used internally before generalization.
_SparseBuffer = _MatrixBuffer


def _identity_collate(batch):
    """Picklable identity collate for IterableDataset batches (spawn-safe)."""
    return batch


def _get_rank_worker_ids() -> tuple[int, int, int, int]:
    rank = 0
    world_size = 1
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    except (ImportError, RuntimeError):
        pass

    worker_id = 0
    num_workers = 1
    worker_info = get_worker_info()
    if worker_info is not None:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
    return rank, world_size, worker_id, num_workers


def _iter_contiguous_blocks(n_positions: int, block_size: int) -> list[tuple[int, int]]:
    if n_positions == 0:
        return []
    blocks: list[tuple[int, int]] = []
    for start in range(0, n_positions, block_size):
        stop = min(start + block_size, n_positions)
        blocks.append((start, stop))
    return blocks


def _resolve_matrix_path(source: ZarrMatrixSource, store_dir: Path | None) -> Path:
    p = Path(source.x_relpath)
    if p.is_absolute():
        return p
    if store_dir is None:
        raise ValueError(
            f"store_dir is required for relative x_relpath {source.x_relpath!r} "
            f"(registry key {source.registry_key!r})."
        )
    return Path(store_dir) / p


def _resolve_csr_group_path(source: ZarrMatrixSource, store_dir: Path | None) -> Path:
    return _resolve_matrix_path(source, store_dir)


def _open_csr_group(group_path: Path) -> CSRDataset:
    """Open a zarr CSR group, with parent-group indexing fallback for nested paths."""
    path_str = str(group_path)
    try:
        zgroup = zarr.open_group(path_str, mode="r")
        csr = sparse_dataset(zgroup)
        if csr.backend == "zarr":
            return csr
    except Exception:
        pass

    parent_path = group_path.parent
    leaf = group_path.name
    if leaf and parent_path != group_path:
        parent = zarr.open_group(str(parent_path), mode="r")
        if leaf in parent:
            csr = sparse_dataset(parent[leaf])
            if csr.backend == "zarr":
                return csr

    raise FileNotFoundError(f"Could not open zarr-backed CSR group at {group_path}")


def _open_dense_array(array_path: Path) -> zarr.Array:
    """Open a zarr-backed dense matrix (2-D array)."""
    node = zarr.open_array(str(array_path), mode="r")
    if node.ndim != 2:
        raise TypeError(f"Expected a 2-D dense zarr array at {array_path}, got ndim={node.ndim}.")
    return node


def _open_matrix_source(source: ZarrMatrixSource, store_dir: Path | None):
    path = _resolve_matrix_path(source, store_dir)
    if source.layout == "dense":
        return _open_dense_array(path)
    csr = _open_csr_group(path)
    if csr.backend != "zarr":
        raise TypeError(
            f"Registry key {source.registry_key!r} must be zarr-backed; got {csr.backend!r}."
        )
    return csr


def _open_matrix_sources(
    store_dir: Path | None,
    sources: dict[str, ZarrMatrixSource],
) -> dict[str, CSRDataset | zarr.Array]:
    return {key: _open_matrix_source(source, store_dir) for key, source in sources.items()}


def _open_csr_sources(
    store_dir: Path | None,
    sources: dict[str, ZarrMatrixSource],
) -> dict[str, CSRDataset]:
    opened: dict[str, CSRDataset] = {}
    for registry_key, source in sources.items():
        if source.layout != "csr":
            raise TypeError(
                f"Registry key {registry_key!r} has layout {source.layout!r}; expected 'csr'."
            )
        group_path = _resolve_csr_group_path(source, store_dir)
        csr = _open_csr_group(group_path)
        if csr.backend != "zarr":
            raise TypeError(
                f"Registry key {registry_key!r} must be zarr-backed; got {csr.backend!r}."
            )
        opened[registry_key] = csr
    return opened


def _zarr_csr_group_path(csr: CSRDataset) -> Path:
    """Extract the on-disk zarr CSR group path from a backed CSRDataset."""
    backend = getattr(csr, "backend", None)
    if backend != "zarr":
        raise TypeError(
            f"Expected a zarr-backed CSRDataset, got backend={backend!r}. "
            "Use to_object(type='backed') on a zarr-backed MuData."
        )
    group = csr.group
    store = group.store
    root = getattr(store, "root", None)
    if root is not None:
        base = Path(root)
        group_path = getattr(group, "path", "") or ""
        return base / group_path if group_path else base

    store_path = str(getattr(group, "store_path", ""))
    if store_path.startswith("file://"):
        store_path = store_path[7:]
    if not store_path:
        raise ValueError("Could not determine zarr CSR group path from CSRDataset.")
    return Path(store_path)


def _zarr_dense_array_path(arr: zarr.Array) -> Path:
    """Extract the on-disk zarr array path from a backed ``zarr.Array``."""
    if not isinstance(arr, zarr.Array):
        raise TypeError(f"Expected a zarr.Array, got {type(arr)!r}.")
    store = arr.store
    root = getattr(store, "root", None)
    if root is not None:
        base = Path(root)
        array_path = getattr(arr, "path", "") or ""
        return base / array_path if array_path else base

    store_path = str(getattr(arr, "store_path", ""))
    if store_path.startswith("file://"):
        store_path = store_path[7:]
    if not store_path:
        raise ValueError("Could not determine zarr array path from zarr.Array.")
    return Path(store_path)


def _infer_matrix_layout(matrix) -> MatrixLayout:
    """Infer CSR vs dense layout from a backed modality matrix."""
    if isinstance(matrix, zarr.Array):
        return "dense"
    if getattr(matrix, "backend", None) == "zarr" and hasattr(matrix, "group"):
        try:
            _zarr_csr_group_path(matrix)
            return "csr"
        except (TypeError, ValueError):
            return "dense"
    if sp.issparse(matrix):
        return "csr"
    return "dense"


def matrix_sources_from_backed_mudata(
    mdata: MuData,
    registry_map: dict[str, str],
    *,
    validate: bool = True,
    n_obs: int | None = None,
    x_suffix: str = "",
    layout: MatrixLayout | Literal["auto"] = "auto",
    store_dir: Path | str | None = None,
) -> dict[str, ZarrMatrixSource]:
    """Build picklable :class:`ZarrMatrixSource` descriptors from a backed MuData.

    When ``layout='auto'``, each modality's layout (CSR vs dense) is inferred
    independently, enabling mixed layouts such as RNA CSR + ADT dense in one
    :class:`ZarrDataset`.
    """
    expected_n_obs = mdata.n_obs if n_obs is None else n_obs
    resolved_store = Path(store_dir) if store_dir is not None else None
    sources: dict[str, ZarrMatrixSource] = {}

    for registry_key, mod_key in registry_map.items():
        if mod_key not in mdata.mod:
            raise KeyError(f"Modality {mod_key!r} not found in mdata.mod.")

        matrix = mdata.mod[mod_key].X
        matrix_layout: MatrixLayout = _infer_matrix_layout(matrix) if layout == "auto" else layout

        if x_suffix:
            if resolved_store is None:
                raise ValueError("store_dir is required when using x_suffix for matrix paths.")
            x_name = f"X{x_suffix}"
            x_relpath = str(resolved_store / "mod" / mod_key / x_name)
        elif matrix_layout == "csr":
            x_relpath = str(_zarr_csr_group_path(matrix))
        else:
            try:
                x_relpath = str(_zarr_dense_array_path(matrix))
            except (TypeError, ValueError):
                if resolved_store is None:
                    raise ValueError("store_dir is required for dense matrix path inference.")
                dense_suffix = x_suffix if x_suffix else "_dense"
                x_relpath = str(resolved_store / "mod" / mod_key / f"X{dense_suffix}")

        sources[registry_key] = ZarrMatrixSource(
            registry_key, x_relpath, layout=matrix_layout
        )

    if validate:
        opened = _open_matrix_sources(None, sources)
        for registry_key, opened_matrix in opened.items():
            n_rows = opened_matrix.shape[0]
            if n_rows != expected_n_obs:
                mod_key = registry_map[registry_key]
                raise ValueError(
                    f"Modality {mod_key!r} has {n_rows} rows, expected {expected_n_obs}."
                )
    return sources


def csr_sources_from_backed_mudata(
    mdata: MuData,
    registry_map: dict[str, str],
    *,
    validate: bool = True,
    n_obs: int | None = None,
) -> dict[str, ZarrMatrixSource]:
    """EXPERIMENTAL: Build picklable :class:`ZarrCSRSource` descriptors from a backed MuData.

    Each worker reopens matrices from these paths independently, so this path is
    safe for ``DataLoader(num_workers > 0)`` unlike passing open ``csr_datasets``.

    Parameters
    ----------
    mdata
        MuData with zarr-backed ``CSRDataset`` matrices (e.g. from
        ``DummyAtlas.to_object(type='backed')``).
    registry_map
        Mapping of scvi registry keys to MuData modality keys, e.g.
        ``{REGISTRY_KEYS.X_KEY: "RNA", REGISTRY_KEYS.ATAC_X_KEY: "ATAC"}``.
    validate
        If True, reopen each source once in the main process and verify row count.
    n_obs
        Expected number of observations; defaults to ``mdata.n_obs``.
    """
    return matrix_sources_from_backed_mudata(
        mdata,
        registry_map,
        validate=validate,
        n_obs=n_obs,
        layout="csr",
    )


def _validate_matrix_datasets(
    matrix_datasets: dict[str, CSRDataset | zarr.Array],
    layouts: dict[str, MatrixLayout] | None = None,
) -> None:
    if not matrix_datasets:
        raise ValueError("`matrix_datasets` must contain at least one matrix.")
    n_obs = None
    for key, matrix in matrix_datasets.items():
        layout = layouts.get(key, "csr") if layouts else "csr"
        if layout == "csr":
            if matrix.backend != "zarr":
                raise TypeError(f"{key!r} must be a zarr-backed CSRDataset, got {matrix.backend!r}.")
        elif not isinstance(matrix, zarr.Array):
            raise TypeError(f"{key!r} must be a zarr.Array for dense layout, got {type(matrix)!r}.")
        if n_obs is None:
            n_obs = matrix.shape[0]
        elif matrix.shape[0] != n_obs:
            raise ValueError("All matrices must have the same number of rows.")


def _validate_matrix_sources(sources: dict[str, ZarrMatrixSource]) -> None:
    if not sources:
        raise ValueError("`matrix_sources` must contain at least one matrix.")
    for key, source in sources.items():
        if source.layout not in ("csr", "dense"):
            raise ValueError(
                f"Registry key {key!r} has unsupported layout {source.layout!r}; "
                "expected 'csr' or 'dense'."
            )


def _validate_csr_datasets(csr_datasets: dict[str, CSRDataset]) -> None:
    _validate_matrix_datasets(csr_datasets, {k: "csr" for k in csr_datasets})


class ZarrDataset(IterableDataset):
    """EXPERIMENTAL: Stream paired modality batches from zarr-backed CSR or dense matrices.

    Each registry key may use a different layout (for example RNA as CSR and ADT as
    dense ``zarr.Array``). Uses sorted within-block reads and a compact shuffle
    buffer, following the TileDB-SOMA / Census streaming pattern.

    Parameters
    ----------
    obs_tensors
        Per-observation arrays keyed by registry field names (e.g. ``batch``, ``labels``,
        ``ind_x``). Each array must have length equal to the number of observations.
    indices
        Global row indices to stream (train or validation subset).
    matrix_datasets
        Optional in-memory mapping of registry keys to open zarr-backed matrices
        (``CSRDataset`` or dense ``zarr.Array``). Used for single-process tests.
        For multi-worker training, pass ``store_dir`` and ``matrix_sources`` instead.
    csr_datasets
        Deprecated alias for ``matrix_datasets`` (CSR-only).
    store_dir
        Root directory of the zarr-backed store.
    matrix_sources
        Mapping of registry keys to :class:`ZarrMatrixSource` descriptors for lazy reopening.
    csr_sources
        Deprecated alias for ``matrix_sources``.
    batch_size
        Minibatch size.
    block_size
        Number of rows per contiguous I/O block.
    shuffle
        Whether to shuffle rows via the shuffle buffer (training).
    shuffle_buffer_blocks
        Number of I/O blocks to accumulate before shuffling and emitting minibatches.
    seed
        Base random seed.
    epoch
        Current epoch; combined with ``seed`` for per-epoch block-order shuffles.
    drop_last
        Drop the final incomplete minibatch.
    """

    def __init__(
        self,
        obs_tensors: dict[str, np.ndarray],
        indices: np.ndarray,
        *,
        matrix_datasets: dict[str, CSRDataset | zarr.Array] | None = None,
        csr_datasets: dict[str, CSRDataset] | None = None,
        store_dir: Path | str | None = None,
        matrix_sources: dict[str, ZarrMatrixSource] | None = None,
        csr_sources: dict[str, ZarrMatrixSource] | None = None,
        batch_size: int = 128,
        block_size: int = 4096,
        shuffle: bool = True,
        shuffle_buffer_blocks: int = 16,
        seed: int = 0,
        epoch: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        if matrix_datasets is None:
            matrix_datasets = csr_datasets
        if matrix_sources is None:
            matrix_sources = csr_sources

        self.obs_tensors = {k: np.asarray(v) for k, v in obs_tensors.items()}
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle
        self.shuffle_buffer_blocks = shuffle_buffer_blocks
        self.seed = seed
        self.epoch = epoch
        self.drop_last = drop_last

        if matrix_datasets is not None and (store_dir is not None or matrix_sources is not None):
            raise ValueError(
                "Pass either `matrix_datasets` or (`store_dir`, `matrix_sources`), not both."
            )
        if matrix_datasets is None:
            if matrix_sources is None:
                raise ValueError(
                    "Provide `matrix_datasets` for single-process use, or `matrix_sources` "
                    "(with optional `store_dir`) for multi-worker streaming."
                )
            _validate_matrix_sources(matrix_sources)
            needs_store_dir = any(
                not Path(source.x_relpath).is_absolute() for source in matrix_sources.values()
            )
            if needs_store_dir and store_dir is None:
                raise ValueError(
                    "store_dir is required when matrix_sources use relative x_relpath values."
                )
            self.store_dir = Path(store_dir) if store_dir is not None else None
            self.matrix_sources = matrix_sources
            self._matrix_datasets = None
            self._matrix_layouts = {k: s.layout for k, s in matrix_sources.items()}
        else:
            layouts = {key: _infer_matrix_layout(matrix) for key, matrix in matrix_datasets.items()}
            _validate_matrix_datasets(matrix_datasets, layouts)
            self.store_dir = None
            self.matrix_sources = None
            self._matrix_datasets = matrix_datasets
            self._matrix_layouts = layouts

        registry_keys = set(
            matrix_datasets.keys() if matrix_datasets else matrix_sources.keys()
        )
        overlap = registry_keys & set(obs_tensors.keys())
        if overlap:
            raise ValueError(
                f"Registry keys overlap with obs tensor keys: {sorted(overlap)}. "
                "Use distinct keys for matrices vs metadata."
            )

        n_obs = next(iter(matrix_datasets.values())).shape[0] if matrix_datasets else None
        if n_obs is None and matrix_sources is not None:
            n_obs = len(next(iter(obs_tensors.values())))
        for key, arr in self.obs_tensors.items():
            if arr.shape[0] != n_obs:
                raise ValueError(
                    f"obs_tensors[{key!r}] length {arr.shape[0]} != n_obs {n_obs}."
                )

        self._matrix_keys = sorted(registry_keys)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic per-epoch shuffles."""
        self.epoch = epoch

    def _get_matrix_datasets(self) -> dict[str, CSRDataset | zarr.Array]:
        if self._matrix_datasets is not None:
            return self._matrix_datasets
        assert self.matrix_sources is not None
        return _open_matrix_sources(self.store_dir, self.matrix_sources)

    def _get_csr_datasets(self) -> dict[str, CSRDataset]:
        """Backward-compatible CSR-only accessor."""
        opened = self._get_matrix_datasets()
        return {
            key: matrix
            for key, matrix in opened.items()
            if self._matrix_layouts.get(key, "csr") == "csr"
        }

    def _read_block(
        self,
        matrix_datasets: dict[str, CSRDataset | zarr.Array],
        row_indices: np.ndarray,
    ) -> dict:
        """Read a block of rows as sparse CSR submatrices or dense slices.

        Row indices are sorted ascending before reading so zarr reads are monotonic
        rather than random-seek. Obs arrays are sliced with the same sorted indices
        so pairing is preserved.
        """
        if len(row_indices) == 0:
            return {}
        row_indices = np.asarray(row_indices, dtype=np.int64)
        sorted_idx = np.sort(row_indices)

        block: dict = {}
        for registry_key in self._matrix_keys:
            layout = self._matrix_layouts.get(registry_key, "csr")
            matrix = matrix_datasets[registry_key]
            if layout == "dense":
                if len(sorted_idx) == 1:
                    sub = np.asarray(matrix[int(sorted_idx[0]) : int(sorted_idx[0]) + 1])
                elif np.all(np.diff(sorted_idx) == 1):
                    sub = np.asarray(matrix[int(sorted_idx[0]) : int(sorted_idx[-1]) + 1])
                else:
                    sub = np.asarray(matrix.oindex[sorted_idx])
            elif len(sorted_idx) == 1:
                sub = matrix[int(sorted_idx[0]) : int(sorted_idx[0]) + 1]
            elif np.all(np.diff(sorted_idx) == 1):
                sub = matrix[int(sorted_idx[0]) : int(sorted_idx[-1]) + 1]
            else:
                sub = matrix[sorted_idx]
            block[registry_key] = sub.tocsr() if layout == "csr" and not sp.isspmatrix_csr(sub) else sub

        for obs_key, obs_arr in self.obs_tensors.items():
            block[obs_key] = obs_arr[sorted_idx]

        return block

    def _make_batch(
        self,
        buffer: _MatrixBuffer,
        row_positions: np.ndarray | slice,
    ) -> dict[str, torch.Tensor]:
        """Build a minibatch from buffered rows (sparse or dense).

        ``row_positions`` may be an index array (shuffled training minibatch) or a
        contiguous ``slice`` (validation window). Call ``buffer.materialize()`` first so
        the per-key reads here are O(batch), not O(buffer).
        """
        out: dict[str, torch.Tensor] = {}
        for key in self._matrix_keys:
            stacked = buffer.stacked_matrix(key)
            layout = buffer.matrix_layouts.get(key, "csr")
            if layout == "dense":
                dense = np.asarray(stacked[row_positions], dtype=np.float32)
            else:
                dense = stacked[row_positions].toarray().astype(np.float32, copy=False)
            out[key] = torch.as_tensor(dense)
        for key in buffer.obs_blocks:
            stacked = buffer.stacked_obs(key)
            values = np.asarray(stacked[row_positions])
            if key in _INT64_OBS_KEYS:
                out[key] = torch.as_tensor(values, dtype=torch.int64).reshape(-1, 1)
            else:
                out[key] = torch.as_tensor(values).reshape(-1, 1)
        return out

    def _emit_buffered_batches(
        self,
        buffer: _MatrixBuffer,
        rng: np.random.Generator,
        *,
        drop_last: bool,
    ):
        """Shuffle buffered rows and yield minibatches (training path)."""
        n_rows = buffer.n_rows
        if n_rows == 0:
            return
        buffer.materialize()
        perm = rng.permutation(n_rows)
        start = 0
        while start + self.batch_size <= n_rows:
            batch_rows = perm[start : start + self.batch_size]
            start += self.batch_size
            yield self._make_batch(buffer, batch_rows)
        remainder = perm[start:]
        if len(remainder) > 0 and not drop_last:
            yield self._make_batch(buffer, remainder)

    def _trim_buffer(self, buffer: _MatrixBuffer, n_trim: int) -> None:
        """Drop the first ``n_trim`` rows from a matrix buffer."""
        if n_trim <= 0:
            return
        if n_trim >= buffer.n_rows:
            buffer.clear()
            return
        for key in self._matrix_keys:
            stacked = buffer.stacked_matrix(key)
            buffer.matrix_blocks[key] = [stacked[n_trim:]]
        for key in list(buffer.obs_blocks.keys()):
            stacked = buffer.stacked_obs(key)
            buffer.obs_blocks[key] = [stacked[n_trim:]]
        buffer.n_rows -= n_trim

    def _emit_streaming_batches(self, buffer: _MatrixBuffer):
        """Emit full minibatches from the front of a buffer (validation path).

        Materializes the carried-over rows plus the new block once, emits contiguous
        slice windows, and trims a single time, instead of re-stacking and recopying the
        buffer for every minibatch.
        """
        if buffer.n_rows < self.batch_size:
            return
        buffer.materialize()
        n_full = buffer.n_rows // self.batch_size
        for i in range(n_full):
            sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
            yield self._make_batch(buffer, sl)
        self._trim_buffer(buffer, n_full * self.batch_size)

    def _rank_indices(self) -> np.ndarray:
        """Restrict indices to an equal per-rank slice for DDP."""
        rank, world_size, _, _ = _get_rank_worker_ids()
        if world_size <= 1:
            return self.indices
        n = len(self.indices)
        per_rank = n // world_size
        if per_rank == 0:
            return self.indices[:0]
        start = rank * per_rank
        stop = start + per_rank
        return self.indices[start:stop]

    def _partition_blocks(self, blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Assign disjoint block subsets to each DataLoader worker (within a rank)."""
        _, _, worker_id, num_workers = _get_rank_worker_ids()
        return [b for i, b in enumerate(blocks) if i % num_workers == worker_id]

    def __iter__(self):
        matrix_datasets = self._get_matrix_datasets()
        rank_indices = self._rank_indices()
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)

        _, world_size, _, _ = _get_rank_worker_ids()
        # Under DDP every rank must emit the same number of batches.
        drop_last = self.drop_last or (self.shuffle and world_size > 1)

        blocks = _iter_contiguous_blocks(len(rank_indices), self.block_size)
        if self.shuffle:
            block_order = rng.permutation(len(blocks))
            blocks = [blocks[i] for i in block_order]
        blocks = self._partition_blocks(blocks)

        if self.shuffle:
            buffer = _MatrixBuffer()
            blocks_in_buffer = 0
            for block_start, block_stop in blocks:
                row_indices = rank_indices[block_start:block_stop]
                block = self._read_block(matrix_datasets, row_indices)
                buffer.append(block, layouts=self._matrix_layouts)
                blocks_in_buffer += 1

                if blocks_in_buffer >= self.shuffle_buffer_blocks:
                    yield from self._emit_buffered_batches(buffer, rng, drop_last=drop_last)
                    buffer.clear()
                    blocks_in_buffer = 0

            if buffer.n_rows > 0:
                yield from self._emit_buffered_batches(buffer, rng, drop_last=drop_last)
        else:
            leftover = _MatrixBuffer()
            for block_start, block_stop in blocks:
                row_indices = rank_indices[block_start:block_stop]
                block = self._read_block(matrix_datasets, row_indices)
                leftover.append(block, layouts=self._matrix_layouts)
                yield from self._emit_streaming_batches(leftover)

            if leftover.n_rows > 0 and not drop_last:
                yield self._make_batch(leftover, np.arange(leftover.n_rows))
