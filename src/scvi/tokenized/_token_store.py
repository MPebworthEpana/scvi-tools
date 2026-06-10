"""Tiered ATAC token store: ragged CSR ids (+ optional values) with mmap/ram/gpu tiers."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse

from scvi import REGISTRY_KEYS
from scvi.data import _constants
from scvi.tokenized._csr_batch_tokenize import csr_batch_to_tokens
from scvi.tokenized._nnz import atac_row_nnz
from scvi.tokenized._tokenizers import tokenize_atac

_HASH_SAMPLE_BYTES = 4 * 1024 * 1024


def _content_hash(atac_x: sparse.spmatrix | np.ndarray) -> str:
    """Cheap cache-validation key (not a cryptographic integrity hash)."""
    h = hashlib.sha256()
    h.update(str(atac_x.shape).encode())
    if sparse.issparse(atac_x):
        csr = atac_x.tocsr()
        h.update(csr.indptr.tobytes())
        nnz_total = int(csr.nnz)
        h.update(str(nnz_total).encode())
        if nnz_total > 0:
            stride = max(1, nnz_total // _HASH_SAMPLE_BYTES)
            h.update(csr.indices[::stride].tobytes())
            h.update(csr.data[::stride].tobytes())
    else:
        arr = np.asarray(atac_x)
        flat = arr.ravel()
        stride = max(1, flat.size // _HASH_SAMPLE_BYTES)
        h.update(flat[::stride].tobytes())
    return h.hexdigest()


def registry_for_checkpoint(registry: dict) -> dict:
    """Return a checkpoint-safe registry copy without per-cell token store payloads."""
    from scvi.tokenized import ATAC_TOKEN_CONFIG_KEY
    from scvi.tokenized._field import AtacTokenConfigField

    reg = copy.copy(registry)
    field_regs = reg.get(_constants._FIELD_REGISTRIES_KEY)
    if field_regs is None or ATAC_TOKEN_CONFIG_KEY not in field_regs:
        return reg
    field_regs = copy.copy(field_regs)
    atac_field = copy.copy(field_regs[ATAC_TOKEN_CONFIG_KEY])
    state = copy.copy(atac_field.get(_constants._STATE_REGISTRY_KEY, {}))
    state.pop(AtacTokenConfigField.TOKEN_STORE_KEY, None)
    state.pop(AtacTokenConfigField.NN_LENGTHS_KEY, None)
    atac_field[_constants._STATE_REGISTRY_KEY] = state
    field_regs[ATAC_TOKEN_CONFIG_KEY] = atac_field
    reg[_constants._FIELD_REGISTRIES_KEY] = field_regs
    return reg


def _truncate_row(
    ids: np.ndarray,
    values: np.ndarray,
    max_tokens: int,
    coord_table: np.ndarray,
    genomic: bool,
    genomic_rank: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(ids) <= max_tokens:
        return ids, values
    order = np.argsort(-values, kind="stable")
    keep = order[:max_tokens]
    if genomic and genomic_rank is not None:
        keep = keep[np.argsort(genomic_rank[ids[keep]], kind="stable")]
    else:
        keep = np.sort(keep)
    return ids[keep], values[keep]


def _resolve_tier(tier: str) -> str:
    if tier == "auto":
        return "gpu" if torch.cuda.is_available() else "ram"
    if tier not in {"gpu", "ram", "mmap"}:
        raise ValueError(f"Unknown token store tier: {tier!r}")
    return tier


@dataclass
class AtacTokenStore:
    """Ragged int32 token ids with int64 indptr; optional float32 values when truncation is needed."""

    ids: np.ndarray
    indptr: np.ndarray
    values: np.ndarray | None
    coord_table: np.ndarray
    genomic_rank: np.ndarray | None
    max_encoder_tokens: int
    genomic: bool
    truncation_possible: bool
    tier: str
    n_obs: int
    content_hash: str
    mmap_dir: str | None = None
    _torch_ids: torch.Tensor | None = None
    _torch_indptr: torch.Tensor | None = None
    _torch_values: torch.Tensor | None = None

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.indptr).astype(np.int64)

    def _row_ids_vals(self, row: int) -> tuple[np.ndarray, np.ndarray | None]:
        start = int(self.indptr[row])
        end = int(self.indptr[row + 1])
        ids = self.ids[start:end]
        vals = None if self.values is None else self.values[start:end]
        return ids, vals

    def _gather_vectorized_numpy(
        self,
        cell_idx: np.ndarray,
        *,
        for_encoder: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        cell_idx = np.asarray(cell_idx, dtype=np.int64).ravel()
        starts = self.indptr[cell_idx]
        lengths = self.indptr[cell_idx + 1] - starts
        if for_encoder:
            out_len = max(self.max_encoder_tokens, 1)
        else:
            out_len = max(int(lengths.max(initial=0)), 1)
        gather_len = max(int(lengths.max(initial=0)), 1)
        ar = np.arange(gather_len, dtype=np.int64)
        mask = ar[None, :] < lengths[:, None]
        gidx = starts[:, None] + ar[None, :]
        gidx_safe = np.clip(gidx, 0, max(len(self.ids) - 1, 0))
        ids_full = np.where(mask, self.ids[gidx_safe], 0).astype(np.int64)
        mask_full = mask

        if (
            for_encoder
            and self.truncation_possible
            and self.values is not None
            and np.any(lengths > self.max_encoder_tokens)
        ):
            trunc_rows = np.where(lengths > self.max_encoder_tokens)[0]
            for i in trunc_rows:
                row_ids, row_vals = self._row_ids_vals(int(cell_idx[i]))
                truncated, _ = _truncate_row(
                    row_ids,
                    row_vals,
                    self.max_encoder_tokens,
                    self.coord_table,
                    self.genomic,
                    self.genomic_rank,
                )
                n = len(truncated)
                ids_full[i, :] = 0
                mask_full[i, :] = False
                if n:
                    ids_full[i, :n] = truncated
                    mask_full[i, :n] = True

        ids = ids_full[:, :out_len]
        mask = mask_full[:, :out_len]
        return ids, mask

    def gather(
        self,
        cell_idx: np.ndarray,
        *,
        for_encoder: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._gather_vectorized_numpy(cell_idx, for_encoder=for_encoder)

    def gather_reference(
        self,
        cell_idx: np.ndarray,
        *,
        for_encoder: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-row reference gather for regression tests."""
        cell_idx = np.asarray(cell_idx, dtype=np.int64).ravel()
        max_len = self.max_encoder_tokens if for_encoder else int(self.lengths.max(initial=0))
        max_len = max(max_len, 1)
        ids = np.zeros((len(cell_idx), max_len), dtype=np.int64)
        mask = np.zeros((len(cell_idx), max_len), dtype=bool)
        for i, row in enumerate(cell_idx):
            row_ids, row_vals = self._row_ids_vals(int(row))
            if for_encoder and self.truncation_possible and len(row_ids) > self.max_encoder_tokens:
                row_ids, _ = _truncate_row(
                    row_ids,
                    row_vals,
                    self.max_encoder_tokens,
                    self.coord_table,
                    self.genomic,
                    self.genomic_rank,
                )
            n = len(row_ids)
            if n:
                ids[i, :n] = row_ids
                mask[i, :n] = True
        return ids, mask

    def gather_torch(
        self,
        cell_idx: np.ndarray,
        device: torch.device,
        *,
        for_encoder: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tier != "gpu":
            ids, mask = self.gather(cell_idx, for_encoder=for_encoder)
            return (
                torch.as_tensor(ids, device=device, dtype=torch.long),
                torch.as_tensor(mask, device=device, dtype=torch.bool),
            )
        if (
            self._torch_ids is None
            or self._torch_indptr is None
            or self._torch_ids.device != device
            or self._torch_indptr.device != device
        ):
            self._torch_ids = torch.as_tensor(self.ids, dtype=torch.int32, device=device)
            self._torch_indptr = torch.as_tensor(self.indptr, dtype=torch.int64, device=device)
            if self.values is not None:
                self._torch_values = torch.as_tensor(self.values, dtype=torch.float32, device=device)
            else:
                self._torch_values = None

        idx = torch.as_tensor(cell_idx, dtype=torch.int64, device=device).long()
        starts = self._torch_indptr[idx]
        lengths = self._torch_indptr[idx + 1] - starts
        if for_encoder:
            out_len = max(self.max_encoder_tokens, 1)
        else:
            out_len = int(lengths.max().item()) if idx.numel() else 1
            out_len = max(out_len, 1)
        gather_len = int(lengths.max().item()) if idx.numel() else 1
        gather_len = max(gather_len, 1)
        ar = torch.arange(gather_len, device=device)
        mask_full = ar.unsqueeze(0) < lengths.unsqueeze(1)
        gidx = starts.unsqueeze(1) + ar.unsqueeze(0)
        gidx_safe = gidx.clamp(0, max(self._torch_ids.numel() - 1, 0))
        ids_full = torch.zeros(idx.shape[0], gather_len, dtype=torch.long, device=device)
        ids_full[mask_full] = self._torch_ids[gidx_safe[mask_full]].long()

        if (
            for_encoder
            and self.truncation_possible
            and self.values is not None
            and (lengths > self.max_encoder_tokens).any()
        ):
            trunc_rows = torch.where(lengths > self.max_encoder_tokens)[0]
            for i in trunc_rows.cpu().numpy():
                row_ids, row_vals = self._row_ids_vals(int(cell_idx[i]))
                truncated, _ = _truncate_row(
                    row_ids,
                    row_vals,
                    self.max_encoder_tokens,
                    self.coord_table,
                    self.genomic,
                    self.genomic_rank,
                )
                n = len(truncated)
                ids_full[i, :] = 0
                mask_full[i, :] = False
                if n:
                    ids_full[i, :n] = torch.as_tensor(truncated, device=device, dtype=torch.long)
                    mask_full[i, :n] = True
        return ids_full[:, :out_len], mask_full[:, :out_len]

    def to_handle(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "mmap_dir": self.mmap_dir,
            "content_hash": self.content_hash,
            "max_encoder_tokens": self.max_encoder_tokens,
            "genomic": self.genomic,
            "truncation_possible": self.truncation_possible,
            "n_obs": self.n_obs,
        }

    @classmethod
    def from_handle(cls, handle: dict[str, Any]) -> AtacTokenStore:
        mmap_dir = handle.get("mmap_dir")
        if not mmap_dir:
            raise ValueError("mmap handle requires mmap_dir")
        meta_path = Path(mmap_dir) / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        ids = np.memmap(Path(mmap_dir) / "ids.bin", dtype=np.int32, mode="r", shape=(meta["n_ids"],))
        indptr = np.memmap(
            Path(mmap_dir) / "indptr.bin", dtype=np.int64, mode="r", shape=(meta["n_obs"] + 1,)
        )
        values = None
        if meta.get("has_values"):
            values = np.memmap(
                Path(mmap_dir) / "values.bin", dtype=np.float32, mode="r", shape=(meta["n_ids"],)
            )
        coord_table = np.load(Path(mmap_dir) / "coord_table.npy")
        genomic_rank = None
        rank_path = Path(mmap_dir) / "genomic_rank.npy"
        if rank_path.exists():
            genomic_rank = np.load(rank_path)
        return cls(
            ids=np.asarray(ids),
            indptr=np.asarray(indptr),
            values=None if values is None else np.asarray(values),
            coord_table=coord_table,
            genomic_rank=genomic_rank,
            max_encoder_tokens=meta["max_encoder_tokens"],
            genomic=meta["genomic"],
            truncation_possible=meta["truncation_possible"],
            tier="mmap",
            n_obs=meta["n_obs"],
            content_hash=meta["content_hash"],
            mmap_dir=str(mmap_dir),
        )


def _fill_chunk_into_store(
    chunk: sparse.csr_matrix,
    *,
    start: int,
    indptr_arr: np.ndarray,
    ids_out: np.ndarray,
    values_out: np.ndarray | None,
    store_values: bool,
    coord_table: np.ndarray,
    genomic_rank: np.ndarray | None,
    max_encoder_tokens: int,
    n_regions: int,
    genomic: bool,
) -> None:
    if store_values:
        for local_row in range(chunk.shape[0]):
            global_row = start + local_row
            row = chunk.getrow(local_row)
            tok = tokenize_atac(
                row.indices.astype(np.int64),
                row.data.astype(np.float32),
                n_regions,
                coord_table,
                genomic=genomic,
                genomic_rank=genomic_rank,
            )
            row_ids = tok["ids"].astype(np.int32)
            row_start = int(indptr_arr[global_row])
            row_end = int(indptr_arr[global_row + 1])
            ids_out[row_start:row_end] = row_ids
            if values_out is not None:
                values_out[row_start:row_end] = tok["values"].astype(np.float32)
    else:
        ids_batch, mask_batch = csr_batch_to_tokens(
            chunk,
            coord_table,
            max_encoder_tokens,
            genomic=genomic,
            genomic_rank=genomic_rank,
        )
        for local_row in range(chunk.shape[0]):
            global_row = start + local_row
            row_mask = mask_batch[local_row]
            row_ids = ids_batch[local_row][row_mask].astype(np.int32)
            row_start = int(indptr_arr[global_row])
            row_end = int(indptr_arr[global_row + 1])
            ids_out[row_start:row_end] = row_ids


def build_token_store(
    atac_x: sparse.spmatrix,
    coord_table: np.ndarray,
    genomic_rank: np.ndarray | None,
    *,
    max_encoder_tokens: int,
    genomic: bool = True,
    tier: str = "auto",
    out_dir: str | Path | None = None,
    chunk_size: int = 8192,
) -> AtacTokenStore:
    """Build a tiered token store from CSR ATAC rows (backed-safe chunked pass)."""
    resolved_tier = _resolve_tier(tier)
    n_obs = int(atac_x.shape[0])
    n_regions = int(atac_x.shape[1])
    content_hash = _content_hash(atac_x)

    nnz = atac_row_nnz(atac_x, chunk_size=chunk_size)
    store_values = int(nnz.max(initial=0)) > max_encoder_tokens
    indptr_arr = np.zeros(n_obs + 1, dtype=np.int64)
    indptr_arr[1:] = np.cumsum(nnz, dtype=np.int64)
    total_ids = int(indptr_arr[-1])

    mmap_dir: Path | None = None
    ids_path: Path | None = None
    indptr_path: Path | None = None
    vals_path: Path | None = None

    if resolved_tier == "mmap":
        if out_dir is None:
            raise ValueError("mmap tier requires out_dir")
        mmap_dir = Path(out_dir)
        mmap_dir.mkdir(parents=True, exist_ok=True)
        ids_path = mmap_dir / "ids.bin"
        indptr_path = mmap_dir / "indptr.bin"
        ids_out: np.ndarray = np.memmap(ids_path, dtype=np.int32, mode="w+", shape=(total_ids,))
        if store_values:
            vals_path = mmap_dir / "values.bin"
            values_out: np.ndarray | None = np.memmap(
                vals_path, dtype=np.float32, mode="w+", shape=(total_ids,)
            )
        else:
            values_out = None
    else:
        ids_out = np.empty(total_ids, dtype=np.int32)
        values_out = np.empty(total_ids, dtype=np.float32) if store_values else None

    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        chunk = atac_x[start:end]
        if sparse.issparse(chunk):
            chunk = chunk.tocsr()
        else:
            chunk = sparse.csr_matrix(chunk)
        _fill_chunk_into_store(
            chunk,
            start=start,
            indptr_arr=indptr_arr,
            ids_out=ids_out,
            values_out=values_out,
            store_values=store_values,
            coord_table=coord_table,
            genomic_rank=genomic_rank,
            max_encoder_tokens=max_encoder_tokens,
            n_regions=n_regions,
            genomic=genomic,
        )

    if store_values and values_out is not None and len(values_out) != len(ids_out):
        raise RuntimeError(
            f"Token store values/ids length mismatch: {len(values_out)} vs {len(ids_out)}"
        )

    if resolved_tier == "mmap":
        assert mmap_dir is not None and ids_path is not None and indptr_path is not None
        ids_out.flush()
        indptr_mm = np.memmap(indptr_path, dtype=np.int64, mode="w+", shape=indptr_arr.shape)
        indptr_mm[:] = indptr_arr
        indptr_mm.flush()
        if values_out is not None:
            values_out.flush()
        np.save(mmap_dir / "coord_table.npy", coord_table)
        if genomic_rank is not None:
            np.save(mmap_dir / "genomic_rank.npy", genomic_rank)
        meta = {
            "n_ids": total_ids,
            "n_obs": n_obs,
            "max_encoder_tokens": max_encoder_tokens,
            "genomic": genomic,
            "truncation_possible": store_values,
            "has_values": store_values,
            "content_hash": content_hash,
        }
        with open(mmap_dir / "meta.json", "w") as f:
            json.dump(meta, f)
        ids_arr = np.memmap(ids_path, dtype=np.int32, mode="r", shape=(total_ids,))
        indptr_read = np.memmap(indptr_path, dtype=np.int64, mode="r", shape=indptr_arr.shape)
        values_arr = None
        if store_values and vals_path is not None:
            values_arr = np.memmap(vals_path, dtype=np.float32, mode="r", shape=(total_ids,))
    else:
        ids_arr = ids_out
        indptr_read = indptr_arr
        values_arr = values_out
        mmap_dir = None

    store = AtacTokenStore(
        ids=ids_arr,
        indptr=indptr_read,
        values=values_arr,
        coord_table=coord_table,
        genomic_rank=genomic_rank,
        max_encoder_tokens=max_encoder_tokens,
        genomic=genomic,
        truncation_possible=store_values,
        tier=resolved_tier,
        n_obs=n_obs,
        content_hash=content_hash,
        mmap_dir=str(mmap_dir) if mmap_dir is not None else None,
    )

    if resolved_tier == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        store._torch_ids = torch.as_tensor(store.ids, dtype=torch.int32, device=device)
        store._torch_indptr = torch.as_tensor(store.indptr, dtype=torch.int64, device=device)
        if store.values is not None:
            store._torch_values = torch.as_tensor(store.values, dtype=torch.float32, device=device)

    return store


def _mutable_token_state(adata_manager) -> dict:
    from scvi.tokenized import ATAC_TOKEN_CONFIG_KEY

    return adata_manager._registry[_constants._FIELD_REGISTRIES_KEY][
        ATAC_TOKEN_CONFIG_KEY
    ][_constants._STATE_REGISTRY_KEY]


def attach_token_store_to_registry(
    state: dict,
    store: AtacTokenStore,
    *,
    nnz_lengths: np.ndarray | None = None,
) -> None:
    from scvi.tokenized._field import AtacTokenConfigField

    state[AtacTokenConfigField.TOKEN_STORE_KEY] = store
    state[AtacTokenConfigField.TOKEN_STORE_HANDLE_KEY] = store.to_handle()
    state[AtacTokenConfigField.NN_LENGTHS_KEY] = (
        nnz_lengths if nnz_lengths is not None else store.lengths
    )


def rebuild_token_store(
    adata_manager,
    *,
    tier: str | None = None,
    out_dir: str | Path | None = None,
    chunk_size: int = 8192,
) -> AtacTokenStore:
    """Rebuild the token store from registry ATAC_X and attach it to the manager."""
    from scvi.tokenized import ATAC_TOKEN_CONFIG_KEY, AtacTokenConfigField

    token_cfg = adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)
    if token_cfg is None:
        raise ValueError("ATAC token config not registered on adata_manager")
    state = _mutable_token_state(adata_manager)

    atac_x = adata_manager.get_from_registry(REGISTRY_KEYS.ATAC_X_KEY)
    coord_table = token_cfg[AtacTokenConfigField.COORD_TABLE_KEY]
    genomic_rank = token_cfg.get(AtacTokenConfigField.GENOMIC_RANK_KEY)
    max_encoder_tokens = token_cfg[AtacTokenConfigField.MAX_TOKENS_KEY]
    genomic = token_cfg.get(AtacTokenConfigField.GENOMIC_KEY, True)

    handle = token_cfg.get(AtacTokenConfigField.TOKEN_STORE_HANDLE_KEY)
    resolved_tier = tier or (handle["tier"] if handle else "auto")
    if out_dir is None and handle is not None:
        out_dir = handle.get("mmap_dir")

    expected_hash = _content_hash(atac_x)
    if handle is not None and handle.get("content_hash") == expected_hash:
        if resolved_tier == "mmap" and out_dir and Path(out_dir).exists():
            meta_path = Path(out_dir) / "meta.json"
            if meta_path.exists():
                store = AtacTokenStore.from_handle({**handle, "mmap_dir": str(out_dir)})
                attach_token_store_to_registry(state, store, nnz_lengths=atac_row_nnz(atac_x))
                return store

    store = build_token_store(
        atac_x,
        coord_table,
        genomic_rank,
        max_encoder_tokens=max_encoder_tokens,
        genomic=genomic,
        tier=resolved_tier,
        out_dir=out_dir,
        chunk_size=chunk_size,
    )
    attach_token_store_to_registry(state, store, nnz_lengths=atac_row_nnz(atac_x))
    return store


def ensure_token_store_for_manager(
    adata_manager,
    *,
    training_manager=None,
    force_ephemeral: bool = False,
) -> AtacTokenStore:
    """Ensure a token store exists for ``adata_manager``, building an ephemeral RAM store when needed."""
    from scvi.tokenized import ATAC_TOKEN_CONFIG_KEY, AtacTokenConfigField

    import numpy as np

    adata = adata_manager.adata
    if "_indices" not in adata.obs.columns:
        adata.obs["_indices"] = np.arange(adata.n_obs)

    token_cfg = adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)
    if token_cfg is None:
        raise ValueError("ATAC token config not registered on adata_manager")
    state = _mutable_token_state(adata_manager)

    atac_x = adata_manager.get_from_registry(REGISTRY_KEYS.ATAC_X_KEY)
    expected_hash = _content_hash(atac_x)
    store = token_cfg.get(AtacTokenConfigField.TOKEN_STORE_KEY)
    if (
        not force_ephemeral
        and store is not None
        and store.content_hash == expected_hash
        and (training_manager is None or adata_manager is training_manager)
    ):
        return store

    tier = "ram" if force_ephemeral or (
        training_manager is not None and adata_manager is not training_manager
    ) else None
    return rebuild_token_store(adata_manager, tier=tier)
