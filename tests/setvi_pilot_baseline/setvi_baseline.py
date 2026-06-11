"""Train SETVI on the pilot mosaic dataset and write UMAP + mixing metrics.

Run from anywhere::

    python MultiVI/tests/setvi_pilot_baseline/setvi_baseline.py

Outputs are written next to this script (``setvi_pilot_latent.npz``,
``setvi_pilot_umap.png``, ``setvi_pilot_umap_leiden.png``,
``setvi_pilot_metrics.json``, ``model/``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_MULTIVI_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_MULTIVI_SRC) not in sys.path:
    sys.path.insert(0, str(_MULTIVI_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import scvi
import umap
from scib_metrics import ilisi_knn, kbet
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- settings ---
HERE = Path(__file__).resolve().parent
ATLAS_ROOT = HERE.parents[2]
MDATA_PATH = ATLAS_ROOT / "complex_object_pilot.h5mu"
OUT_DIR = HERE

BATCH_KEY = "data_type"
MAX_EPOCHS = 150
SEED = 0
BATCH_SIZE = 128
TRAIN_SIZE = 0.85
N_NEIGHBORS = 15
MAX_ATAC_TOKENS = 8192
ATAC_TOKEN_STORE = "auto"
LENGTH_BUCKETING = True
BUCKET_MULT = 50
# region_factors is always on for SETVI (shared decoder baseline + encoder rarity prior).
REGION_FACTORS = True

ST_D_MODEL = 128
ST_N_LAYERS = 2
ST_N_INDUCING = 32
ST_N_HEADS = 4
ST_DROPOUT = 0.0
USE_CARDINALITY_FILM = False
USE_SAMPLING_CORRECTION = False
# Reuse region_factors as a capped per-peak rarity prior on the encoder attention.
# Cap C = 5.0 -> frequency floor exp(-5) ~= 0.67%; peaks rarer than that are flattened.
USE_PEAK_SALIENCE_PRIOR = True
PEAK_SALIENCE_CAP = 5.0
USE_COUNTS_IN_ENCODER = True

MATMUL_PRECISION = "high"
TRAIN_PRECISION = "32-true"
DL_NUM_WORKERS = 0
DL_PIN_MEMORY = True
DL_PERSISTENT_WORKERS = True
COMPILE_MODEL = False
TRAINER_BENCHMARK = True
N_EPOCHS_KL_WARMUP = 50

# Collapse-trace diagnostic: cells per data_type group sampled each epoch.
TRACE_PER_GROUP = 400

REF_LATENT_PATH = HERE.parent / "multivi_pilot_baseline" / "multivi_pilot_latent.npz"


def _apply_perf_settings(matmul_precision: str = MATMUL_PRECISION) -> None:
    import torch

    if matmul_precision != "highest":
        torch.set_float32_matmul_precision(matmul_precision)


def _perf_settings_dict(
    *,
    matmul_precision: str | None = None,
    train_precision: str | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    compile_model: bool | None = None,
    trainer_benchmark: bool | None = None,
) -> dict:
    matmul_precision = MATMUL_PRECISION if matmul_precision is None else matmul_precision
    train_precision = TRAIN_PRECISION if train_precision is None else train_precision
    num_workers = DL_NUM_WORKERS if num_workers is None else num_workers
    pin_memory = DL_PIN_MEMORY if pin_memory is None else pin_memory
    persistent_workers = DL_PERSISTENT_WORKERS if persistent_workers is None else persistent_workers
    compile_model = COMPILE_MODEL if compile_model is None else compile_model
    trainer_benchmark = TRAINER_BENCHMARK if trainer_benchmark is None else trainer_benchmark
    nw = max(0, int(num_workers))
    return {
        "matmul_precision": matmul_precision,
        "train_precision": train_precision,
        "dl_num_workers": nw,
        "dl_pin_memory": bool(pin_memory),
        "dl_persistent_workers": bool(persistent_workers) and nw > 0,
        "compile_model": bool(compile_model),
        "trainer_benchmark": bool(trainer_benchmark),
    }


def _build_train_kwargs(
    *,
    max_epochs: int,
    adversarial: bool = True,
    n_epochs_kl_warmup: int | None = None,
    bucketing: bool | None = None,
    matmul_precision: str | None = None,
    train_precision: str | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    compile_model: bool | None = None,
    trainer_benchmark: bool | None = None,
) -> dict:
    matmul_precision = MATMUL_PRECISION if matmul_precision is None else matmul_precision
    train_precision = TRAIN_PRECISION if train_precision is None else train_precision
    num_workers = DL_NUM_WORKERS if num_workers is None else num_workers
    pin_memory = DL_PIN_MEMORY if pin_memory is None else pin_memory
    persistent_workers = DL_PERSISTENT_WORKERS if persistent_workers is None else persistent_workers
    compile_model = COMPILE_MODEL if compile_model is None else compile_model
    trainer_benchmark = TRAINER_BENCHMARK if trainer_benchmark is None else trainer_benchmark
    bucketing = LENGTH_BUCKETING if bucketing is None else bucketing
    if n_epochs_kl_warmup is None:
        n_epochs_kl_warmup = N_EPOCHS_KL_WARMUP
    _apply_perf_settings(matmul_precision)
    nw = max(0, int(num_workers))
    if persistent_workers and nw == 0:
        persistent_workers = False
    return {
        "max_epochs": max_epochs,
        "batch_size": BATCH_SIZE,
        "train_size": TRAIN_SIZE,
        "accelerator": "auto",
        "precision": train_precision,
        "benchmark": trainer_benchmark,
        "datasplitter_kwargs": {
            "num_workers": nw,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        },
        "plan_kwargs": {"compile": compile_model},
        "adversarial_mixing": adversarial,
        "n_epochs_kl_warmup": n_epochs_kl_warmup,
        "atac_length_bucketing": bucketing,
        "bucket_mult": BUCKET_MULT,
    }


def _load_mdata(mdata_path: Path, subset_data_type: str | None = None) -> md.MuData:
    if not mdata_path.exists():
        raise FileNotFoundError(f"Dataset not found: {mdata_path}")
    logger.info("Loading %s", mdata_path)
    mdata = md.read_h5mu(mdata_path)
    if subset_data_type is not None:
        if BATCH_KEY not in mdata.obs:
            raise KeyError(f"Cannot subset by {BATCH_KEY!r}; key is missing from mdata.obs")
        observed = mdata.obs[BATCH_KEY].astype(str)
        mask = observed.str.lower() == subset_data_type.lower()
        n_selected = int(mask.sum())
        if n_selected == 0:
            available = sorted(observed.unique().tolist())
            raise ValueError(
                f"No cells found where {BATCH_KEY} == {subset_data_type!r}. "
                f"Available values: {available}"
            )
        mdata = mdata[mask.to_numpy()].copy()
        logger.info(
            "Subset %s == %s -> %d cells",
            BATCH_KEY,
            subset_data_type,
            n_selected,
        )
        batch_col = mdata.obs[BATCH_KEY]
        if str(batch_col.dtype) == "category":
            mdata.obs[BATCH_KEY] = batch_col.cat.remove_unused_categories()
    logger.info(
        "cells=%d  RNA=%d  ATAC=%d",
        mdata.n_obs,
        mdata.mod["RNA"].n_vars,
        mdata.mod["ATAC"].n_vars,
    )
    return mdata


def _neighbors(latent: np.ndarray, n_neighbors: int):
    return pynndescent(
        latent.astype(np.float32),
        n_neighbors=n_neighbors,
        random_state=SEED,
        n_jobs=1,
    )


def _leiden_labels(latent: np.ndarray, n_neighbors: int = N_NEIGHBORS) -> np.ndarray:
    import scanpy as sc
    from anndata import AnnData

    adata = AnnData(latent.astype(np.float32))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", random_state=SEED)
    try:
        sc.tl.leiden(
            adata, flavor="igraph", n_iterations=2, directed=False, random_state=SEED
        )
    except TypeError:
        sc.tl.leiden(adata, random_state=SEED)
    return adata.obs["leiden"].to_numpy()


def _bio_conservation(latent: np.ndarray, obs_names: np.ndarray) -> dict:
    out = {
        "nmi_vs_rna_pseudo_label": float("nan"),
        "ari_vs_rna_pseudo_label": float("nan"),
        "nmi_vs_multivi_leiden": float("nan"),
        "ari_vs_multivi_leiden": float("nan"),
        "ref_nmi_vs_rna_pseudo_label": float("nan"),
        "n_bio_cells": 0,
    }
    try:
        if not REF_LATENT_PATH.exists():
            logger.warning("Reference latent %s not found; skipping bio metrics", REF_LATENT_PATH)
            return out
        ref = np.load(REF_LATENT_PATH, allow_pickle=True)
        ref_names = ref["obs_names"].astype(str)
        ref_pos = {n: i for i, n in enumerate(ref_names)}
        names = obs_names.astype(str)
        order = [ref_pos[n] for n in names if n in ref_pos]
        keep = np.array([n in ref_pos for n in names], dtype=bool)
        if keep.sum() < 10:
            logger.warning("Too few shared cells with reference (%d); skipping", int(keep.sum()))
            return out

        setvi_leiden = _leiden_labels(latent[keep])
        out["n_bio_cells"] = int(keep.sum())
        if "rna_pseudo_label" in ref.files:
            rna_lab = ref["rna_pseudo_label"].astype(str)[order]
            out["nmi_vs_rna_pseudo_label"] = float(
                normalized_mutual_info_score(rna_lab, setvi_leiden)
            )
            out["ari_vs_rna_pseudo_label"] = float(adjusted_rand_score(rna_lab, setvi_leiden))
            if "leiden" in ref.files:
                out["ref_nmi_vs_rna_pseudo_label"] = float(
                    normalized_mutual_info_score(rna_lab, ref["leiden"].astype(str)[order])
                )
        if "leiden" in ref.files:
            ref_leiden = ref["leiden"].astype(str)[order]
            out["nmi_vs_multivi_leiden"] = float(
                normalized_mutual_info_score(ref_leiden, setvi_leiden)
            )
            out["ari_vs_multivi_leiden"] = float(adjusted_rand_score(ref_leiden, setvi_leiden))
    except Exception as exc:
        logger.warning("bio-conservation metric failed: %r", exc)
    return out


def _last_val(history: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        val = history.get(key)
        if val is not None:
            return float(np.asarray(val).ravel()[-1])
    return float("nan")


def _plot_umap(
    emb: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    *,
    n_latent: int,
    max_epochs: int,
    title: str | None = None,
    cmap_name: str = "tab10",
    legend_ncol: int = 1,
    legend_fontsize: int = 9,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    cats = sorted(set(labels.tolist()))
    cmap = plt.get_cmap(cmap_name)
    for i, cat in enumerate(cats):
        mask = labels == cat
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            color=cmap(i % cmap.N),
            label=f"{cat} (n={int(mask.sum())})",
            alpha=0.7,
            linewidths=0,
        )
    if title is None:
        title = (
            f"SETVI pilot — {BATCH_KEY}\n"
            f"{max_epochs} epochs (n={len(labels)}, d={n_latent})"
        )
    ax.set_title(title)
    ax.legend(
        markerscale=2,
        frameon=False,
        loc="best",
        ncol=legend_ncol,
        fontsize=legend_fontsize,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- collapse-trace instrumentation -------------------------------------------------
# These helpers record, at the end of every training epoch, the per-modality latent
# std (collapse signal) and per-cell KL on a small fixed subset of cells, plus the
# Set-Transformer accessibility-encoder output. Together with the global training
# pressures (adversarial loss, recon losses, KL) pulled from ``model.history`` this
# pinpoints *when* and *under which pressure* the ATAC path collapses.


def _select_trace_indices(data_type: np.ndarray, per_group: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sel = []
    for g in np.unique(data_type):
        idx = np.where(data_type == g)[0]
        if len(idx) > per_group:
            idx = rng.choice(idx, per_group, replace=False)
        sel.append(idx)
    return np.sort(np.concatenate(sel))


def _participation_ratio(x: np.ndarray) -> float:
    """Effective dimensionality from singular-value spectrum."""
    if x.shape[0] < 2:
        return float("nan")
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    ev = s**2
    total = ev.sum()
    if total <= 0:
        return float("nan")
    ev = ev / total
    return float((ev.sum() ** 2) / (ev**2).sum())


def _acc_depth_metrics(
    am: np.ndarray, data_type: np.ndarray, depth: np.ndarray, group: str
) -> tuple[float, float]:
    """Return (|corr(PC1, depth)|, participation ratio) for accessibility latent."""
    mask = data_type == group
    if int(mask.sum()) < 3:
        return float("nan"), float("nan")
    sub = am[mask]
    depth_sub = depth[mask]
    xc = sub - sub.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    pc1 = xc @ vt[0]
    if np.std(pc1) < 1e-12 or np.std(depth_sub) < 1e-12:
        corr = float("nan")
    else:
        corr = abs(float(np.corrcoef(pc1, depth_sub)[0, 1]))
    return corr, _participation_ratio(sub)


def _atac_peak_counts(mdata: md.MuData, obs_names: np.ndarray) -> np.ndarray:
    """Per-cell open-peak counts aligned to ``obs_names``."""
    import scipy.sparse as sp

    atac = mdata.mod["ATAC"]
    x = atac.X
    if sp.issparse(x):
        nnz = np.asarray((x > 0).sum(axis=1)).ravel()
    else:
        nnz = np.asarray((np.asarray(x) > 0).sum(axis=1)).ravel()
    pos = {n: i for i, n in enumerate(atac.obs_names.astype(str))}
    return nnz[np.array([pos[n] for n in obs_names.astype(str)], dtype=np.int64)].astype(
        np.float64
    )


def _latent_group_stats(
    q_m: np.ndarray, q_v: np.ndarray, data_type: np.ndarray, group: str
) -> tuple[float, float]:
    """Return (mean per-dim std of q_m, mean per-cell KL vs N(0,1)) for one group."""
    mask = data_type == group
    if int(mask.sum()) == 0:
        return float("nan"), float("nan")
    m = q_m[mask]
    v = np.clip(q_v[mask], 1e-12, None)
    std = float(np.mean(np.std(m, axis=0)))
    kl = 0.5 * (m**2 + v - np.log(v) - 1.0)
    kl = float(np.mean(np.sum(kl, axis=1)))
    return std, kl


def _make_collapse_trace_callback(
    model,
    data_type_sub: np.ndarray,
    indices: np.ndarray,
    depth_sub: np.ndarray,
    *,
    batch_size: int,
    groups: tuple[str, ...],
):
    """Build a Lightning callback that traces latent collapse during training."""
    import lightning.pytorch as pl
    import torch

    class _CollapseTraceCallback(pl.Callback):
        def __init__(self):
            super().__init__()
            self.records: list[dict] = []

        def on_train_epoch_end(self, trainer, pl_module):
            was_trained = getattr(model, "is_trained_", False)
            try:
                model.is_trained_ = True
                model.module.eval()
                with torch.no_grad():
                    jm, jv = model.get_latent_representation(
                        indices=indices,
                        return_dist=True,
                        modality="joint",
                        batch_size=batch_size,
                    )
                    am, av = model.get_latent_representation(
                        indices=indices,
                        return_dist=True,
                        modality="accessibility",
                        batch_size=batch_size,
                    )
                rec = {"epoch": int(trainer.current_epoch)}
                for g in groups:
                    std, kl = _latent_group_stats(jm, jv, data_type_sub, g)
                    rec[f"joint_std_{g}"] = std
                    rec[f"joint_kl_{g}"] = kl
                    if g in ("ATAC", "multiome"):
                        std_a, kl_a = _latent_group_stats(am, av, data_type_sub, g)
                        rec[f"acc_std_{g}"] = std_a
                        rec[f"acc_kl_{g}"] = kl_a
                        dcorr, eff_rank = _acc_depth_metrics(
                            am, data_type_sub, depth_sub, g
                        )
                        rec[f"acc_depth_corr_{g}"] = dcorr
                        rec[f"acc_eff_rank_{g}"] = eff_rank
                self.records.append(rec)
                logger.info(
                    "[trace] epoch %d  joint_std ATAC=%.4g RNA=%.4g multi=%.4g | "
                    "joint_kl ATAC=%.3g RNA=%.3g multi=%.3g | "
                    "acc_depth_corr ATAC=%.3g multi=%.3g | acc_eff_rank ATAC=%.2f multi=%.2f",
                    rec["epoch"],
                    rec.get("joint_std_ATAC", float("nan")),
                    rec.get("joint_std_RNA", float("nan")),
                    rec.get("joint_std_multiome", float("nan")),
                    rec.get("joint_kl_ATAC", float("nan")),
                    rec.get("joint_kl_RNA", float("nan")),
                    rec.get("joint_kl_multiome", float("nan")),
                    rec.get("acc_depth_corr_ATAC", float("nan")),
                    rec.get("acc_depth_corr_multiome", float("nan")),
                    rec.get("acc_eff_rank_ATAC", float("nan")),
                    rec.get("acc_eff_rank_multiome", float("nan")),
                )
            except Exception as exc:  # never break training for a diagnostic
                logger.warning("[trace] epoch trace failed: %r", exc)
            finally:
                model.is_trained_ = was_trained
                model.module.train()

    return _CollapseTraceCallback()


def _history_array(history: dict, key: str) -> np.ndarray | None:
    obj = history.get(key) if isinstance(history, dict) else None
    if obj is None:
        return None
    try:
        return np.asarray(obj).ravel().astype(float)
    except Exception:
        return None


def _plot_collapse_trace(
    records: list[dict],
    history: dict,
    *,
    kl_warmup: int,
    adversarial: bool,
    max_epochs: int,
    out_path: Path,
) -> None:
    if not records:
        return
    ep = np.array([r["epoch"] for r in records], dtype=float)

    def col(name: str) -> np.ndarray:
        return np.array([r.get(name, np.nan) for r in records], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(9, 15), sharex=True)

    ax = axes[0]
    ax.plot(ep, col("joint_std_ATAC"), color="tab:blue", label="joint std — ATAC")
    ax.plot(ep, col("joint_std_RNA"), color="tab:orange", label="joint std — RNA")
    ax.plot(ep, col("joint_std_multiome"), color="tab:green", label="joint std — multiome")
    ax.plot(ep, col("acc_std_ATAC"), color="tab:blue", ls="--", label="ST-acc std — ATAC")
    ax.plot(ep, col("acc_std_multiome"), color="tab:green", ls="--", label="ST-acc std — multiome")
    ax.set_yscale("log")
    ax.set_ylabel("latent std (mean over dims, log)")
    ax.set_title("Per-modality latent std — collapse onset")
    ax.legend(fontsize=8, ncol=2, frameon=False)

    ax = axes[1]
    ax.plot(ep, col("joint_kl_ATAC"), color="tab:blue", label="KL — ATAC")
    ax.plot(ep, col("joint_kl_RNA"), color="tab:orange", label="KL — RNA")
    ax.plot(ep, col("joint_kl_multiome"), color="tab:green", label="KL — multiome")
    ax.set_yscale("log")
    ax.set_ylabel("KL per cell (q vs N(0,1), log)")
    ax.set_title("Per-modality KL (effective KL pressure = KL x weight)")
    # KL-warmup weight ramp on a twin axis to show when KL pressure turns on.
    axw = ax.twinx()
    klw = np.clip((ep + 1.0) / max(kl_warmup, 1), 0.0, 1.0)
    axw.plot(ep, klw, color="grey", ls="--", lw=1, label="KL weight")
    axw.set_ylabel("KL warmup weight")
    axw.set_ylim(-0.02, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axw.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False)

    ax = axes[2]
    ax.plot(ep, col("acc_depth_corr_ATAC"), color="tab:blue", label="|corr(PC1, depth)| — ATAC")
    ax.plot(
        ep,
        col("acc_depth_corr_multiome"),
        color="tab:green",
        label="|corr(PC1, depth)| — multiome",
    )
    ax.set_ylabel("|depth correlation|")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("ATAC accessibility latent vs sequencing depth")
    axw = ax.twinx()
    axw.plot(ep, col("acc_eff_rank_ATAC"), color="tab:blue", ls="--", label="eff rank — ATAC")
    axw.plot(
        ep,
        col("acc_eff_rank_multiome"),
        color="tab:green",
        ls="--",
        label="eff rank — multiome",
    )
    axw.set_ylabel("effective rank (of 8)")
    axw.set_ylim(0, 8.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axw.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper right")

    ax = axes[3]
    adv = _history_array(history, "adversarial_loss_train")
    recon = _history_array(history, "reconstruction_loss_train")
    klg = _history_array(history, "kl_local_train")
    handles = []
    if recon is not None:
        (line,) = ax.plot(
            np.arange(len(recon)), recon, color="tab:purple", label="reconstruction_loss_train"
        )
        handles.append(line)
    if klg is not None:
        (line,) = ax.plot(
            np.arange(len(klg)), klg, color="tab:brown", alpha=0.7, label="kl_local_train"
        )
        handles.append(line)
    ax.set_yscale("log")
    ax.set_ylabel("loss (log)")
    ax.set_xlabel("epoch")
    if adv is not None:
        ax2 = ax.twinx()
        (line,) = ax2.plot(
            np.arange(len(adv)), adv, color="tab:red", alpha=0.8, label="adversarial_loss_train"
        )
        ax2.set_ylabel("adversarial loss")
        handles.append(line)
    ax.set_title("Global training pressures")
    if handles:
        ax.legend(handles, [h.get_label() for h in handles], fontsize=8, frameon=False)

    for a in axes:
        a.axvline(kl_warmup, color="grey", ls=":", lw=1)
    fig.suptitle(
        f"SETVI collapse trace — adversarial={adversarial}, kl_warmup={kl_warmup}, "
        f"{max_epochs} epochs (dotted line = KL warmup end)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_collapse_trace(
    records: list[dict],
    history: dict,
    suffix: str,
    *,
    adversarial: bool,
    n_epochs_kl_warmup: int,
    max_epochs: int,
) -> None:
    if not records:
        logger.warning("Collapse trace empty; nothing to write.")
        return
    payload = {
        "adversarial": adversarial,
        "n_epochs_kl_warmup": n_epochs_kl_warmup,
        "max_epochs": max_epochs,
        "trace_per_group": TRACE_PER_GROUP,
        "records": records,
    }
    (OUT_DIR / f"setvi_collapse_trace{suffix}.json").write_text(json.dumps(payload, indent=2))
    _plot_collapse_trace(
        records,
        history,
        kl_warmup=n_epochs_kl_warmup,
        adversarial=adversarial,
        max_epochs=max_epochs,
        out_path=OUT_DIR / f"setvi_collapse_trace{suffix}.png",
    )
    logger.info("Wrote collapse trace (%d epochs) to %s", len(records), OUT_DIR)


def main(
    *,
    adversarial: bool = True,
    max_epochs: int = MAX_EPOCHS,
    n_epochs_kl_warmup: int = N_EPOCHS_KL_WARMUP,
    region_factors: bool = REGION_FACTORS,
    st_d_model: int = ST_D_MODEL,
    st_n_layers: int = ST_N_LAYERS,
    st_n_inducing: int = ST_N_INDUCING,
    st_n_heads: int = ST_N_HEADS,
    st_dropout: float = ST_DROPOUT,
    use_cardinality_film: bool = USE_CARDINALITY_FILM,
    use_sampling_correction: bool = USE_SAMPLING_CORRECTION,
    use_peak_salience_prior: bool = USE_PEAK_SALIENCE_PRIOR,
    peak_salience_cap: float = PEAK_SALIENCE_CAP,
    use_counts_in_encoder: bool = USE_COUNTS_IN_ENCODER,
    tag: str = "",
    mdata_path: Path = MDATA_PATH,
    subset_data_type: str | None = None,
    matmul_precision: str = MATMUL_PRECISION,
    train_precision: str = TRAIN_PRECISION,
    num_workers: int = DL_NUM_WORKERS,
    pin_memory: bool = DL_PIN_MEMORY,
    persistent_workers: bool = DL_PERSISTENT_WORKERS,
    compile_model: bool = COMPILE_MODEL,
    trainer_benchmark: bool = TRAINER_BENCHMARK,
    trace_collapse: bool = False,
    atac_token_store: str = ATAC_TOKEN_STORE,
) -> None:
    scvi.settings.seed = SEED
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    mdata = _load_mdata(mdata_path, subset_data_type=subset_data_type)

    if (
        subset_data_type is not None
        and subset_data_type.lower() == "atac"
        and not adversarial
    ):
        logger.info(
            "ATAC-only subset detected: enabling adversarial alignment path."
        )
        adversarial = True

    scvi.model.SETVI.setup_mudata(
        mdata,
        batch_key=BATCH_KEY,
        modalities={"rna_layer": "RNA", "atac_layer": "ATAC"},
        max_atac_tokens=MAX_ATAC_TOKENS,
        atac_token_store=atac_token_store,
    )
    model = scvi.model.SETVI(
        mdata,
        st_d_model=st_d_model,
        st_n_layers=st_n_layers,
        st_n_inducing=st_n_inducing,
        st_n_heads=st_n_heads,
        st_dropout=st_dropout,
        use_cardinality_film=use_cardinality_film,
        use_sampling_correction=use_sampling_correction,
        use_peak_salience_prior=use_peak_salience_prior,
        peak_salience_cap=peak_salience_cap,
        use_counts_in_encoder=use_counts_in_encoder,
    )

    alignment_mode = "adversarial" if adversarial else "standard"
    perf = _perf_settings_dict(
        matmul_precision=matmul_precision,
        train_precision=train_precision,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        compile_model=compile_model,
        trainer_benchmark=trainer_benchmark,
    )
    logger.info(
        (
            "Training SETVI (%d epochs, batch_key=%s, alignment=%s, "
            "kl_warmup=%d, batch_size=%d, max_atac_tokens=%d, bucketing=%s, "
            "atac_token_store=%s, perf=%s)"
        ),
        max_epochs,
        BATCH_KEY,
        alignment_mode,
        n_epochs_kl_warmup,
        BATCH_SIZE,
        MAX_ATAC_TOKENS,
        LENGTH_BUCKETING,
        atac_token_store,
        perf,
    )
    trace_cb = None
    if trace_collapse:
        data_type_full = mdata.obs[BATCH_KEY].astype(str).to_numpy()
        groups = tuple(np.unique(data_type_full).tolist())
        trace_idx = _select_trace_indices(data_type_full, TRACE_PER_GROUP, SEED)
        trace_dt = data_type_full[trace_idx]
        trace_names = np.asarray(mdata.obs.index, dtype=object)[trace_idx]
        trace_depth = _atac_peak_counts(mdata, trace_names)
        trace_cb = _make_collapse_trace_callback(
            model,
            trace_dt,
            trace_idx,
            trace_depth,
            batch_size=max(BATCH_SIZE, 256),
            groups=groups,
        )
        logger.info(
            "Collapse trace enabled: %d cells (%s)",
            len(trace_idx),
            {g: int((trace_dt == g).sum()) for g in groups},
        )

    train_kwargs = _build_train_kwargs(
        max_epochs=max_epochs,
        adversarial=adversarial,
        n_epochs_kl_warmup=n_epochs_kl_warmup,
        bucketing=LENGTH_BUCKETING,
        matmul_precision=matmul_precision,
        train_precision=train_precision,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        compile_model=compile_model,
        trainer_benchmark=trainer_benchmark,
    )
    if trace_cb is not None:
        train_kwargs["callbacks"] = [trace_cb]
    t0 = time.perf_counter()
    model.train(**train_kwargs)
    train_seconds = time.perf_counter() - t0
    logger.info(
        "Training finished in %.1f s (%.2f s/epoch)",
        train_seconds,
        train_seconds / max_epochs,
    )

    latent = model.get_latent_representation(batch_size=BATCH_SIZE)
    data_type = mdata.obs[BATCH_KEY].astype(str).to_numpy()
    obs_names = np.asarray(mdata.obs.index, dtype=object)

    nbr = _neighbors(latent, N_NEIGHBORS)
    unique_data_types = np.unique(data_type)
    kbet_val = float("nan")
    ilisi_val = float("nan")
    silhouette_val = float("nan")
    if len(unique_data_types) >= 2:
        try:
            kbet_result = kbet(nbr, data_type)
            kbet_val = float(kbet_result[0] if isinstance(kbet_result, tuple) else kbet_result)
            ilisi_val = float(ilisi_knn(nbr, data_type))
            silhouette_val = float(silhouette_score(latent.astype(np.float32), data_type))
        except Exception as exc:
            logger.warning("Mixing metrics failed: %r", exc)
    else:
        logger.info(
            "Skipping data_type mixing metrics: only one class present (%s).",
            unique_data_types.tolist(),
        )
    bio = _bio_conservation(latent, obs_names)
    leiden = _leiden_labels(latent, n_neighbors=N_NEIGHBORS).astype(str)

    metrics = {
        "model": "SETVI",
        "tag": tag,
        "alignment_mode": alignment_mode,
        "adversarial_mixing": adversarial,
        "n_epochs_kl_warmup": n_epochs_kl_warmup,
        "region_factors": region_factors,
        "use_peak_salience_prior": use_peak_salience_prior,
        "peak_salience_cap": peak_salience_cap,
        "atac_token_store": atac_token_store,
        "csr_token_streaming": False,
        "precompute_atac_tokens": True,
        "st_d_model": st_d_model,
        "st_n_layers": st_n_layers,
        "st_n_inducing": st_n_inducing,
        "st_n_heads": st_n_heads,
        "st_dropout": st_dropout,
        "use_cardinality_film": use_cardinality_film,
        "use_sampling_correction": use_sampling_correction,
        "use_counts_in_encoder": use_counts_in_encoder,
        "dataset": str(mdata_path),
        "subset_data_type": subset_data_type,
        "batch_key": BATCH_KEY,
        "max_epochs": max_epochs,
        "seed": SEED,
        "n_neighbors": N_NEIGHBORS,
        "n_obs": int(mdata.n_obs),
        "n_rna_vars": int(mdata.mod["RNA"].n_vars),
        "n_atac_vars": int(mdata.mod["ATAC"].n_vars),
        "train_seconds": round(train_seconds, 2),
        "train_seconds_per_epoch": round(train_seconds / max_epochs, 2),
        "atac_length_bucketing": LENGTH_BUCKETING,
        "bucket_mult": BUCKET_MULT,
        "max_atac_tokens": MAX_ATAC_TOKENS,
        **perf,
        "recon_loss": _last_val(
            model.history,
            ("reconstruction_loss_validation", "validation_loss", "elbo_validation"),
        ),
        "reconstruction_loss_expression_train": _last_val(
            model.history, ("reconstruction_loss_expression_train",),
        ),
        "reconstruction_loss_accessibility_train": _last_val(
            model.history, ("reconstruction_loss_accessibility_train",),
        ),
        "n_unique_data_type": int(len(unique_data_types)),
        "unique_data_type": unique_data_types.tolist(),
        "n_leiden_clusters": int(len(np.unique(leiden))),
        "adversarial_loss_train": _last_val(
            model.history, ("adversarial_loss_train", "adversarial_loss"),
        ),
        "kbet_data_type": kbet_val,
        "ilisi_data_type": ilisi_val,
        "silhouette_data_type": silhouette_val,
        **bio,
    }

    emb = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=SEED).fit_transform(
        latent.astype(np.float32)
    )

    np.savez_compressed(
        OUT_DIR / f"setvi_pilot_latent{suffix}.npz",
        latent=latent,
        data_type=data_type,
        obs_names=obs_names,
        umap=emb,
        leiden=leiden,
    )
    _plot_umap(
        emb,
        data_type,
        OUT_DIR / f"setvi_pilot_umap{suffix}.png",
        n_latent=latent.shape[1],
        max_epochs=max_epochs,
    )
    _plot_umap(
        emb,
        leiden,
        OUT_DIR / f"setvi_pilot_umap_leiden{suffix}.png",
        n_latent=latent.shape[1],
        max_epochs=max_epochs,
        title=(
            f"SETVI pilot — Leiden (res=scanpy-default, k={N_NEIGHBORS})\n"
            f"{int(metrics['n_leiden_clusters'])} clusters, n={len(leiden)}, d={latent.shape[1]}"
        ),
        cmap_name="tab20",
        legend_ncol=2,
        legend_fontsize=7,
    )
    (OUT_DIR / f"setvi_pilot_metrics{suffix}.json").write_text(json.dumps(metrics, indent=2))
    model.save(OUT_DIR / f"model{suffix}", overwrite=True)

    if trace_cb is not None:
        try:
            _save_collapse_trace(
                trace_cb.records,
                model.history,
                suffix,
                adversarial=adversarial,
                n_epochs_kl_warmup=n_epochs_kl_warmup,
                max_epochs=max_epochs,
            )
        except Exception as exc:
            logger.warning("Failed to write collapse trace: %r", exc)

    print(json.dumps(metrics, indent=2))
    logger.info("Wrote artifacts to %s", OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Training epochs (default: {MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--adversarial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use adversarial batch mixing (default: True).",
    )
    parser.add_argument(
        "--kl-warmup",
        type=int,
        default=N_EPOCHS_KL_WARMUP,
        help=f"KL warmup epochs (default: {N_EPOCHS_KL_WARMUP}).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix for output files.",
    )
    parser.add_argument(
        "--mdata-path",
        type=Path,
        default=MDATA_PATH,
        help=f"Path to .h5mu dataset (default: {MDATA_PATH}).",
    )
    parser.add_argument(
        "--subset-data-type",
        type=str,
        default=None,
        help=f"Subset to rows where `{BATCH_KEY}` matches this value.",
    )
    parser.add_argument(
        "--atac-only",
        action="store_true",
        help=f"Shortcut for `--subset-data-type ATAC`.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--max-atac-tokens",
        type=int,
        default=MAX_ATAC_TOKENS,
        help=f"Max ATAC peaks per cell (default: {MAX_ATAC_TOKENS}).",
    )
    parser.add_argument(
        "--atac-length-bucketing",
        action=argparse.BooleanOptionalAction,
        default=LENGTH_BUCKETING,
        help=f"Enable ATAC length bucketing (default: {LENGTH_BUCKETING}).",
    )
    parser.add_argument(
        "--st-d-model",
        type=int,
        default=ST_D_MODEL,
        help=f"Set Transformer d_model (default: {ST_D_MODEL}).",
    )
    parser.add_argument(
        "--st-n-layers",
        type=int,
        default=ST_N_LAYERS,
        help=f"Number of ISAB layers (default: {ST_N_LAYERS}).",
    )
    parser.add_argument(
        "--st-n-inducing",
        type=int,
        default=ST_N_INDUCING,
        help=f"Number of inducing points (default: {ST_N_INDUCING}).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=TRAIN_PRECISION,
        help=f"Lightning precision (default: {TRAIN_PRECISION}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DL_NUM_WORKERS,
        help=f"DataLoader workers (default: {DL_NUM_WORKERS}).",
    )
    parser.add_argument(
        "--atac-token-store",
        type=str,
        default=ATAC_TOKEN_STORE,
        choices=("auto", "gpu", "ram", "mmap"),
        help=f"ATAC token store tier (default: {ATAC_TOKEN_STORE}).",
    )
    parser.add_argument(
        "--trace-collapse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record per-modality latent std + KL each epoch to diagnose collapse.",
    )
    parser.add_argument(
        "--use-cardinality-film",
        action=argparse.BooleanOptionalAction,
        default=USE_CARDINALITY_FILM,
        help="Inject set cardinality via CardinalityFiLM (default: False after depth-collapse fix).",
    )
    parser.add_argument(
        "--peak-salience-prior",
        action=argparse.BooleanOptionalAction,
        default=USE_PEAK_SALIENCE_PRIOR,
        help="Reuse region_factors as a capped rarity prior on encoder attention (default: True).",
    )
    parser.add_argument(
        "--peak-salience-cap",
        type=float,
        default=PEAK_SALIENCE_CAP,
        help=(
            f"Cap on the per-peak rarity weight (default: {PEAK_SALIENCE_CAP}; "
            "freq floor exp(-cap))."
        ),
    )
    parser.add_argument(
        "--counts-in-encoder",
        action=argparse.BooleanOptionalAction,
        default=USE_COUNTS_IN_ENCODER,
        help=(
            "Route ATAC token counts through ValueMLP+log1p in the encoder embedding "
            f"(default: {USE_COUNTS_IN_ENCODER})."
        ),
    )
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    MAX_ATAC_TOKENS = args.max_atac_tokens
    LENGTH_BUCKETING = args.atac_length_bucketing
    TRAIN_PRECISION = args.precision
    DL_NUM_WORKERS = args.num_workers
    scvi.settings.seed = SEED
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subset_data_type = "ATAC" if args.atac_only else args.subset_data_type
    _apply_perf_settings(MATMUL_PRECISION)
    if not args.mdata_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.mdata_path}")
    main(
        adversarial=args.adversarial,
        max_epochs=args.max_epochs,
        n_epochs_kl_warmup=args.kl_warmup,
        st_d_model=args.st_d_model,
        st_n_layers=args.st_n_layers,
        st_n_inducing=args.st_n_inducing,
        tag=args.tag,
        mdata_path=args.mdata_path,
        subset_data_type=subset_data_type,
        train_precision=TRAIN_PRECISION,
        num_workers=DL_NUM_WORKERS,
        trace_collapse=args.trace_collapse,
        atac_token_store=args.atac_token_store,
        use_cardinality_film=args.use_cardinality_film,
        use_peak_salience_prior=args.peak_salience_prior,
        peak_salience_cap=args.peak_salience_cap,
        use_counts_in_encoder=args.counts_in_encoder,
    )
