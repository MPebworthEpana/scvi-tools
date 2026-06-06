"""Train MultiVI on the pilot mosaic dataset and write UMAP + mixing metrics.

Run from anywhere::

    python MultiVI/tests/multivi_pilot_baseline/multivi_baseline.py

Outputs are written next to this script (``multivi_pilot_latent.npz``,
``multivi_pilot_umap.png``, ``multivi_pilot_umap_leiden.png``,
``multivi_pilot_metrics.json``, ``model/``).
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from pathlib import Path

import anndata as ad
import igraph
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import scanpy as sc
import scvi
import umap
from scib_metrics import clisi_knn, ilisi_knn, kbet
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent
from scipy.sparse import spmatrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

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
LEIDEN_RESOLUTION = 1.0
RNA_PCA_DIMS = 50
PSEUDO_LABEL_KEY = "rna_pseudo_label"


def _neighbors(latent: np.ndarray, n_neighbors: int, *, seed: int = SEED) -> NeighborsResults:
    return pynndescent(
        latent.astype(np.float32),
        n_neighbors=n_neighbors,
        random_state=seed,
        n_jobs=1,
    )


def _last_val(history: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        val = history.get(key)
        if val is not None:
            return float(np.asarray(val).ravel()[-1])
    return float("nan")


def _leiden_clusters(
    nbr: NeighborsResults,
    resolution: float = LEIDEN_RESOLUTION,
    seed: int = SEED,
) -> np.ndarray:
    """Leiden clustering on a kNN connectivity graph."""
    conn = nbr.knn_graph_connectivities
    if not isinstance(conn, spmatrix):
        raise TypeError("Expected sparse connectivity graph from NeighborsResults.")
    rng = random.Random(seed)
    igraph.set_random_number_generator(rng)
    graph = igraph.Graph.Weighted_Adjacency(conn, mode="directed")
    graph.to_undirected(mode="each")
    clustering = graph.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=resolution,
    )
    return np.asarray(clustering.membership, dtype=str)


def _rna_present_mask(mdata: md.MuData) -> np.ndarray:
    rna = mdata.mod["RNA"].X
    if hasattr(rna, "toarray"):
        sums = np.asarray(rna.sum(axis=1)).ravel()
    else:
        sums = np.asarray(rna.sum(axis=1)).ravel()
    return sums > 0


def _transfer_pseudo_labels(
    latent: np.ndarray,
    nbr: NeighborsResults,
    pseudo_labels: np.ndarray,
    rna_present: np.ndarray,
    *,
    seed: int = SEED,
) -> np.ndarray:
    """Assign RNA pseudo-labels to ATAC-only cells via latent kNN majority vote."""
    out = pseudo_labels.copy()
    missing = ~rna_present
    if not missing.any():
        return out

    rna_indices = np.flatnonzero(rna_present)
    rna_labels = pseudo_labels[rna_indices]
    rng = np.random.default_rng(seed)

    for cell_idx in np.flatnonzero(missing):
        neighbor_idx = nbr.indices[cell_idx]
        rna_neighbor_mask = rna_present[neighbor_idx]
        if not rna_neighbor_mask.any():
            dists = np.linalg.norm(
                latent[rna_indices] - latent[cell_idx], axis=1
            )
            k = min(N_NEIGHBORS, len(rna_indices))
            nearest = rna_indices[np.argpartition(dists, k - 1)[:k]]
            votes = pseudo_labels[nearest]
        else:
            votes = pseudo_labels[neighbor_idx[rna_neighbor_mask]]
        if len(votes) == 0:
            out[cell_idx] = str(rng.integers(0, 1_000_000))
            continue
        out[cell_idx] = Counter(votes.tolist()).most_common(1)[0][0]
    return out


def _rna_pseudo_labels(
    mdata: md.MuData,
    latent: np.ndarray,
    nbr: NeighborsResults,
    *,
    seed: int = SEED,
    n_neighbors: int = N_NEIGHBORS,
    resolution: float = LEIDEN_RESOLUTION,
) -> np.ndarray:
    """Leiden clusters on RNA PCA for RNA-present cells; kNN transfer for ATAC-only."""
    rna_present = _rna_present_mask(mdata)
    n_obs = mdata.n_obs
    pseudo = np.full(n_obs, "", dtype=object)

    rna_adata = ad.AnnData(
        X=mdata.mod["RNA"][rna_present].X.copy(),
        obs=mdata.obs.loc[rna_present].copy(),
    )
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    n_comps = min(RNA_PCA_DIMS, rna_adata.n_vars - 1, rna_adata.n_obs - 1)
    sc.pp.pca(rna_adata, n_comps=n_comps, random_state=seed)
    sc.pp.neighbors(rna_adata, n_neighbors=n_neighbors, random_state=seed)
    sc.tl.leiden(
        rna_adata,
        resolution=resolution,
        key_added=PSEUDO_LABEL_KEY,
        random_state=seed,
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )

    rna_indices = np.flatnonzero(rna_present)
    pseudo[rna_indices] = rna_adata.obs[PSEUDO_LABEL_KEY].astype(str).to_numpy()
    pseudo = _transfer_pseudo_labels(latent, nbr, pseudo.astype(str), rna_present, seed=seed)
    return pseudo.astype(str)


def bio_conservation_metrics(
    latent: np.ndarray,
    nbr: NeighborsResults,
    leiden: np.ndarray,
    pseudo_labels: np.ndarray,
) -> dict[str, float]:
    """NMI, ARI, cLISI, and silhouette for integrated Leiden vs RNA pseudo-labels."""
    latent_f = latent.astype(np.float32)
    nmi = float(normalized_mutual_info_score(pseudo_labels, leiden, average_method="arithmetic"))
    ari = float(adjusted_rand_score(pseudo_labels, leiden))
    clisi = float(clisi_knn(nbr, pseudo_labels))
    sil = float(silhouette_score(latent_f, pseudo_labels))
    return {
        "n_leiden_clusters": float(len(np.unique(leiden))),
        "n_rna_pseudo_clusters": float(len(np.unique(pseudo_labels))),
        "nmi_leiden_rna_pseudo": nmi,
        "ari_leiden_rna_pseudo": ari,
        "clisi_rna_pseudo": clisi,
        "silhouette_rna_pseudo": sil,
    }


def _plot_umap(
    emb: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    *,
    title: str,
    cmap_name: str = "tab10",
    legend_ncol: int = 1,
    legend_fontsize: int = 9,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    cats = sorted(set(labels.tolist()), key=lambda x: (str(x).isdigit(), str(x)))
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
    ax.set_title(title)
    ax.legend(
        markerscale=2,
        frameon=False,
        loc="best",
        fontsize=legend_fontsize,
        ncol=legend_ncol,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    scvi.settings.seed = SEED
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", MDATA_PATH)
    mdata = md.read_h5mu(MDATA_PATH)
    logger.info(
        "cells=%d  RNA=%d  ATAC=%d",
        mdata.n_obs,
        mdata.mod["RNA"].n_vars,
        mdata.mod["ATAC"].n_vars,
    )

    scvi.model.MULTIVI.setup_mudata(
        mdata,
        batch_key=BATCH_KEY,
        modalities={"rna_layer": "RNA", "atac_layer": "ATAC"},
    )
    model = scvi.model.MULTIVI(mdata)

    logger.info("Training MultiVI (%d epochs, batch_key=%s)", MAX_EPOCHS, BATCH_KEY)
    t0 = time.perf_counter()
    model.train(
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        train_size=TRAIN_SIZE,
        accelerator="auto",
        adversarial_mixing=True,  # MultiVI default; adversarial target = BATCH_KEY
    )
    train_seconds = time.perf_counter() - t0
    logger.info(
        "Training finished in %.1f s (%.2f s/epoch)",
        train_seconds,
        train_seconds / MAX_EPOCHS,
    )

    latent = model.get_latent_representation()
    data_type = mdata.obs[BATCH_KEY].astype(str).to_numpy()

    nbr = _neighbors(latent, N_NEIGHBORS)
    kbet_result = kbet(nbr, data_type)
    kbet_val = float(kbet_result[0] if isinstance(kbet_result, tuple) else kbet_result)

    logger.info("Computing RNA pseudo-labels and integrated Leiden clusters")
    pseudo_labels = _rna_pseudo_labels(mdata, latent, nbr)
    leiden = _leiden_clusters(nbr)
    bio_metrics = bio_conservation_metrics(latent, nbr, leiden, pseudo_labels)

    metrics = {
        "dataset": str(MDATA_PATH),
        "batch_key": BATCH_KEY,
        "max_epochs": MAX_EPOCHS,
        "seed": SEED,
        "n_neighbors": N_NEIGHBORS,
        "leiden_resolution": LEIDEN_RESOLUTION,
        "rna_pca_dims": RNA_PCA_DIMS,
        "n_obs": int(mdata.n_obs),
        "n_rna_vars": int(mdata.mod["RNA"].n_vars),
        "n_atac_vars": int(mdata.mod["ATAC"].n_vars),
        "train_seconds": round(train_seconds, 2),
        "train_seconds_per_epoch": round(train_seconds / MAX_EPOCHS, 2),
        "recon_loss": _last_val(
            model.history,
            ("reconstruction_loss_validation", "validation_loss", "elbo_validation"),
        ),
        "kbet_data_type": kbet_val,
        "ilisi_data_type": float(ilisi_knn(nbr, data_type)),
        "silhouette_data_type": float(silhouette_score(latent.astype(np.float32), data_type)),
        **bio_metrics,
    }

    emb = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=SEED).fit_transform(
        latent.astype(np.float32)
    )

    np.savez_compressed(
        OUT_DIR / "multivi_pilot_latent.npz",
        latent=latent,
        data_type=data_type,
        obs_names=np.asarray(mdata.obs.index, dtype=object),
        umap=emb,
        leiden=leiden,
        rna_pseudo_label=pseudo_labels,
    )
    n_latent = latent.shape[1]
    _plot_umap(
        emb,
        data_type,
        OUT_DIR / "multivi_pilot_umap.png",
        title=(
            f"MultiVI pilot — {BATCH_KEY}\n"
            f"{MAX_EPOCHS} epochs (n={len(data_type)}, d={n_latent})"
        ),
    )
    _plot_umap(
        emb,
        leiden,
        OUT_DIR / "multivi_pilot_umap_leiden.png",
        title=(
            f"MultiVI pilot — Leiden (res={LEIDEN_RESOLUTION}, k={N_NEIGHBORS})\n"
            f"{int(bio_metrics['n_leiden_clusters'])} clusters, "
            f"n={len(leiden)}, d={n_latent}"
        ),
        cmap_name="tab20",
        legend_ncol=2,
        legend_fontsize=7,
    )
    (OUT_DIR / "multivi_pilot_metrics.json").write_text(json.dumps(metrics, indent=2))
    model.save(OUT_DIR / "model", overwrite=True)

    print(json.dumps(metrics, indent=2))
    logger.info("Wrote artifacts to %s", OUT_DIR)


if __name__ == "__main__":
    main()
