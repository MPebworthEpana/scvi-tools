"""Train MultiVI on the pilot mosaic dataset and write UMAP + mixing metrics.

Run from anywhere::

    python MultiVI/tests/multivi_pilot_baseline/multivi_baseline.py

Outputs are written next to this script (``multivi_pilot_latent.npz``,
``multivi_pilot_umap.png``, ``multivi_pilot_metrics.json``, ``model/``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import scvi
import umap
from scib_metrics import ilisi_knn, kbet
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.metrics import silhouette_score

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


def _neighbors(latent: np.ndarray, n_neighbors: int):
    return pynndescent(
        latent.astype(np.float32),
        n_neighbors=n_neighbors,
        random_state=SEED,
        n_jobs=1,
    )


def _last_val(history: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        val = history.get(key)
        if val is not None:
            return float(np.asarray(val).ravel()[-1])
    return float("nan")


def _plot_umap(emb: np.ndarray, labels: np.ndarray, out_path: Path, *, n_latent: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    cats = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(cats):
        mask = labels == cat
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            color=cmap(i),
            label=f"{cat} (n={int(mask.sum())})",
            alpha=0.7,
            linewidths=0,
        )
    ax.set_title(
        f"MultiVI pilot — {BATCH_KEY}\n"
        f"{MAX_EPOCHS} epochs (n={len(labels)}, d={n_latent})"
    )
    ax.legend(markerscale=2, frameon=False, loc="best")
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

    metrics = {
        "dataset": str(MDATA_PATH),
        "batch_key": BATCH_KEY,
        "max_epochs": MAX_EPOCHS,
        "seed": SEED,
        "n_neighbors": N_NEIGHBORS,
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
    )
    _plot_umap(emb, data_type, OUT_DIR / "multivi_pilot_umap.png", n_latent=latent.shape[1])
    (OUT_DIR / "multivi_pilot_metrics.json").write_text(json.dumps(metrics, indent=2))
    model.save(OUT_DIR / "model", overwrite=True)

    print(json.dumps(metrics, indent=2))
    logger.info("Wrote artifacts to %s", OUT_DIR)


if __name__ == "__main__":
    main()
