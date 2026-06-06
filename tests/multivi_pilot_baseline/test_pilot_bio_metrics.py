"""Tests for MultiVI pilot bio-conservation metric helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_BASELINE_PATH = Path(__file__).resolve().parent / "multivi_baseline.py"
_spec = importlib.util.spec_from_file_location("multivi_baseline", _BASELINE_PATH)
_baseline = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_baseline)


def test_bio_conservation_metrics_finite_and_bounded():
    rng = np.random.default_rng(0)
    n_cells = 40
    latent = rng.normal(size=(n_cells, 8)).astype(np.float32)
    nbr = _baseline._neighbors(latent, n_neighbors=5, seed=0)
    leiden = _baseline._leiden_clusters(nbr, resolution=1.0, seed=0)
    pseudo = np.array([str(i % 4) for i in range(n_cells)])

    metrics = _baseline.bio_conservation_metrics(latent, nbr, leiden, pseudo)

    assert metrics["n_leiden_clusters"] >= 1
    assert metrics["n_rna_pseudo_clusters"] == 4
    assert 0.0 <= metrics["nmi_leiden_rna_pseudo"] <= 1.0
    assert -1.0 <= metrics["ari_leiden_rna_pseudo"] <= 1.0
    assert np.isfinite(metrics["clisi_rna_pseudo"])
    assert np.isfinite(metrics["silhouette_rna_pseudo"])


def test_transfer_pseudo_labels_fills_atac_only():
    rng = np.random.default_rng(1)
    n_cells = 20
    latent = rng.normal(size=(n_cells, 4)).astype(np.float32)
    nbr = _baseline._neighbors(latent, n_neighbors=5, seed=0)
    rna_present = np.array([True] * 12 + [False] * 8)
    pseudo = np.array([str(i % 3) for i in range(12)] + [""] * 8, dtype=str)

    filled = _baseline._transfer_pseudo_labels(latent, nbr, pseudo, rna_present, seed=0)

    assert filled[:12].tolist() == pseudo[:12].tolist()
    assert all(filled[12:])
    assert len(set(filled[12:])) >= 1
