"""Parse ATAC peak names into a coordinate table."""

from __future__ import annotations

import re

import numpy as np

_CHROM_RE = re.compile(r"^(?:chr)?([0-9XYM]+|MT)[:\-]([0-9]+)[\-:]([0-9]+)")


def _extract_chrom(name: str) -> str | None:
    m = _CHROM_RE.match(str(name))
    if m is None:
        return None
    return m.group(1).upper()


def _canonical_chrom_id(chrom: str) -> int | None:
    chrom = chrom.upper()
    if chrom in ("M", "MT"):
        return 24
    if chrom == "X":
        return 22
    if chrom == "Y":
        return 23
    try:
        return int(chrom)
    except ValueError:
        return None


def _chrom_to_id(chrom: str) -> int:
    """Legacy single-contig id (non-canonical contigs use salted hash)."""
    chrom = chrom.upper()
    canonical = _canonical_chrom_id(chrom)
    if canonical is not None:
        return canonical
    return 25 + (hash(chrom) % 39)


def build_chrom_vocab(peak_names) -> dict[str, int]:
    """Build a deterministic, collision-free contig -> id vocabulary."""
    contigs: set[str] = set()
    for name in peak_names:
        chrom = _extract_chrom(name)
        if chrom is not None:
            contigs.add(chrom)

    vocab: dict[str, int] = {}
    next_id = 25
    for chrom in sorted(c for c in contigs if _canonical_chrom_id(c) is None):
        vocab[chrom] = next_id
        next_id += 1
    for chrom in contigs:
        if chrom not in vocab:
            cid = _canonical_chrom_id(chrom)
            if cid is not None:
                vocab[chrom] = cid
    return vocab


def parse_peak_name(name: str, vocab: dict[str, int] | None = None) -> tuple[int, int, int]:
    """Return ``(chrom_id, start, end)`` for a single peak name."""
    m = _CHROM_RE.match(str(name))
    if m is None:
        return (0, 0, 0)
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    chrom_u = chrom.upper()
    if vocab is not None:
        chrom_id = vocab.get(chrom_u, 0)
    else:
        chrom_id = _chrom_to_id(chrom)
    return (chrom_id, start, end)


def build_coord_table(peak_names, vocab: dict[str, int] | None = None) -> np.ndarray:
    """Build the ``(n_peaks, 3)`` int64 ``coord_table`` from peak names."""
    if vocab is None:
        vocab = build_chrom_vocab(peak_names)
    out = np.zeros((len(peak_names), 3), dtype=np.int64)
    for i, name in enumerate(peak_names):
        out[i] = parse_peak_name(name, vocab=vocab)
    return out


def build_genomic_rank(coord_table: np.ndarray) -> np.ndarray:
    """Return ``genomic_rank[p]`` = sort order of peak ``p`` by ``(chrom, start)``."""
    coord_table = np.asarray(coord_table, dtype=np.int64)
    order = np.lexsort((coord_table[:, 1], coord_table[:, 0]))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank
