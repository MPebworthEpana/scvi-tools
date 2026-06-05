"""Parse ATAC peak names into a coordinate table."""

from __future__ import annotations

import re

import numpy as np

_CHROM_RE = re.compile(r"^(?:chr)?([0-9XYM]+|MT)[:\-]([0-9]+)[\-:]([0-9]+)")


def _chrom_to_id(chrom: str) -> int:
    chrom = chrom.upper()
    if chrom in ("M", "MT"):
        return 24
    if chrom == "X":
        return 22
    if chrom == "Y":
        return 23
    try:
        n = int(chrom)
    except ValueError:
        return 25 + (hash(chrom) % 39)
    return n


def parse_peak_name(name: str) -> tuple[int, int, int]:
    """Return ``(chrom_id, start, end)`` for a single peak name."""
    m = _CHROM_RE.match(str(name))
    if m is None:
        return (0, 0, 0)
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    return (_chrom_to_id(chrom), start, end)


def build_coord_table(peak_names) -> np.ndarray:
    """Build the ``(n_peaks, 3)`` int64 ``coord_table`` from peak names."""
    out = np.zeros((len(peak_names), 3), dtype=np.int64)
    for i, name in enumerate(peak_names):
        out[i] = parse_peak_name(name)
    return out


def build_genomic_rank(coord_table: np.ndarray) -> np.ndarray:
    """Return ``genomic_rank[p]`` = sort order of peak ``p`` by ``(chrom, start)``."""
    coord_table = np.asarray(coord_table, dtype=np.int64)
    order = np.lexsort((coord_table[:, 1], coord_table[:, 0]))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank
