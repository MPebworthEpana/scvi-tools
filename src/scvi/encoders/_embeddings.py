"""ATAC peak embeddings from coordinates + hashed residual."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_encoding(pos: torch.Tensor, d: int, max_period: float = 1e9) -> torch.Tensor:
    half = d // 2
    device = pos.device
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / max(half, 1)
    )
    angles = pos.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < d:
        emb = torch.cat([emb, torch.zeros(*emb.shape[:-1], 1, device=device)], dim=-1)
    return emb


class ValueMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(torch.log1p(value.clamp(min=0)).unsqueeze(-1))


class HashedResidual(nn.Module):
    def __init__(self, d: int, n_buckets: int = 65536, k: int = 4):
        super().__init__()
        if d % k != 0:
            raise ValueError("d must be divisible by k")
        self.B, self.k = n_buckets, k
        self.table = nn.Embedding(n_buckets, d // k)
        self.register_buffer(
            "mult",
            torch.tensor(
                [2654435761, 40503, 2246822519, 3266489917, 668265263, 374761393, 3144134277, 1103515245][:k],
                dtype=torch.long,
            ),
        )
        self.register_buffer("off", torch.arange(1, k + 1, dtype=torch.long))

    def forward(self, peak_ids: torch.Tensor) -> torch.Tensor:
        pid = peak_ids.unsqueeze(-1)
        buckets = (pid * self.mult + self.off) % self.B
        looked = self.table(buckets)
        return looked.reshape(*peak_ids.shape, -1)


class AtacPeakEmbedding(nn.Module):
    def __init__(
        self,
        d: int,
        n_chrom: int = 65,
        n_length_bins: int = 16,
        hash_buckets: int = 65536,
        hash_k: int = 4,
        binarize_values: bool = True,
    ):
        super().__init__()
        self.d = d
        self.chrom_emb = nn.Embedding(n_chrom, d, padding_idx=0)
        self.length_emb = nn.Embedding(n_length_bins, d)
        self.n_length_bins = n_length_bins
        self.residual = HashedResidual(d, hash_buckets, hash_k)
        self.value_emb = ValueMLP(d)
        self.binarize_values = binarize_values

    def _length_bin(self, length: torch.Tensor) -> torch.Tensor:
        bins = torch.log2(length.clamp(min=1).float()).long()
        return bins.clamp(0, self.n_length_bins - 1)

    def forward(self, peak_ids, chrom, pos, length, value) -> torch.Tensor:
        if self.binarize_values:
            value = (value > 0).to(value.dtype)
        return (
            self.chrom_emb(chrom)
            + sinusoidal_encoding(pos, self.d)
            + self.length_emb(self._length_bin(length))
            + self.residual(peak_ids)
            + self.value_emb(value)
        )
