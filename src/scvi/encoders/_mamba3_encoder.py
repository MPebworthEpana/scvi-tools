"""Bidirectional Mamba3 ATAC encoder (Mamba3-only, no fallback)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from scvi.encoders._checkpointed import maybe_checkpoint

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba3
except ImportError as err:
    Mamba3 = None
    _MAMBA3_IMPORT_ERROR = err
else:
    _MAMBA3_IMPORT_ERROR = None


class _BiMamba3Layer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        headdim: int = 64,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        chunk_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "Mamba3 is required for MAMBAVAE. Install mamba-ssm from source "
                "(see https://github.com/state-spaces/mamba)."
            ) from _MAMBA3_IMPORT_ERROR
        block_kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba3(**block_kwargs)
        self.bwd = Mamba3(**block_kwargs)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        mamba_dtype = self.fwd.in_proj.weight.dtype
        h = h.to(dtype=mamba_dtype)
        out = torch.cat([self.fwd(h), self.bwd(h.flip(1)).flip(1)], dim=-1)
        out = out.to(dtype=x.dtype)
        return x + self.proj(out)


class BidirectionalMamba3Encoder(nn.Module):
    """Token sequence encoder using stacked bidirectional Mamba3 blocks."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        d_state: int = 128,
        headdim: int = 64,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        chunk_size: int = 16,
        n_heads: int = 4,
        dtype: torch.dtype = torch.bfloat16,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "Mamba3 is required for MAMBAVAE. Install mamba-ssm from source "
                "(see https://github.com/state-spaces/mamba)."
            ) from _MAMBA3_IMPORT_ERROR
        self.layers = nn.ModuleList(
            [
                _BiMamba3Layer(
                    d_model=d_model,
                    d_state=d_state,
                    headdim=headdim,
                    is_mimo=is_mimo,
                    mimo_rank=mimo_rank,
                    chunk_size=chunk_size,
                    dtype=dtype,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        from scvi.encoders._attention import AttentionPooling

        self.pool = AttentionPooling(d_model, n_heads)
        self.use_checkpoint = use_checkpoint
        self.boundary_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self._last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        chrom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = tokens * token_mask.unsqueeze(-1)
        if chrom is not None:
            prev = F.pad(chrom[:, :-1], (1, 0), value=-1)
            is_boundary = (chrom != prev) & token_mask
            x = x + is_boundary.unsqueeze(-1).float() * self.boundary_emb
        for layer in self.layers:
            x = maybe_checkpoint(layer, x, enabled=self.use_checkpoint)
        x = self.norm(x)
        pooled, token_weights = self.pool(x, token_mask)
        self._last_attn_weights = token_weights
        return pooled
