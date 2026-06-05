"""Attention pooling for ATAC sequence readout."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Learned query pools a sequence -> vector + per-token weights."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dh = n_heads, d_model // n_heads
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None):
        B, L, _ = x.shape
        q = self.q_proj(self.query.expand(B, 1, -1)).reshape(B, 1, self.h, self.dh).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.h, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)
        ctx = (attn @ v).transpose(1, 2).reshape(B, 1, self.h * self.dh)
        pooled = self.out(ctx)[:, 0]
        token_weights = attn[:, :, 0, :].mean(1)
        return pooled, token_weights
