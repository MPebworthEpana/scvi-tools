"""Set Transformer blocks for permutation-invariant set encoding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RowwiseFF(nn.Module):
    """Row-wise feed-forward block applied to each set element."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiheadAttentionBlock(nn.Module):
    """MAB(X, Y): equivariant in X, invariant to permutations of Y."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.rff = RowwiseFF(d_model, dropout=dropout)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n_q, _ = q.shape
        n_k = k.shape[1]
        qh = self.q_proj(q).view(b, n_q, self.n_heads, self.d_head).transpose(1, 2)
        kh = self.k_proj(k).view(b, n_k, self.n_heads, self.d_head).transpose(1, 2)
        vh = self.v_proj(v).view(b, n_k, self.n_heads, self.d_head).transpose(1, 2)

        attn_mask = None
        if key_mask is not None or logit_bias is not None:
            attn_mask = torch.zeros(b, n_k, device=q.device, dtype=qh.dtype)
            if logit_bias is not None:
                attn_mask = attn_mask + logit_bias
            if key_mask is not None:
                attn_mask = attn_mask.masked_fill(~key_mask, float("-inf"))
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

        dropout_p = self.dropout.p if self.training else 0.0
        ctx = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        ctx = torch.nan_to_num(ctx)
        ctx = ctx.transpose(1, 2).reshape(b, n_q, self.d_model)
        return self.out_proj(ctx)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x + self._attend(x, y, y, key_mask=key_mask, logit_bias=logit_bias))
        return self.norm2(h + self.rff(h))


class SetAttentionBlock(nn.Module):
    """SAB(X) = MAB(X, X)."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d_model, n_heads=n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.mab(x, x, key_mask=key_mask)


class InducedSetAttentionBlock(nn.Module):
    """ISAB_m(X): inducing-point bottleneck, cost O(n m)."""

    def __init__(
        self,
        d_model: int,
        n_inducing: int = 32,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(n_inducing, d_model) * 0.02)
        self.mab1 = MultiheadAttentionBlock(d_model, n_heads=n_heads, dropout=dropout)
        self.mab2 = MultiheadAttentionBlock(d_model, n_heads=n_heads, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        inducing = self.inducing.unsqueeze(0).expand(b, -1, -1)
        h = self.mab1(inducing, x, key_mask=key_mask, logit_bias=logit_bias)
        return self.mab2(x, h)


class PoolingByMultiheadAttention(nn.Module):
    """PMA_k(Z): permutation-invariant pooling with k learnable seed queries."""

    def __init__(
        self,
        d_model: int,
        n_seeds: int = 1,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_sampling_correction: bool = False,
    ):
        super().__init__()
        self.n_seeds = n_seeds
        self.use_sampling_correction = use_sampling_correction
        self.seeds = nn.Parameter(torch.randn(n_seeds, d_model) * 0.02)
        self.rff = RowwiseFF(d_model, dropout=dropout)
        self.mab = MultiheadAttentionBlock(d_model, n_heads=n_heads, dropout=dropout)

    def forward(
        self,
        z: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        inclusion_prob: torch.Tensor | None = None,
        peak_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = z.shape[0]
        z_ff = self.rff(z)
        seeds = self.seeds.unsqueeze(0).expand(b, -1, -1)
        logit_bias = None
        if self.use_sampling_correction and inclusion_prob is not None:
            log_pi = torch.log(inclusion_prob.clamp(min=1e-8))
            logit_bias = -log_pi
        if peak_logit_bias is not None:
            logit_bias = (
                peak_logit_bias if logit_bias is None else logit_bias + peak_logit_bias
            )
        pooled = self.mab(seeds, z_ff, key_mask=key_mask, logit_bias=logit_bias)
        if self.n_seeds == 1:
            return pooled[:, 0]
        return pooled


class CardinalityFiLM(nn.Module):
    """FiLM modulation from observation cardinality (n, N, coverage)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )
        # Near-identity modulation at init: (1 + small) * z + small
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def _cardinality_features(
        self,
        n_obs: torch.Tensor,
        n_total: torch.Tensor,
    ) -> torch.Tensor:
        n = n_obs.float().clamp(min=1.0)
        n_total_f = n_total.float().clamp(min=1.0)
        coverage = n / n_total_f
        log_n = torch.log(n)
        log_n_total = torch.log(n_total_f)
        log_ratio = torch.log(n_total_f / n)
        return torch.cat([n, n_total_f, coverage, log_n, log_n_total, log_ratio], dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        n_obs: torch.Tensor,
        n_total: torch.Tensor,
    ) -> torch.Tensor:
        u = self._cardinality_features(n_obs, n_total)
        gamma_beta = self.mlp(u)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma) * z + beta


class SetTransformerEncoder(nn.Module):
    """Stack of ISAB blocks followed by PMA pooling and optional cardinality FiLM."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_inducing: int = 32,
        n_heads: int = 4,
        n_seeds: int = 1,
        dropout: float = 0.0,
        use_cardinality_film: bool = True,
        use_sampling_correction: bool = False,
    ):
        super().__init__()
        self.use_cardinality_film = use_cardinality_film
        self.layers = nn.ModuleList(
            [
                InducedSetAttentionBlock(
                    d_model,
                    n_inducing=n_inducing,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.pool = PoolingByMultiheadAttention(
            d_model,
            n_seeds=n_seeds,
            n_heads=n_heads,
            dropout=dropout,
            use_sampling_correction=use_sampling_correction,
        )
        self.cardinality = CardinalityFiLM(d_model) if use_cardinality_film else None

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        n_total: int | None = None,
        peak_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, key_mask=key_mask, logit_bias=peak_logit_bias)
        n_obs = None
        inclusion_prob = None
        if key_mask is not None and n_total is not None and n_total > 0:
            n_obs = key_mask.sum(dim=1, keepdim=True).float()
            inclusion_prob = n_obs / float(n_total)
            inclusion_prob = inclusion_prob.expand(-1, key_mask.shape[1])
            inclusion_prob = inclusion_prob * key_mask.float()
        pooled = self.pool(
            h,
            key_mask=key_mask,
            inclusion_prob=inclusion_prob,
            peak_logit_bias=peak_logit_bias,
        )
        if self.cardinality is not None and n_obs is not None and n_total is not None:
            n_total_t = torch.full_like(n_obs, float(n_total))
            pooled = self.cardinality(pooled, n_obs, n_total_t)
        return pooled
