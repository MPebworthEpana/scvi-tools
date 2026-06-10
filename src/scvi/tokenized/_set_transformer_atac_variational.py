"""Variational encoder facade for Set Transformer ATAC encoding."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from scvi.nn._base_components import _identity
from scvi.tokenized._embeddings import AtacPeakEmbedding
from scvi.tokenized._set_transformer import SetTransformerEncoder


class SetTransformerAtacVariationalEncoder(nn.Module):
    """Encode padded ATAC peak tokens with a Set Transformer into a variational latent."""

    def __init__(
        self,
        n_latent: int,
        coord_table: torch.Tensor,
        n_regions: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_inducing: int = 32,
        n_heads: int = 4,
        dropout: float = 0.0,
        n_batch: int = 0,
        n_cat_list: Iterable[int] | None = None,
        n_continuous_cov: int = 0,
        encode_covariates: bool = False,
        latent_distribution: str = "normal",
        var_eps: float = 1e-4,
        binarize: bool = True,
        use_cardinality_film: bool = True,
        use_sampling_correction: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latent = n_latent
        self.n_regions = n_regions
        self.encode_covariates = encode_covariates
        self.latent_distribution = latent_distribution
        self.var_eps = var_eps
        extra_cats = list(n_cat_list or [])
        if n_batch > 0 and (not extra_cats or extra_cats[0] != n_batch):
            self.n_cat_list = [n_batch] + extra_cats
        else:
            self.n_cat_list = extra_cats

        n_chrom = max(int(coord_table[:, 0].max().item()) + 1, 25)
        self.embedding = AtacPeakEmbedding(
            d_model,
            n_chrom=n_chrom,
            binarize_values=binarize,
            value_as_bias=True,
        )
        self.embedding.build_static_cache(coord_table.long())
        self.backbone = SetTransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_inducing=n_inducing,
            n_heads=n_heads,
            dropout=dropout,
            use_cardinality_film=use_cardinality_film,
            use_sampling_correction=use_sampling_correction,
        )
        self.head_norm = nn.LayerNorm(d_model)

        cov_dim = 0
        if encode_covariates:
            cov_dim = sum(c for c in self.n_cat_list if c > 1) + n_continuous_cov
        self.cov_dim = cov_dim
        head_in = d_model + cov_dim
        self.mean_encoder = nn.Linear(head_in, n_latent)
        self.var_encoder = nn.Linear(head_in, n_latent)
        if latent_distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity

    def _encode_covariates(
        self,
        pooled: torch.Tensor,
        batch_index: torch.Tensor,
        cat_list: tuple[torch.Tensor, ...],
        cont_covs: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.encode_covariates:
            return pooled
        cov_tensors = []
        if cont_covs is not None:
            cov_tensors.append(cont_covs)
        for n_cat, cat in zip(self.n_cat_list, (batch_index, *cat_list), strict=False):
            if n_cat <= 1:
                continue
            if cat.size(1) != n_cat:
                cov_tensors.append(F.one_hot(cat.squeeze(-1).long(), n_cat).float())
            else:
                cov_tensors.append(cat.float())
        if not cov_tensors:
            return pooled
        return torch.cat([pooled, *cov_tensors], dim=-1)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        batch_index: torch.Tensor,
        *cat_list: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        peak_logit_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = token_mask.to(token_ids.dtype)
        tokens = self.embedding.embed_ids(token_ids, values)
        pooled = self.backbone(
            tokens,
            key_mask=token_mask.bool(),
            n_total=self.n_regions,
            peak_logit_bias=peak_logit_bias,
        )
        pooled = self.head_norm(pooled)
        h = self._encode_covariates(pooled, batch_index, cat_list, cont_covs)
        q_m = self.mean_encoder(h)
        log_v = self.var_encoder(h).clamp(-10.0, 10.0)
        q_v = torch.exp(log_v) + max(self.var_eps, 1e-4)
        z = self.z_transformation(Normal(q_m, q_v.sqrt()).rsample())
        return q_m, q_v, z
