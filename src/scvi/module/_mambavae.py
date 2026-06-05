"""MULTIVAE variant with a Mamba3 tokenized ATAC encoder."""

from __future__ import annotations

import torch
from torch.distributions import Normal

from scvi import REGISTRY_KEYS
from scvi.encoders._constants import ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY
from scvi.encoders._mamba_atac_variational import MambaAtacVariationalEncoder
from scvi.module._multivae import MULTIVAE, mix_modalities
from scvi.module.base._decorators import auto_move_data


class MAMBAVAE(MULTIVAE):
    """MULTIVAE with Mamba3 ATAC encoder; decoders and losses unchanged."""

    def __init__(
        self,
        *args,
        coord_table=None,
        max_atac_tokens: int = 8192,
        atac_genomic_sort: bool = True,
        mamba_d_model: int = 128,
        mamba_n_layers: int = 4,
        mamba_use_checkpoint: bool = False,
        mamba3_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_atac_tokens = max_atac_tokens
        self.atac_genomic_sort = atac_genomic_sort
        if self.n_input_regions > 0:
            if coord_table is None:
                raise ValueError("coord_table is required for MAMBAVAE when ATAC is enabled.")
            cat_list = [self.n_batch] + list(self.n_cats_per_cov or [])
            encoder_cat_list = cat_list if self.encode_covariates else None
            self.z_encoder_accessibility = MambaAtacVariationalEncoder(
                n_latent=self.n_latent,
                coord_table=torch.as_tensor(coord_table, dtype=torch.long),
                d_model=mamba_d_model,
                n_layers=mamba_n_layers,
                n_batch=self.n_batch,
                n_cat_list=encoder_cat_list,
                n_continuous_cov=self.n_continuous_cov,
                encode_covariates=self.encode_covariates,
                latent_distribution=self.latent_distribution,
                binarize=True,
                use_checkpoint=mamba_use_checkpoint,
                mamba3_kwargs=mamba3_kwargs,
            )

    def _get_inference_input(self, tensors):
        input_dict = super()._get_inference_input(tensors)
        if ATAC_TOKEN_IDS_KEY in tensors:
            input_dict["atac_token_ids"] = tensors[ATAC_TOKEN_IDS_KEY]
            input_dict["atac_token_mask"] = tensors[ATAC_TOKEN_MASK_KEY]
        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        y,
        batch_index,
        cont_covs,
        cat_covs,
        label,
        cell_idx,
        size_factor,
        atac_token_ids=None,
        atac_token_mask=None,
        n_samples=1,
    ) -> dict[str, torch.Tensor]:
        if self.n_input_genes == 0:
            x_rna = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            x_rna = x[:, : self.n_input_genes]
        if self.n_input_regions == 0:
            x_atac = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]

        if atac_token_mask is not None:
            mask_acc = atac_token_mask.any(dim=1)
        else:
            mask_acc = x_atac.sum(dim=1) > 0
        mask_expr = x_rna.sum(dim=1) > 0
        mask_pro = y.sum(dim=1) > 0

        if cont_covs is not None and self.encode_covariates:
            encoder_input_expression = torch.cat((x_rna, cont_covs), dim=-1)
            encoder_input_accessibility = torch.cat((x_atac, cont_covs), dim=-1)
            encoder_input_protein = torch.cat((y, cont_covs), dim=-1)
        else:
            encoder_input_expression = x_rna
            encoder_input_accessibility = x_atac
            encoder_input_protein = y

        if cat_covs is not None and self.encode_covariates:
            categorical_input = tuple(torch.split(cat_covs, 1, dim=1))
        else:
            categorical_input = ()

        if atac_token_ids is None or atac_token_mask is None:
            raise ValueError(
                "MAMBAVAE requires ATAC token tensors. Use MambaAnnDataLoader / MambaDataSplitter."
            )
        qzm_acc, qzv_acc, z_acc = self.z_encoder_accessibility(
            atac_token_ids.long(),
            atac_token_mask.bool(),
            batch_index,
            *categorical_input,
            cont_covs=cont_covs if self.encode_covariates else None,
        )
        qzm_expr, qzv_expr, z_expr = self.z_encoder_expression(
            encoder_input_expression, batch_index, *categorical_input
        )
        qzm_pro, qzv_pro, z_pro = self.z_encoder_protein(
            encoder_input_protein, batch_index, *categorical_input
        )

        if self.use_size_factor_key:
            libsize_expr = torch.log(size_factor[:, [0]] + 1e-6)
            libsize_acc = size_factor[:, [1]]
        else:
            libsize_expr = self.l_encoder_expression(
                encoder_input_expression, batch_index, *categorical_input
            )
            libsize_acc = self.l_encoder_accessibility(
                encoder_input_accessibility, batch_index, *categorical_input
            )

        if self.modality_weights == "cell":
            weights = self.mod_weights[cell_idx, :]
        else:
            weights = self.mod_weights.unsqueeze(0).expand(len(cell_idx), -1)

        qz_m = mix_modalities(
            (qzm_expr, qzm_acc, qzm_pro), (mask_expr, mask_acc, mask_pro), weights
        )
        qz_v = mix_modalities(
            (qzv_expr, qzv_acc, qzv_pro),
            (mask_expr, mask_acc, mask_pro),
            weights,
            torch.sqrt,
        )

        if n_samples > 1:

            def unsqz(zt, n_s):
                return zt.unsqueeze(0).expand((n_s, zt.size(0), zt.size(1)))

            untran_za = Normal(qzm_acc, qzv_acc.sqrt()).sample((n_samples,))
            z_acc = self.z_encoder_accessibility.z_transformation(untran_za)
            untran_ze = Normal(qzm_expr, qzv_expr.sqrt()).sample((n_samples,))
            z_expr = self.z_encoder_expression.z_transformation(untran_ze)
            untran_zp = Normal(qzm_pro, qzv_pro.sqrt()).sample((n_samples,))
            z_pro = self.z_encoder_protein.z_transformation(untran_zp)

            libsize_expr = unsqz(libsize_expr, n_samples)
            libsize_acc = unsqz(libsize_acc, n_samples)

        untran_z = Normal(qz_m, qz_v.sqrt()).rsample()
        z = self.z_encoder_accessibility.z_transformation(untran_z)

        return {
            "x": x,
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "z_expr": z_expr,
            "qzm_expr": qzm_expr,
            "qzv_expr": qzv_expr,
            "z_acc": z_acc,
            "qzm_acc": qzm_acc,
            "qzv_acc": qzv_acc,
            "z_pro": z_pro,
            "qzm_pro": qzm_pro,
            "qzv_pro": qzv_pro,
            "libsize_expr": libsize_expr,
            "libsize_acc": libsize_acc,
        }
