"""MULTIVAE variant with a Set Transformer tokenized ATAC encoder."""

from __future__ import annotations

import torch

from scvi import REGISTRY_KEYS
from scvi.encoders._constants import ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY
from scvi.encoders._set_transformer_atac_variational import SetTransformerAtacVariationalEncoder
from scvi.module._mambavae import MAMBAVAE


class SETVAE(MAMBAVAE):
    """MAMBAVAE with a Set Transformer ATAC encoder instead of Mamba3."""

    def __init__(
        self,
        *args,
        coord_table=None,
        st_d_model: int = 128,
        st_n_layers: int = 2,
        st_n_inducing: int = 32,
        st_n_heads: int = 4,
        st_dropout: float = 0.0,
        use_cardinality_film: bool = True,
        use_sampling_correction: bool = False,
        **kwargs,
    ):
        self.st_d_model = st_d_model
        self.st_n_layers = st_n_layers
        self.st_n_inducing = st_n_inducing
        self.st_n_heads = st_n_heads
        self.st_dropout = st_dropout
        self.use_cardinality_film = use_cardinality_film
        self.use_sampling_correction = use_sampling_correction
        super().__init__(*args, coord_table=coord_table, **kwargs)
        if self.n_input_regions > 0:
            if coord_table is not None:
                coord_tensor = torch.as_tensor(coord_table, dtype=torch.long)
            else:
                coord_tensor = self.z_encoder_accessibility.coord_table
            encoder_cat_list = (
                list(self.n_cats_per_cov or []) if self.encode_covariates else None
            )
            self.z_encoder_accessibility = SetTransformerAtacVariationalEncoder(
                n_latent=self.n_latent,
                coord_table=coord_tensor,
                n_regions=self.n_input_regions,
                d_model=st_d_model,
                n_layers=st_n_layers,
                n_inducing=st_n_inducing,
                n_heads=st_n_heads,
                dropout=st_dropout,
                n_batch=self.n_batch,
                n_cat_list=encoder_cat_list,
                n_continuous_cov=self.n_continuous_cov,
                encode_covariates=self.encode_covariates,
                latent_distribution=self.latent_distribution,
                binarize=True,
                var_eps=0.0,
                use_cardinality_film=use_cardinality_film,
                use_sampling_correction=use_sampling_correction,
            )

    def _get_inference_input(self, tensors):
        """Support token-only batches without dense RNA/ATAC matrices."""
        x = tensors.get(REGISTRY_KEYS.X_KEY, None)
        x_atac = tensors.get(REGISTRY_KEYS.ATAC_X_KEY, None)
        if x is not None and x_atac is not None:
            x = torch.cat((x, x_atac), dim=-1)
        elif x is None:
            x = x_atac
        if x is None:
            ref = tensors.get(ATAC_TOKEN_IDS_KEY, tensors.get(REGISTRY_KEYS.BATCH_KEY))
            if not torch.is_tensor(ref):
                ref = torch.as_tensor(ref)
            batch_size = ref.shape[0]
            device = ref.device if torch.is_tensor(ref) else None
            width = max(self.n_input_genes, 0) + max(self.n_input_regions, 0)
            width = max(width, 1)
            x = torch.zeros(batch_size, width, device=device, requires_grad=False)
        if self.n_input_proteins == 0:
            y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cell_idx = tensors.get(REGISTRY_KEYS.INDICES_KEY).long().ravel()
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
        label = tensors[REGISTRY_KEYS.LABELS_KEY]
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        input_dict = {
            "x": x,
            "y": y,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "label": label,
            "cell_idx": cell_idx,
            "size_factor": size_factor,
        }
        if ATAC_TOKEN_IDS_KEY in tensors:
            input_dict["atac_token_ids"] = tensors[ATAC_TOKEN_IDS_KEY]
            input_dict["atac_token_mask"] = tensors[ATAC_TOKEN_MASK_KEY]
        return input_dict
