"""MULTIVAE variant with a Set Transformer tokenized ATAC encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from scvi import REGISTRY_KEYS
from torch.distributions import kl_divergence as kld

from scvi.module._multivae import MULTIVAE, get_reconstruction_loss_protein, mix_modalities
from scvi.module.base import LossOutput, auto_move_data
from scvi.tokenized import (
    ATAC_TOKEN_IDS_KEY,
    ATAC_TOKEN_MASK_KEY,
    SetTransformerAtacVariationalEncoder,
)


class SETVAE(MULTIVAE):
    """MULTIVAE with a Set Transformer ATAC encoder and token-native ATAC data."""

    def __init__(
        self,
        *args,
        coord_table=None,
        max_atac_tokens: int = 8192,
        st_d_model: int = 128,
        st_n_layers: int = 2,
        st_n_inducing: int = 32,
        st_n_heads: int = 4,
        st_dropout: float = 0.0,
        use_cardinality_film: bool = False,
        use_sampling_correction: bool = False,
        use_peak_salience_prior: bool = True,
        peak_salience_cap: float = 5.0,
        **kwargs,
    ):
        self.max_atac_tokens = max_atac_tokens
        self.st_d_model = st_d_model
        self.st_n_layers = st_n_layers
        self.st_n_inducing = st_n_inducing
        self.st_n_heads = st_n_heads
        self.st_dropout = st_dropout
        self.use_cardinality_film = use_cardinality_film
        self.use_sampling_correction = use_sampling_correction
        self.use_peak_salience_prior = use_peak_salience_prior
        self.peak_salience_cap = peak_salience_cap
        self._coord_table = coord_table
        self._token_store = None
        # SETVI always learns per-peak region_factors; they are shared with the encoder
        # peak-salience prior (rarity weighting of attention), so force them on.
        kwargs["region_factors"] = True
        super().__init__(*args, **kwargs)
        if self.n_input_regions > 0:
            if coord_table is not None:
                coord_tensor = torch.as_tensor(coord_table, dtype=torch.long)
            else:
                raise ValueError("SETVAE requires coord_table when n_input_regions > 0.")
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

    def set_token_store(self, store) -> None:
        """Attach a GPU-tier token store for in-module gathering."""
        self._token_store = store

    def _peak_salience_bias(self, token_ids: torch.Tensor) -> torch.Tensor | None:
        """Per-token rarity prior derived from the (detached) region_factors.

        ``sigmoid(region_factors_j)`` approximates peak ``j``'s open-frequency ``f_j``, so
        ``softplus(-region_factors_j) = -log sigmoid(region_factors_j) ~= -log f_j`` is the
        peak's surprisal (IDF). Used as an additive attention-logit bias it makes pooling
        up-weight rare/discriminative peaks. The cap flattens the weight for extremely rare
        (likely-noise) peaks: a cap ``C`` corresponds to a frequency floor ``exp(-C)``.
        Detached so the encoder only reads region_factors (no encoder->rf feedback loop).
        """
        if not self.use_peak_salience_prior or self.region_factors is None:
            return None
        salience = torch.clamp(
            F.softplus(-self.region_factors.detach()), max=self.peak_salience_cap
        )
        return salience[token_ids]

    @staticmethod
    def _accessibility_target_from_tokens(
        ids: torch.Tensor, mask: torch.Tensor, n_regions: int
    ) -> torch.Tensor:
        target = torch.zeros(ids.shape[0], n_regions, device=ids.device, dtype=torch.float32)
        rows = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1).expand_as(ids)
        valid = mask & (ids >= 0) & (ids < n_regions)
        target[rows[valid], ids[valid]] = 1.0
        return target

    def _resolve_atac_tokens(
        self,
        tensors,
        *,
        for_encoder: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ATAC_TOKEN_IDS_KEY in tensors and for_encoder:
            ids = tensors[ATAC_TOKEN_IDS_KEY]
            mask = tensors[ATAC_TOKEN_MASK_KEY]
            if not torch.is_tensor(ids):
                ids = torch.as_tensor(ids, device=device, dtype=torch.long)
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
            return ids.long(), mask.bool()
        if self._token_store is not None and REGISTRY_KEYS.INDICES_KEY in tensors:
            cell_idx = tensors[REGISTRY_KEYS.INDICES_KEY].long().ravel()
            if self._token_store.tier == "gpu":
                return self._token_store.gather_torch(
                    cell_idx.cpu().numpy(),
                    device,
                    for_encoder=for_encoder,
                )
            ids, mask = self._token_store.gather(
                cell_idx.cpu().numpy(),
                for_encoder=for_encoder,
            )
            return (
                torch.as_tensor(ids, device=device, dtype=torch.long),
                torch.as_tensor(mask, device=device, dtype=torch.bool),
            )
        raise ValueError("SETVAE requires ATAC token tensors or an attached token store.")

    def _get_inference_input(self, tensors):
        """Support token-only batches without the wide ATAC matrix."""
        x = tensors.get(REGISTRY_KEYS.X_KEY, None)
        if x is None:
            ref = tensors.get(ATAC_TOKEN_IDS_KEY, tensors.get(REGISTRY_KEYS.BATCH_KEY))
            if not torch.is_tensor(ref):
                ref = torch.as_tensor(ref)
            batch_size = ref.shape[0]
            device = ref.device if torch.is_tensor(ref) else None
            width = max(self.n_input_genes, 1)
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
        if ATAC_TOKEN_IDS_KEY in tensors or (
            self._token_store is not None and REGISTRY_KEYS.INDICES_KEY in tensors
        ):
            device = x.device
            enc_ids, enc_mask = self._resolve_atac_tokens(
                tensors, for_encoder=True, device=device
            )
            input_dict["atac_token_ids"] = enc_ids
            input_dict["atac_token_mask"] = enc_mask
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

        if atac_token_mask is not None:
            mask_acc = atac_token_mask.any(dim=1)
        elif self.n_input_regions > 0:
            mask_acc = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        else:
            mask_acc = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        mask_expr = x_rna.sum(dim=1) > 0
        mask_pro = y.sum(dim=1) > 0

        if cont_covs is not None and self.encode_covariates:
            encoder_input_expression = torch.cat((x_rna, cont_covs), dim=-1)
            encoder_input_protein = torch.cat((y, cont_covs), dim=-1)
        else:
            encoder_input_expression = x_rna
            encoder_input_protein = y

        if cat_covs is not None and self.encode_covariates:
            categorical_input = tuple(torch.split(cat_covs, 1, dim=1))
        else:
            categorical_input = ()

        if self.n_input_regions > 0:
            if atac_token_ids is None or atac_token_mask is None:
                raise ValueError(
                    "SETVAE requires ATAC token tensors. Use SetAnnDataLoader / SetDataSplitter."
                )
            peak_logit_bias = self._peak_salience_bias(atac_token_ids.long())
            qzm_acc, qzv_acc, z_acc = self.z_encoder_accessibility(
                atac_token_ids.long(),
                atac_token_mask.bool(),
                batch_index,
                *categorical_input,
                cont_covs=cont_covs if self.encode_covariates else None,
                peak_logit_bias=peak_logit_bias,
            )
        else:
            qzm_acc = torch.zeros(x.shape[0], self.n_latent, device=x.device)
            qzv_acc = torch.ones(x.shape[0], self.n_latent, device=x.device)
            z_acc = qzm_acc

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
            if self.n_input_regions > 0 and atac_token_mask is not None:
                libsize_acc = atac_token_mask.sum(dim=1, keepdim=True).float() / float(
                    self.max_atac_tokens
                )
            else:
                libsize_acc = torch.zeros(x.shape[0], 1, device=x.device)

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
            "atac_token_mask": atac_token_mask,
            "mask_acc": mask_acc,
            "mask_expr": mask_expr,
            "mask_pro": mask_pro,
        }

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """Compute loss using token-reconstructed accessibility targets."""
        x = inference_outputs["x"]
        x_rna = x[:, : self.n_input_genes]
        if self.n_input_regions > 0:
            tgt_ids, tgt_mask = self._resolve_atac_tokens(
                tensors,
                for_encoder=False,
                device=generative_outputs["p"].device,
            )
            x_atac = self._accessibility_target_from_tokens(
                tgt_ids.long(),
                tgt_mask.bool(),
                self.n_input_regions,
            )
        else:
            x_atac = torch.zeros(
                x.shape[0],
                1,
                device=x.device,
                dtype=torch.float32,
                requires_grad=False,
            )
        if self.n_input_proteins == 0:
            y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        mask_expr = inference_outputs["mask_expr"]
        mask_acc = inference_outputs["mask_acc"]
        mask_pro = inference_outputs["mask_pro"]

        p = generative_outputs["p"]
        libsize_acc = inference_outputs["libsize_acc"]
        rl_accessibility = self.get_reconstruction_loss_accessibility(x_atac, p, libsize_acc)

        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        rl_expression = self.get_reconstruction_loss_expression(
            x_rna, px_rate, px_r, px_dropout
        )

        if mask_pro.sum().gt(0):
            py_ = generative_outputs["py_"]
            rl_protein = get_reconstruction_loss_protein(y, py_, None)
        else:
            rl_protein = torch.zeros(x.shape[0], device=x.device, requires_grad=False)

        recon_loss_expression = rl_expression * mask_expr
        recon_loss_accessibility = rl_accessibility * mask_acc
        recon_loss_protein = rl_protein * mask_pro
        recon_loss = recon_loss_expression + recon_loss_accessibility + recon_loss_protein

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        kl_div_z = kld(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(0, 1),
        ).sum(dim=1)

        kl_div_paired = self._compute_mod_penalty(
            (inference_outputs["qzm_expr"], inference_outputs["qzv_expr"]),
            (inference_outputs["qzm_acc"], inference_outputs["qzv_acc"]),
            (inference_outputs["qzm_pro"], inference_outputs["qzv_pro"]),
            mask_expr,
            mask_acc,
            mask_pro,
        )

        kl_local_for_warmup = kl_div_z
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_div_paired
        loss = torch.mean(recon_loss + weighted_kl_local)

        recon_losses = {
            "reconstruction_loss_expression": recon_loss_expression,
            "reconstruction_loss_accessibility": recon_loss_accessibility,
            "reconstruction_loss_protein": recon_loss_protein,
        }
        kl_local = {
            "kl_divergence_z": kl_div_z,
            "kl_divergence_paired": kl_div_paired,
        }

        if self.extra_payload_autotune:
            extra_metrics_payload = {
                "z": inference_outputs["z"],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
            }
        else:
            extra_metrics_payload = {}

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_losses,
            kl_local=kl_local,
            extra_metrics=extra_metrics_payload,
        )
