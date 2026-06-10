"""MULTIVAE variant with a Mamba3 tokenized ATAC encoder."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kld

from scvi import REGISTRY_KEYS
from scvi.encoders._constants import ATAC_TOKEN_IDS_KEY, ATAC_TOKEN_MASK_KEY
from scvi.encoders._mamba_atac_variational import MambaAtacVariationalEncoder
from scvi.module._mamba_atac_decoder import MambaAtacDecoder
from scvi.module._multivae import MULTIVAE, get_reconstruction_loss_protein, mix_modalities
from scvi.module.base import LossOutput
from scvi.module.base._decorators import auto_move_data
from scvi.nn._contrastive import augmented_view_dcl, symmetric_dcl
from scvi.nn._optimal_transport import multiome_anchored_ot_loss, multiome_pseudo_dcl_loss

ReconstructionNormalizationMode = Literal[
    "none", "per_feature", "balanced", "ema", "uncertainty"
]
UnpairedAlignmentMode = Literal["ot_bridge", "pseudo_dcl"]
PseudoNegativeMode = Literal["same_mod", "same_plus_cross"]
TrainingPhase = Literal["unimodal", "align_warmup", "joint"]
UnimodalPretrainModality = Literal["rna", "atac"]
AtacLossMode = Literal["dense", "balanced_subsample"]
_TRAINING_PHASE_CODE = {"unimodal": 0.0, "align_warmup": 1.0, "joint": 2.0}


def sample_balanced_atac_loss_indices(
    atac_token_ids: torch.Tensor,
    atac_token_mask: torch.Tensor,
    n_regions: int,
    negatives_per_positive: int = 1,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Build padded positive/negative peak ids and binary targets for subsampled ATAC BCE."""
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be >= 1.")
    device = atac_token_ids.device
    batch_size, max_tokens = atac_token_ids.shape
    max_pos = max_tokens
    max_samples = max_pos * (1 + negatives_per_positive)

    peak_ids = torch.full((batch_size, max_samples), -1, dtype=torch.long, device=device)
    targets = torch.zeros((batch_size, max_samples), dtype=torch.float32, device=device)
    sample_mask = torch.zeros((batch_size, max_samples), dtype=torch.bool, device=device)

    pos_slot = atac_token_mask.long().cumsum(dim=1) - 1
    pos_rows, pos_cols = atac_token_mask.nonzero(as_tuple=True)
    pos_slots = pos_slot[pos_rows, pos_cols]
    peak_ids[pos_rows, pos_slots] = atac_token_ids[pos_rows, pos_cols].long()
    targets[pos_rows, pos_slots] = 1.0
    sample_mask[pos_rows, pos_slots] = True

    n_pos_per_row = atac_token_mask.sum(dim=1).long()
    desired_neg = n_pos_per_row * negatives_per_positive
    max_neg = max_pos * negatives_per_positive

    pos_ids = atac_token_ids.long().clamp(min=0, max=n_regions - 1)
    pos_one_hot = torch.zeros(
        batch_size, n_regions, dtype=torch.bool, device=device
    )
    pos_one_hot.scatter_(1, pos_ids, atac_token_mask)
    available = ~pos_one_hot

    rand_kwargs: dict = {"device": device}
    if generator is None or generator.device.type == device.type:
        rand_kwargs["generator"] = generator
    keys = torch.rand((batch_size, n_regions), **rand_kwargs)
    keys = keys.masked_fill(~available, 2.0)

    neg_vals, neg_idx = keys.topk(max_neg, dim=1, largest=False)
    neg_valid = neg_vals < 1.0
    neg_rank = neg_valid.long().cumsum(dim=1) - 1
    neg_sel = neg_valid & (neg_rank < desired_neg.unsqueeze(1))

    neg_rows, neg_cols = neg_sel.nonzero(as_tuple=True)
    neg_slots = max_pos + neg_rank[neg_rows, neg_cols]
    peak_ids[neg_rows, neg_slots] = neg_idx[neg_rows, neg_cols]
    sample_mask[neg_rows, neg_slots] = True

    return {
        "loss_peak_ids": peak_ids,
        "loss_targets": targets,
        "loss_sample_mask": sample_mask,
    }


def _resolve_reconstruction_normalization(
    reconstruction_normalization: ReconstructionNormalizationMode,
    normalize_reconstruction_loss: bool,
) -> ReconstructionNormalizationMode:
    if normalize_reconstruction_loss:
        return "per_feature"
    return reconstruction_normalization


class MAMBAVAE(MULTIVAE):
    """MULTIVAE with Mamba3 ATAC encoder and optional Mamba ATAC decoder."""

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
        atac_pooling_mode: str = "mlp_only",
        atac_decoder_module: Literal["peakvi", "mamba"] = "peakvi",
        atac_decoder_output_mode: Literal["deterministic", "variational_bernoulli"] = (
            "deterministic"
        ),
        atac_decoder_kl_weight: float = 1.0,
        atac_loss_mode: AtacLossMode = "dense",
        atac_negatives_per_positive: int = 1,
        atac_subsample_rescale: bool = True,
        mamba_decoder_kwargs: dict | None = None,
        adversarial_data_type: bool = False,
        lambda_dcl: float = 1.0,
        dcl_temperature: float = 0.1,
        contrast_proj_dim: int = 64,
        lambda_mu_anchor: float = 0.5,
        lambda_ot: float = 0.2,
        sinkhorn_eps: float = 0.05,
        bridge_k: int = 5,
        bridge_tau: float = 0.1,
        lambda_bridge_inner: float = 0.01,
        lambda_aug: float = 0.1,
        aug_noise_std: float = 0.01,
        bridge_detach_gallery: bool = True,
        use_projection_memory_bank: bool = True,
        memory_bank_size: int = 4096,
        combine_adversarial_vertical: bool = False,
        n_epochs_vertical_warmup: int = 10,
        unpaired_alignment_mode: UnpairedAlignmentMode = "ot_bridge",
        lambda_pseudo_dcl: float = 0.0,
        pseudo_negative_mode: PseudoNegativeMode = "same_mod",
        normalize_reconstruction_loss: bool = False,
        reconstruction_normalization: ReconstructionNormalizationMode = "none",
        recon_ema_beta: float = 0.99,
        max_recon_weight_ratio: float | None = 10.0,
        log_var_clamp: float = 5.0,
        **kwargs,
    ):
        self.normalize_reconstruction_loss = normalize_reconstruction_loss
        self.reconstruction_normalization = _resolve_reconstruction_normalization(
            reconstruction_normalization, normalize_reconstruction_loss
        )
        self.recon_ema_beta = recon_ema_beta
        self.max_recon_weight_ratio = max_recon_weight_ratio
        self.log_var_clamp = log_var_clamp
        self.adversarial_data_type = adversarial_data_type
        self.combine_adversarial_vertical = combine_adversarial_vertical
        self.lambda_dcl = lambda_dcl
        self.dcl_temperature = dcl_temperature
        self.contrast_proj_dim = contrast_proj_dim
        self.lambda_mu_anchor = lambda_mu_anchor
        self.lambda_ot = lambda_ot
        self.sinkhorn_eps = sinkhorn_eps
        self.bridge_k = bridge_k
        self.bridge_tau = bridge_tau
        self.lambda_bridge_inner = lambda_bridge_inner
        self.lambda_aug = lambda_aug
        self.aug_noise_std = aug_noise_std
        self.bridge_detach_gallery = bridge_detach_gallery
        self.use_projection_memory_bank = use_projection_memory_bank
        self.memory_bank_size = memory_bank_size
        self.n_epochs_vertical_warmup = n_epochs_vertical_warmup
        self.vertical_warmup_scale = 1.0
        if unpaired_alignment_mode not in ("ot_bridge", "pseudo_dcl"):
            raise ValueError(
                "unpaired_alignment_mode must be 'ot_bridge' or 'pseudo_dcl', "
                f"got {unpaired_alignment_mode!r}."
            )
        if pseudo_negative_mode not in ("same_mod", "same_plus_cross"):
            raise ValueError(
                "pseudo_negative_mode must be 'same_mod' or 'same_plus_cross', "
                f"got {pseudo_negative_mode!r}."
            )
        self.unpaired_alignment_mode = unpaired_alignment_mode
        self.lambda_pseudo_dcl = lambda_pseudo_dcl
        self.pseudo_negative_mode = pseudo_negative_mode
        self.pretrain_epochs = 0
        self.training_phase: TrainingPhase = "joint"
        self.unimodal_pretrain_active = False
        self.unimodal_pretrain_modality: UnimodalPretrainModality = "atac"
        self.alignment_warmup_active = False
        self.atac_decoder_module = atac_decoder_module
        self.atac_decoder_output_mode = atac_decoder_output_mode
        self.atac_decoder_kl_weight = atac_decoder_kl_weight
        self.atac_pooling_mode = atac_pooling_mode
        if atac_decoder_output_mode == "variational_bernoulli" and atac_decoder_module != "mamba":
            raise ValueError(
                "atac_decoder_output_mode='variational_bernoulli' requires atac_decoder_module='mamba'."
            )
        if atac_loss_mode not in ("dense", "balanced_subsample"):
            raise ValueError(
                "atac_loss_mode must be 'dense' or 'balanced_subsample', "
                f"got {atac_loss_mode!r}."
            )
        if atac_loss_mode == "balanced_subsample" and atac_decoder_module != "peakvi":
            raise ValueError(
                "atac_loss_mode='balanced_subsample' requires atac_decoder_module='peakvi'."
            )
        if atac_negatives_per_positive < 1:
            raise ValueError("atac_negatives_per_positive must be >= 1.")
        self.atac_loss_mode = atac_loss_mode
        self.atac_negatives_per_positive = int(atac_negatives_per_positive)
        self.atac_subsample_rescale = bool(atac_subsample_rescale)

        super().__init__(*args, **kwargs)
        self._init_reconstruction_balancing()
        self.max_atac_tokens = max_atac_tokens
        self.atac_genomic_sort = atac_genomic_sort
        if self.n_input_regions > 0 and hasattr(self, "l_encoder_accessibility"):
            del self.l_encoder_accessibility
        if self.n_input_regions > 0:
            if coord_table is None:
                raise ValueError("coord_table is required for MAMBAVAE when ATAC is enabled.")
            coord_tensor = torch.as_tensor(coord_table, dtype=torch.long)
            encoder_cat_list = (
                list(self.n_cats_per_cov or []) if self.encode_covariates else None
            )
            self.z_encoder_accessibility = MambaAtacVariationalEncoder(
                n_latent=self.n_latent,
                coord_table=coord_tensor,
                d_model=mamba_d_model,
                n_layers=mamba_n_layers,
                n_batch=self.n_batch,
                n_cat_list=encoder_cat_list,
                n_continuous_cov=self.n_continuous_cov,
                encode_covariates=self.encode_covariates,
                latent_distribution=self.latent_distribution,
                binarize=True,
                # Match the FC RNA/protein encoders' variance floor (var_eps=0) so the
                # posterior-variance scale feeding mix_modalities is symmetric across modalities.
                var_eps=0.0,
                use_checkpoint=mamba_use_checkpoint,
                pooling_mode=atac_pooling_mode,
                mamba3_kwargs=mamba3_kwargs,
            )
            if atac_decoder_module == "mamba":
                cat_list = [self.n_batch] + list(self.n_cats_per_cov or [])
                decoder_kwargs = dict(mamba_decoder_kwargs or {})
                self.z_decoder_accessibility = MambaAtacDecoder(
                    n_input=self.n_latent + self.n_continuous_cov,
                    n_output=self.n_input_regions,
                    coord_table=coord_tensor,
                    d_model=decoder_kwargs.pop("d_model", mamba_d_model),
                    n_layers=decoder_kwargs.pop("n_layers", mamba_n_layers),
                    n_cat_list=cat_list,
                    use_batch_norm=self.use_batch_norm_decoder,
                    use_layer_norm=self.use_layer_norm_decoder,
                    deep_inject_covariates=self.deeply_inject_covariates,
                    output_mode=atac_decoder_output_mode,
                    use_checkpoint=decoder_kwargs.pop(
                        "use_checkpoint", mamba_use_checkpoint
                    ),
                    mamba3_kwargs=decoder_kwargs.pop("mamba3_kwargs", mamba3_kwargs),
                    **decoder_kwargs,
                )

        self.contrast_proj = None
        needs_contrast = (
            lambda_dcl > 0
            or lambda_ot > 0
            or lambda_aug > 0
            or lambda_pseudo_dcl > 0
        )
        if (
            (not adversarial_data_type or combine_adversarial_vertical)
            and needs_contrast
            and self.n_input_genes > 0
            and self.n_input_regions > 0
        ):
            pd = contrast_proj_dim
            self.contrast_proj = nn.Sequential(
                nn.Linear(self.n_latent, pd),
                nn.ReLU(),
                nn.Linear(pd, pd),
            )
            if use_projection_memory_bank:
                self.register_buffer(
                    "proj_r_bank",
                    torch.zeros(memory_bank_size, pd),
                    persistent=False,
                )
                self.register_buffer(
                    "proj_a_bank",
                    torch.zeros(memory_bank_size, pd),
                    persistent=False,
                )
                self.register_buffer(
                    "proj_bank_ptr",
                    torch.zeros((), dtype=torch.long),
                    persistent=False,
                )
                self.register_buffer(
                    "proj_bank_filled",
                    torch.zeros((), dtype=torch.long),
                    persistent=False,
                )

    def _init_reconstruction_balancing(self) -> None:
        mode = self.reconstruction_normalization
        if mode == "ema":
            self.register_buffer("ema_recon_rna", torch.tensor(1.0))
            self.register_buffer("ema_recon_atac", torch.tensor(1.0))
        if mode == "uncertainty":
            self.log_var_rna = nn.Parameter(torch.zeros(()))
            self.log_var_atac = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _static_recon_weights(
        mode: ReconstructionNormalizationMode,
        n_genes: int,
        n_regions: int,
    ) -> tuple[float, float]:
        if mode == "none":
            return 1.0, 1.0
        if mode == "per_feature":
            w_rna = 1.0 / n_genes if n_genes > 0 else 1.0
            w_atac = 1.0 / n_regions if n_regions > 0 else 1.0
            return w_rna, w_atac
        if mode == "balanced":
            ref = max(n_genes, n_regions, 1)
            w_rna = ref / n_genes if n_genes > 0 else 1.0
            w_atac = ref / n_regions if n_regions > 0 else 1.0
            return w_rna, w_atac
        return 1.0, 1.0

    def _clamp_recon_weight(self, weight: torch.Tensor | float) -> torch.Tensor | float:
        if self.max_recon_weight_ratio is None:
            return weight
        if isinstance(weight, torch.Tensor):
            return weight.clamp(max=float(self.max_recon_weight_ratio))
        return min(float(weight), float(self.max_recon_weight_ratio))

    def _update_recon_ema(self, modality: Literal["rna", "atac"], value: torch.Tensor) -> None:
        if self.reconstruction_normalization != "ema" or not self.training:
            return
        beta = self.recon_ema_beta
        mean_val = value.mean().detach()
        buffer = self.ema_recon_rna if modality == "rna" else self.ema_recon_atac
        buffer.mul_(beta).add_(mean_val, alpha=1.0 - beta)

    def _ema_recon_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-8
        if self.n_input_regions >= self.n_input_genes:
            anchor = self.ema_recon_atac
        else:
            anchor = self.ema_recon_rna
        w_rna = self._clamp_recon_weight(anchor / (self.ema_recon_rna + eps))
        w_atac = self._clamp_recon_weight(anchor / (self.ema_recon_atac + eps))
        return w_rna.detach(), w_atac.detach()

    def _uncertainty_precision(self, modality: Literal["rna", "atac"]) -> torch.Tensor:
        log_var = self.log_var_rna if modality == "rna" else self.log_var_atac
        log_var = log_var.clamp(-self.log_var_clamp, self.log_var_clamp)
        return 0.5 * torch.exp(-log_var)

    def reconstruction_weight_report(self) -> dict[str, float]:
        mode = self.reconstruction_normalization
        if mode == "uncertainty":
            w_rna = float(self._uncertainty_precision("rna").detach().cpu())
            w_atac = float(self._uncertainty_precision("atac").detach().cpu())
        elif mode == "ema":
            w_rna_t, w_atac_t = self._ema_recon_weights()
            w_rna = float(w_rna_t.cpu())
            w_atac = float(w_atac_t.cpu())
        else:
            w_rna, w_atac = self._static_recon_weights(
                mode, self.n_input_genes, self.n_input_regions
            )
        return {
            "reconstruction_normalization": mode,
            "recon_weight_rna": w_rna,
            "recon_weight_atac": w_atac,
        }

    def _weight_reconstruction_loss(
        self,
        rl: torch.Tensor,
        *,
        modality: Literal["rna", "atac"],
    ) -> torch.Tensor:
        mode = self.reconstruction_normalization
        if mode == "none":
            return rl
        if mode == "uncertainty":
            return rl * self._uncertainty_precision(modality)
        if mode == "ema":
            self._update_recon_ema(modality, rl)
            w_rna, w_atac = self._ema_recon_weights()
            weight = w_rna if modality == "rna" else w_atac
            return rl * weight
        w_rna, w_atac = self._static_recon_weights(
            mode, self.n_input_genes, self.n_input_regions
        )
        weight = w_rna if modality == "rna" else w_atac
        return rl * weight

    @property
    def atac_pretrain_active(self) -> bool:
        return (
            self.unimodal_pretrain_active
            and self.unimodal_pretrain_modality == "atac"
        )

    @atac_pretrain_active.setter
    def atac_pretrain_active(self, value: bool) -> None:
        if value:
            self.unimodal_pretrain_active = True
            self.unimodal_pretrain_modality = "atac"
            self.training_phase = "unimodal"
        elif self.unimodal_pretrain_modality == "atac":
            self.unimodal_pretrain_active = False

    def _training_phase_metric(self, device: torch.device) -> torch.Tensor:
        if self.unimodal_pretrain_active:
            phase: TrainingPhase = "unimodal"
        elif self.alignment_warmup_active:
            phase = "align_warmup"
        else:
            phase = self.training_phase
        return torch.tensor(_TRAINING_PHASE_CODE[phase], device=device)

    def _uses_variational_atac_decoder(self) -> bool:
        return (
            self.atac_decoder_module == "mamba"
            and self.atac_decoder_output_mode == "variational_bernoulli"
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
        elif x.shape[1] >= self.n_input_genes + self.n_input_regions:
            x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        else:
            x_atac = torch.zeros(
                x.shape[0],
                self.n_input_regions,
                device=x.device,
                dtype=x.dtype,
                requires_grad=False,
            )

        if atac_token_mask is not None:
            mask_acc = atac_token_mask.any(dim=1)
        else:
            mask_acc = x_atac.sum(dim=1) > 0
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
            libsize_acc = atac_token_mask.sum(dim=1, keepdim=True).float() / float(
                self.max_atac_tokens
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

        if self.unimodal_pretrain_active:
            if self.unimodal_pretrain_modality == "rna":
                qz_m = qzm_expr
                qz_v = qzv_expr
            else:
                qz_m = qzm_acc
                qz_v = qzv_acc

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

        if self.unimodal_pretrain_active and n_samples > 1:
            z = z_expr if self.unimodal_pretrain_modality == "rna" else z_acc
        elif self.unimodal_pretrain_active:
            untran_z = Normal(qz_m, qz_v.sqrt()).rsample()
            encoder = (
                self.z_encoder_expression
                if self.unimodal_pretrain_modality == "rna"
                else self.z_encoder_accessibility
            )
            z = encoder.z_transformation(untran_z)
        else:
            untran_z = Normal(qz_m, qz_v.sqrt()).rsample()
            z = self.z_encoder_accessibility.z_transformation(untran_z)

        x_out = x
        if (
            self.n_input_regions > 0
            and x.shape[1] < self.n_input_genes + self.n_input_regions
        ):
            x_out = torch.cat([x[:, : self.n_input_genes], x_atac], dim=-1)

        return {
            "x": x_out,
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
        }

    def _use_atac_subsample_training(self) -> bool:
        return (
            self.atac_loss_mode == "balanced_subsample"
            and self.training
            and self.n_input_regions > 0
            and self.atac_decoder_module == "peakvi"
        )

    def _build_atac_decoder_input(
        self,
        z: torch.Tensor,
        qz_m: torch.Tensor,
        cont_covs: torch.Tensor | None,
        *,
        use_z_mean: bool = False,
    ) -> torch.Tensor:
        latent = z if not use_z_mean else qz_m
        if cont_covs is None:
            return latent
        if latent.dim() != cont_covs.dim():
            return torch.cat(
                [latent, cont_covs.unsqueeze(0).expand(latent.size(0), -1, -1)],
                dim=-1,
            )
        return torch.cat([latent, cont_covs], dim=-1)

    def _decode_atac_subset(
        self,
        decoder_input: torch.Tensor,
        peak_ids: torch.Tensor,
        batch_index: torch.Tensor,
        cat_covs: torch.Tensor | None,
    ) -> torch.Tensor:
        if cat_covs is not None:
            categorical_input = tuple(torch.split(cat_covs, 1, dim=1))
        else:
            categorical_input = ()
        return self.z_decoder_accessibility.forward_peaks(
            decoder_input,
            peak_ids,
            batch_index,
            *categorical_input,
        )

    def _accessibility_bce_subsampled(
        self,
        p_sel: torch.Tensor,
        targets: torch.Tensor,
        sample_mask: torch.Tensor,
        libsize_acc: torch.Tensor,
        peak_ids: torch.Tensor,
    ) -> torch.Tensor:
        reg_factor = (
            torch.sigmoid(self.region_factors)
            if self.region_factors is not None
            else 1.0
        )
        if isinstance(reg_factor, torch.Tensor):
            reg = reg_factor[peak_ids.clamp(min=0)]
        else:
            reg = reg_factor
        pred = p_sel * libsize_acc * reg
        bce = F.binary_cross_entropy(pred, targets, reduction="none")
        rl = (bce * sample_mask).sum(dim=-1)
        if self.atac_subsample_rescale:
            n_sampled = sample_mask.sum(dim=1).float().clamp(min=1.0)
            rl = rl * (float(self.n_input_regions) / n_sampled)
        if self.n_input_regions > 0:
            rl = self._weight_reconstruction_loss(rl, modality="atac")
        return rl

    def _atac_mask_from_tensors(
        self, tensors: dict, inference_outputs: dict
    ) -> torch.Tensor:
        token_mask = inference_outputs.get("atac_token_mask")
        if token_mask is None and ATAC_TOKEN_MASK_KEY in tensors:
            token_mask = tensors[ATAC_TOKEN_MASK_KEY]
        if token_mask is not None:
            return token_mask.any(dim=1)
        x = inference_outputs["x"]
        x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        return x_atac.sum(dim=1) > 0

    def _balanced_subsample_atac_loss(
        self,
        tensors: dict,
        inference_outputs: dict,
        generative_kwargs: dict,
    ) -> torch.Tensor:
        if ATAC_TOKEN_IDS_KEY not in tensors or ATAC_TOKEN_MASK_KEY not in tensors:
            raise ValueError(
                "balanced_subsample ATAC loss requires atac_token_ids and atac_token_mask."
            )
        samples = sample_balanced_atac_loss_indices(
            tensors[ATAC_TOKEN_IDS_KEY].long(),
            tensors[ATAC_TOKEN_MASK_KEY].bool(),
            self.n_input_regions,
            self.atac_negatives_per_positive,
        )
        decoder_input = self._build_atac_decoder_input(
            inference_outputs["z"],
            inference_outputs["qz_m"],
            generative_kwargs.get("cont_covs"),
        )
        p_sel = self._decode_atac_subset(
            decoder_input,
            samples["loss_peak_ids"],
            generative_kwargs["batch_index"],
            generative_kwargs.get("cat_covs"),
        )
        return self._accessibility_bce_subsampled(
            p_sel,
            samples["loss_targets"],
            samples["loss_sample_mask"],
            inference_outputs["libsize_acc"],
            samples["loss_peak_ids"],
        )

    @auto_move_data
    def generative(
        self,
        z,
        qz_m,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        libsize_expr=None,
        use_z_mean=False,
        label: torch.Tensor = None,
        transform_batch: int | None = None,
    ):
        if not self._use_atac_subsample_training():
            outputs = super().generative(
                z,
                qz_m,
                batch_index,
                cont_covs=cont_covs,
                cat_covs=cat_covs,
                libsize_expr=libsize_expr,
                use_z_mean=use_z_mean,
                label=label,
                transform_batch=transform_batch,
            )
            if self._uses_variational_atac_decoder():
                decoder_kl = getattr(self.z_decoder_accessibility, "last_decoder_kl", None)
                if decoder_kl is not None:
                    outputs["decoder_kl"] = decoder_kl
            return outputs

        if cat_covs is not None:
            categorical_input = tuple(torch.split(cat_covs, 1, dim=1))
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        decoder_input = self._build_atac_decoder_input(
            z, qz_m, cont_covs, use_z_mean=use_z_mean
        )

        px_scale, _, px_rate, px_dropout = self.z_decoder_expression(
            self.gene_dispersion,
            decoder_input,
            libsize_expr,
            batch_index,
            *categorical_input,
            label,
        )
        if self.gene_dispersion == "gene-label":
            px_r = F.linear(
                F.one_hot(label.squeeze(-1), self.n_labels).float(), self.px_r
            )
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(
                F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r
            )
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        py_, log_pro_back_mean = self.z_decoder_pro(
            decoder_input, batch_index, *categorical_input
        )
        if self.protein_dispersion == "protein-label":
            py_r = F.linear(F.one_hot(label.squeeze(-1), self.n_labels).float(), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(
                F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.py_r
            )
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)
        py_["r"] = py_r

        return {
            "p": None,
            "px_scale": px_scale,
            "px_r": torch.exp(self.px_r),
            "px_rate": px_rate,
            "px_dropout": px_dropout,
            "py_": py_,
            "log_pro_back_mean": log_pro_back_mean,
        }

    def get_reconstruction_loss_expression(self, x, px_rate, px_r, px_dropout):
        """Expression reconstruction loss with optional RNA/ATAC balancing (protein untouched).

        Modes: ``none`` (default), ``per_feature``, ``balanced``, ``ema``, ``uncertainty``.
        Only RNA and ATAC are reweighted; KL and protein recon are unchanged.
        """
        rl = super().get_reconstruction_loss_expression(x, px_rate, px_r, px_dropout)
        if self.n_input_genes > 0:
            rl = self._weight_reconstruction_loss(rl, modality="rna")
        return rl

    def get_reconstruction_loss_accessibility(self, x, p, d):
        """Accessibility reconstruction loss with optional RNA/ATAC balancing (protein untouched)."""
        rl = super().get_reconstruction_loss_accessibility(x, p, d)
        if self.n_input_regions > 0:
            rl = self._weight_reconstruction_loss(rl, modality="atac")
        return rl

    def _use_vertical_alignment(self) -> bool:
        if self.n_input_genes == 0 or self.n_input_regions == 0:
            return False
        if not self.adversarial_data_type:
            return True
        return self.combine_adversarial_vertical

    def _enqueue_projection_bank(self, proj_r: torch.Tensor, proj_a: torch.Tensor) -> None:
        if not self.use_projection_memory_bank or not hasattr(self, "proj_r_bank"):
            return
        n = proj_r.shape[0]
        if n == 0:
            return
        bank_size = self.proj_r_bank.shape[0]
        ptr = int(self.proj_bank_ptr.item())
        filled = int(self.proj_bank_filled.item())
        end = min(ptr + n, bank_size)
        first_chunk = end - ptr
        self.proj_r_bank[ptr:end] = proj_r[:first_chunk].detach()
        self.proj_a_bank[ptr:end] = proj_a[:first_chunk].detach()
        overflow = n - first_chunk
        if overflow > 0:
            self.proj_r_bank[:overflow] = proj_r[first_chunk:].detach()
            self.proj_a_bank[:overflow] = proj_a[first_chunk:].detach()
            ptr = overflow
            filled = bank_size
        else:
            ptr = end if end < bank_size else 0
            filled = min(filled + n, bank_size)
        self.proj_bank_ptr.fill_(ptr)
        self.proj_bank_filled.fill_(filled)

    def _gallery_with_bank(
        self, proj_batch: torch.Tensor, bank: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_projection_memory_bank or not hasattr(self, "proj_r_bank"):
            return proj_batch
        filled = int(self.proj_bank_filled.item())
        if filled == 0:
            return proj_batch
        return torch.cat([proj_batch, bank[:filled]], dim=0)

    def _compute_vertical_loss(self, inference_outputs) -> dict[str, torch.Tensor]:
        device = inference_outputs["z"].device
        zero = torch.zeros((), device=device)
        out = {
            "dcl_loss_paired": zero,
            "dcl_loss": zero,
            "ot_loss": zero,
            "bridge_loss": zero,
            "pseudo_dcl_loss": zero,
            "aug_loss_rna": zero,
            "aug_loss_atac": zero,
            "mu_anchor_loss": zero,
            "vertical_loss": zero,
            "n_paired": zero,
            "n_rna_only": zero,
            "n_atac_only": zero,
        }
        if getattr(self, "unimodal_pretrain_active", False):
            out["_vertical_loss_tensor"] = zero
            return out
        if self.contrast_proj is None or not self._use_vertical_alignment():
            out["_vertical_loss_tensor"] = zero
            return out

        x = inference_outputs["x"]
        x_rna = x[:, : self.n_input_genes]
        x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        mask_expr = x_rna.sum(dim=1) > 0
        token_mask = inference_outputs.get("atac_token_mask")
        if token_mask is not None:
            mask_acc = token_mask.any(dim=1)
        else:
            mask_acc = x_atac.sum(dim=1) > 0
        paired = mask_expr & mask_acc
        rna_only = mask_expr & ~mask_acc
        atac_only = ~mask_expr & mask_acc

        out["n_paired"] = paired.float().sum()
        out["n_rna_only"] = rna_only.float().sum()
        out["n_atac_only"] = atac_only.float().sum()

        qzm_expr = inference_outputs["qzm_expr"]
        qzm_acc = inference_outputs["qzm_acc"]

        dcl_paired = zero
        mu_anchor = zero
        if paired.any():
            qe = qzm_expr[paired]
            qa = qzm_acc[paired]
            if self.lambda_mu_anchor > 0:
                mu_anchor = ((qe - qa) ** 2).sum(-1).mean()
            if self.lambda_dcl > 0:
                pr = self.contrast_proj(qe)
                pa = self.contrast_proj(qa)
                dcl_paired = symmetric_dcl(
                    pr, pa, temperature=self.dcl_temperature,
                )
                if self.training:
                    self._enqueue_projection_bank(pr, pa)

        ot_total = zero
        bridge = zero
        pseudo_dcl = zero
        warmup = self.vertical_warmup_scale
        have_unpaired = bool(rna_only.any() or atac_only.any())
        use_ot_bridge = (
            self.unpaired_alignment_mode == "ot_bridge"
            and self.lambda_ot > 0
            and have_unpaired
        )
        use_pseudo_dcl = (
            self.unpaired_alignment_mode == "pseudo_dcl"
            and self.lambda_pseudo_dcl > 0
            and have_unpaired
        )
        if use_ot_bridge or use_pseudo_dcl:
            pd = self.contrast_proj_dim
            proj_r_p = (
                self.contrast_proj(qzm_expr[paired])
                if paired.any()
                else qzm_expr.new_zeros((0, pd))
            )
            proj_a_p = (
                self.contrast_proj(qzm_acc[paired])
                if paired.any()
                else qzm_acc.new_zeros((0, pd))
            )
            bank_r = (
                self.proj_r_bank
                if hasattr(self, "proj_r_bank")
                else proj_r_p.new_zeros((0, pd))
            )
            bank_a = (
                self.proj_a_bank
                if hasattr(self, "proj_a_bank")
                else proj_a_p.new_zeros((0, pd))
            )
            gallery_r = self._gallery_with_bank(proj_r_p, bank_r)
            gallery_a = self._gallery_with_bank(proj_a_p, bank_a)
            proj_r_u = (
                self.contrast_proj(qzm_expr[rna_only])
                if rna_only.any()
                else qzm_expr.new_zeros((0, pd))
            )
            proj_a_u = (
                self.contrast_proj(qzm_acc[atac_only])
                if atac_only.any()
                else qzm_acc.new_zeros((0, pd))
            )
            if use_ot_bridge:
                ot_total, _, bridge = multiome_anchored_ot_loss(
                    gallery_r,
                    gallery_a,
                    proj_r_u,
                    proj_a_u,
                    epsilon=self.sinkhorn_eps,
                    bridge_k=self.bridge_k,
                    bridge_tau=self.bridge_tau,
                    bridge_detach_gallery=self.bridge_detach_gallery,
                    lambda_bridge_inner=self.lambda_bridge_inner,
                    ot_anchor_r=proj_r_p,
                    ot_anchor_a=proj_a_p,
                )
            elif use_pseudo_dcl:
                pseudo_dcl = multiome_pseudo_dcl_loss(
                    gallery_r,
                    gallery_a,
                    proj_r_u,
                    proj_a_u,
                    bridge_k=self.bridge_k,
                    bridge_tau=self.bridge_tau,
                    bridge_detach_gallery=self.bridge_detach_gallery,
                    temperature=self.dcl_temperature,
                    pseudo_negative_mode=self.pseudo_negative_mode,
                )

        aug_rna = zero
        aug_atac = zero
        if self.lambda_aug > 0 and warmup > 0:
            if mask_expr.sum() >= 2:
                aug_rna = augmented_view_dcl(
                    qzm_expr[mask_expr],
                    self.contrast_proj,
                    noise_std=self.aug_noise_std,
                    temperature=self.dcl_temperature,
                )
            if mask_acc.sum() >= 2:
                aug_atac = augmented_view_dcl(
                    qzm_acc[mask_acc],
                    self.contrast_proj,
                    noise_std=self.aug_noise_std,
                    temperature=self.dcl_temperature,
                )

        vertical = (
            self.lambda_dcl * dcl_paired
            + warmup * self.lambda_ot * ot_total
            + warmup * self.lambda_pseudo_dcl * pseudo_dcl
            + warmup * self.lambda_aug * (aug_rna + aug_atac)
            + self.lambda_mu_anchor * mu_anchor
        )

        out.update({
            "dcl_loss_paired": dcl_paired.detach(),
            "dcl_loss": dcl_paired.detach(),
            "ot_loss": ot_total.detach(),
            "bridge_loss": bridge.detach(),
            "pseudo_dcl_loss": pseudo_dcl.detach(),
            "aug_loss_rna": aug_rna.detach(),
            "aug_loss_atac": aug_atac.detach(),
            "mu_anchor_loss": mu_anchor.detach(),
            "vertical_loss": vertical.detach(),
        })
        out["_vertical_loss_tensor"] = vertical
        return out

    def _loss_rna_pretrain(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Pure RNA pretraining objective: masked expression recon + KL(q(z_expr))."""
        x = inference_outputs["x"]
        x_rna = x[:, : self.n_input_genes]
        mask_expr = x_rna.sum(dim=1) > 0

        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        rl_expression = self.get_reconstruction_loss_expression(
            x_rna, px_rate, px_r, px_dropout
        )
        recon_loss_expression = rl_expression * mask_expr

        batch_size = x.shape[0]
        device = x.device
        zero_recon = torch.zeros(batch_size, device=device, requires_grad=False)

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        kl_div_z = kld(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(0, 1),
        ).sum(dim=1)
        kl_div_paired = torch.zeros(batch_size, device=device, requires_grad=False)
        weighted_kl = kl_weight * kl_div_z

        loss = torch.mean(recon_loss_expression + weighted_kl)

        recon_losses = {
            "reconstruction_loss_expression": recon_loss_expression,
            "reconstruction_loss_accessibility": zero_recon,
            "reconstruction_loss_protein": zero_recon,
        }
        kl_local = {
            "kl_divergence_z": kl_div_z,
            "kl_divergence_paired": kl_div_paired,
        }

        vert = self._compute_vertical_loss(inference_outputs)
        extra = {
            key: val if not isinstance(val, torch.Tensor) else val.detach()
            for key, val in vert.items()
            if not key.startswith("_")
        }
        extra["training_phase_code"] = self._training_phase_metric(device)

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_losses,
            kl_local=kl_local,
            extra_metrics=extra,
        )

    def _loss_alignment_warmup(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Alignment-only warmup: vertical bundle only, recon/KL zeroed."""
        batch_size = inference_outputs["x"].shape[0]
        device = inference_outputs["x"].device
        zero = torch.zeros(batch_size, device=device, requires_grad=False)

        vert = self._compute_vertical_loss(inference_outputs)
        loss = vert["_vertical_loss_tensor"]

        recon_losses = {
            "reconstruction_loss_expression": zero,
            "reconstruction_loss_accessibility": zero,
            "reconstruction_loss_protein": zero,
        }
        kl_local = {
            "kl_divergence_z": zero,
            "kl_divergence_paired": zero,
        }
        extra = {
            key: val if not isinstance(val, torch.Tensor) else val.detach()
            for key, val in vert.items()
            if not key.startswith("_")
        }
        extra["training_phase_code"] = self._training_phase_metric(device)

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_losses,
            kl_local=kl_local,
            extra_metrics=extra,
        )

    def _loss_atac_pretrain(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Pure ATAC pretraining objective: masked ATAC recon + KL(q(z_acc))."""
        mask_acc = self._atac_mask_from_tensors(tensors, inference_outputs)

        if self._use_atac_subsample_training():
            generative_kwargs = self._get_generative_input(tensors, inference_outputs)
            rl_accessibility = self._balanced_subsample_atac_loss(
                tensors, inference_outputs, generative_kwargs
            )
        else:
            x = inference_outputs["x"]
            x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
            p = generative_outputs["p"]
            libsize_acc = inference_outputs["libsize_acc"]
            rl_accessibility = self.get_reconstruction_loss_accessibility(
                x_atac, p, libsize_acc
            )
        recon_loss_accessibility = rl_accessibility * mask_acc

        batch_size = inference_outputs["x"].shape[0]
        device = inference_outputs["x"].device
        zero_recon = torch.zeros(batch_size, device=device, requires_grad=False)

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        kl_div_z = kld(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(0, 1),
        ).sum(dim=1)
        kl_div_paired = torch.zeros(batch_size, device=device, requires_grad=False)
        weighted_kl = kl_weight * kl_div_z

        loss = torch.mean(recon_loss_accessibility + weighted_kl)

        recon_losses = {
            "reconstruction_loss_expression": zero_recon,
            "reconstruction_loss_accessibility": recon_loss_accessibility,
            "reconstruction_loss_protein": zero_recon,
        }
        kl_local = {
            "kl_divergence_z": kl_div_z,
            "kl_divergence_paired": kl_div_paired,
        }

        vert = self._compute_vertical_loss(inference_outputs)
        extra = {
            key: val if not isinstance(val, torch.Tensor) else val.detach()
            for key, val in vert.items()
            if not key.startswith("_")
        }
        extra["training_phase_code"] = self._training_phase_metric(device)

        if "decoder_kl" in generative_outputs:
            decoder_kl = generative_outputs["decoder_kl"] * mask_acc
            weighted_decoder_kl = self.atac_decoder_kl_weight * decoder_kl
            loss = loss + torch.mean(weighted_decoder_kl)
            kl_local["kl_divergence_decoder"] = weighted_decoder_kl

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_losses,
            kl_local=kl_local,
            extra_metrics=extra,
        )

    def _loss_joint_balanced_subsample(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Joint ELBO with token-only subsampled ATAC reconstruction."""
        x = inference_outputs["x"]
        x_rna = x[:, : self.n_input_genes]
        if self.n_input_proteins == 0:
            y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = self._atac_mask_from_tensors(tensors, inference_outputs)
        mask_pro = y.sum(dim=1) > 0

        generative_kwargs = self._get_generative_input(tensors, inference_outputs)
        rl_accessibility = self._balanced_subsample_atac_loss(
            tensors, inference_outputs, generative_kwargs
        )

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
        weighted_kl_local = kl_weight * kl_div_z + kl_div_paired
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
        extra_metrics_payload = (
            {
                "z": inference_outputs["z"],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
            }
            if self.extra_payload_autotune
            else {}
        )
        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_losses,
            kl_local=kl_local,
            extra_metrics=extra_metrics_payload,
        )

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        if self.unimodal_pretrain_active:
            if self.unimodal_pretrain_modality == "rna":
                return self._loss_rna_pretrain(
                    tensors, inference_outputs, generative_outputs, kl_weight
                )
            return self._loss_atac_pretrain(
                tensors, inference_outputs, generative_outputs, kl_weight
            )
        if self.alignment_warmup_active:
            return self._loss_alignment_warmup(
                tensors, inference_outputs, generative_outputs, kl_weight
            )
        if self._use_atac_subsample_training():
            loss_output = self._loss_joint_balanced_subsample(
                tensors, inference_outputs, generative_outputs, kl_weight
            )
        else:
            loss_output = super().loss(
                tensors, inference_outputs, generative_outputs, kl_weight
            )
        if self.reconstruction_normalization == "uncertainty":
            loss_output.loss = loss_output.loss + 0.5 * (
                self.log_var_rna + self.log_var_atac
            )
        vert = self._compute_vertical_loss(inference_outputs)
        loss_output.loss = loss_output.loss + vert["_vertical_loss_tensor"]
        extra = dict(loss_output.extra_metrics or {})
        for key, val in vert.items():
            if key.startswith("_"):
                continue
            extra[key] = val if not isinstance(val, torch.Tensor) else val.detach()
        for key in (
            "reconstruction_loss_expression",
            "reconstruction_loss_accessibility",
        ):
            term = loss_output.reconstruction_loss.get(key)
            if term is not None:
                extra[key] = term.mean().detach()
        for key in ("recon_weight_rna", "recon_weight_atac"):
            extra[key] = torch.tensor(
                self.reconstruction_weight_report()[key],
                device=loss_output.loss.device,
            )
        extra["training_phase_code"] = self._training_phase_metric(
            loss_output.loss.device
        )
        loss_output.extra_metrics = extra

        if "decoder_kl" not in generative_outputs:
            return loss_output

        mask_acc = self._atac_mask_from_tensors(tensors, inference_outputs)
        decoder_kl = generative_outputs["decoder_kl"] * mask_acc
        weighted_decoder_kl = self.atac_decoder_kl_weight * decoder_kl
        loss_output.loss = loss_output.loss + torch.mean(weighted_decoder_kl)
        loss_output.kl_local = {
            **loss_output.kl_local,
            "kl_divergence_decoder": weighted_decoder_kl,
        }
        return loss_output
