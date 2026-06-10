"""SETVI: MultiVI with Set Transformer ATAC encoder and token-native data path."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from mudata import MuData

from scvi import REGISTRY_KEYS
from scvi.data import _constants, AnnDataManager, fields
from scvi.data._utils import _get_adata_minify_type
from scvi.model._multivi import MULTIVI
from scvi.module import SETVAE
from scvi.tokenized import (
    ATAC_TOKEN_CONFIG_KEY,
    AtacTokenConfigField,
    MuDataAtacTokenField,
    SetAnnDataLoader,
    SetDataSplitter,
    build_coord_table,
)
from scvi.tokenized._precompute import build_and_attach_token_store
from scvi.tokenized._token_store import (
    ensure_token_store_for_manager,
    rebuild_token_store,
    registry_for_checkpoint,
)
from scvi.utils._docstrings import setup_anndata_dsp

if TYPE_CHECKING:
    from anndata import AnnData
    from torch import Tensor

    from scvi._types import AnnOrMuData

logger = logging.getLogger(__name__)


class SETVI(MULTIVI):
    """MultiVI integration with a Set Transformer ATAC encoder and token-native ATAC data."""

    _module_cls = SETVAE
    _data_splitter_cls = SetDataSplitter
    _data_loader_cls = SetAnnDataLoader

    def __init__(
        self,
        adata: AnnOrMuData,
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
        # SETVI always learns per-peak region_factors: they double as the decoder baseline
        # and, shared with the encoder, as the capped rarity prior on attention. Not a knob.
        if "region_factors" in kwargs and not kwargs["region_factors"]:
            logger.warning(
                "SETVI always uses region_factors=True (shared with the encoder peak-salience "
                "prior); ignoring region_factors=False."
            )
        kwargs.pop("region_factors", None)
        if coord_table is None:
            manager = SETVI._get_most_recent_anndata_manager(adata, required=True)
            token_cfg = manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)
            coord_table = token_cfg[AtacTokenConfigField.COORD_TABLE_KEY]
        super().__init__(
            adata,
            coord_table=coord_table,
            max_atac_tokens=max_atac_tokens,
            st_d_model=st_d_model,
            st_n_layers=st_n_layers,
            st_n_inducing=st_n_inducing,
            st_n_heads=st_n_heads,
            st_dropout=st_dropout,
            use_cardinality_film=use_cardinality_film,
            use_sampling_correction=use_sampling_correction,
            use_peak_salience_prior=use_peak_salience_prior,
            peak_salience_cap=peak_salience_cap,
            **kwargs,
        )
        self.init_params_ = self._get_init_params(locals())
        self.init_params_["non_kwargs"]["coord_table"] = None
        self._model_summary_string = self._model_summary_string.replace(
            "MultiVI Model", "SETVI Model"
        )
        self._ensure_token_store()
        self._attach_token_store_to_module()

    def _ensure_token_store(self) -> None:
        token_cfg = self.adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)
        if token_cfg is None:
            return
        if token_cfg.get(AtacTokenConfigField.TOKEN_STORE_KEY) is not None:
            return
        handle = token_cfg.get(AtacTokenConfigField.TOKEN_STORE_HANDLE_KEY)
        if handle is None:
            return
        rebuild_token_store(
            self.adata_manager,
            tier=handle.get("tier"),
            out_dir=handle.get("mmap_dir"),
        )

    def _attach_token_store_to_module(self) -> None:
        token_cfg = self.adata_manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)
        if token_cfg is None:
            return
        store = token_cfg.get(AtacTokenConfigField.TOKEN_STORE_KEY)
        if store is not None:
            self.module.set_token_store(store)

    def _get_user_attributes(self):
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]
        out = []
        for name, value in attributes:
            if name == "registry_":
                value = registry_for_checkpoint(value)
            out.append((name, value))
        return out

    def _ensure_store_for_manager(self, adata_manager, *, force_ephemeral: bool = False):
        return ensure_token_store_for_manager(
            adata_manager,
            training_manager=self.adata_manager,
            force_ephemeral=force_ephemeral,
        )

    @contextmanager
    def _use_inference_store(self, adata: AnnOrMuData | None):
        if adata is None:
            adata = self.adata
            manager = self.adata_manager
        else:
            adata = self._validate_anndata(adata)
            manager = self.get_anndata_manager(adata)
        force = manager is not self.adata_manager
        store = self._ensure_store_for_manager(manager, force_ephemeral=force)
        prev = self.module._token_store
        self.module.set_token_store(store)
        try:
            yield
        finally:
            self.module.set_token_store(prev)

    def get_latent_representation(
        self,
        adata: AnnOrMuData | None = None,
        modality: Literal["joint", "expression", "accessibility"] = "joint",
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
        return_dist: bool = False,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
    ) -> np.ndarray:
        with self._use_inference_store(adata):
            return super().get_latent_representation(
                adata=adata,
                modality=modality,
                indices=indices,
                give_mean=give_mean,
                batch_size=batch_size,
                return_dist=return_dist,
                dataloader=dataloader,
            )

    def get_reconstruction_error(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
        return_mean: bool = True,
        data_loader_kwargs: dict | None = None,
        **kwargs,
    ) -> dict[str, float]:
        with self._use_inference_store(adata):
            return super().get_reconstruction_error(
                adata=adata,
                indices=indices,
                batch_size=batch_size,
                dataloader=dataloader,
                return_mean=return_mean,
                data_loader_kwargs=data_loader_kwargs,
                **kwargs,
            )

    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-4,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        check_val_every_n_epoch: int | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 50,
        adversarial_mixing: bool = True,
        atac_length_bucketing: bool = True,
        bucket_mult: int = 50,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        """Train SETVI with token-native dataloader defaults."""
        self._attach_token_store_to_module()
        datasplitter_kwargs = datasplitter_kwargs or {}
        datasplitter_kwargs.setdefault("load_sparse_tensor", False)
        datasplitter_kwargs.setdefault("pin_memory", pin_memory)
        if num_workers == 0:
            persistent_workers = False
        datasplitter_kwargs.setdefault("num_workers", num_workers)
        datasplitter_kwargs.setdefault("persistent_workers", persistent_workers)
        datasplitter_kwargs.setdefault("atac_length_bucketing", atac_length_bucketing)
        datasplitter_kwargs.setdefault("bucket_mult", bucket_mult)
        return super().train(
            max_epochs=max_epochs,
            lr=lr,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            weight_decay=weight_decay,
            eps=eps,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            adversarial_mixing=adversarial_mixing,
            datasplitter_kwargs=datasplitter_kwargs,
            plan_kwargs=plan_kwargs,
            **kwargs,
        )

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_mudata(
        cls,
        mdata: MuData,
        rna_layer: str | None = None,
        atac_layer: str | None = None,
        protein_layer: str | None = None,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        idx_layer: str | None = None,
        modalities: dict[str, str] | None = None,
        max_atac_tokens: int = 8192,
        atac_genomic_sort: bool = True,
        coord_table=None,
        atac_token_store: str = "auto",
        atac_token_store_dir: str | Path | None = None,
        **kwargs,
    ):
        """%(summary_mdata)s.

        Registers ATAC token config and builds a tiered token store from CSR rows.
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        if modalities is None:
            raise ValueError("Modalities cannot be None.")
        modalities = cls._create_modalities_attr_dict(modalities, setup_method_args)

        desired_order = []
        if modalities.rna_layer is not None:
            desired_order.append(modalities.rna_layer)
        if modalities.atac_layer is not None:
            desired_order.append(modalities.atac_layer)
        if modalities.protein_layer is not None:
            desired_order.append(modalities.protein_layer)

        current_order = list(mdata.mod.keys())
        needs_reorder = current_order[: len(desired_order)] != desired_order
        if needs_reorder:
            reordered_keys = desired_order + [k for k in current_order if k not in desired_order]
            backing_dict = mdata._mod
            snapshot = {k: backing_dict[k] for k in reordered_keys}
            backing_dict.clear()
            backing_dict.update(snapshot)
            mdata.update()

        import numpy as np

        mdata.obs["_indices"] = np.arange(mdata.n_obs)

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=modalities.batch_key,
        )
        mudata_fields = [
            batch_field,
            fields.MuDataCategoricalObsField(
                REGISTRY_KEYS.LABELS_KEY,
                None,
                mod_key=None,
            ),
            fields.MuDataNumericalJointObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY,
                size_factor_key,
                mod_key=None,
                required=False,
            ),
            fields.MuDataCategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY,
                categorical_covariate_keys,
                mod_key=modalities.categorical_covariate_keys,
            ),
            fields.MuDataNumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY,
                continuous_covariate_keys,
                mod_key=modalities.continuous_covariate_keys,
            ),
            fields.MuDataNumericalObsField(
                REGISTRY_KEYS.INDICES_KEY,
                "_indices",
                mod_key=modalities.idx_layer,
                required=False,
            ),
        ]
        if modalities.rna_layer is not None:
            mudata_fields.append(
                fields.MuDataLayerField(
                    REGISTRY_KEYS.X_KEY,
                    rna_layer,
                    mod_key=modalities.rna_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )
        if modalities.atac_layer is not None:
            mudata_fields.append(
                fields.MuDataLayerField(
                    REGISTRY_KEYS.ATAC_X_KEY,
                    atac_layer,
                    mod_key=modalities.atac_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )
            if coord_table is None:
                coord_table = build_coord_table(list(mdata[modalities.atac_layer].var_names))
            mudata_fields.append(
                MuDataAtacTokenField(
                    coord_table=coord_table,
                    max_atac_tokens=max_atac_tokens,
                    genomic=atac_genomic_sort,
                    mod_key=modalities.atac_layer,
                )
            )
        if modalities.protein_layer is not None:
            mudata_fields.append(
                fields.MuDataProteinLayerField(
                    REGISTRY_KEYS.PROTEIN_EXP_KEY,
                    protein_layer,
                    mod_key=modalities.protein_layer,
                    use_batch_mask=True,
                    batch_field=batch_field,
                    is_count_data=True,
                    mod_required=True,
                )
            )
        mdata_minify_type = _get_adata_minify_type(mdata)
        if mdata_minify_type is not None:
            mudata_fields += cls._get_fields_for_mudata_minification(mdata_minify_type)

        adata_manager = AnnDataManager(fields=mudata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(mdata, **kwargs)
        if modalities.atac_layer is not None:
            atac_x = adata_manager.get_from_registry(REGISTRY_KEYS.ATAC_X_KEY)
            store_dir = atac_token_store_dir
            if store_dir is None and getattr(mdata, "filename", None):
                store_dir = Path(mdata.filename).parent / ".setvi_token_store"
            build_and_attach_token_store(
                adata_manager,
                atac_x,
                tier=atac_token_store,
                out_dir=str(store_dir) if store_dir is not None else None,
            )
            state = adata_manager._registry[_constants._FIELD_REGISTRIES_KEY][
                ATAC_TOKEN_CONFIG_KEY
            ][_constants._STATE_REGISTRY_KEY]
            state["atac_token_store_tier"] = state[AtacTokenConfigField.TOKEN_STORE_HANDLE_KEY][
                "tier"
            ]
        cls.register_manager(adata_manager)
