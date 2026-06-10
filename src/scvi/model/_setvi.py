"""SETVI: MultiVI with Set Transformer ATAC encoder and CSR token streaming."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mudata import MuData

from scvi import REGISTRY_KEYS
from scvi.data import _constants, AnnDataManager, fields
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields._atac_token_field import AtacTokenConfigField, MuDataAtacTokenField
from scvi.dataloaders._set_dataloader import SetAnnDataLoader
from scvi.dataloaders._set_splitter import SetDataSplitter
from scvi.encoders._constants import ATAC_TOKEN_CONFIG_KEY
from scvi.encoders._coords import build_coord_table
from scvi.encoders._store_nnz_lengths import store_atac_nnz_lengths
from scvi.model._mambavi import MAMBAVI
from scvi.module import SETVAE
from scvi.utils._docstrings import setup_anndata_dsp

if TYPE_CHECKING:
    from scvi._types import AnnOrMuData


class SETVI(MAMBAVI):
    """MultiVI integration with a Set Transformer ATAC encoder and CSR token streaming.

    SETVI encodes scATAC as unordered sets of open peaks using ISAB + PMA pooling
    (see ``SetTransformer.md``), streams tokens directly from CSR rows at batch time,
    and produces a MultiVI-compatible variational latent for ATAC-only or multimodal data.
    """

    _module_cls = SETVAE
    _data_splitter_cls = SetDataSplitter
    _data_loader_cls = SetAnnDataLoader

    def __init__(
        self,
        adata: AnnOrMuData,
        coord_table=None,
        st_d_model: int = 128,
        st_n_layers: int = 2,
        st_n_inducing: int = 32,
        st_n_heads: int = 4,
        st_dropout: float = 0.0,
        use_cardinality_film: bool = True,
        use_sampling_correction: bool = False,
        atac_loss_mode: Literal["dense", "balanced_subsample"] = "balanced_subsample",
        **kwargs,
    ):
        if coord_table is None:
            manager = SETVI._get_most_recent_anndata_manager(adata, required=True)
            token_cfg = manager.get_state_registry(ATAC_TOKEN_CONFIG_KEY)
            coord_table = token_cfg[AtacTokenConfigField.COORD_TABLE_KEY]
        super().__init__(
            adata,
            coord_table=coord_table,
            atac_loss_mode=atac_loss_mode,
            st_d_model=st_d_model,
            st_n_layers=st_n_layers,
            st_n_inducing=st_n_inducing,
            st_n_heads=st_n_heads,
            st_dropout=st_dropout,
            use_cardinality_film=use_cardinality_film,
            use_sampling_correction=use_sampling_correction,
            **kwargs,
        )
        self._model_summary_string = self._model_summary_string.replace(
            "MultiVI Model", "SETVI Model"
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
        **kwargs,
    ):
        """%(summary_mdata)s.

        Registers ATAC token config over all peaks and stores per-cell nnz lengths
        for CSR streaming (no setup-time token precomputation).
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
            store_atac_nnz_lengths(adata_manager, atac_x)
            state = adata_manager._registry[_constants._FIELD_REGISTRIES_KEY][
                ATAC_TOKEN_CONFIG_KEY
            ][_constants._STATE_REGISTRY_KEY]
            state[AtacTokenConfigField.PRECOMPUTED_KEY] = False
        cls.register_manager(adata_manager)
