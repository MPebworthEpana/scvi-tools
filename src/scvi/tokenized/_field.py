"""Metadata field for ATAC tokenization config."""

from __future__ import annotations

import numpy as np
import rich.table

from scvi import REGISTRY_KEYS
from scvi.data.fields._base_field import BaseAnnDataField
from scvi.data.fields._mudata import BaseMuDataWrapperClass
from scvi.tokenized._constants import ATAC_TOKEN_CONFIG_KEY
from scvi.tokenized._coords import build_chrom_vocab, build_coord_table, build_genomic_rank


class AtacTokenConfigField(BaseAnnDataField):
    """Stores ATAC tokenization config; sequences may be precomputed or streamed from CSR."""

    COORD_TABLE_KEY = "coord_table"
    CHROM_VOCAB_KEY = "chrom_vocab"
    GENOMIC_RANK_KEY = "genomic_rank"
    MAX_TOKENS_KEY = "max_atac_tokens"
    GENOMIC_KEY = "genomic"
    PRECOMPUTED_KEY = "precomputed"
    PRECOMPUTED_IDS_KEY = "precomputed_token_ids"
    PRECOMPUTED_LENGTHS_KEY = "precomputed_token_lengths"
    PRECOMPUTED_VALUES_KEY = "precomputed_token_values"
    NN_LENGTHS_KEY = "nnz_lengths"
    TOKEN_STORE_KEY = "token_store"
    TOKEN_STORE_HANDLE_KEY = "token_store_handle"

    def __init__(
        self,
        coord_table: np.ndarray | None = None,
        max_atac_tokens: int = 8192,
        genomic: bool = True,
    ) -> None:
        super().__init__()
        self._coord_table = coord_table
        self._max_atac_tokens = max_atac_tokens
        self._genomic = genomic

    @property
    def registry_key(self) -> str:
        return ATAC_TOKEN_CONFIG_KEY

    @property
    def attr_name(self) -> str:
        return ""

    @property
    def attr_key(self) -> str | None:
        return None

    @property
    def is_empty(self) -> bool:
        return False

    def get_data_registry(self) -> dict:
        return {}

    def validate_field(self, adata) -> None:
        return None

    def register_field(self, adata) -> dict:
        peak_names = np.asarray(adata.var_names)
        vocab = build_chrom_vocab(peak_names)
        if self._coord_table is None:
            coord_table = build_coord_table(peak_names, vocab=vocab)
        else:
            coord_table = np.asarray(self._coord_table, dtype=np.int64)
        return {
            self.COORD_TABLE_KEY: coord_table,
            self.CHROM_VOCAB_KEY: vocab,
            self.GENOMIC_RANK_KEY: build_genomic_rank(coord_table),
            self.MAX_TOKENS_KEY: int(self._max_atac_tokens),
            self.GENOMIC_KEY: bool(self._genomic),
            self.PRECOMPUTED_KEY: False,
            "atac_source_key": REGISTRY_KEYS.ATAC_X_KEY,
        }

    def transfer_field(self, state_registry: dict, adata_target, **kwargs) -> dict:
        coord_table = np.asarray(state_registry[self.COORD_TABLE_KEY], dtype=np.int64)
        out = {
            self.COORD_TABLE_KEY: coord_table,
            self.CHROM_VOCAB_KEY: dict(
                state_registry.get(self.CHROM_VOCAB_KEY, build_chrom_vocab(adata_target.var_names))
            ),
            self.GENOMIC_RANK_KEY: state_registry.get(
                self.GENOMIC_RANK_KEY, build_genomic_rank(coord_table)
            ),
            self.MAX_TOKENS_KEY: int(state_registry[self.MAX_TOKENS_KEY]),
            self.GENOMIC_KEY: bool(state_registry[self.GENOMIC_KEY]),
            self.PRECOMPUTED_KEY: bool(state_registry.get(self.PRECOMPUTED_KEY, False)),
            "atac_source_key": state_registry.get("atac_source_key", REGISTRY_KEYS.ATAC_X_KEY),
        }
        if AtacTokenConfigField.TOKEN_STORE_HANDLE_KEY in state_registry:
            out[AtacTokenConfigField.TOKEN_STORE_HANDLE_KEY] = dict(
                state_registry[AtacTokenConfigField.TOKEN_STORE_HANDLE_KEY]
            )
        elif out[self.PRECOMPUTED_KEY]:
            out[self.PRECOMPUTED_IDS_KEY] = np.asarray(
                state_registry[self.PRECOMPUTED_IDS_KEY], dtype=np.int64
            )
            out[self.PRECOMPUTED_LENGTHS_KEY] = np.asarray(
                state_registry[self.PRECOMPUTED_LENGTHS_KEY], dtype=np.int64
            )
            if self.PRECOMPUTED_VALUES_KEY in state_registry:
                out[self.PRECOMPUTED_VALUES_KEY] = np.asarray(
                    state_registry[self.PRECOMPUTED_VALUES_KEY], dtype=np.float32
                )
        return out

    def get_summary_stats(self, state_registry: dict) -> dict:
        return {"max_atac_tokens": state_registry[self.MAX_TOKENS_KEY]}

    def view_state_registry(self, state_registry: dict) -> rich.table.Table | None:
        table = rich.table.Table(title=f"{self.registry_key} State Registry")
        table.add_column("Key", justify="right")
        table.add_column("Value")
        table.add_row("max_atac_tokens", str(state_registry[self.MAX_TOKENS_KEY]))
        table.add_row("genomic", str(state_registry[self.GENOMIC_KEY]))
        return table


class MuDataAtacTokenField(BaseMuDataWrapperClass):
    """MuData wrapper that keeps token config out of the dense data registry."""

    def __init__(
        self,
        coord_table: np.ndarray | None = None,
        max_atac_tokens: int = 8192,
        genomic: bool = True,
        mod_key: str | None = None,
        mod_required: bool = False,
    ) -> None:
        super().__init__(mod_key=mod_key, mod_required=mod_required)
        self._adata_field = AtacTokenConfigField(
            coord_table=coord_table,
            max_atac_tokens=max_atac_tokens,
            genomic=genomic,
        )

    def get_data_registry(self) -> dict:
        return {}
