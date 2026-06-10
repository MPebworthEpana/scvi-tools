"""Metadata field for Mamba ATAC tokenization config."""

from __future__ import annotations

import numpy as np
import rich.table

from scvi import REGISTRY_KEYS
from scvi.encoders._constants import ATAC_TOKEN_CONFIG_KEY
from scvi.encoders._coords import build_coord_table, build_genomic_rank

from ._base_field import BaseAnnDataField
from ._mudata import BaseMuDataWrapperClass


class AtacTokenConfigField(BaseAnnDataField):
    """Stores ATAC tokenization config; sequences are precomputed at setup."""

    COORD_TABLE_KEY = "coord_table"
    GENOMIC_RANK_KEY = "genomic_rank"
    MAX_TOKENS_KEY = "max_atac_tokens"
    GENOMIC_KEY = "genomic"
    PRECOMPUTED_KEY = "precomputed"
    PRECOMPUTED_IDS_KEY = "precomputed_token_ids"
    PRECOMPUTED_LENGTHS_KEY = "precomputed_token_lengths"
    NN_LENGTHS_KEY = "nnz_lengths"

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
        if self._coord_table is None:
            self._coord_table = build_coord_table(np.asarray(adata.var_names))
        coord_table = np.asarray(self._coord_table, dtype=np.int64)
        return {
            self.COORD_TABLE_KEY: coord_table,
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
            self.GENOMIC_RANK_KEY: state_registry.get(
                self.GENOMIC_RANK_KEY, build_genomic_rank(coord_table)
            ),
            self.MAX_TOKENS_KEY: int(state_registry[self.MAX_TOKENS_KEY]),
            self.GENOMIC_KEY: bool(state_registry[self.GENOMIC_KEY]),
            self.PRECOMPUTED_KEY: bool(state_registry.get(self.PRECOMPUTED_KEY, False)),
            "atac_source_key": state_registry.get("atac_source_key", REGISTRY_KEYS.ATAC_X_KEY),
        }
        if out[self.PRECOMPUTED_KEY]:
            out[self.PRECOMPUTED_IDS_KEY] = np.asarray(
                state_registry[self.PRECOMPUTED_IDS_KEY], dtype=np.int64
            )
            out[self.PRECOMPUTED_LENGTHS_KEY] = np.asarray(
                state_registry[self.PRECOMPUTED_LENGTHS_KEY], dtype=np.int64
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
