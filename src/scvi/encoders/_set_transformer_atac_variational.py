"""Variational Set Transformer ATAC encoder.

Canonical definition lives in :mod:`scvi.tokenized._set_transformer_atac_variational`;
re-exported here so the Mamba encoder stack and the SetVI stack share a single source of
truth.
"""

from scvi.tokenized._set_transformer_atac_variational import (
    SetTransformerAtacVariationalEncoder,
)

__all__ = ["SetTransformerAtacVariationalEncoder"]
