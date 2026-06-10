"""Set Transformer encoder building blocks.

Canonical definitions live in :mod:`scvi.tokenized._set_transformer`; re-exported here
so the Mamba encoder stack and the SetVI stack share a single source of truth.
"""

from scvi.tokenized._set_transformer import SetTransformerEncoder

__all__ = ["SetTransformerEncoder"]
