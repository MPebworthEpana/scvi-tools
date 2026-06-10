"""Store per-cell ATAC nnz counts for CSR streaming length bucketing."""

from __future__ import annotations

import numpy as np

from scvi.data import _constants
from scvi.tokenized._constants import ATAC_TOKEN_CONFIG_KEY
from scvi.tokenized._field import AtacTokenConfigField
from scvi.tokenized._nnz import atac_row_nnz


def store_atac_nnz_lengths(adata_manager, atac_x) -> None:
    """Patch the ATAC token state registry with per-cell nnz lengths."""
    nnz = atac_row_nnz(atac_x)
    state = adata_manager._registry[_constants._FIELD_REGISTRIES_KEY][ATAC_TOKEN_CONFIG_KEY][
        _constants._STATE_REGISTRY_KEY
    ]
    state[AtacTokenConfigField.NN_LENGTHS_KEY] = np.asarray(nnz, dtype=np.int64)
