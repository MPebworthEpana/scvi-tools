"""Public exports for Mamba ATAC encoders."""

from scvi.encoders._constants import (
    ATAC_TOKEN_CONFIG_KEY,
    ATAC_TOKEN_IDS_KEY,
    ATAC_TOKEN_MASK_KEY,
)
from scvi.encoders._coords import build_coord_table, build_genomic_rank, parse_peak_name
from scvi.encoders._csr_batch_tokenize import csr_batch_to_tokens
from scvi.encoders._mamba3_encoder import BidirectionalMamba3Encoder
from scvi.encoders._mamba_atac_variational import MambaAtacVariationalEncoder
from scvi.encoders._precompute_tokens import atac_row_nnz, precompute_atac_token_sequences
from scvi.encoders._tokenizers import tokenize_atac

__all__ = [
    "ATAC_TOKEN_CONFIG_KEY",
    "ATAC_TOKEN_IDS_KEY",
    "ATAC_TOKEN_MASK_KEY",
    "BidirectionalMamba3Encoder",
    "MambaAtacVariationalEncoder",
    "atac_row_nnz",
    "build_coord_table",
    "build_genomic_rank",
    "csr_batch_to_tokens",
    "parse_peak_name",
    "precompute_atac_token_sequences",
    "tokenize_atac",
]
