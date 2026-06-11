"""Shared ATAC token-streaming infrastructure independent of MambaVAE."""

from scvi.tokenized._constants import (
    ATAC_TOKEN_CONFIG_KEY,
    ATAC_TOKEN_IDS_KEY,
    ATAC_TOKEN_MASK_KEY,
    ATAC_TOKEN_VALUES_KEY,
)
from scvi.tokenized._coords import (
    build_chrom_vocab,
    build_coord_table,
    build_genomic_rank,
    parse_peak_name,
)
from scvi.tokenized._csr_batch_tokenize import csr_batch_to_tokens
from scvi.tokenized._dataloader import SetAnnDataLoader
from scvi.tokenized._dataset import SetAnnTorchDataset
from scvi.tokenized._embeddings import AtacPeakEmbedding
from scvi.tokenized._field import AtacTokenConfigField, MuDataAtacTokenField
from scvi.tokenized._length_bucket_sampler import LengthBucketedBatchSampler
from scvi.tokenized._nnz import atac_row_nnz
from scvi.tokenized._precompute import (
    build_and_attach_token_store,
    precompute_atac_token_sequences,
    store_precomputed_atac_tokens,
)
from scvi.tokenized._token_store import (
    AtacTokenStore,
    build_token_store,
    ensure_token_store_for_manager,
    registry_for_checkpoint,
)
from scvi.tokenized._set_transformer import (
    CardinalityFiLM,
    DeepSetEncoder,
    InducedSetAttentionBlock,
    MultiheadAttentionBlock,
    PoolingByMultiheadAttention,
    SetAttentionBlock,
    SetTransformerEncoder,
)
from scvi.tokenized._set_transformer_atac_variational import SetTransformerAtacVariationalEncoder
from scvi.tokenized._splitter import SetDataSplitter
from scvi.tokenized._store_nnz_lengths import store_atac_nnz_lengths
from scvi.tokenized._subsample import sample_balanced_atac_loss_indices
from scvi.tokenized._tokenizers import tokenize_atac

__all__ = [
    "ATAC_TOKEN_CONFIG_KEY",
    "ATAC_TOKEN_IDS_KEY",
    "ATAC_TOKEN_MASK_KEY",
    "ATAC_TOKEN_VALUES_KEY",
    "AtacPeakEmbedding",
    "AtacTokenConfigField",
    "AtacTokenStore",
    "CardinalityFiLM",
    "DeepSetEncoder",
    "InducedSetAttentionBlock",
    "LengthBucketedBatchSampler",
    "MuDataAtacTokenField",
    "MultiheadAttentionBlock",
    "PoolingByMultiheadAttention",
    "SetAnnDataLoader",
    "SetAnnTorchDataset",
    "SetAttentionBlock",
    "SetDataSplitter",
    "SetTransformerAtacVariationalEncoder",
    "SetTransformerEncoder",
    "atac_row_nnz",
    "build_and_attach_token_store",
    "build_chrom_vocab",
    "build_coord_table",
    "ensure_token_store_for_manager",
    "registry_for_checkpoint",
    "build_genomic_rank",
    "build_token_store",
    "csr_batch_to_tokens",
    "parse_peak_name",
    "precompute_atac_token_sequences",
    "sample_balanced_atac_loss_indices",
    "store_atac_nnz_lengths",
    "store_precomputed_atac_tokens",
    "tokenize_atac",
]
