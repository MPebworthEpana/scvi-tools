# Set Transformer ATAC Encoding in SETVI

This document maps the mathematics in `SetTransformer.md` to the SETVI implementation
and describes the token-native ATAC data path.

SETVI inherits from **MultiVI** (not MambaVI). Shared ATAC token-streaming infrastructure
lives in the independent `scvi.tokenized` sub-package.

Implementation references:

| Component | Source file |
|-----------|-------------|
| Token-streaming package | `src/scvi/tokenized/` |
| Tiered token store | `src/scvi/tokenized/_token_store.py` |
| Set Transformer blocks | `src/scvi/tokenized/_set_transformer.py` |
| ATAC variational encoder | `src/scvi/tokenized/_set_transformer_atac_variational.py` |
| Peak embeddings | `src/scvi/tokenized/_embeddings.py` |
| CSR batch tokenization | `src/scvi/tokenized/_csr_batch_tokenize.py` |
| Token-native dataset | `src/scvi/tokenized/_dataset.py` |
| Module integration | `src/scvi/module/_setvae.py` |
| Model API | `src/scvi/model/_setvi.py` |

## Set Transformer blocks

- **MAB(X, Y)**: multihead attention with pre-LN residuals and row-wise FFN.
- **ISAB_m(X)**: inducing points compress the set in `O(n m)` before set elements attend back.
- **PMA_k(Z)**: learnable seed queries pool the set to `k` invariant summaries.
- **CardinalityFiLM**: optional; conditions the pooled vector on `(n, N, n/N, log n, log N, log(N/n))`.
  **Default `use_cardinality_film=False`.** With FiLM on, depth (open-peak count) is injected into
  the ATAC latent fingerprint and collapses the embedding to a 1D depth curve (corr ~ −1 with peak
  count). Depth is already supplied to the PeakVI decoder via `libsize_acc` in `SETVAE.inference`,
  so FiLM is redundant for reconstruction and harmful for biology in `z`.
- **Head LayerNorm**: `SetTransformerAtacVariationalEncoder` applies `nn.LayerNorm(d_model)` to the
  PMA+FiLM pooled fingerprint before the Gaussian heads, mirroring the FC RNA encoder's trailing
  LayerNorm and keeping ATAC/RNA posterior scales matched.
- **PMA with sampling correction**: optional `-log(pi_i)` logit bias for inclusion-reweighted pooling.

## Token-native ATAC data path

SETVI never ships the wide `(n_obs, n_regions)` sparse ATAC matrix across dataloader workers.
Instead:

1. `setup_mudata` registers a deterministic `coord_table`, `chrom_vocab`, and genomic rank.
2. A tiered **token store** (`auto` / `gpu` / `ram` / `mmap`) is built once from CSR rows.
3. `SetAnnTorchDataset` gathers encoder tokens per batch (`ram`/`mmap`) or defers to the module (`gpu`).
4. `SETVAE.loss` reconstructs the accessibility target from full target tokens on GPU.
5. Checkpoints persist only a lightweight store **handle**; the live store is rebuilt on load.

### Token store tiers

| Tier | Where tokens live | Who gathers encoder tokens |
|------|-------------------|----------------------------|
| `gpu` | CUDA buffers on the store | `SETVAE` via `gather_torch` |
| `ram` | In-memory ragged arrays | `SetAnnTorchDataset` |
| `mmap` | Memory-mapped files | `SetAnnTorchDataset` (reopened on load) |

### Lean checkpoints

`SETVI._get_user_attributes` strips `token_store` and per-cell `nnz_lengths` from the serialized
registry. Only `token_store_handle`, `coord_table`, `chrom_vocab`, and `genomic_rank` are saved
(all `O(n_regions)`, independent of `n_obs`).

### Query / reference inference

`get_latent_representation(adata=query)` and `get_reconstruction_error(adata=query)` build an
ephemeral RAM token store keyed to the query manager (with `_indices = arange(n_obs)`), temporarily
swap it onto the module, and restore the training store afterward.

## Differences from MambaVI

SETVI does **not** inherit MambaVAE features:

- No vertical bridge / OT alignment
- No reconstruction normalization modes
- No `balanced_subsample` ATAC loss
- No staged / unimodal pretrain training phases

## Variational contract

`SetTransformerAtacVariationalEncoder` returns `(q_m, q_v, z)` with shape `(batch, n_latent)`,
matching `MambaAtacVariationalEncoder` and plugging into `mix_modalities` unchanged.

## Performance defaults

- **Dataloader**: `train()` defaults to `num_workers=0`, `load_sparse_tensor=False` (wide ATAC
  never crosses worker IPC).
- **Length bucketing**: `atac_length_bucketing=True` groups cells of similar ATAC nnz.
- **Vectorized gather**: ragged-to-padded gather in `AtacTokenStore` (per-row fallback only for
  truncated encoder rows).
- **Static peak embeddings**: `AtacPeakEmbedding.embed_ids` uses a setup-time cache.
- **SDPA attention**: Set Transformer blocks use `F.scaled_dot_product_attention`.

### Checkpoint compatibility

Encoder architecture changes (static peak cache, `value_bias`, deterministic `chrom_vocab` sizing)
invalidate checkpoints saved before those updates.

## Usage

```python
from scvi.model import SETVI

SETVI.setup_mudata(
    mdata,
    modalities={"atac_layer": "accessibility", "rna_layer": "rna"},
    batch_key="batch",
    atac_token_store="auto",
)
model = SETVI(mdata, st_n_layers=2, st_n_inducing=32)
model.train(max_epochs=100)
latent = model.get_latent_representation()
query_latent = model.get_latent_representation(adata=query_mdata)
```
