# Set Transformer ATAC Encoding in SETVI

This document maps the mathematics in `SetTransformer.md` to the SETVI implementation
and describes how CSR token streaming differs from the MAMBAVI precompute path.

Implementation references:

| Component | Source file |
|-----------|-------------|
| Set Transformer blocks | `src/scvi/encoders/_set_transformer.py` |
| ATAC variational encoder | `src/scvi/encoders/_set_transformer_atac_variational.py` |
| Peak embeddings | `src/scvi/encoders/_embeddings.py` |
| CSR batch tokenization | `src/scvi/encoders/_csr_batch_tokenize.py` |
| CSR streaming dataset | `src/scvi/dataloaders/_set_dataset.py` |
| Module integration | `src/scvi/module/_setvae.py` |
| Model API | `src/scvi/model/_setvi.py` |

## Set Transformer blocks

- **MAB(X, Y)**: multihead attention with pre-LN residuals and row-wise FFN.
- **ISAB_m(X)**: inducing points compress the set in `O(n m)` before set elements attend back.
- **PMA_k(Z)**: learnable seed queries pool the set to `k` invariant summaries.
- **CardinalityFiLM**: conditions the pooled vector on `(n, N, n/N, log n, log N, log(N/n))`.
- **PMA with sampling correction**: optional `-log(pi_i)` logit bias for inclusion-reweighted pooling.

## CSR streaming vs MAMBAVI precompute

MAMBAVI precomputes padded `(n_obs, max_len)` token id arrays at setup and loads them in
`MambaAnnTorchDataset`. SETVI instead:

1. Registers `coord_table` and genomic rank for all peaks at setup.
2. Stores per-cell `nnz` lengths (cheap CSR `indptr` scan) for optional length bucketing.
3. Tokenizes open peaks from CSR row slices in `SetAnnTorchDataset.__getitem__` via
   `csr_batch_to_tokens`.
4. Defaults to `atac_loss_mode="balanced_subsample"`, excluding dense `ATAC_X` from train
   batches so the matrix is never densified during training.

## Variational contract

`SetTransformerAtacVariationalEncoder` returns `(q_m, q_v, z)` with shape `(batch, n_latent)`,
matching `MambaAtacVariationalEncoder` and plugging into `mix_modalities` unchanged.

## Usage

```python
from scvi.model import SETVI

SETVI.setup_mudata(mdata, modalities={"atac_layer": "accessibility"}, batch_key="batch")
model = SETVI(mdata, st_n_layers=2, st_n_inducing=32)
model.train(max_epochs=100)
latent = model.get_latent_representation()
```
