# MultiVI pilot bio-conservation evaluation

## Summary

- Extends the MultiVI pilot baseline to report **Leiden cluster UMAP** and **scIB-style biological conservation metrics** on the mosaic pilot dataset.
- Uses **RNA-derived pseudo-labels** (Leiden on RNA PCA) as a biological reference because `complex_object_pilot.h5mu` has no cell-type annotations (`data_type` only).
- Transfers pseudo-labels to ATAC-only cells via latent kNN majority vote so metrics cover the full cohort.
- **No changes to MultiVI model code** — evaluation-only PR.

## Metrics

### Batch correction (unchanged)

| Metric | Key | Higher = better mixing? |
|--------|-----|-------------------------|
| kBET | `kbet_data_type` | Yes |
| iLISI | `ilisi_data_type` | Yes |
| Silhouette (batch) | `silhouette_data_type` | Lower magnitude / near 0 preferred |

### Biological conservation (new)

| Metric | Key | Interpretation |
|--------|-----|----------------|
| Leiden cluster count | `n_leiden_clusters` | Descriptive |
| RNA pseudo cluster count | `n_rna_pseudo_clusters` | Descriptive |
| NMI | `nmi_leiden_rna_pseudo` | Agreement: integrated Leiden vs RNA pseudo-labels |
| ARI | `ari_leiden_rna_pseudo` | Same, chance-adjusted |
| cLISI | `clisi_rna_pseudo` | Local pseudo-label purity in latent neighborhoods |
| Silhouette (bio) | `silhouette_rna_pseudo` | Global separation of pseudo-types in latent space |

## Limitations

- **Pseudo-labels are not ground truth.** They approximate biology from RNA structure only.
- **ATAC-only labels are transferred** from RNA-present neighbors in integrated latent space; treat full-cohort NMI/ARI as an integration benchmark, not annotation accuracy.
- Compare methods using the **same protocol** (k=15, Leiden res=1.0, RNA PCA 50 dims); do not compare absolute scores across datasets.

## Outputs

| File | Contents |
|------|----------|
| `multivi_pilot_umap.png` | UMAP colored by `data_type` |
| `multivi_pilot_umap_leiden.png` | UMAP colored by integrated Leiden clusters |
| `multivi_pilot_latent.npz` | `latent`, `umap`, `data_type`, `leiden`, `rna_pseudo_label`, `obs_names` |
| `multivi_pilot_metrics.json` | Batch + bio metrics |

## Test plan

```bash
cd MultiVI
PYTHONPATH=src python -m pytest tests/multivi_pilot_baseline/test_pilot_bio_metrics.py -v
```

Full pilot run (requires `complex_object_pilot.h5mu`, GPU recommended, ~150 epochs):

```bash
PYTHONPATH=src python tests/multivi_pilot_baseline/multivi_baseline.py
```

## Example metrics (MultiVI, 150 epochs, `complex_object_pilot.h5mu`)

```json
{
  "kbet_data_type": 0.42,
  "ilisi_data_type": 0.26,
  "silhouette_data_type": -0.007,
  "n_leiden_clusters": 23,
  "n_rna_pseudo_clusters": 17,
  "nmi_leiden_rna_pseudo": 0.49,
  "ari_leiden_rna_pseudo": 0.22,
  "clisi_rna_pseudo": 0.97,
  "silhouette_rna_pseudo": 0.006
}
```

Bio metrics are also written to `multivi_pilot_metrics.json` after running the baseline locally.
