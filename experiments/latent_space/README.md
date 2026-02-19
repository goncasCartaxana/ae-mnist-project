# Experiment: Latent Space Organization

## Goal
Evaluate how well each autoencoder variant organizes the latent space — 
whether digits cluster meaningfully and whether the space generalizes smoothly.

## Models Compared
| Run | Architecture | Latent Dim |
|-----|-------------|------------|
| `<run_name>` | `<e.g. Vanilla AE>` | `<e.g. 32>` |
| `<run_name>` | `<e.g. VAE>` | `<e.g. 64>` |

## Metrics
- `<e.g. Silhouette Score>` — `<brief description>`
- `<e.g. KL Divergence>` — `<brief description>`

## Visualizations
- `<e.g. t-SNE plot>` — `<brief description>`
- `<e.g. Interpolation grid>` — `<brief description>`

## Structure
```
latent_space/
├── results/
│   └── <run_name>/
│       ├── tsne.png              # t-SNE or UMAP plot of latent space
│       ├── interpolation.png     # interpolation between digit classes
│       └── metrics.json          # clustering/organization scores
├── summary.csv                   # aggregated metrics across all runs
├── compare.py                    # produces comparison plots
└── README.md
```

## How to Run
```bash
python experiments/latent_space/compare.py --runs <run_name_1> <run_name_2>
```

## Results Summary
_To be filled after running the experiment._

| Run | Silhouette Score | `<metric>` |
|-----|-----------------|------------|
| `<run_name>` | `<value>` | `<value>` |

## Notes
`<Any observations, caveats, or decisions made during this experiment.>`

