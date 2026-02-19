# Experiment: Reconstruction Quality

## Goal
Compare how well each autoencoder variant reconstructs MNIST digits from the test set.

## Models Compared
| Run | Architecture | Latent Dim |
|-----|-------------|------------ |
| `<run_name>` | `<e.g. Vanilla AE>` | `<e.g. 32>` |
| `<run_name>` | `<e.g. VAE>` | `<e.g. 64>` |

## Metrics
- `<e.g. MSE>` — `<brief description>`
- `<e.g. SSIM>` — `<brief description>`

## Structure
```
reconstruction/
├── results/
│   └── <run_name>/
│       ├── reconstructed/    # .png reconstructions per digit
│       └── metrics.json      # metric scores for this run
├── summary.csv               # aggregated metrics across all runs
├── compare.py                # loads results and produces comparison plots
└── README.md
```

## How to Run
```bash
python experiments/reconstruction/compare.py --runs <run_name_1> <run_name_2>
```

## Results Summary
_To be filled after running the experiment._

| Run | MSE | SSIM |
|-----|-----|------|
| `<run_name>` | `<value>` | `<value>` |

## Notes
`<Any observations, caveats, or decisions made during this experiment.>`



