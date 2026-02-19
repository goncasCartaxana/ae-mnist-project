# Experiment: Latent Space Classification

## Goal
Train a classifier on top of each model's frozen latent space and compare 
classification performance — a proxy for how semantically meaningful each 
latent representation is.

## Models Compared
| Run | Architecture | Latent Dim |
|-----|-------------|------------|
| `<run_name>` | `<e.g. Vanilla AE>` | `<e.g. 32>` |
| `<run_name>` | `<e.g. VAE>` | `<e.g. 64>` |

## Classifier
- Architecture: `<e.g. Linear layer, MLP>`
- Input: frozen latent representation from each autoencoder
- Trained on: `<e.g. MNIST train split>`
- Evaluated on: `<e.g. MNIST test split>`

## Metrics
- `<e.g. Accuracy>` — `<brief description>`
- `<e.g. F1 Score>` — `<brief description>`

## Structure
```
classification/
├── classifier/               # the classifier trained on latent representations
│   ├── config.yaml           # classifier architecture and training config
│   └── model.pt              # saved classifier weights
├── results/
│   └── <run_name>/
│       └── metrics.json      # accuracy, f1, confusion matrix, etc.
├── summary.csv               # aggregated metrics across all runs
├── compare.py                # produces comparison plots and tables
└── README.md
```

## How to Run
```bash
# Step 1: train the classifier on a run's latent space
python experiments/classification/train_classifier.py --run <run_name>

# Step 2: compare across all runs
python experiments/classification/compare.py --runs <run_name_1> <run_name_2>
```

## Results Summary
_To be filled after running the experiment._

| Run | Accuracy | F1 Score |
|-----|----------|----------|
| `<run_name>` | `<value>` | `<value>` |

## Notes
`<Any observations, caveats, or decisions made during this experiment.>`
