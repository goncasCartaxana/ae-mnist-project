# MNIST Autoencoder: Reconstruction and Representation Learning

A modular Python project focused on building an Autoencoder for the MNIST dataset. 

## 🎯 Project Objectives

- **Primary: Image Reconstruction** - Train AutoEncoder Models (and variants) to recreate (compress and reconstruct) handwritten digits.

- **Secondary: Analyze the learned representations** - Visualize the distribution of latent vectors and observe how the model naturally clusters different digit classes.


## 📂 Project Structure
├── data/           # Dataset storage and loading logic
├── experiments/    # Experiment tracking: models, logs, and metrics
├── models/         # Architectures (Encoders, Decoders, VAEs)
├── training/       # Training loops and loss functions
└── main.py         # Orchestration script (Config -> Data -> Model)


### Experiments
```
experiments/        # Saves each experiment       
├── exp001_vae_base/
├── exp002_vae_deep/
└── exp003_aae/
```

* The `exp001_` prefix: Ensures experiments are sorted via 3-digit numbers (e.g. 001).

#### Per Experiment
```
exp001_vae_base/
├── config.yaml             # Defines model's hyperparameters
└── results/
    ├── model.pth               # Model of last epoch
    ├── best_model.pth          # Model at best epoch (Saved weights for the lowest validation loss).
    ├── metrics.json            # Readable Summary: config, train_losses, test_losses, final/best
    └── training_history.csv    # Per-epoch logs for loss visualization (ready for Excel, pandas, matplotlib, etc)
```

### Models
```
├── models/
│   ├── __init__.py
|   ├── vanilla_encoder.py  # MLP layers
│   ├── vae_encoder.py      # MLP layers  w/ mean_log 
│   ├── decoder.py          # Generic MLP Decoder used by all models
│   ├── variational_ae.py   # VAE Class (Combines VAEEncoder + Decoder)
|   └── vanilla_ae.py       # Vanilla AE Class (Combines VanillaEncoder + Decoder)
```


### training
```
├── training/  
│   ├── __init__.py      
|   ├── Trainer      # The Trainer class: manages loops, checkpoints, and logging
|   └── losses       # Reconstruction (MSE/BCE) and KL-Divergence functions
```

### data
```
├── data/
│   ├── mnist/          # [Ignored] Raw .gz IDX files (Auto-downloaded)
│   └── dataloader.py   # PyTorch Dataset and DataLoader wrappers
```

#### mnist
```
mnist/
├── train-images-idx3-ubyte.gz    # Train images (60,000)
├── train-labels-idx1-ubyte.gz    # Train labels (60,000)  
├── t10k-images-idx3-ubyte.gz     # Test images  (10,000)
└── t10k-labels-idx1-ubyte.gz     # Test labels  (10,000)
```

- IDX1 (labels): 1 dim → magic=2049 → (magic, n labels) = [magic][n_labels][0][1][2]...
- IDX3 (images): 3 dim → magic=2051 → (magic, n, 28×28) = [magic][n][28][28][pixels]




