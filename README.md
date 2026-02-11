# MNIST Autoencoder: Reconstruction and Representation Learning

A modular PyTorch implementation of AutoEncoder variants (e.g. Vanilla AE, VAE) for image reconstruction and latent space analysis on the MNIST dataset.


## 🎯 Objectives
- **Primary**: Recreate (compress & reconstruct) MNIST digits 
- **Secondary**: Visualize latent space clustering by digit class.

## 📂 Project Structure
```
├── data/                   # MNIST .gz + dataloaders
├── experiments/            # Per-exp: config + results/
├── models/                 # Architectures (Encoders, Decoders, AEs)
├── training/               # Trainer + losses (MSE, KL)
└── main.py                 # Orchestration script via config.yaml
```

### Per Experiment
```
experiments/exp001_<name>/  
├── config.yaml                 # Hyperparameters
└── results/                    # Created while and after training
    ├── model.pth               # Model of last epoch
    ├── best_model.pth          # Model at lowest validation loss
    ├── metrics.json            # Summary: config and losses & nº of epochs
    └── training_history.csv    # Per-epoch logs for loss visualization
```
* The `exp001_` prefix: Ensures experiments are sorted via 3-digit numbers (e.g. 001).


### Models
Define model classes:
```
models/
├── __init__.py
├── vanilla_encoder.py  # MLP layers 
├── vae_encoder.py      # MLP layers  w/ mean_log 
├── decoder.py          # Generic MLP Decoder
├── variational_ae.py   # VAE Class
└── vanilla_ae.py       # Vanilla AE Class
```

### training
```
├── training/  
│   ├── __init__.py      
|   ├── Trainer      # The Trainer class: manages loops, checkpoints, and logging
|   └── losses       # Defines loss functions: Reconstruction (MSE/BCE) and KL-Divergence
```

### data
```
data/
├── mnist/          # Auto-downloaded .gz files
└── dataloader.py   # PyTorch Dataset and DataLoader wrappers
```

## ⚙️ Config-Driven Experiments
### Config Structure
```yaml
# Model: Architecture selection + params
model_type: vae
model_params:
  input_dim: 784
  hidden_dims: [512,256]
  latent_dim: 32

# Training: Training hyperparameters
loss_type: vae
lr: 0.001
epochs: 50

# Data: Data pipeline params
batch_size: 128
data_root: "data"
```

### loss_type & model_type
They serve different purposes:
**`model_type`**: "Which model file/class to load"  
**`loss_type`**: "How to compute the loss function"

## 🛠️ Quick Start
Usage of `uv` is recommended because:
- Way faster than pip, auto virtual env, reliable env.
- Uses both pyproject.toml + uv.lock.
- creates .venv if needed.

```bash
# Install dependencies 
uv sync
 
# Run experiments
cd <experiment_path>
uv run "<path>/main.py"
```

Alternative (pip):
```bash
# python -m venv .venv     # Create
# source .venv/bin/activate  # Activate (every new terminal)
pip install -r requirements.txt
python main.py --config experiments/exp001_vae_base/config.yaml
```

## 🖼️ Data Format (MNIST)
```
mnist/
├── train-images-idx3-ubyte.gz    # Train images (60,000)
├── train-labels-idx1-ubyte.gz    # Train labels (60,000)  
├── t10k-images-idx3-ubyte.gz     # Test images  (10,000)
└── t10k-labels-idx1-ubyte.gz     # Test labels  (10,000)
```
- `idx1-ubyte`: labels (1 number per image) 
- `idx3-ubyte`: images (count + height + width + pixels)
- `*-images-idx3-ubyte.gz`: [header][60k][pixels]
- `*-labels-idx1-ubyte.gz`: [header][60k][0-9 labels]

