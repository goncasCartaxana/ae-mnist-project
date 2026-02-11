"""
Main script to run the training process for the MNIST autoencoder project. 
It initializes the Trainer class, readies the dataloader, gets the configuration file, and starts the training loop. 
After training, saves the results (models and metrics) in a structured format for easy analysis and reproducibility.
"""

import sys
from pathlib import Path
import yaml
import importlib


# Allow Python to see local packages # may not be needed
ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

from data.dataloader import get_mnist_loaders
from training.trainer import Trainer

def load_model_class(model_type: str):
    """
    Dynamically loads model class from models/{model_type}.py.
    
    Args:
        model_type (str): 'vae', 'aae', 'vanilla_ae' etc.
        
    Returns:
        ModelClass: e.g. Vae, VanillaAe (nn.Module subclass) 
        Convention: Which is the model_type in CamelCase.
        
    Raises:
        AttributeError: If class not found in models/{model_type}.py
    """

    module_name = f"models.{model_type}"
    class_name = "".join(word.capitalize() for word in model_type.split("_"))
    # Does CamelCase: vanilla_ae -> VanillaAe & variational_ae -> VariationalAe
    
    print(f"DEBUG: Loading {model_type} → {module_name}.{class_name}")

    module = importlib.import_module(module_name)
      
    return getattr(module, class_name)

def main():
    """
    Run complete VAE/AE experiment from config.yaml.
    """

    print("Starting AE MNIST experiment...")

    # Load config.yaml
    exp_dir = Path.cwd()   # (e.g. experiments/exp001_vae_base)
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"Config loaded: {config_path}")

    # Load model dynamically
    print(f"  model_type = '{config['model_type']}'")
    ModelClass = load_model_class(config["model_type"])

    print("Building model...")
    model = ModelClass(**config['model_params'])

    # Create dataloaders
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist_loaders(config)
    
    # Train
    print("Starting training...")
    trainer = Trainer(model, config = config)
    trainer.train(train_loader, test_loader)
    trainer.save_results(exp_dir)

    print(f"Complete! Results: {exp_dir}/results/")

if __name__ == "__main__":
    main()
