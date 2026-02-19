#!/usr/bin/env python
"""
Reconstruct MNIST digits 0-9 using trained autoencoder models.
Uses same indices, via a seed, as create_original_digits.py for comparison.
"""

import argparse
import json
import os
import torch
import yaml
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pathlib import Path

# reconstruct_digits.py
def load_model(model_type, model_params, checkpoint_path, device):
    """Simple model loader"""

    model_map = {
        # config.model_type: (module_name, class_name)
        'vanilla_autoencoder': ('models.vanilla_autoencoder ', 'VanillaAutoencoder'),
        'variational_autoencoder': ('models.variational_autoencoder', 'VariationalAutoencoder'),
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(model_map)}")
    
    module_name, class_name = model_map[model_type]
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)
    model = model_class(**model_params)   
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Extract Weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    # Parsing
    print("Parsing inputs...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', required=True, help='Path to expXXX folder')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference-dir', default='reference_digits')
    args = parser.parse_args()
    
    # Paths
    print("Determining paths...")
    config_path = os.path.join(args.experiment_dir, 'config.yaml')
    checkpoint_path = os.path.join(args.experiment_dir, 'results', 'best_model.pth')
    recreate_dir = os.path.join(args.experiment_dir, 'recreate')
    os.makedirs(recreate_dir, exist_ok=True)
    reference_indices_path = os.path.join(args.reference_dir, f'selected_indices_seed{args.seed}.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    print("Loading config...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_type = config['model_type']
    model_params = config['model_params']
    latent_dim = model_params['latent_dim']
    print(f"Loading {model_type} (latent_dim={latent_dim}) from {checkpoint_path}")
    
    # Load selected indices
    with open(reference_indices_path) as f:
        indices_data = json.load(f)
    assert indices_data['seed'] == args.seed, "Seed mismatch!"
    
    selected_indices = indices_data['indices']
    
    # Load test dataset (same transform as training)
    transform = transforms.ToTensor() 
    test_dataset = datasets.MNIST(
        root=Path(args.data_root),
        train=False,
        download=True,
        transform=transform,
    )
    

    # Load originals and get reconstructions
    reconstructions_vis = []
    with torch.no_grad():
        model = load_model(model_type, model_params, checkpoint_path, device)
        
        for digit in range(10):
            idx = selected_indices[str(digit)]
            img, label = test_dataset[idx]
            
            # Flatten for model (matches training)
            img_flat = img.view(1, -1).to(device)  # [1, 784]
            
            # Forward pass
            if model_type == 'variational_autoencoder':
                recon_flat, *_ = model(img_flat)  # [1, 784]
            else:
                recon_flat = model(img_flat)      # [1, 784]
            
            # Reshape back to image
            recon_img = recon_flat.view(28, 28).cpu()  # [28, 28]
            recon_vis = torch.clamp(recon_img, 0, 1)
            
            reconstructions_vis.append(recon_vis)

    # Create 2x5 grid for reconstructions ONLY
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    # Plot each reconstructed digit
    for digit in range(10):
        row = digit // 5
        col = digit % 5
        ax = axes[row, col]
        
        # Show image
        ax.imshow(reconstructions_vis[digit], cmap='gray', interpolation='nearest')
        ax.axis('off')
        
        # Add simple label below
        ax.text(0.5, -0.15, str(digit),
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=18, color='black')

    # Adjust spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.4,
                    left=0.05, right=0.95,
                    top=0.92, bottom=0.05)

    # Title
    title = f"MNIST Reconstructions ({model_type.replace('_', ' ').title()}, Latent dim={latent_dim}, Seed={args.seed})"
    fig.suptitle(title, fontsize=16, weight='bold')

    # Save
    output_path = os.path.join(recreate_dir, f'reconstructions_only_seed{args.seed}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"✨ Saved reconstructions: {output_path}")

if __name__ == '__main__':
    main()
