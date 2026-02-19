#!/usr/bin/env python
"""
Create stable MNIST original digits grid (0-9) for all experiments.
"""
import argparse
import json
import os
import random
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data', help='MNIST data root')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducible selection')
    parser.add_argument('--output-dir', default='reference_digits', help='Where to save originals')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Create test dataset
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(
        root=Path(args.data_root),
        train=False,
        download=True,
        transform=transform,
    )
    
    # Shuffle ALL indices deterministically
    all_indices = list(range(len(test_dataset)))
    random.shuffle(all_indices)
    
    # Find first occurrence of each digit 0-9
    selected_indices = {}
    for idx in all_indices:
        _, label = test_dataset[idx]
        if label not in selected_indices:
            selected_indices[label] = idx
        if len(selected_indices) == 10:
            break
    
    # Save indices
    indices_path = os.path.join(args.output_dir, f'selected_indices_seed{args.seed}.json')
    with open(indices_path, 'w') as f:
        json.dump({'seed': args.seed, 'indices': selected_indices}, f, indent=2)
    
    # Load the 10 selected images
    images = []
    for digit in range(10):
        idx = selected_indices[digit]
        img, _ = test_dataset[idx]
        img = img.squeeze().numpy()  # [28, 28]
        images.append(img)
    
    # Create clean minimalistic figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    
    # Add title
    fig.suptitle(f'MNIST Original Digits (Seed {args.seed})', 
                 fontsize=16, y=0.98)
    
    # Plot each digit
    for digit in range(10):
        row = digit // 5
        col = digit % 5
        ax = axes[row, col]
        
        # Show image
        ax.imshow(images[digit], cmap='gray', interpolation='nearest')
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
    
    # Save
    output_path = os.path.join(args.output_dir, f'originals_seed{args.seed}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', pad_inches=0.2)
    plt.close()
    
    print(f"Saved originals: {output_path}")
    print(f"Indices saved: {indices_path}")
    print("Selected indices:", [selected_indices[i] for i in range(10)])

if __name__ == '__main__':
    main()