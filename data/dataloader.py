"""
The dataloader module creates data loaders for the MNIST dataset, allowing for configurable batch size and data directory.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_loaders(config: dict):
    batch_size = config.get("batch_size", 128)
    data_root = config.get("data_root", "./data")

    transform = transforms.ToTensor()

    # 
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


