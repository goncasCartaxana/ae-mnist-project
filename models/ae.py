import torch.nn as nn
import torch

from .ae_encoder import AeEncoder
from .decoder import Decoder


class Ae(nn.Module):
    """
    Vanilla Autoencoder (AE) for MNIST reconstruction.
    
    Architecture: Encoder → Latent code → Decoder
    Input → deterministic latent z → Reconstruction (no KL divergence)
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Args:
            input_dim (int): Flattened MNIST size (28×28 = 784)
            hidden_dims (list): Encoder/decoder layer sizes (e.g. [512, 256, 128])
            latent_dim (int): Bottleneck dimension (e.g. 32)
        """
        super(Ae, self).__init__()
        
        # Symmetric encoder/decoder via config
        self.encoder = AeEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)
    
    def forward(self, x):
        """
        Full AE forward pass for training.
        
        Args:
            x: Input batch [batch_size, input_dim] (e.g. [128, 784])
        Returns:
            x_hat: Reconstruction [batch_size, input_dim]
        """
        z = self.encoder(x)           # Deterministic output
        x_hat = self.decoder(z)
        
        return x_hat
