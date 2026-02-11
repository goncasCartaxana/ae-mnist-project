import torch.nn as nn
import torch

from .vae_encoder import VaeEncoder  
from .decoder import Decoder



class Vae(nn.Module):
    """
    Variational Autoencoder (VAE) for MNIST generation.
    
    Architecture: Encoder → Reparameterization → Decoder
    Input → latent space (μ, logσ²) → z ~ N(μ, σ) → Reconstruction
    """
    
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Args:
            input_dim (int): Flattened MNIST size (28×28 = 784)
            hidden_dims (list): Encoder/decoder layer sizes (e.g. [512, 256, 128])
            latent_dim (int): Latent space dimension (e.g. 20)
        """
        super(Vae, self).__init__()
        
        # Symmetric encoder/decoder via config
        self.encoder = VaeEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)
    
    def reparameterization(self, mean, log_var):
        """
        Reparameterization trick: z = μ + σ * ε where ε ~ N(0,1)
        
        Args:
            mean:    [batch_size, latent_dim]
            log_var: [batch_size, latent_dim]
        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std 
    
    def forward(self, x):
        """
        Full VAE forward pass for training.
        
        Args:
            x: Input batch [batch_size, input_dim] (e.g. [128, 784])
        Returns:
            (x_hat, mean, log_var):
                - x_hat:   Reconstruction [batch_size, input_dim]
                - mean:    Latent mean [batch_size, latent_dim]
                - log_var: Latent log-variance [batch_size, latent_dim]
        """

        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        
        return x_hat, mean, log_var

