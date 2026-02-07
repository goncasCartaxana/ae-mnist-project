import torch.nn as nn
import torch

from .vae_encoder import VAEEncoder  
from .decoder import Decoder



class VAE(nn.Module):
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        
        # Defines encoder/decoder via config
        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)
    
    def reparameterization(self, mean, log_var):
        """
        Reparameterization trick to sample from N(mean, std) from N(0,1).
        z = μ + σ * ε    where ε ~ N(0,1)
        """
        std = torch.exp(0.5 * log_var)  # std = exp(0.5 * log_var)
        eps = torch.randn_like(std)     # ε ~ N(0,1)
        return mean + eps * std         # z = μ + σ * ε
    
    def forward(self, x):
        # Encode → sample → decode
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        
        return x_hat, mean, log_var





