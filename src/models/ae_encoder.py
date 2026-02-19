import torch.nn as nn


class AeEncoder(nn.Module): 
    """
    Vanilla AE Encoder: Maps input → latent code (deterministic).
    
    Args:
        input_dim: Flattened MNIST (28*28 = 784)
        hidden_dims: List of hidden layer sizes (e.g. [512, 256, 128])
        latent_dim: Bottleneck latent space dimension (e.g. 32)
    
    Returns: z - deterministic latent representation [batch_size, latent_dim]
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Args:
            input_dim (int): Flattened input size (MNIST: 784)
            hidden_dims (list): Hidden layer sizes (e.g. [512, 256, 128])
            latent_dim (int): Bottleneck dimension (e.g. 32)
        """
        super(AeEncoder, self).__init__()

        self.hidden_dims = hidden_dims

        # Build hidden layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
        
        self.hidden_layers = nn.ModuleList(layers)

        # Output layer 
        self.fc_latent = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        """
        Forward pass: input → latent code.
        
        Args:
            x: Input tensor [batch_size, input_dim] (e.g. [32, 784])
            
        Returns:
            z: Latent code [batch_size, latent_dim] (e.g. [32, 32])
        """
        # Process through hidden layers
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
        
        # SINGLE output → deterministic latent code
        z = self.fc_latent(h)
        return z
