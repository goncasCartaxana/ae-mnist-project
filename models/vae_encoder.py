import torch.nn as nn

class VaeEncoder(nn.Module): 
    """
    VAE Encoder: Maps input → latent space (μ, logσ²).
    
    Args:
        input_dim: Flattened MNIST (28*28 = 784)
        hidden_dims: List of hidden layer sizes (e.g. [512, 256, 128])
        latent_dim: Output latent space dimension (e.g. 20)
    
    Returns: (mean, log_var) for reparameterization
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """        
        Args:
            input_dim (int): Flattened input size (MNIST: 784)
            hidden_dims (list): Hidden layer sizes (e.g. [512, 256, 128])
            latent_dim (int): VAE latent space dimension (e.g. 20)
        """

        super(VaeEncoder, self).__init__()

        # List of hidden layers
        self.hidden_dims = hidden_dims

        # Build layers dynamically
        layers = []
        dims = [input_dim] + hidden_dims
        
        # Neural network of fully connected layers
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1])) # fc neurons
            layers.append(nn.LeakyReLU(0.2)) # activation functions
        
        self.hidden_layers = nn.ModuleList(layers)

        # Output layers for VAE's Latent Space
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)


    def forward(self, x):
        """
        Forward pass: input → latent parameters.
        
        Args:
            x: Input tensor [batch_size, input_dim] (e.g. [32, 784])
            
        Returns:
            (mean, log_var): Latent space parameters 
            - mean:    [batch_size, latent_dim] (e.g. [32, 20])
            - log_var: [batch_size, latent_dim] (e.g. [32, 20])
            
        For reparameterization: z = mean + std * epsilon
        """

        # Start with input
        h = x
        # Process through each layer
        for layer in self.hidden_layers:
            h = layer(h)
        # Output
        return self.fc_mean(h), self.fc_logvar(h)

