import torch.nn as nn
import torch


class Decoder(nn.Module):
    
    def __init__(self, latent_dim, hidden_dims, output_dim):
        """
        Defines the layers of the decoder network.
        """

        super(Decoder, self).__init__()
        
        # List of hidden layers (reversed for decoder)
        self.hidden_dims = hidden_dims  # [512, 256, 128]

        # Build layers dynamically
        layers = []
        dims = [latent_dim] + hidden_dims

        # Neural network of fully connected layers
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))     # fc neurons
            layers.append(nn.LeakyReLU(0.2))                 # activation functions
        
        # Last layer WITHOUT activation (activation in forward)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.hidden_layers = nn.ModuleList(layers)


    def forward(self, x):
        """
        Defines the forward pass of the decoder network
        Input: z (latent vector)
        Output: x_hat (reconstructed data)
        """
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
        return torch.sigmoid(h)  # Normalize to [0,1] for MNIST

