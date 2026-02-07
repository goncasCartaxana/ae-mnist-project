import torch.nn as nn

class VAEEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """ 
        Defines the layers of the encoder network.
        """ 
        super(VAEEncoder, self).__init__()

        # list of hidden layers (from YAML!)
        self.hidden_dims = hidden_dims # e.g. [512, 256, 128]

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
        Defines the forward pass of the encoder network
        Input:
            x: input data points
        Output: 
            mean, log_var: latent space's mean and log variance vectors 
        """
        # start with input
        h = x
        # Process through each layer
        for layer in self.hidden_layers:
            h = layer(h)
        # Output
        return self.fc_mean(h), self.fc_logvar(h)

