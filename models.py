import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, z_dim)
        
    def forward(self,x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin1(x))
        z = self.lin2(x)
        return z

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(z_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)
        self.n = int(np.sqrt(input_dim))
    
    def forward(self, z):
        z = F.relu(self.lin1(z))
        x = self.lin2(z)
        return x.reshape((-1, 1, self.n, self.n))

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        
    def forward(self,x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin1(x))
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class Autoencoder(nn.Module):
    @classmethod
    def init_variational(cls, input_dim, hidden_dim, z_dim, **args):
        encoder = VAE(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim)
        decoder = Decoder(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim)
        return cls(encoder, decoder)

    @classmethod
    def init_vanilla(cls, input_dim, hidden_dim, z_dim):
        encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim)
        decoder = Decoder(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim)
        return cls(encoder, decoder)

    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return torch.sigmoid(z.reshape(x.shape))
