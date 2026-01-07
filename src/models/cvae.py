"""
Conditional Variational Autoencoder (HARD task option).
Conditioned on genre labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder.
    Conditioned on genre one-hot vector (10-dim).
    """
    
    def __init__(self, 
                 input_dim: int, 
                 condition_dim: int = 10,
                 latent_dim: int = 16, 
                 hidden_dims: list = None):
        """
        Initialize CVAE.
        
        Args:
            input_dim: Input feature dimension
            condition_dim: Condition dimension (number of genres)
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
        """
        super(CVAE, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Encoder: concatenate input with condition
        encoder_layers = []
        prev_dim = input_dim + condition_dim  # Concatenate x and c
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder: concatenate latent with condition
        decoder_layers = []
        prev_dim = latent_dim + condition_dim  # Concatenate z and c
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x, c):
        """
        Encode input conditioned on c.
        
        Args:
            x: Input features
            c: Condition (genre one-hot)
            
        Returns:
            mu, logvar
        """
        # Concatenate input with condition
        xc = torch.cat([x, c], dim=1)
        h = self.encoder(xc)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """
        Decode from latent conditioned on c.
        
        Args:
            z: Latent code
            c: Condition (genre one-hot)
            
        Returns:
            Reconstructed input
        """
        # Concatenate latent with condition
        zc = torch.cat([z, c], dim=1)
        x_recon = self.decoder(zc)
        return x_recon
    
    def forward(self, x, c):
        """
        Forward pass.
        
        Args:
            x: Input features
            c: Condition (genre one-hot)
            
        Returns:
            x_recon, mu, logvar, z
        """
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar, z
    
    def sample(self, num_samples: int, condition: torch.Tensor, device: torch.device):
        """
        Sample from conditional prior p(z|c).
        
        Args:
            num_samples: Number of samples
            condition: Condition vector (genre one-hot)
            device: Device
            
        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Expand condition to match batch size
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).expand(num_samples, -1)
        
        samples = self.decode(z, condition)
        return samples


def cvae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    CVAE loss = Reconstruction loss + KL divergence.
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_cvae(model: CVAE,
               features: np.ndarray,
               conditions: np.ndarray,
               epochs: int = 50,
               batch_size: int = 64,
               learning_rate: float = 1e-3,
               beta: float = 1.0,
               device: torch.device = None) -> dict:
    """
    Train CVAE.
    
    Args:
        model: CVAE model
        features: Feature array
        conditions: Condition array (genre one-hot)
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        beta: Beta parameter for KL term
        device: Device to train on
        
    Returns:
        Dictionary with training history
    """
    if device is None:
        device = torch.device('cpu')
    
    model = model.to(device)
    
    # Prepare data
    dataset = TensorDataset(
        torch.FloatTensor(features),
        torch.FloatTensor(conditions)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    recon_losses = []
    kl_losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        for batch in dataloader:
            x, c = batch
            x = x.to(device)
            c = c.to(device)
            
            # Forward
            x_recon, mu, logvar, z = model(x, c)
            
            # Loss
            loss, recon_loss, kl_loss = cvae_loss(x_recon, x, mu, logvar, beta)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(x)
            epoch_recon += recon_loss.item() * len(x)
            epoch_kl += kl_loss.item() * len(x)
        
        epoch_loss /= len(features)
        epoch_recon /= len(features)
        epoch_kl /= len(features)
        
        train_losses.append(epoch_loss)
        recon_losses.append(epoch_recon)
        kl_losses.append(epoch_kl)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, "
                        f"Recon: {epoch_recon:.6f}, KL: {epoch_kl:.6f}")
    
    return {
        'train_losses': train_losses,
        'recon_losses': recon_losses,
        'kl_losses': kl_losses
    }


