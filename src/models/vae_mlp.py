"""
MLP Variational Autoencoder (EASY task).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from tqdm import tqdm


class MLPVAE(nn.Module):
    """
    Multi-Layer Perceptron Variational Autoencoder.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims: list = None):
        """
        Initialize MLP VAE.
        
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
        """
        super(MLPVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
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
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Encode input to latent parameters.
        
        Returns:
            mu, logvar
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass.
        
        Returns:
            x_recon, mu, logvar, z
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def sample(self, num_samples: int, device: torch.device):
        """
        Sample from prior p(z) = N(0, I).
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence.
    
    Args:
        x_recon: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL term (beta-VAE)
        
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence: KL(q(z|x) || p(z))
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Normalize by batch size
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_vae_mlp(model: MLPVAE,
                  features: np.ndarray,
                  epochs: int = 50,
                  batch_size: int = 64,
                  learning_rate: float = 1e-3,
                  beta: float = 1.0,
                  device: torch.device = None) -> dict:
    """
    Train MLP VAE.
    
    Args:
        model: MLPVAE model
        features: Feature array
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
    dataset = TensorDataset(torch.FloatTensor(features))
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
            x = batch[0].to(device)
            
            # Forward
            x_recon, mu, logvar, z = model(x)
            
            # Loss
            loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta)
            
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


