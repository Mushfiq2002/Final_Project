"""
Beta-VAE (HARD task option).
Beta-VAE is a VAE with weighted KL term for better disentanglement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging

from .vae_mlp import MLPVAE, vae_loss as mlp_vae_loss
from .vae_conv import ConvVAE, vae_conv_loss


class BetaVAE(nn.Module):
    """
    Beta-VAE wrapper that can use either MLP or Conv architecture.
    """
    
    def __init__(self, base_model: nn.Module, beta: float = 4.0):
        """
        Initialize Beta-VAE.
        
        Args:
            base_model: Base VAE model (MLPVAE or ConvVAE)
            beta: Beta parameter (>1 for stronger disentanglement)
        """
        super(BetaVAE, self).__init__()
        self.base_model = base_model
        self.beta = beta
    
    def encode(self, x):
        return self.base_model.encode(x)
    
    def reparameterize(self, mu, logvar):
        return self.base_model.reparameterize(mu, logvar)
    
    def decode(self, z):
        return self.base_model.decode(z)
    
    def forward(self, x):
        return self.base_model.forward(x)
    
    @property
    def latent_dim(self):
        return self.base_model.latent_dim


def train_beta_vae(model: BetaVAE,
                   features: np.ndarray,
                   epochs: int = 50,
                   batch_size: int = 64,
                   learning_rate: float = 1e-3,
                   device: torch.device = None,
                   is_conv: bool = False) -> dict:
    """
    Train Beta-VAE.
    
    Args:
        model: BetaVAE model
        features: Feature array
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        is_conv: Whether using convolutional architecture
        
    Returns:
        Dictionary with training history
    """
    if device is None:
        device = torch.device('cpu')
    
    model = model.to(device)
    beta = model.beta
    
    # Ensure features have correct shape for conv
    if is_conv and features.ndim == 3:
        features = np.expand_dims(features, axis=1)
    
    # Prepare data
    dataset = TensorDataset(torch.FloatTensor(features))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    recon_losses = []
    kl_losses = []
    
    # Select loss function
    loss_fn = vae_conv_loss if is_conv else mlp_vae_loss
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            # Forward
            x_recon, mu, logvar, z = model(x)
            
            # Loss with beta weighting
            loss, recon_loss, kl_loss = loss_fn(x_recon, x, mu, logvar, beta)
            
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
                        f"Recon: {epoch_recon:.6f}, KL: {epoch_kl:.6f} (beta={beta})")
    
    return {
        'train_losses': train_losses,
        'recon_losses': recon_losses,
        'kl_losses': kl_losses
    }


