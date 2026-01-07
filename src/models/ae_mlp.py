"""
MLP Autoencoder (baseline).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from tqdm import tqdm


class MLPAE(nn.Module):
    """
    Multi-Layer Perceptron Autoencoder.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims: list = None):
        """
        Initialize MLP Autoencoder.
        
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
        """
        super(MLPAE, self).__init__()
        
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
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
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
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def train_ae(model: MLPAE,
             features: np.ndarray,
             epochs: int = 30,
             batch_size: int = 64,
             learning_rate: float = 1e-3,
             device: torch.device = None) -> dict:
    """
    Train autoencoder.
    
    Args:
        model: MLPAE model
        features: Feature array
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
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
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            # Forward
            x_recon, z = model(x)
            
            # Reconstruction loss (MSE)
            loss = F.mse_loss(x_recon, x)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(x)
        
        epoch_loss /= len(features)
        train_losses.append(epoch_loss)
        
        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")
    
    return {'train_losses': train_losses}


