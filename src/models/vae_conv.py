"""
Convolutional Variational Autoencoder (MEDIUM task).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from tqdm import tqdm


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for log-mel spectrograms.
    Input shape: (batch, 1, n_mels, time_frames)
    """
    
    def __init__(self, input_shape: tuple = (1, 128, 431), latent_dim: int = 32):
        """
        Initialize Conv VAE.
        
        Args:
            input_shape: (channels, height, width) = (1, n_mels, time_frames)
            latent_dim: Latent space dimension
        """
        super(ConvVAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        channels, height, width = input_shape
        
        # Encoder
        self.encoder = nn.Sequential(
            # Conv1: (1, 128, 431) -> (32, 64, 215)
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Conv2: (32, 64, 215) -> (64, 32, 107)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Conv3: (64, 32, 107) -> (128, 16, 53)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Conv4: (128, 16, 53) -> (256, 8, 26)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        # Calculate flattened dimension after convolutions
        # For input (1, 128, 431): after 4 conv layers with stride 2
        # Height: 128 -> 64 -> 32 -> 16 -> 8
        # Width: 431 -> 215 -> 107 -> 53 -> 26
        self.flatten_dim = 256 * 8 * 26  # 53248
        
        # Latent space
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder input
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            # DeConv1: (256, 8, 26) -> (128, 16, 53)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # DeConv2: (128, 16, 53) -> (64, 32, 107)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # DeConv3: (64, 32, 107) -> (32, 64, 215)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # DeConv4: (32, 64, 215) -> (1, 128, 431)
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def encode(self, x):
        """
        Encode input to latent parameters.
        
        Returns:
            mu, logvar
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick with clamping for numerical stability."""
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-20, max=20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 26)  # Reshape to conv shape
        x_recon = self.decoder(h)
        
        # Crop/pad to match input shape if needed
        _, _, h_out, w_out = x_recon.shape
        _, h_target, w_target = self.input_shape
        
        if h_out != h_target or w_out != w_target:
            x_recon = F.interpolate(x_recon, size=(h_target, w_target), mode='bilinear')
        
        return x_recon
    
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


def vae_conv_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    Conv VAE loss = Reconstruction loss + KL divergence.
    """
    # Reconstruction loss (MSE or BCE)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # Clamp logvar for numerical stability in KL computation
    logvar = torch.clamp(logvar, min=-20, max=20)
    
    # KL divergence with numerical stability
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Ensure no NaN
    if torch.isnan(kl_loss):
        kl_loss = torch.tensor(0.0, device=x.device)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_vae_conv(model: ConvVAE,
                   features: np.ndarray,
                   epochs: int = 50,
                   batch_size: int = 32,
                   learning_rate: float = 1e-3,
                   beta: float = 1.0,
                   device: torch.device = None) -> dict:
    """
    Train Conv VAE.
    
    Args:
        model: ConvVAE model
        features: Feature array (N, C, H, W) or (N, H, W)
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
    
    # Ensure features have channel dimension
    if features.ndim == 3:
        features = np.expand_dims(features, axis=1)
    
    # Replace NaN values and normalize features for numerical stability
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize to [0, 1] range for sigmoid output
    feat_min = features.min()
    feat_max = features.max()
    if feat_max - feat_min > 0:
        features = (features - feat_min) / (feat_max - feat_min)
    
    # Prepare data
    dataset = TensorDataset(torch.FloatTensor(features))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
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
            loss, recon_loss, kl_loss = vae_conv_loss(x_recon, x, mu, logvar, beta)
            
            # Backward with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


