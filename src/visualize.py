"""
Visualization functions for clustering results.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional
import logging


def plot_tsne(features: np.ndarray,
              labels_true: np.ndarray,
              labels_pred: Optional[np.ndarray] = None,
              title: str = "t-SNE Visualization",
              genre_names: Optional[np.ndarray] = None,
              perplexity: int = 30,
              random_state: int = 42) -> plt.Figure:
    """
    Create t-SNE visualization.
    
    Args:
        features: Feature array
        labels_true: Ground truth labels
        labels_pred: Optional predicted cluster labels
        title: Plot title
        genre_names: Optional genre names for coloring
        perplexity: t-SNE perplexity
        random_state: Random seed
        
    Returns:
        Matplotlib figure
    """
    logging.info("Computing t-SNE...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    features_2d = tsne.fit_transform(features)
    
    # Create figure
    if labels_pred is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    
    # Plot by true labels
    ax = axes[0]
    
    if genre_names is not None:
        # Use genre names for legend
        unique_genres = np.unique(genre_names)
        for genre in unique_genres:
            mask = genre_names == genre
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      alpha=0.6, s=20, label=genre)
        ax.set_title(f"{title} - True Genres")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                           c=labels_true, cmap='tab10', alpha=0.6, s=20)
        ax.set_title(f"{title} - True Labels")
        plt.colorbar(scatter, ax=ax)
    
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)
    
    # Plot by predicted labels
    if labels_pred is not None:
        ax = axes[1]
        
        # Filter out noise (-1)
        valid_mask = labels_pred != -1
        scatter = ax.scatter(features_2d[valid_mask, 0], features_2d[valid_mask, 1],
                           c=labels_pred[valid_mask], cmap='tab10', alpha=0.6, s=20)
        
        # Plot noise points separately
        if (~valid_mask).any():
            ax.scatter(features_2d[~valid_mask, 0], features_2d[~valid_mask, 1],
                      c='gray', alpha=0.3, s=10, label='Noise')
        
        ax.set_title(f"{title} - Predicted Clusters")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    return fig


def plot_umap(features: np.ndarray,
              labels_true: np.ndarray,
              labels_pred: Optional[np.ndarray] = None,
              title: str = "UMAP Visualization",
              genre_names: Optional[np.ndarray] = None,
              n_neighbors: int = 15,
              min_dist: float = 0.1,
              random_state: int = 42) -> plt.Figure:
    """
    Create UMAP visualization.
    
    Args:
        features: Feature array
        labels_true: Ground truth labels
        labels_pred: Optional predicted cluster labels
        title: Plot title
        genre_names: Optional genre names for coloring
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed
        
    Returns:
        Matplotlib figure
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn not installed. Install with: pip install umap-learn")
    
    logging.info("Computing UMAP...")
    
    # Compute UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                       min_dist=min_dist, random_state=random_state)
    features_2d = reducer.fit_transform(features)
    
    # Create figure
    if labels_pred is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    
    # Plot by true labels
    ax = axes[0]
    
    if genre_names is not None:
        unique_genres = np.unique(genre_names)
        for genre in unique_genres:
            mask = genre_names == genre
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      alpha=0.6, s=20, label=genre)
        ax.set_title(f"{title} - True Genres")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                           c=labels_true, cmap='tab10', alpha=0.6, s=20)
        ax.set_title(f"{title} - True Labels")
        plt.colorbar(scatter, ax=ax)
    
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.3)
    
    # Plot by predicted labels
    if labels_pred is not None:
        ax = axes[1]
        
        valid_mask = labels_pred != -1
        scatter = ax.scatter(features_2d[valid_mask, 0], features_2d[valid_mask, 1],
                           c=labels_pred[valid_mask], cmap='tab10', alpha=0.6, s=20)
        
        if (~valid_mask).any():
            ax.scatter(features_2d[~valid_mask, 0], features_2d[~valid_mask, 1],
                      c='gray', alpha=0.3, s=10, label='Noise')
        
        ax.set_title(f"{title} - Predicted Clusters")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    return fig


def plot_reconstruction_error_hist(original: np.ndarray,
                                   reconstructed: np.ndarray,
                                   title: str = "Reconstruction Error Distribution") -> plt.Figure:
    """
    Plot histogram of reconstruction errors.
    
    Args:
        original: Original features
        reconstructed: Reconstructed features
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Compute MSE per sample
    mse_per_sample = np.mean((original - reconstructed) ** 2, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mse_per_sample, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.axvline(np.mean(mse_per_sample), color='red', linestyle='--',
               label=f'Mean: {np.mean(mse_per_sample):.4f}')
    ax.axvline(np.median(mse_per_sample), color='green', linestyle='--',
               label=f'Median: {np.median(mse_per_sample):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_spectrogram_reconstruction(original_specs: np.ndarray,
                                   reconstructed_specs: np.ndarray,
                                   n_examples: int = 8,
                                   indices: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Plot original vs reconstructed spectrograms.
    
    Args:
        original_specs: Original spectrograms (N, C, H, W) or (N, H, W)
        reconstructed_specs: Reconstructed spectrograms
        n_examples: Number of examples to plot
        indices: Optional specific indices to plot
        
    Returns:
        Matplotlib figure
    """
    if indices is None:
        indices = np.random.choice(len(original_specs), n_examples, replace=False)
    else:
        n_examples = len(indices)
    
    # Handle channel dimension
    if original_specs.ndim == 4:
        original_specs = original_specs[:, 0, :, :]  # Take first channel
        reconstructed_specs = reconstructed_specs[:, 0, :, :]
    
    # Create grid
    fig, axes = plt.subplots(n_examples, 2, figsize=(8, 2 * n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Original
        ax = axes[i, 0]
        im = ax.imshow(original_specs[idx], aspect='auto', origin='lower', cmap='viridis')
        if i == 0:
            ax.set_title("Original")
        ax.set_ylabel(f"Sample {idx}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Reconstructed
        ax = axes[i, 1]
        im = ax.imshow(reconstructed_specs[idx], aspect='auto', origin='lower', cmap='viridis')
        if i == 0:
            ax.set_title("Reconstructed")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def plot_latent_space_2d(latent_features: np.ndarray,
                         labels_true: np.ndarray,
                         title: str = "2D Latent Space",
                         genre_names: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Plot 2D latent space (if latent_dim == 2).
    
    Args:
        latent_features: 2D latent features
        labels_true: Ground truth labels
        title: Plot title
        genre_names: Optional genre names
        
    Returns:
        Matplotlib figure
    """
    if latent_features.shape[1] != 2:
        raise ValueError(f"Expected 2D latent features, got {latent_features.shape[1]}D")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if genre_names is not None:
        unique_genres = np.unique(genre_names)
        for genre in unique_genres:
            mask = genre_names == genre
            ax.scatter(latent_features[mask, 0], latent_features[mask, 1],
                      alpha=0.6, s=20, label=genre)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        scatter = ax.scatter(latent_features[:, 0], latent_features[:, 1],
                           c=labels_true, cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax)
    
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_training_curves(train_losses: list,
                         val_losses: Optional[list] = None,
                         title: str = "Training Curves") -> plt.Figure:
    """
    Plot training (and validation) loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    
    if val_losses is not None:
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


