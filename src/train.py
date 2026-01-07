"""
Main training script for VAE models.
"""
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import Config
from utils import setup_logging, set_seed, save_checkpoint, get_device
from dataset import load_features_and_labels, create_fused_representation
from models.vae_mlp import MLPVAE, train_vae_mlp
from models.vae_conv import ConvVAE, train_vae_conv
from models.vae_beta import BetaVAE, train_beta_vae
from models.cvae import CVAE, train_cvae
from clustering import cluster_with_algorithm
from evaluation import evaluate_clustering, log_metrics, save_metrics_to_csv, create_confusion_matrix
from visualize import (plot_tsne, plot_umap, plot_reconstruction_error_hist,
                      plot_spectrogram_reconstruction, plot_training_curves)


def extract_latent_features(model, features, device, is_conv=False, conditions=None):
    """
    Extract latent representations from trained model.
    
    Args:
        model: Trained VAE/CVAE model
        features: Input features
        device: Device
        is_conv: Whether model is convolutional
        conditions: Optional conditions for CVAE
        
    Returns:
        Latent features array
    """
    model.eval()
    
    # Ensure correct shape
    if is_conv and features.ndim == 3:
        features = np.expand_dims(features, axis=1)
    
    with torch.no_grad():
        x = torch.FloatTensor(features).to(device)
        
        if conditions is not None:
            # CVAE
            c = torch.FloatTensor(conditions).to(device)
            mu, logvar = model.encode(x, c)
        else:
            # VAE
            mu, logvar = model.encode(x)
        
        # Use mean of latent distribution
        z = mu.cpu().numpy()
    
    return z


def get_reconstructions(model, features, device, is_conv=False, conditions=None):
    """
    Get reconstructions from model.
    
    Args:
        model: Trained model
        features: Input features
        device: Device
        is_conv: Whether convolutional
        conditions: Optional conditions for CVAE
        
    Returns:
        Tuple of (original, reconstructed)
    """
    model.eval()
    
    if is_conv and features.ndim == 3:
        features = np.expand_dims(features, axis=1)
    
    with torch.no_grad():
        x = torch.FloatTensor(features).to(device)
        
        if conditions is not None:
            # CVAE
            c = torch.FloatTensor(conditions).to(device)
            x_recon, mu, logvar, z = model(x, c)
        else:
            # VAE
            x_recon, mu, logvar, z = model(x)
        
        x_recon = x_recon.cpu().numpy()
    
    return features, x_recon


def train_vae_mlp_pipeline(config: Config, args):
    """
    Train MLP-VAE pipeline (EASY task).
    """
    logging.info("\n" + "=" * 60)
    logging.info("Training MLP-VAE (EASY Task)")
    logging.info("=" * 60)
    
    device = get_device(device_str=args.device)
    
    # Load MFCC features
    features_dict, labels, metadata_df = load_features_and_labels(
        config.metadata_path,
        config.features_dir,
        modalities=['mfcc'],
        split='train'
    )
    features_train = features_dict['mfcc']
    
    # Load test data
    features_dict_test, labels_test, metadata_df_test = load_features_and_labels(
        config.metadata_path,
        config.features_dir,
        modalities=['mfcc'],
        split='test'
    )
    features_test = features_dict_test['mfcc']
    
    logging.info(f"Train features: {features_train.shape}")
    logging.info(f"Test features: {features_test.shape}")
    
    # Create model
    input_dim = features_train.shape[1]
    model = MLPVAE(input_dim=input_dim, latent_dim=config.latent_dim)
    logging.info(f"Model: MLP-VAE, Latent dim: {config.latent_dim}")
    
    # Train
    history = train_vae_mlp(
        model=model,
        features=features_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        beta=config.beta,
        device=device
    )
    
    # Save checkpoint
    checkpoint_path = os.path.join(config.checkpoints_dir, 'vae_mlp.pt')
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Model saved: {checkpoint_path}")
    
    # Save training curves
    fig_train = plot_training_curves(history['train_losses'], title="MLP-VAE Training")
    fig_train.savefig(os.path.join(config.figures_dir, 'training_vae_mlp.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_train)
    
    # Extract latent features for test set
    latent_test = extract_latent_features(model, features_test, device)
    
    # Save latents
    latents_path = os.path.join(config.metrics_dir, 'latents_vae_mlp.npy')
    np.save(latents_path, latent_test)
    logging.info(f"Latent features saved: {latents_path}")
    
    # Clustering on latent features
    labels_pred = cluster_with_algorithm(
        latent_test, 'kmeans', n_clusters=config.n_clusters,
        random_state=config.seed
    )
    
    # Evaluate
    metrics = evaluate_clustering(latent_test, labels_pred, labels_test)
    log_metrics(metrics, "MLP-VAE + K-Means Results")
    
    # Save metrics
    experiment_info = {
        'experiment_name': 'vae_mlp_kmeans',
        'model': 'MLP-VAE',
        'representation': 'vae_latent',
        'clustering_algo': 'kmeans',
        'split': 'test',
        'latent_dim': config.latent_dim
    }
    metrics_path = os.path.join(config.metrics_dir, 'metrics.csv')
    save_metrics_to_csv(metrics, metrics_path, experiment_info, append=True)
    
    # Visualizations
    genre_names = metadata_df_test['genre'].values
    
    # t-SNE
    fig_tsne = plot_tsne(latent_test, labels_test, labels_pred,
                        title="MLP-VAE Latent Space (t-SNE)",
                        genre_names=genre_names)
    fig_tsne.savefig(os.path.join(config.figures_dir, 'tsne_vae_mlp.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_tsne)
    
    # UMAP (if available)
    try:
        fig_umap = plot_umap(latent_test, labels_test, labels_pred,
                            title="MLP-VAE Latent Space (UMAP)",
                            genre_names=genre_names)
        fig_umap.savefig(os.path.join(config.figures_dir, 'umap_vae_mlp.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_umap)
    except ImportError:
        logging.info("UMAP not available, skipping")
    
    # Reconstruction error
    features_orig, features_recon = get_reconstructions(model, features_test, device)
    fig_recon = plot_reconstruction_error_hist(features_orig, features_recon,
                                               title="MLP-VAE Reconstruction Error")
    fig_recon.savefig(os.path.join(config.figures_dir, 'recon_error_vae_mlp.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_recon)
    
    # Confusion matrix
    confusion = create_confusion_matrix(labels_test, labels_pred, genre_names)
    confusion_path = os.path.join(config.metrics_dir, 'confusion_vae_mlp.csv')
    confusion.to_csv(confusion_path)
    
    logging.info("MLP-VAE pipeline complete!")


def train_vae_conv_pipeline(config: Config, args):
    """
    Train Conv-VAE pipeline (MEDIUM task).
    """
    logging.info("\n" + "=" * 60)
    logging.info("Training Conv-VAE (MEDIUM Task)")
    logging.info("=" * 60)
    
    device = get_device(device_str=args.device)
    
    # Load log-mel features and TF-IDF
    modalities = ['logmel']
    if args.multimodal:
        modalities.append('tfidf_svd')
    
    features_dict, labels, metadata_df = load_features_and_labels(
        config.metadata_path,
        config.features_dir,
        modalities=modalities,
        split='train'
    )
    features_train = features_dict['logmel']
    
    features_dict_test, labels_test, metadata_df_test = load_features_and_labels(
        config.metadata_path,
        config.features_dir,
        modalities=modalities,
        split='test'
    )
    features_test = features_dict_test['logmel']
    
    logging.info(f"Train features: {features_train.shape}")
    logging.info(f"Test features: {features_test.shape}")
    
    # Create model
    input_shape = (1, config.n_mels, config.mel_time_frames)
    model = ConvVAE(input_shape=input_shape, latent_dim=config.latent_dim)
    logging.info(f"Model: Conv-VAE, Latent dim: {config.latent_dim}")
    
    # Train
    history = train_vae_conv(
        model=model,
        features=features_train,
        epochs=config.epochs,
        batch_size=config.batch_size // 2,  # Smaller batch for conv
        learning_rate=config.learning_rate,
        beta=config.beta,
        device=device
    )
    
    # Save checkpoint
    checkpoint_path = os.path.join(config.checkpoints_dir, 'vae_conv.pt')
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Model saved: {checkpoint_path}")
    
    # Save training curves
    fig_train = plot_training_curves(history['train_losses'], title="Conv-VAE Training")
    fig_train.savefig(os.path.join(config.figures_dir, 'training_vae_conv.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_train)
    
    # Extract latent features
    latent_test = extract_latent_features(model, features_test, device, is_conv=True)
    
    # Multimodal fusion
    if args.multimodal:
        logging.info("Creating multimodal fusion (audio + lyrics)...")
        lyrics_test = features_dict_test['tfidf_svd']
        # Fuse: concatenate audio latent + lyrics
        fused_test = np.concatenate([latent_test, lyrics_test], axis=1)
        clustering_features = fused_test
        representation_name = 'conv_vae_audio_lyrics'
    else:
        clustering_features = latent_test
        representation_name = 'conv_vae_latent'
    
    # Save features
    latents_path = os.path.join(config.metrics_dir, f'latents_vae_conv.npy')
    np.save(latents_path, clustering_features)
    logging.info(f"Features saved: {latents_path}")
    
    # Try multiple clustering algorithms
    clustering_algos = ['kmeans', 'agglomerative']
    
    for algo in clustering_algos:
        logging.info(f"\nClustering with {algo}...")
        labels_pred = cluster_with_algorithm(
            clustering_features, algo, n_clusters=config.n_clusters,
            random_state=config.seed
        )
        
        # Evaluate
        metrics = evaluate_clustering(clustering_features, labels_pred, labels_test)
        log_metrics(metrics, f"Conv-VAE + {algo} Results")
        
        # Save metrics
        experiment_info = {
            'experiment_name': f'vae_conv_{algo}',
            'model': 'Conv-VAE',
            'representation': representation_name,
            'clustering_algo': algo,
            'split': 'test',
            'latent_dim': config.latent_dim,
            'multimodal': args.multimodal
        }
        metrics_path = os.path.join(config.metrics_dir, 'metrics.csv')
        save_metrics_to_csv(metrics, metrics_path, experiment_info, append=True)
    
    # Visualizations (use kmeans results)
    labels_pred = cluster_with_algorithm(
        clustering_features, 'kmeans', n_clusters=config.n_clusters,
        random_state=config.seed
    )
    genre_names = metadata_df_test['genre'].values
    
    fig_tsne = plot_tsne(clustering_features, labels_test, labels_pred,
                        title="Conv-VAE Latent Space (t-SNE)",
                        genre_names=genre_names)
    fig_tsne.savefig(os.path.join(config.figures_dir, 'tsne_vae_conv.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_tsne)
    
    # Spectrogram reconstructions
    features_orig, features_recon = get_reconstructions(model, features_test[:16], device, is_conv=True)
    fig_spec = plot_spectrogram_reconstruction(features_orig, features_recon, n_examples=8)
    fig_spec.savefig(os.path.join(config.figures_dir, 'recon_spectrograms_vae_conv.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_spec)
    
    logging.info("Conv-VAE pipeline complete!")


def train_cvae_pipeline(config: Config, args):
    """
    Train CVAE pipeline (HARD task).
    """
    logging.info("\n" + "=" * 60)
    logging.info("Training CVAE (HARD Task)")
    logging.info("=" * 60)
    
    device = get_device(device_str=args.device)
    
    # Load features with all modalities for full fusion
    modalities = ['mfcc', 'tfidf_svd'] if not args.skip_lyrics else ['mfcc']
    
    features_dict, labels, metadata_df = load_features_and_labels(
        config.metadata_path,
        config.features_dir,
        modalities=modalities,
        split='train'
    )
    
    features_dict_test, labels_test, metadata_df_test = load_features_and_labels(
        config.metadata_path,
        config.features_dir,
        modalities=modalities,
        split='test'
    )
    
    # Use MFCC as primary input
    features_train = features_dict['mfcc']
    features_test = features_dict_test['mfcc']
    
    # Prepare conditions (genre one-hot)
    n_samples_train = len(features_train)
    n_samples_test = len(features_test)
    conditions_train = np.zeros((n_samples_train, config.n_genres))
    conditions_test = np.zeros((n_samples_test, config.n_genres))
    
    for i, label in enumerate(labels):
        conditions_train[i, int(label)] = 1.0
    for i, label in enumerate(labels_test):
        conditions_test[i, int(label)] = 1.0
    
    logging.info(f"Train features: {features_train.shape}")
    logging.info(f"Conditions: {conditions_train.shape}")
    
    # Create model
    input_dim = features_train.shape[1]
    model = CVAE(input_dim=input_dim, condition_dim=config.n_genres, latent_dim=config.latent_dim)
    logging.info(f"Model: CVAE, Latent dim: {config.latent_dim}")
    
    # Train
    history = train_cvae(
        model=model,
        features=features_train,
        conditions=conditions_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        beta=config.beta,
        device=device
    )
    
    # Save checkpoint
    checkpoint_path = os.path.join(config.checkpoints_dir, 'cvae.pt')
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Model saved: {checkpoint_path}")
    
    # Extract latent features
    latent_test = extract_latent_features(model, features_test, device, conditions=conditions_test)
    
    # Full multimodal fusion: audio latent + lyrics + genre
    if not args.skip_lyrics and 'tfidf_svd' in features_dict_test:
        lyrics_test = features_dict_test['tfidf_svd']
        fused_test = np.concatenate([latent_test, lyrics_test, conditions_test], axis=1)
        representation_name = 'cvae_audio_lyrics_genre'
        logging.info(f"Fused representation: {fused_test.shape}")
    else:
        fused_test = np.concatenate([latent_test, conditions_test], axis=1)
        representation_name = 'cvae_audio_genre'
    
    # Save features
    latents_path = os.path.join(config.metrics_dir, 'latents_cvae.npy')
    np.save(latents_path, fused_test)
    
    # Clustering with multiple algorithms
    clustering_algos = ['kmeans', 'agglomerative']
    
    for algo in clustering_algos:
        logging.info(f"\nClustering with {algo}...")
        labels_pred = cluster_with_algorithm(
            fused_test, algo, n_clusters=config.n_clusters,
            random_state=config.seed
        )
        
        # Evaluate with all metrics
        metrics = evaluate_clustering(fused_test, labels_pred, labels_test)
        log_metrics(metrics, f"CVAE + {algo} Results")
        
        # Save metrics
        experiment_info = {
            'experiment_name': f'cvae_{algo}',
            'model': 'CVAE',
            'representation': representation_name,
            'clustering_algo': algo,
            'split': 'test',
            'latent_dim': config.latent_dim
        }
        metrics_path = os.path.join(config.metrics_dir, 'metrics.csv')
        save_metrics_to_csv(metrics, metrics_path, experiment_info, append=True)
    
    # Visualizations
    labels_pred = cluster_with_algorithm(
        fused_test, 'kmeans', n_clusters=config.n_clusters,
        random_state=config.seed
    )
    genre_names = metadata_df_test['genre'].values
    
    fig_tsne = plot_tsne(fused_test, labels_test, labels_pred,
                        title="CVAE Fused Space (t-SNE)",
                        genre_names=genre_names)
    fig_tsne.savefig(os.path.join(config.figures_dir, 'tsne_cvae.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_tsne)
    
    # Reconstruction error
    features_orig, features_recon = get_reconstructions(model, features_test, device, conditions=conditions_test)
    fig_recon = plot_reconstruction_error_hist(features_orig, features_recon,
                                               title="CVAE Reconstruction Error")
    fig_recon.savefig(os.path.join(config.figures_dir, 'recon_error_cvae.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_recon)
    
    logging.info("CVAE pipeline complete!")


def main():
    parser = argparse.ArgumentParser(description="Train VAE models for music clustering")
    parser.add_argument('--mode', type=str, required=True,
                       choices=['vae_mlp', 'vae_conv', 'beta_vae', 'cvae', 'all'],
                       help='Training mode')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='Device (cpu/cuda/mps/auto)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--latent_dim', type=int, default=None,
                       help='Latent dimension (overrides config)')
    parser.add_argument('--beta', type=float, default=None,
                       help='Beta parameter for beta-VAE (overrides config)')
    parser.add_argument('--multimodal', action='store_true',
                       help='Use multimodal fusion (audio + lyrics)')
    parser.add_argument('--skip_lyrics', action='store_true',
                       help='Skip lyrics features (for CVAE)')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override config with CLI args
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.latent_dim is not None:
        config.latent_dim = args.latent_dim
    if args.beta is not None:
        config.beta = args.beta
    
    # Setup
    setup_logging()
    set_seed(config.seed)
    config.ensure_directories()
    
    logging.info("=" * 60)
    logging.info("VAE Training Pipeline")
    logging.info("=" * 60)
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Epochs: {config.epochs}")
    logging.info(f"Latent dim: {config.latent_dim}")
    logging.info(f"Beta: {config.beta}")
    
    # Run training
    if args.mode == 'vae_mlp' or args.mode == 'all':
        train_vae_mlp_pipeline(config, args)
    
    if args.mode == 'vae_conv' or args.mode == 'all':
        train_vae_conv_pipeline(config, args)
    
    if args.mode == 'cvae' or args.mode == 'all':
        train_cvae_pipeline(config, args)
    
    if args.mode == 'beta_vae':
        logging.warning("Beta-VAE mode: Use vae_mlp or vae_conv with --beta > 1.0")
        if config.beta <= 1.0:
            logging.warning(f"Beta={config.beta}, setting to 4.0 for disentanglement")
            config.beta = 4.0
        # Run with increased beta
        args.mode = 'vae_mlp'  # or 'vae_conv' depending on input
        train_vae_mlp_pipeline(config, args)
    
    logging.info("\n" + "=" * 60)
    logging.info("Training complete!")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()

