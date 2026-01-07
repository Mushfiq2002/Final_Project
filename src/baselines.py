"""
Baseline clustering methods (PCA+KMeans, Direct KMeans, AE+KMeans, Spectral).
"""
import os
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import Config
from utils import setup_logging, set_seed
from dataset import load_features_and_labels
from clustering import run_kmeans, run_spectral
from evaluation import evaluate_clustering, log_metrics, save_metrics_to_csv, create_confusion_matrix
from visualize import plot_tsne, plot_umap


def run_pca_kmeans(features: np.ndarray, 
                   labels_true: np.ndarray,
                   n_components: int = 50,
                   n_clusters: int = 10,
                   random_state: int = 42) -> tuple:
    """
    Run PCA + KMeans baseline.
    
    Args:
        features: Input features
        labels_true: Ground truth labels
        n_components: Number of PCA components
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Tuple of (pca_features, cluster_labels, metrics)
    """
    logging.info("\n" + "=" * 60)
    logging.info("Running PCA + K-Means Baseline")
    logging.info("=" * 60)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    n_components = min(n_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    features_pca = pca.fit_transform(features_scaled)
    
    logging.info(f"PCA: {features.shape[1]} -> {n_components} dims")
    logging.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # K-Means clustering
    labels_pred = run_kmeans(features_pca, n_clusters, random_state, normalize=False)
    
    # Evaluate
    metrics = evaluate_clustering(features_pca, labels_pred, labels_true)
    log_metrics(metrics, "PCA + K-Means Results")
    
    return features_pca, labels_pred, metrics


def run_direct_kmeans(features: np.ndarray,
                      labels_true: np.ndarray,
                      n_clusters: int = 10,
                      random_state: int = 42) -> tuple:
    """
    Run K-Means directly on features.
    
    Args:
        features: Input features
        labels_true: Ground truth labels
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Tuple of (features, cluster_labels, metrics)
    """
    logging.info("\n" + "=" * 60)
    logging.info("Running Direct K-Means Baseline")
    logging.info("=" * 60)
    
    # K-Means clustering (with normalization)
    labels_pred = run_kmeans(features, n_clusters, random_state, normalize=True)
    
    # Evaluate (use normalized features for metrics)
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    metrics = evaluate_clustering(features_normalized, labels_pred, labels_true)
    log_metrics(metrics, "Direct K-Means Results")
    
    return features_normalized, labels_pred, metrics


def run_autoencoder_kmeans(features: np.ndarray,
                           labels_true: np.ndarray,
                           latent_dim: int = 16,
                           n_clusters: int = 10,
                           config: Config = None) -> tuple:
    """
    Run Autoencoder + KMeans baseline.
    
    Args:
        features: Input features
        labels_true: Ground truth labels
        latent_dim: Latent dimension for AE
        n_clusters: Number of clusters
        config: Configuration object
        
    Returns:
        Tuple of (latent_features, cluster_labels, metrics)
    """
    import torch
    from models.ae_mlp import MLPAE, train_ae
    
    logging.info("\n" + "=" * 60)
    logging.info("Running Autoencoder + K-Means Baseline")
    logging.info("=" * 60)
    
    if config is None:
        config = Config()
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train Autoencoder
    device = torch.device('cpu')
    input_dim = features.shape[1]
    
    model = MLPAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    
    logging.info(f"Training Autoencoder (input={input_dim}, latent={latent_dim})...")
    train_ae(
        model=model,
        features=features_scaled,
        epochs=30,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        device=device
    )
    
    # Extract latent representations
    model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        latent_features = model.encode(features_tensor).cpu().numpy()
    
    logging.info(f"Latent features shape: {latent_features.shape}")
    
    # K-Means clustering on latent features
    labels_pred = run_kmeans(latent_features, n_clusters, config.seed, normalize=False)
    
    # Evaluate
    metrics = evaluate_clustering(latent_features, labels_pred, labels_true)
    log_metrics(metrics, "Autoencoder + K-Means Results")
    
    return latent_features, labels_pred, metrics


def run_spectral_clustering(features: np.ndarray,
                            labels_true: np.ndarray,
                            n_clusters: int = 10,
                            random_state: int = 42) -> tuple:
    """
    Run Spectral Clustering baseline.
    
    Args:
        features: Input features
        labels_true: Ground truth labels
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Tuple of (features, cluster_labels, metrics)
    """
    logging.info("\n" + "=" * 60)
    logging.info("Running Spectral Clustering Baseline")
    logging.info("=" * 60)
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Spectral clustering
    labels_pred = run_spectral(features_normalized, n_clusters, random_state, normalize=False)
    
    # Evaluate
    metrics = evaluate_clustering(features_normalized, labels_pred, labels_true)
    log_metrics(metrics, "Spectral Clustering Results")
    
    return features_normalized, labels_pred, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline clustering methods"
    )
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['pca_kmeans', 'direct_kmeans', 'ae_kmeans', 'spectral'],
                       help='Baseline methods to run')
    parser.add_argument('--split', type=str, default='test',
                       help='Data split to use (train/val/test)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate t-SNE/UMAP visualizations')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    if args.seed is not None:
        config.seed = args.seed
    
    # Setup
    setup_logging()
    set_seed(config.seed)
    config.ensure_directories()
    
    logging.info("=" * 60)
    logging.info("Baseline Clustering Methods")
    logging.info("=" * 60)
    logging.info(f"Methods: {args.methods}")
    logging.info(f"Split: {args.split}")
    
    # Load features and labels
    features_dict, labels, metadata_df = load_features_and_labels(
        config.metadata_path,
        config.features_dir,
        modalities=['mfcc'],
        split=args.split
    )
    
    features = features_dict['mfcc']
    logging.info(f"Loaded features: {features.shape}")
    logging.info(f"Labels: {labels.shape}")
    
    # Output path for metrics
    metrics_path = os.path.join(config.metrics_dir, 'metrics.csv')
    
    # Run baselines
    results = {}
    
    if 'pca_kmeans' in args.methods:
        features_pca, labels_pca, metrics_pca = run_pca_kmeans(
            features, labels, n_components=50, n_clusters=config.n_clusters,
            random_state=config.seed
        )
        results['pca_kmeans'] = (features_pca, labels_pca, metrics_pca)
        
        # Save metrics
        experiment_info = {
            'experiment_name': 'baseline_pca_kmeans',
            'model': 'PCA+KMeans',
            'representation': 'mfcc_pca',
            'clustering_algo': 'kmeans',
            'split': args.split
        }
        save_metrics_to_csv(metrics_pca, metrics_path, experiment_info, append=True)
        
        # Save confusion matrix
        confusion = create_confusion_matrix(labels, labels_pca, metadata_df['genre'].values)
        confusion_path = os.path.join(config.metrics_dir, 'confusion_pca_kmeans.csv')
        confusion.to_csv(confusion_path)
        logging.info(f"Confusion matrix saved: {confusion_path}")
    
    if 'direct_kmeans' in args.methods:
        features_norm, labels_direct, metrics_direct = run_direct_kmeans(
            features, labels, n_clusters=config.n_clusters,
            random_state=config.seed
        )
        results['direct_kmeans'] = (features_norm, labels_direct, metrics_direct)
        
        experiment_info = {
            'experiment_name': 'baseline_direct_kmeans',
            'model': 'DirectKMeans',
            'representation': 'mfcc_raw',
            'clustering_algo': 'kmeans',
            'split': args.split
        }
        save_metrics_to_csv(metrics_direct, metrics_path, experiment_info, append=True)
        
        confusion = create_confusion_matrix(labels, labels_direct, metadata_df['genre'].values)
        confusion_path = os.path.join(config.metrics_dir, 'confusion_direct_kmeans.csv')
        confusion.to_csv(confusion_path)
    
    if 'ae_kmeans' in args.methods:
        features_ae, labels_ae, metrics_ae = run_autoencoder_kmeans(
            features, labels, latent_dim=config.latent_dim,
            n_clusters=config.n_clusters, config=config
        )
        results['ae_kmeans'] = (features_ae, labels_ae, metrics_ae)
        
        experiment_info = {
            'experiment_name': 'baseline_ae_kmeans',
            'model': 'AE+KMeans',
            'representation': 'ae_latent',
            'clustering_algo': 'kmeans',
            'split': args.split,
            'latent_dim': config.latent_dim
        }
        save_metrics_to_csv(metrics_ae, metrics_path, experiment_info, append=True)
        
        confusion = create_confusion_matrix(labels, labels_ae, metadata_df['genre'].values)
        confusion_path = os.path.join(config.metrics_dir, 'confusion_ae_kmeans.csv')
        confusion.to_csv(confusion_path)
    
    if 'spectral' in args.methods:
        features_spec, labels_spec, metrics_spec = run_spectral_clustering(
            features, labels, n_clusters=config.n_clusters,
            random_state=config.seed
        )
        results['spectral'] = (features_spec, labels_spec, metrics_spec)
        
        experiment_info = {
            'experiment_name': 'baseline_spectral',
            'model': 'SpectralClustering',
            'representation': 'mfcc_raw',
            'clustering_algo': 'spectral',
            'split': args.split
        }
        save_metrics_to_csv(metrics_spec, metrics_path, experiment_info, append=True)
        
        confusion = create_confusion_matrix(labels, labels_spec, metadata_df['genre'].values)
        confusion_path = os.path.join(config.metrics_dir, 'confusion_spectral.csv')
        confusion.to_csv(confusion_path)
    
    # Visualizations
    if args.visualize:
        logging.info("\nGenerating visualizations...")
        genre_names = metadata_df['genre'].values
        
        for method_name, (feats, preds, _) in results.items():
            # t-SNE
            fig_tsne = plot_tsne(
                feats, labels, preds,
                title=f"t-SNE: {method_name}",
                genre_names=genre_names
            )
            tsne_path = os.path.join(config.figures_dir, f'tsne_{method_name}.png')
            fig_tsne.savefig(tsne_path, dpi=150, bbox_inches='tight')
            plt.close(fig_tsne)
            logging.info(f"t-SNE plot saved: {tsne_path}")
            
            # UMAP (if available)
            try:
                fig_umap = plot_umap(
                    feats, labels, preds,
                    title=f"UMAP: {method_name}",
                    genre_names=genre_names
                )
                umap_path = os.path.join(config.figures_dir, f'umap_{method_name}.png')
                fig_umap.savefig(umap_path, dpi=150, bbox_inches='tight')
                plt.close(fig_umap)
                logging.info(f"UMAP plot saved: {umap_path}")
            except ImportError:
                logging.info("UMAP not available, skipping")
    
    logging.info("\n" + "=" * 60)
    logging.info("Baseline experiments complete!")
    logging.info(f"Metrics saved to: {metrics_path}")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()


