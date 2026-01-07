"""
Evaluation metrics for clustering.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from typing import Dict, Optional
import logging


def compute_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute clustering purity.
    
    Purity = (1/N) * sum_k(max_j |cluster_k ∩ class_j|)
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
        
    Returns:
        Purity score (higher is better, max=1.0)
    """
    # Filter out noise points (label -1) from DBSCAN
    valid_mask = labels_pred != -1
    if not valid_mask.any():
        return 0.0
    
    labels_true = labels_true[valid_mask]
    labels_pred = labels_pred[valid_mask]
    
    # Compute purity
    n_samples = len(labels_true)
    contingency_matrix = pd.crosstab(labels_pred, labels_true)
    
    # Sum of max counts per cluster
    purity = contingency_matrix.max(axis=1).sum() / n_samples
    
    return purity


def compute_unsupervised_metrics(features: np.ndarray,
                                 labels_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute unsupervised clustering metrics.
    
    Args:
        features: Feature array
        labels_pred: Predicted cluster labels
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # Filter out noise points for metrics that don't handle them
    valid_mask = labels_pred != -1
    n_clusters = len(set(labels_pred[valid_mask]))
    
    if n_clusters < 2:
        logging.warning("Less than 2 clusters, returning dummy metrics")
        return {
            'silhouette': 0.0,
            'calinski_harabasz': 0.0,
            'davies_bouldin': 0.0,
            'n_clusters': n_clusters,
            'n_noise': int((~valid_mask).sum())
        }
    
    # Use only valid samples for metric computation
    features_valid = features[valid_mask]
    labels_valid = labels_pred[valid_mask]
    
    try:
        # Silhouette score (higher is better, range [-1, 1])
        metrics['silhouette'] = silhouette_score(features_valid, labels_valid)
    except Exception as e:
        logging.warning(f"Failed to compute silhouette: {e}")
        metrics['silhouette'] = 0.0
    
    try:
        # Calinski-Harabasz index (higher is better)
        metrics['calinski_harabasz'] = calinski_harabasz_score(features_valid, labels_valid)
    except Exception as e:
        logging.warning(f"Failed to compute Calinski-Harabasz: {e}")
        metrics['calinski_harabasz'] = 0.0
    
    try:
        # Davies-Bouldin index (lower is better)
        metrics['davies_bouldin'] = davies_bouldin_score(features_valid, labels_valid)
    except Exception as e:
        logging.warning(f"Failed to compute Davies-Bouldin: {e}")
        metrics['davies_bouldin'] = 0.0
    
    # Number of clusters and noise points
    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = int((~valid_mask).sum())
    
    return metrics


def compute_supervised_metrics(labels_true: np.ndarray,
                               labels_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute supervised clustering metrics using ground truth labels.
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # Filter out noise points
    valid_mask = labels_pred != -1
    
    if not valid_mask.any():
        logging.warning("All samples labeled as noise, returning zero metrics")
        return {
            'ari': 0.0,
            'nmi': 0.0,
            'purity': 0.0
        }
    
    labels_true_valid = labels_true[valid_mask]
    labels_pred_valid = labels_pred[valid_mask]
    
    try:
        # Adjusted Rand Index (higher is better, max=1.0)
        metrics['ari'] = adjusted_rand_score(labels_true_valid, labels_pred_valid)
    except Exception as e:
        logging.warning(f"Failed to compute ARI: {e}")
        metrics['ari'] = 0.0
    
    try:
        # Normalized Mutual Information (higher is better, max=1.0)
        metrics['nmi'] = normalized_mutual_info_score(labels_true_valid, labels_pred_valid)
    except Exception as e:
        logging.warning(f"Failed to compute NMI: {e}")
        metrics['nmi'] = 0.0
    
    try:
        # Purity (higher is better, max=1.0)
        metrics['purity'] = compute_purity(labels_true, labels_pred)
    except Exception as e:
        logging.warning(f"Failed to compute purity: {e}")
        metrics['purity'] = 0.0
    
    return metrics


def evaluate_clustering(features: np.ndarray,
                       labels_pred: np.ndarray,
                       labels_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all clustering evaluation metrics.
    
    Args:
        features: Feature array
        labels_pred: Predicted cluster labels
        labels_true: Optional ground truth labels for supervised metrics
        
    Returns:
        Dictionary of all metrics
    """
    all_metrics = {}
    
    # Unsupervised metrics
    unsup_metrics = compute_unsupervised_metrics(features, labels_pred)
    all_metrics.update(unsup_metrics)
    
    # Supervised metrics if ground truth available
    if labels_true is not None:
        sup_metrics = compute_supervised_metrics(labels_true, labels_pred)
        all_metrics.update(sup_metrics)
    
    return all_metrics


def create_confusion_matrix(labels_true: np.ndarray,
                           labels_pred: np.ndarray,
                           genre_names: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Create confusion-like matrix showing cluster vs genre distribution.
    
    Args:
        labels_true: Ground truth genre labels
        labels_pred: Predicted cluster labels
        genre_names: Optional array of genre names matching labels_true
        
    Returns:
        DataFrame with cluster-genre distribution
    """
    # Filter out noise
    valid_mask = labels_pred != -1
    labels_true = labels_true[valid_mask]
    labels_pred = labels_pred[valid_mask]
    
    if genre_names is not None:
        genre_names = genre_names[valid_mask]
    
    # Create contingency table
    if genre_names is not None:
        confusion = pd.crosstab(
            labels_pred, genre_names,
            rownames=['Cluster'], colnames=['Genre']
        )
    else:
        confusion = pd.crosstab(
            labels_pred, labels_true,
            rownames=['Cluster'], colnames=['Genre']
        )
    
    # Add totals
    confusion['Total'] = confusion.sum(axis=1)
    
    return confusion


def log_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Log metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for logging
    """
    if prefix:
        logging.info(f"\n{prefix}")
        logging.info("=" * 60)
    
    # Unsupervised metrics
    if 'silhouette' in metrics:
        logging.info(f"Silhouette Score:       {metrics['silhouette']:.4f}")
    if 'calinski_harabasz' in metrics:
        logging.info(f"Calinski-Harabasz:      {metrics['calinski_harabasz']:.2f}")
    if 'davies_bouldin' in metrics:
        logging.info(f"Davies-Bouldin:         {metrics['davies_bouldin']:.4f}")
    
    # Supervised metrics
    if 'ari' in metrics:
        logging.info(f"Adjusted Rand Index:    {metrics['ari']:.4f}")
    if 'nmi' in metrics:
        logging.info(f"Normalized Mutual Info: {metrics['nmi']:.4f}")
    if 'purity' in metrics:
        logging.info(f"Purity:                 {metrics['purity']:.4f}")
    
    # Cluster info
    if 'n_clusters' in metrics:
        logging.info(f"Number of clusters:     {metrics['n_clusters']}")
    if 'n_noise' in metrics and metrics['n_noise'] > 0:
        logging.info(f"Noise points:           {metrics['n_noise']}")


def save_metrics_to_csv(metrics: Dict[str, float],
                        output_path: str,
                        experiment_info: Optional[Dict[str, str]] = None,
                        append: bool = True):
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to output CSV
        experiment_info: Additional experiment information (model, representation, etc.)
        append: Whether to append to existing file
    """
    import os
    
    # Combine experiment info and metrics
    record = {}
    if experiment_info:
        record.update(experiment_info)
    record.update(metrics)
    
    # Convert to DataFrame
    df_new = pd.DataFrame([record])
    
    # Append or create new
    if append and os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_path, index=False)
    else:
        df_new.to_csv(output_path, index=False)
    
    logging.info(f"Metrics saved to: {output_path}")


