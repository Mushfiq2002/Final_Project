"""
Clustering algorithms for music clustering.
"""
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
import logging
from typing import Optional, Dict, Any


def run_kmeans(features: np.ndarray,
               n_clusters: int = 10,
               random_state: int = 42,
               normalize: bool = True) -> np.ndarray:
    """
    Run K-Means clustering.
    
    Args:
        features: Feature array (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random seed
        normalize: Whether to normalize features
        
    Returns:
        Cluster labels
    """
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    logging.info(f"Running K-Means (k={n_clusters})...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
        max_iter=300
    )
    
    labels = kmeans.fit_predict(features)
    
    logging.info(f"K-Means complete. Unique clusters: {len(np.unique(labels))}")
    
    return labels


def run_agglomerative(features: np.ndarray,
                     n_clusters: int = 10,
                     linkage: str = 'ward',
                     normalize: bool = True) -> np.ndarray:
    """
    Run Agglomerative Clustering.
    
    Args:
        features: Feature array
        n_clusters: Number of clusters
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        normalize: Whether to normalize features
        
    Returns:
        Cluster labels
    """
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    logging.info(f"Running Agglomerative Clustering (k={n_clusters}, linkage={linkage})...")
    
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    
    labels = agg.fit_predict(features)
    
    logging.info(f"Agglomerative complete. Unique clusters: {len(np.unique(labels))}")
    
    return labels


def run_dbscan(features: np.ndarray,
               eps: float = 0.5,
               min_samples: int = 5,
               normalize: bool = True) -> np.ndarray:
    """
    Run DBSCAN clustering.
    
    Args:
        features: Feature array
        eps: Maximum distance between samples
        min_samples: Minimum samples in a neighborhood
        normalize: Whether to normalize features
        
    Returns:
        Cluster labels (-1 for noise)
    """
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    logging.info(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(features)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    logging.info(f"DBSCAN complete. Clusters: {n_clusters}, Noise points: {n_noise}")
    
    return labels


def run_spectral(features: np.ndarray,
                n_clusters: int = 10,
                random_state: int = 42,
                normalize: bool = True) -> np.ndarray:
    """
    Run Spectral Clustering.
    
    Args:
        features: Feature array
        n_clusters: Number of clusters
        random_state: Random seed
        normalize: Whether to normalize features
        
    Returns:
        Cluster labels
    """
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    logging.info(f"Running Spectral Clustering (k={n_clusters})...")
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        random_state=random_state,
        affinity='nearest_neighbors',
        n_neighbors=10,
        assign_labels='kmeans'
    )
    
    labels = spectral.fit_predict(features)
    
    logging.info(f"Spectral complete. Unique clusters: {len(np.unique(labels))}")
    
    return labels


def cluster_with_algorithm(features: np.ndarray,
                           algorithm: str,
                           n_clusters: int = 10,
                           random_state: int = 42,
                           **kwargs) -> np.ndarray:
    """
    Unified interface for clustering.
    
    Args:
        features: Feature array
        algorithm: Algorithm name ('kmeans', 'agglomerative', 'dbscan', 'spectral')
        n_clusters: Number of clusters (not used for DBSCAN)
        random_state: Random seed
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Cluster labels
    """
    if algorithm == 'kmeans':
        return run_kmeans(features, n_clusters, random_state, 
                         normalize=kwargs.get('normalize', True))
    
    elif algorithm == 'agglomerative':
        return run_agglomerative(features, n_clusters,
                                linkage=kwargs.get('linkage', 'ward'),
                                normalize=kwargs.get('normalize', True))
    
    elif algorithm == 'dbscan':
        return run_dbscan(features,
                         eps=kwargs.get('eps', 0.5),
                         min_samples=kwargs.get('min_samples', 5),
                         normalize=kwargs.get('normalize', True))
    
    elif algorithm == 'spectral':
        return run_spectral(features, n_clusters, random_state,
                           normalize=kwargs.get('normalize', True))
    
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def find_optimal_eps(features: np.ndarray, 
                     k: int = 4,
                     normalize: bool = True) -> float:
    """
    Find optimal eps for DBSCAN using k-distance plot heuristic.
    
    Args:
        features: Feature array
        k: Number of neighbors to consider
        normalize: Whether to normalize features
        
    Returns:
        Suggested eps value
    """
    from sklearn.neighbors import NearestNeighbors
    
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(features)
    distances, _ = neighbors.kneighbors(features)
    
    # Sort k-distances
    k_distances = np.sort(distances[:, -1])
    
    # Use elbow at 90th percentile as heuristic
    eps = np.percentile(k_distances, 90)
    
    logging.info(f"Suggested DBSCAN eps: {eps:.4f}")
    
    return eps


