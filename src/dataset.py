"""
PyTorch Dataset for music clustering.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy import sparse
from typing import Optional, List, Dict, Any


class MusicClusteringDataset(Dataset):
    """
    Dataset for music clustering with multimodal features.
    """
    
    def __init__(self,
                 metadata_df: pd.DataFrame,
                 features_dir: str,
                 modalities: List[str] = ['mfcc'],
                 split: Optional[str] = None):
        """
        Initialize dataset.
        
        Args:
            metadata_df: Metadata DataFrame
            features_dir: Directory containing feature files
            modalities: List of modalities to include: 
                       'mfcc', 'logmel', 'tfidf', 'tfidf_svd', 'minilm'
            split: Optional split filter ('train', 'val', 'test')
        """
        self.metadata_df = metadata_df.copy()
        self.features_dir = features_dir
        self.modalities = modalities
        
        # Filter by split if specified
        if split is not None:
            if 'split' in self.metadata_df.columns:
                self.metadata_df = self.metadata_df[
                    self.metadata_df['split'] == split
                ].reset_index(drop=True)
        
        # Load features
        self.features = {}
        self._load_features()
        
        # Store indices mapping
        self.indices = self.metadata_df.index.tolist()
    
    def _load_features(self):
        """Load all requested feature modalities."""
        
        # MFCC stats
        if 'mfcc' in self.modalities:
            mfcc_path = os.path.join(self.features_dir, 'mfcc_stats.npy')
            if os.path.exists(mfcc_path):
                mfcc_all = np.load(mfcc_path)
                self.features['mfcc'] = mfcc_all[self.metadata_df.index]
            else:
                raise FileNotFoundError(f"MFCC features not found: {mfcc_path}")
        
        # Log-mel spectrogram
        if 'logmel' in self.modalities:
            logmel_path = os.path.join(self.features_dir, 'logmel.npy')
            if os.path.exists(logmel_path):
                logmel_all = np.load(logmel_path)
                self.features['logmel'] = logmel_all[self.metadata_df.index]
            else:
                raise FileNotFoundError(f"Log-mel features not found: {logmel_path}")
        
        # TF-IDF (sparse)
        if 'tfidf' in self.modalities:
            tfidf_path = os.path.join(self.features_dir, 'lyrics_tfidf.npz')
            if os.path.exists(tfidf_path):
                tfidf_all = sparse.load_npz(tfidf_path)
                self.features['tfidf'] = tfidf_all[self.metadata_df.index].toarray()
            else:
                raise FileNotFoundError(f"TF-IDF features not found: {tfidf_path}")
        
        # TF-IDF SVD (dense)
        if 'tfidf_svd' in self.modalities:
            svd_path = os.path.join(self.features_dir, 'lyrics_tfidf_svd.npy')
            if os.path.exists(svd_path):
                svd_all = np.load(svd_path)
                self.features['tfidf_svd'] = svd_all[self.metadata_df.index]
            else:
                raise FileNotFoundError(f"TF-IDF SVD features not found: {svd_path}")
        
        # Sentence embeddings (MiniLM)
        if 'minilm' in self.modalities:
            minilm_path = os.path.join(self.features_dir, 'lyrics_minilm.npy')
            if os.path.exists(minilm_path):
                minilm_all = np.load(minilm_path)
                self.features['minilm'] = minilm_all[self.metadata_df.index]
            else:
                raise FileNotFoundError(f"MiniLM features not found: {minilm_path}")
    
    def __len__(self) -> int:
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with available modalities and metadata
        """
        sample = {}
        
        # Add features for each modality
        for modality in self.modalities:
            if modality in self.features:
                feature = self.features[modality][idx]
                
                # Convert to tensor
                if modality == 'logmel':
                    # Add channel dimension for conv nets: (1, n_mels, time)
                    feature = torch.FloatTensor(feature).unsqueeze(0)
                else:
                    feature = torch.FloatTensor(feature)
                
                sample[modality] = feature
        
        # Add metadata
        row = self.metadata_df.iloc[idx]
        sample['clip_id'] = row['clip_id']
        sample['genre'] = row['genre']
        sample['genre_id'] = int(row['genre_id'])
        
        # Genre one-hot encoding
        genre_onehot = torch.zeros(10)  # 10 genres
        genre_onehot[sample['genre_id']] = 1.0
        sample['genre_onehot'] = genre_onehot
        
        return sample
    
    def get_feature_array(self, modality: str) -> np.ndarray:
        """
        Get full feature array for a modality.
        
        Args:
            modality: Feature modality name
            
        Returns:
            Feature array
        """
        if modality not in self.features:
            raise ValueError(f"Modality '{modality}' not loaded")
        return self.features[modality]
    
    def get_labels(self) -> np.ndarray:
        """Get genre labels."""
        return self.metadata_df['genre_id'].values
    
    def get_genre_names(self) -> np.ndarray:
        """Get genre names."""
        return self.metadata_df['genre'].values


def load_features_and_labels(metadata_path: str,
                             features_dir: str,
                             modalities: List[str],
                             split: Optional[str] = None) -> tuple:
    """
    Convenience function to load features and labels as numpy arrays.
    
    Args:
        metadata_path: Path to metadata CSV
        features_dir: Directory with feature files
        modalities: List of modalities to load
        split: Optional split filter
        
    Returns:
        Tuple of (features_dict, labels, metadata_df)
    """
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    
    # Filter by split
    if split is not None and 'split' in metadata_df.columns:
        metadata_df = metadata_df[metadata_df['split'] == split].reset_index(drop=True)
    
    # Load features
    features_dict = {}
    
    for modality in modalities:
        if modality == 'mfcc':
            path = os.path.join(features_dir, 'mfcc_stats.npy')
            features_dict['mfcc'] = np.load(path)[metadata_df.index]
        
        elif modality == 'logmel':
            path = os.path.join(features_dir, 'logmel.npy')
            features_dict['logmel'] = np.load(path)[metadata_df.index]
        
        elif modality == 'tfidf':
            path = os.path.join(features_dir, 'lyrics_tfidf.npz')
            tfidf = sparse.load_npz(path)
            features_dict['tfidf'] = tfidf[metadata_df.index].toarray()
        
        elif modality == 'tfidf_svd':
            path = os.path.join(features_dir, 'lyrics_tfidf_svd.npy')
            features_dict['tfidf_svd'] = np.load(path)[metadata_df.index]
        
        elif modality == 'minilm':
            path = os.path.join(features_dir, 'lyrics_minilm.npy')
            features_dict['minilm'] = np.load(path)[metadata_df.index]
    
    # Get labels
    labels = metadata_df['genre_id'].values
    
    return features_dict, labels, metadata_df


def create_fused_representation(features_dict: Dict[str, np.ndarray],
                                include_modalities: List[str]) -> np.ndarray:
    """
    Create fused multimodal representation by concatenation.
    
    Args:
        features_dict: Dictionary of feature arrays
        include_modalities: List of modalities to include in fusion
        
    Returns:
        Fused feature array
    """
    fused_parts = []
    
    for modality in include_modalities:
        if modality not in features_dict:
            raise ValueError(f"Modality '{modality}' not in features_dict")
        
        features = features_dict[modality]
        
        # Flatten if needed (e.g., for logmel)
        if features.ndim > 2:
            n_samples = features.shape[0]
            features = features.reshape(n_samples, -1)
        
        fused_parts.append(features)
    
    # Concatenate along feature dimension
    fused = np.concatenate(fused_parts, axis=1)
    
    return fused

