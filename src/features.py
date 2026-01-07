"""
Feature extraction script: MFCC stats, log-mel spectrograms, and TF-IDF.
"""
import os
import argparse
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from config import Config
from utils import setup_logging, set_seed


def extract_mfcc_stats(audio_path: str, config: Config) -> np.ndarray:
    """
    Extract MFCC statistics (mean + std) from audio file.
    
    Args:
        audio_path: Path to audio file
        config: Configuration object
        
    Returns:
        Feature vector of shape (n_mfcc * 2,) = (40,)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=config.sample_rate, mono=True)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length
        )
        
        # Compute statistics
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Concatenate
        features = np.concatenate([mfcc_mean, mfcc_std])
        
        return features
    
    except Exception as e:
        logging.error(f"Error extracting MFCC from {audio_path}: {e}")
        # Return zeros as fallback
        return np.zeros(config.n_mfcc * 2)


def extract_logmel_spectrogram(audio_path: str, config: Config) -> np.ndarray:
    """
    Extract log-mel spectrogram from audio file.
    
    Args:
        audio_path: Path to audio file
        config: Configuration object
        
    Returns:
        Log-mel spectrogram of shape (n_mels, time_frames)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=config.sample_rate, mono=True)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        
        # Pad or crop to fixed time dimension
        if log_mel.shape[1] < config.mel_time_frames:
            # Pad
            pad_width = config.mel_time_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        elif log_mel.shape[1] > config.mel_time_frames:
            # Crop
            log_mel = log_mel[:, :config.mel_time_frames]
        
        return log_mel
    
    except Exception as e:
        logging.error(f"Error extracting log-mel from {audio_path}: {e}")
        # Return zeros as fallback
        return np.zeros((config.n_mels, config.mel_time_frames))


def extract_audio_features(df: pd.DataFrame, config: Config, force: bool = False):
    """
    Extract all audio features and save to disk.
    
    Args:
        df: Metadata DataFrame
        config: Configuration object
        force: Force re-extraction even if files exist
    """
    mfcc_path = os.path.join(config.features_dir, 'mfcc_stats.npy')
    logmel_path = os.path.join(config.features_dir, 'logmel.npy')
    
    # Check if already extracted
    if not force and os.path.exists(mfcc_path) and os.path.exists(logmel_path):
        logging.info("Audio features already exist. Use --force to re-extract.")
        return
    
    logging.info("Extracting audio features...")
    
    n_samples = len(df)
    mfcc_features = np.zeros((n_samples, config.n_mfcc * 2), dtype=np.float32)
    logmel_features = np.zeros(
        (n_samples, config.n_mels, config.mel_time_frames), 
        dtype=np.float32
    )
    
    for idx, row in tqdm(df.iterrows(), total=n_samples, desc="Audio features"):
        clip_path = row['clip_path']
        
        # Extract MFCC stats
        mfcc_features[idx] = extract_mfcc_stats(clip_path, config)
        
        # Extract log-mel spectrogram
        logmel_features[idx] = extract_logmel_spectrogram(clip_path, config)
    
    # Save features
    os.makedirs(config.features_dir, exist_ok=True)
    np.save(mfcc_path, mfcc_features)
    np.save(logmel_path, logmel_features)
    
    logging.info(f"MFCC stats saved: {mfcc_path} (shape: {mfcc_features.shape})")
    logging.info(f"Log-mel saved: {logmel_path} (shape: {logmel_features.shape})")


def extract_tfidf_features(df: pd.DataFrame, config: Config, force: bool = False):
    """
    Extract TF-IDF features from lyrics text.
    
    Args:
        df: Metadata DataFrame with lyrics_text column
        config: Configuration object
        force: Force re-extraction even if files exist
    """
    tfidf_path = os.path.join(config.features_dir, 'lyrics_tfidf.npz')
    
    # Check if already extracted
    if not force and os.path.exists(tfidf_path):
        logging.info("TF-IDF features already exist. Use --force to re-extract.")
        return
    
    # Check if lyrics are available
    if 'lyrics_text' not in df.columns or df['lyrics_text'].isna().all():
        logging.warning("No lyrics text found in metadata. Skipping TF-IDF extraction.")
        logging.info("Run transcribe.py first to generate lyrics.")
        return
    
    logging.info("Extracting TF-IDF features...")
    
    # Prepare text data
    texts = df['lyrics_text'].fillna('[NO_LYRICS]').tolist()
    
    # Check if we have meaningful text
    meaningful_texts = [t for t in texts if len(t) >= config.min_lyrics_length]
    if len(meaningful_texts) < 100:
        logging.warning(f"Only {len(meaningful_texts)} clips have meaningful lyrics.")
        logging.warning("TF-IDF features may not be informative.")
    
    # Extract TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=config.tfidf_max_features,
        stop_words='english',
        lowercase=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Save TF-IDF matrix (sparse)
        sparse.save_npz(tfidf_path, tfidf_matrix)
        logging.info(f"TF-IDF saved: {tfidf_path} (shape: {tfidf_matrix.shape})")
        
        # Also create and save reduced dense version using SVD
        svd_path = os.path.join(config.features_dir, 'lyrics_tfidf_svd.npy')
        n_components = min(config.tfidf_svd_components, tfidf_matrix.shape[1])
        
        if n_components > 0:
            svd = TruncatedSVD(n_components=n_components, random_state=config.seed)
            tfidf_dense = svd.fit_transform(tfidf_matrix)
            np.save(svd_path, tfidf_dense)
            logging.info(f"TF-IDF SVD saved: {svd_path} (shape: {tfidf_dense.shape})")
            logging.info(f"Explained variance: {svd.explained_variance_ratio_.sum():.3f}")
    
    except Exception as e:
        logging.error(f"Error extracting TF-IDF: {e}")


def extract_sentence_embeddings(df: pd.DataFrame, config: Config, 
                               force: bool = False):
    """
    Extract sentence embeddings using MiniLM (optional, if available).
    
    Args:
        df: Metadata DataFrame with lyrics_text column
        config: Configuration object
        force: Force re-extraction even if files exist
    """
    embeddings_path = os.path.join(config.features_dir, 'lyrics_minilm.npy')
    
    # Check if already extracted
    if not force and os.path.exists(embeddings_path):
        logging.info("Sentence embeddings already exist. Use --force to re-extract.")
        return
    
    # Check if lyrics are available
    if 'lyrics_text' not in df.columns or df['lyrics_text'].isna().all():
        logging.warning("No lyrics text found. Skipping sentence embeddings.")
        return
    
    try:
        from sentence_transformers import SentenceTransformer
        
        logging.info("Extracting sentence embeddings with MiniLM...")
        logging.info("This may take a while on CPU...")
        
        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Prepare texts
        texts = df['lyrics_text'].fillna('[NO_LYRICS]').tolist()
        
        # Extract embeddings
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Save
        np.save(embeddings_path, embeddings)
        logging.info(f"Sentence embeddings saved: {embeddings_path} (shape: {embeddings.shape})")
    
    except ImportError:
        logging.info("sentence-transformers not installed. Skipping.")
        logging.info("Install with: pip install sentence-transformers")
    except Exception as e:
        logging.error(f"Error extracting sentence embeddings: {e}")


def verify_features(config: Config):
    """
    Verify that all feature files exist and have correct shapes.
    
    Args:
        config: Configuration object
    """
    logging.info("Verifying features...")
    
    # Load metadata to get expected number of samples
    if not os.path.exists(config.metadata_path):
        logging.error(f"Metadata not found: {config.metadata_path}")
        return
    
    df = pd.read_csv(config.metadata_path)
    n_samples = len(df)
    
    # Check MFCC
    mfcc_path = os.path.join(config.features_dir, 'mfcc_stats.npy')
    if os.path.exists(mfcc_path):
        mfcc = np.load(mfcc_path)
        expected_shape = (n_samples, config.n_mfcc * 2)
        if mfcc.shape == expected_shape:
            logging.info(f"✓ MFCC stats: {mfcc.shape}")
        else:
            logging.error(f"✗ MFCC shape mismatch: {mfcc.shape} vs {expected_shape}")
    else:
        logging.warning(f"✗ MFCC not found: {mfcc_path}")
    
    # Check log-mel
    logmel_path = os.path.join(config.features_dir, 'logmel.npy')
    if os.path.exists(logmel_path):
        logmel = np.load(logmel_path)
        expected_shape = (n_samples, config.n_mels, config.mel_time_frames)
        if logmel.shape == expected_shape:
            logging.info(f"✓ Log-mel: {logmel.shape}")
        else:
            logging.error(f"✗ Log-mel shape mismatch: {logmel.shape} vs {expected_shape}")
    else:
        logging.warning(f"✗ Log-mel not found: {logmel_path}")
    
    # Check TF-IDF
    tfidf_path = os.path.join(config.features_dir, 'lyrics_tfidf.npz')
    if os.path.exists(tfidf_path):
        tfidf = sparse.load_npz(tfidf_path)
        if tfidf.shape[0] == n_samples:
            logging.info(f"✓ TF-IDF: {tfidf.shape}")
        else:
            logging.error(f"✗ TF-IDF shape mismatch: {tfidf.shape[0]} vs {n_samples}")
    else:
        logging.info(f"ℹ TF-IDF not found (run transcribe.py first): {tfidf_path}")
    
    # Check TF-IDF SVD
    svd_path = os.path.join(config.features_dir, 'lyrics_tfidf_svd.npy')
    if os.path.exists(svd_path):
        svd_feats = np.load(svd_path)
        logging.info(f"✓ TF-IDF SVD: {svd_feats.shape}")
    
    # Check sentence embeddings (optional)
    emb_path = os.path.join(config.features_dir, 'lyrics_minilm.npy')
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
        logging.info(f"✓ Sentence embeddings: {emb.shape}")
    else:
        logging.info(f"ℹ Sentence embeddings not found (optional): {emb_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio and text features from GTZAN clips"
    )
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to metadata CSV (overrides config)')
    parser.add_argument('--audio_only', action='store_true',
                       help='Extract only audio features (MFCC, log-mel)')
    parser.add_argument('--text_only', action='store_true',
                       help='Extract only text features (TF-IDF)')
    parser.add_argument('--sentence_embeddings', action='store_true',
                       help='Also extract sentence embeddings (requires sentence-transformers)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-extraction even if files exist')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify existing features')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    if args.metadata:
        config.metadata_path = args.metadata
    
    # Setup
    setup_logging()
    set_seed(config.seed)
    config.ensure_directories()
    
    logging.info("=" * 60)
    logging.info("Feature Extraction Pipeline")
    logging.info("=" * 60)
    
    # Verify only mode
    if args.verify_only:
        verify_features(config)
        return
    
    # Load metadata
    if not os.path.exists(config.metadata_path):
        logging.error(f"Metadata not found: {config.metadata_path}")
        logging.error("Run preprocess.py first!")
        return
    
    df = pd.read_csv(config.metadata_path)
    logging.info(f"Loaded metadata: {len(df)} clips")
    
    # Extract features
    if not args.text_only:
        extract_audio_features(df, config, force=args.force)
    
    if not args.audio_only:
        extract_tfidf_features(df, config, force=args.force)
        
        if args.sentence_embeddings:
            extract_sentence_embeddings(df, config, force=args.force)
    
    # Verify
    verify_features(config)
    
    logging.info("=" * 60)
    logging.info("Feature extraction complete!")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()


