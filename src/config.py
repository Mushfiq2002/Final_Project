"""
Configuration management for the music clustering pipeline.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class Config:
    """Main configuration class for the pipeline."""
    
    # Paths
    project_root: str = "."
    data_dir: str = "data"
    raw_gtzan_dir: str = "data/raw_gtzan"
    clips_dir: str = "data/clips_10s"
    features_dir: str = "data/features"
    metadata_path: str = "data/metadata.csv"
    results_dir: str = "results"
    checkpoints_dir: str = "results/checkpoints"
    metrics_dir: str = "results/metrics"
    figures_dir: str = "results/figures"
    
    # Audio processing
    sample_rate: int = 22050
    clip_duration: float = 10.0  # seconds
    n_clips_per_track: int = 3
    expected_total_clips: int = 3000  # 1000 tracks * 3 clips
    
    # Feature extraction
    n_mfcc: int = 20
    n_mels: int = 128
    hop_length: int = 512
    n_fft: int = 2048
    mel_time_frames: int = 431  # For 10s at sr=22050, hop=512: ~431 frames
    
    # Text features
    tfidf_max_features: int = 5000
    tfidf_svd_components: int = 128
    min_lyrics_length: int = 15
    
    # Dataset splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # GTZAN specifics
    n_genres: int = 10
    genres: List[str] = field(default_factory=lambda: [
        'blues', 'classical', 'country', 'disco', 'hiphop',
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ])
    
    # Training
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    device: str = "auto"  # 'auto', 'cpu', 'cuda', or 'mps'
    
    # Model architecture
    latent_dim: int = 16
    beta: float = 1.0  # for beta-VAE
    
    # Clustering
    n_clusters: int = 10
    
    # Whisper transcription
    whisper_model_size: str = "tiny"  # tiny/base/small/medium
    whisper_compute_type: str = "int8"  # int8 for CPU, float16 for GPU
    
    def __post_init__(self):
        """Ensure all paths are absolute and create directories if needed."""
        # Make paths absolute if they're relative
        if not os.path.isabs(self.project_root):
            self.project_root = os.path.abspath(self.project_root)
        
        # Update all paths to be absolute
        for attr in ['data_dir', 'raw_gtzan_dir', 'clips_dir', 'features_dir',
                     'metadata_path', 'results_dir', 'checkpoints_dir',
                     'metrics_dir', 'figures_dir']:
            path = getattr(self, attr)
            if not os.path.isabs(path):
                setattr(self, attr, os.path.join(self.project_root, path))
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        # Convert to dict, excluding methods
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def ensure_directories(self):
        """Create all necessary directories."""
        dirs = [
            self.data_dir, self.clips_dir, self.features_dir,
            self.results_dir, self.checkpoints_dir,
            self.metrics_dir, self.figures_dir
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


# Default global config instance
default_config = Config()


