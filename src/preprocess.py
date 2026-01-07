"""
Preprocessing script: Generate 10-second clips from GTZAN dataset.
Creates metadata.csv with train/val/test splits.
"""
import os
import argparse
import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf

from config import Config
from utils import setup_logging, set_seed


def find_gtzan_tracks(gtzan_dir: str, genres: List[str]) -> Dict[str, List[str]]:
    """
    Find all GTZAN tracks organized by genre.
    
    Args:
        gtzan_dir: Path to GTZAN root directory
        genres: List of genre names
        
    Returns:
        Dictionary mapping genre to list of track paths
    """
    tracks_by_genre = {}
    
    # Try common GTZAN folder structures
    possible_structures = [
        os.path.join(gtzan_dir, 'genres_original'),  # genres_original/blues/*.wav
        os.path.join(gtzan_dir, 'genres'),            # genres/blues/*.wav
        gtzan_dir                                      # gtzan/blues/*.wav
    ]
    
    for base_path in possible_structures:
        if os.path.exists(base_path):
            for genre in genres:
                genre_path = os.path.join(base_path, genre)
                if os.path.exists(genre_path):
                    wav_files = sorted([
                        os.path.join(genre_path, f)
                        for f in os.listdir(genre_path)
                        if f.endswith('.wav') or f.endswith('.au')
                    ])
                    if wav_files:
                        tracks_by_genre[genre] = wav_files
            
            if tracks_by_genre:
                logging.info(f"Found GTZAN tracks in: {base_path}")
                break
    
    if not tracks_by_genre:
        error_msg = f"""
        GTZAN dataset not found!
        
        Expected structure (any of these):
        {gtzan_dir}/genres_original/<genre>/*.wav
        {gtzan_dir}/genres/<genre>/*.wav
        {gtzan_dir}/<genre>/*.wav
        
        Where <genre> is one of: {', '.join(genres)}
        
        Please place GTZAN dataset in: {gtzan_dir}
        """
        raise FileNotFoundError(error_msg)
    
    # Verify counts
    for genre, tracks in tracks_by_genre.items():
        logging.info(f"Genre '{genre}': {len(tracks)} tracks")
    
    total_tracks = sum(len(tracks) for tracks in tracks_by_genre.values())
    logging.info(f"Total tracks found: {total_tracks}")
    
    if total_tracks < 1000:
        logging.warning(f"Expected 1000 tracks, found {total_tracks}")
    
    return tracks_by_genre


def has_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which('ffmpeg') is not None


def create_clip_ffmpeg(input_path: str, output_path: str, 
                       start_sec: float, duration: float, sr: int) -> bool:
    """
    Create audio clip using ffmpeg (faster method).
    
    Args:
        input_path: Input audio file
        output_path: Output clip path
        start_sec: Start time in seconds
        duration: Clip duration in seconds
        sr: Sample rate
        
    Returns:
        True if successful
    """
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ss', str(start_sec),
            '-t', str(duration),
            '-ar', str(sr),
            '-ac', '1',  # mono
            '-y',  # overwrite
            '-loglevel', 'error',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        logging.warning(f"ffmpeg failed: {e}")
        return False


def create_clip_librosa(input_path: str, output_path: str,
                        start_sec: float, duration: float, sr: int) -> bool:
    """
    Create audio clip using librosa (fallback method).
    
    Args:
        input_path: Input audio file
        output_path: Output clip path
        start_sec: Start time in seconds
        duration: Clip duration in seconds
        sr: Sample rate
        
    Returns:
        True if successful
    """
    try:
        # Load audio
        y, orig_sr = librosa.load(input_path, sr=sr, mono=True, 
                                  offset=start_sec, duration=duration)
        
        # Ensure exact duration (pad if needed)
        expected_samples = int(duration * sr)
        if len(y) < expected_samples:
            y = np.pad(y, (0, expected_samples - len(y)), mode='constant')
        elif len(y) > expected_samples:
            y = y[:expected_samples]
        
        # Save
        sf.write(output_path, y, sr)
        return True
    except Exception as e:
        logging.error(f"librosa failed for {input_path}: {e}")
        return False


def create_clips(tracks_by_genre: Dict[str, List[str]],
                output_dir: str,
                config: Config,
                force: bool = False) -> pd.DataFrame:
    """
    Create 10-second clips from GTZAN tracks.
    
    Args:
        tracks_by_genre: Dictionary of genre -> track paths
        output_dir: Output directory for clips
        config: Configuration object
        force: Force regeneration of existing clips
        
    Returns:
        DataFrame with clip metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for ffmpeg
    use_ffmpeg = has_ffmpeg()
    if use_ffmpeg:
        logging.info("Using ffmpeg for clip generation (fast)")
    else:
        logging.info("ffmpeg not found, using librosa (slower)")
    
    metadata_records = []
    clip_count = 0
    
    for genre_id, (genre, track_paths) in enumerate(sorted(tracks_by_genre.items())):
        logging.info(f"Processing genre: {genre} (ID: {genre_id})")
        
        for track_idx, track_path in enumerate(tqdm(track_paths, desc=genre)):
            track_name = Path(track_path).stem
            
            # Create 3 clips per track: [0-10s], [10-20s], [20-30s]
            for seg_idx in range(config.n_clips_per_track):
                start_sec = seg_idx * config.clip_duration
                
                # Generate clip ID
                clip_id = f"{genre}_{track_idx:03d}_{seg_idx}"
                clip_filename = f"{clip_id}.wav"
                clip_path = os.path.join(output_dir, clip_filename)
                
                # Skip if exists and not forcing
                if os.path.exists(clip_path) and not force:
                    clip_duration = librosa.get_duration(path=clip_path)
                else:
                    # Create clip
                    if use_ffmpeg:
                        success = create_clip_ffmpeg(
                            track_path, clip_path, start_sec, 
                            config.clip_duration, config.sample_rate
                        )
                    else:
                        success = create_clip_librosa(
                            track_path, clip_path, start_sec,
                            config.clip_duration, config.sample_rate
                        )
                    
                    if not success:
                        logging.error(f"Failed to create clip: {clip_id}")
                        continue
                    
                    clip_duration = config.clip_duration
                
                # Add metadata record
                metadata_records.append({
                    'clip_id': clip_id,
                    'genre': genre,
                    'genre_id': genre_id,
                    'track_path': track_path,
                    'track_name': track_name,
                    'clip_path': clip_path,
                    'start_sec': start_sec,
                    'end_sec': start_sec + config.clip_duration,
                    'duration_sec': clip_duration,
                    'segment_idx': seg_idx,
                    'track_idx': track_idx,
                    'lyrics_text': '',
                    'lyrics_status': 'pending'
                })
                
                clip_count += 1
    
    logging.info(f"Created {clip_count} clips")
    
    # Create DataFrame
    df = pd.DataFrame(metadata_records)
    return df


def create_splits(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Create train/val/test splits at track level to avoid leakage.
    
    Args:
        df: Metadata DataFrame
        config: Configuration object
        
    Returns:
        DataFrame with 'split' column added
    """
    logging.info("Creating train/val/test splits...")
    
    # Get unique tracks (genre + track_idx identifies a unique track)
    df['track_id'] = df['genre'] + '_' + df['track_idx'].astype(str)
    unique_tracks = df['track_id'].unique()
    
    # Shuffle tracks
    np.random.shuffle(unique_tracks)
    
    # Split tracks
    n_tracks = len(unique_tracks)
    n_train = int(n_tracks * config.train_ratio)
    n_val = int(n_tracks * config.val_ratio)
    
    train_tracks = unique_tracks[:n_train]
    val_tracks = unique_tracks[n_train:n_train + n_val]
    test_tracks = unique_tracks[n_train + n_val:]
    
    # Assign splits to all clips
    df['split'] = 'test'
    df.loc[df['track_id'].isin(train_tracks), 'split'] = 'train'
    df.loc[df['track_id'].isin(val_tracks), 'split'] = 'val'
    
    # Log split statistics
    split_counts = df['split'].value_counts()
    for split_name in ['train', 'val', 'test']:
        count = split_counts.get(split_name, 0)
        logging.info(f"{split_name}: {count} clips ({count/len(df)*100:.1f}%)")
    
    # Remove temporary track_id column
    df = df.drop(columns=['track_id'])
    
    return df


def verify_clips(df: pd.DataFrame, config: Config):
    """
    Verify that all clips exist and have reasonable durations.
    
    Args:
        df: Metadata DataFrame
        config: Configuration object
    """
    logging.info("Verifying clips...")
    
    missing_clips = []
    invalid_durations = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying"):
        clip_path = row['clip_path']
        
        if not os.path.exists(clip_path):
            missing_clips.append(row['clip_id'])
            continue
        
        # Check duration
        try:
            duration = librosa.get_duration(path=clip_path)
            if abs(duration - config.clip_duration) > 0.5:  # tolerance of 0.5s
                invalid_durations.append((row['clip_id'], duration))
        except Exception as e:
            logging.error(f"Error checking {clip_path}: {e}")
    
    if missing_clips:
        logging.error(f"Missing {len(missing_clips)} clips: {missing_clips[:5]}...")
    
    if invalid_durations:
        logging.warning(f"Found {len(invalid_durations)} clips with unexpected duration")
        for clip_id, dur in invalid_durations[:5]:
            logging.warning(f"  {clip_id}: {dur:.2f}s")
    
    if not missing_clips and not invalid_durations:
        logging.info("✓ All clips verified successfully!")
    
    # Final count check
    expected = config.expected_total_clips
    actual = len(df)
    logging.info(f"Expected: {expected} clips, Actual: {actual} clips")
    
    if actual < expected * 0.95:  # Allow 5% tolerance
        logging.warning(f"Clip count is below expected ({actual} < {expected})")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GTZAN: Generate 10s clips and metadata"
    )
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--gtzan_dir', type=str, default=None,
                       help='Path to GTZAN dataset (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for clips (overrides config)')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration of existing clips')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify existing clips, do not generate')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override config with CLI args
    if args.gtzan_dir:
        config.raw_gtzan_dir = args.gtzan_dir
    if args.output_dir:
        config.clips_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    
    # Setup
    setup_logging()
    set_seed(config.seed)
    config.ensure_directories()
    
    logging.info("=" * 60)
    logging.info("GTZAN Preprocessing Pipeline")
    logging.info("=" * 60)
    logging.info(f"GTZAN directory: {config.raw_gtzan_dir}")
    logging.info(f"Output directory: {config.clips_dir}")
    logging.info(f"Metadata path: {config.metadata_path}")
    logging.info(f"Sample rate: {config.sample_rate} Hz")
    logging.info(f"Clip duration: {config.clip_duration}s")
    logging.info(f"Random seed: {config.seed}")
    
    # If verify only mode
    if args.verify_only:
        if os.path.exists(config.metadata_path):
            df = pd.read_csv(config.metadata_path)
            verify_clips(df, config)
        else:
            logging.error(f"Metadata file not found: {config.metadata_path}")
        return
    
    # Find GTZAN tracks
    tracks_by_genre = find_gtzan_tracks(config.raw_gtzan_dir, config.genres)
    
    # Create clips
    df = create_clips(tracks_by_genre, config.clips_dir, config, force=args.force)
    
    # Create splits
    df = create_splits(df, config)
    
    # Save metadata
    df.to_csv(config.metadata_path, index=False)
    logging.info(f"Metadata saved to: {config.metadata_path}")
    logging.info(f"Total records: {len(df)}")
    
    # Verify clips
    verify_clips(df, config)
    
    logging.info("=" * 60)
    logging.info("Preprocessing complete!")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()


