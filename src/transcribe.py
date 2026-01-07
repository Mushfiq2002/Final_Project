"""
Transcribe audio clips using Whisper to generate "lyrics" text.
"""
import os
import argparse
import logging
from typing import Optional
import pandas as pd
from tqdm import tqdm

from config import Config
from utils import setup_logging, set_seed


def transcribe_with_faster_whisper(clip_path: str,
                                   model,
                                   min_length: int = 15) -> tuple:
    """
    Transcribe audio clip using faster-whisper.
    
    Args:
        clip_path: Path to audio clip
        model: Faster-whisper model
        min_length: Minimum text length to consider valid
        
    Returns:
        Tuple of (text, status)
        status: 'ok', 'empty', or 'failed'
    """
    try:
        # Transcribe
        segments, info = model.transcribe(clip_path, beam_size=5)
        
        # Collect text
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        text = ' '.join(text_parts).strip()
        
        # Check if meaningful
        if len(text) >= min_length:
            return text, 'ok'
        else:
            return '[NO_LYRICS]', 'empty'
    
    except Exception as e:
        logging.error(f"Transcription failed for {clip_path}: {e}")
        return '[FAILED]', 'failed'


def transcribe_with_openai_whisper(clip_path: str,
                                   model,
                                   min_length: int = 15) -> tuple:
    """
    Transcribe audio clip using openai-whisper (fallback).
    
    Args:
        clip_path: Path to audio clip
        model: Whisper model
        min_length: Minimum text length
        
    Returns:
        Tuple of (text, status)
    """
    try:
        result = model.transcribe(clip_path)
        text = result['text'].strip()
        
        if len(text) >= min_length:
            return text, 'ok'
        else:
            return '[NO_LYRICS]', 'empty'
    
    except Exception as e:
        logging.error(f"Transcription failed for {clip_path}: {e}")
        return '[FAILED]', 'failed'


def transcribe_clips(metadata_path: str,
                    config: Config,
                    model_size: str = 'tiny',
                    device: str = 'cpu',
                    compute_type: str = 'int8',
                    use_faster_whisper: bool = True,
                    force: bool = False,
                    max_clips: Optional[int] = None) -> pd.DataFrame:
    """
    Transcribe all audio clips in metadata.
    
    Args:
        metadata_path: Path to metadata CSV
        config: Configuration object
        model_size: Whisper model size (tiny/base/small/medium)
        device: Device (cpu/cuda)
        compute_type: Compute type (int8/float16)
        use_faster_whisper: Whether to use faster-whisper
        force: Force re-transcription
        max_clips: Maximum clips to process (for testing)
        
    Returns:
        Updated metadata DataFrame
    """
    # Load metadata
    df = pd.read_csv(metadata_path)
    logging.info(f"Loaded metadata: {len(df)} clips")
    
    # Check which clips need transcription
    if not force:
        # Skip clips already transcribed
        need_transcription = (
            (df['lyrics_status'] == 'pending') | 
            (df['lyrics_status'].isna()) |
            (df['lyrics_text'] == '')
        )
        clips_to_process = df[need_transcription].copy()
        logging.info(f"Clips already transcribed: {len(df) - len(clips_to_process)}")
        logging.info(f"Clips to process: {len(clips_to_process)}")
    else:
        clips_to_process = df.copy()
        logging.info(f"Force mode: processing all {len(clips_to_process)} clips")
    
    if len(clips_to_process) == 0:
        logging.info("All clips already transcribed!")
        return df
    
    # Limit for testing
    if max_clips is not None and max_clips < len(clips_to_process):
        clips_to_process = clips_to_process.iloc[:max_clips]
        logging.info(f"Limited to {max_clips} clips for testing")
    
    # Load Whisper model
    # Note: Whisper doesn't support MPS well due to sparse tensor operations
    # Force CPU for transcription - it's fast enough for audio processing
    if device == 'mps':
        logging.warning("MPS not supported for Whisper (sparse tensor limitations). Using CPU instead.")
        logging.info("Note: MPS works for train.py, but Whisper transcription runs best on CPU.")
        device = 'cpu'
    
    logging.info(f"Loading Whisper model: {model_size} on {device}")
    
    if use_faster_whisper:
        try:
            from faster_whisper import WhisperModel
            
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            transcribe_fn = transcribe_with_faster_whisper
            logging.info("Using faster-whisper")
        
        except ImportError:
            logging.warning("faster-whisper not installed, falling back to openai-whisper")
            use_faster_whisper = False
    
    if not use_faster_whisper:
        try:
            import whisper
            
            model = whisper.load_model(model_size, device=device)
            transcribe_fn = transcribe_with_openai_whisper
            logging.info("Using openai-whisper")
        
        except ImportError:
            raise ImportError(
                "Neither faster-whisper nor openai-whisper is installed!\n"
                "Install with: pip install faster-whisper\n"
                "Or: pip install openai-whisper"
            )
    
    # Transcribe clips
    logging.info("Starting transcription...")
    
    for idx, row in tqdm(clips_to_process.iterrows(), total=len(clips_to_process), desc="Transcribing"):
        clip_path = row['clip_path']
        
        # Check if file exists
        if not os.path.exists(clip_path):
            logging.warning(f"Clip not found: {clip_path}")
            df.loc[idx, 'lyrics_text'] = '[FILE_NOT_FOUND]'
            df.loc[idx, 'lyrics_status'] = 'failed'
            continue
        
        # Transcribe
        text, status = transcribe_fn(clip_path, model, config.min_lyrics_length)
        
        # Update DataFrame
        df.loc[idx, 'lyrics_text'] = text
        df.loc[idx, 'lyrics_status'] = status
    
    # Save updated metadata
    df.to_csv(metadata_path, index=False)
    logging.info(f"Updated metadata saved: {metadata_path}")
    
    # Log statistics
    status_counts = df['lyrics_status'].value_counts()
    logging.info("\nTranscription statistics:")
    for status, count in status_counts.items():
        logging.info(f"  {status}: {count} ({count/len(df)*100:.1f}%)")
    
    # Show some examples
    ok_samples = df[df['lyrics_status'] == 'ok'].head(5)
    if len(ok_samples) > 0:
        logging.info("\nExample transcriptions:")
        for idx, row in ok_samples.iterrows():
            text = row['lyrics_text'][:100] + '...' if len(row['lyrics_text']) > 100 else row['lyrics_text']
            logging.info(f"  {row['clip_id']}: {text}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio clips using Whisper"
    )
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to metadata CSV (overrides config)')
    parser.add_argument('--model_size', type=str, default='tiny',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device for transcription (cpu recommended for Whisper)')
    parser.add_argument('--compute_type', type=str, default='int8',
                       choices=['int8', 'float16', 'float32'],
                       help='Compute type (int8 for CPU, float16 for GPU)')
    parser.add_argument('--use_openai_whisper', action='store_true',
                       help='Use openai-whisper instead of faster-whisper')
    parser.add_argument('--force', action='store_true',
                       help='Force re-transcription of all clips')
    parser.add_argument('--max_clips', type=int, default=None,
                       help='Maximum clips to process (for testing)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    if args.metadata:
        config.metadata_path = args.metadata
    if args.seed is not None:
        config.seed = args.seed
    
    # Setup
    setup_logging()
    set_seed(config.seed)
    
    logging.info("=" * 60)
    logging.info("Whisper Transcription Pipeline")
    logging.info("=" * 60)
    logging.info(f"Model size: {args.model_size}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Compute type: {args.compute_type}")
    logging.info(f"Metadata: {config.metadata_path}")
    
    # Check metadata exists
    if not os.path.exists(config.metadata_path):
        logging.error(f"Metadata not found: {config.metadata_path}")
        logging.error("Run preprocess.py first!")
        return
    
    # Transcribe
    use_faster = not args.use_openai_whisper
    
    df = transcribe_clips(
        metadata_path=config.metadata_path,
        config=config,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        use_faster_whisper=use_faster,
        force=args.force,
        max_clips=args.max_clips
    )
    
    logging.info("=" * 60)
    logging.info("Transcription complete!")
    logging.info("=" * 60)
    logging.info("Next step: Run features.py to extract TF-IDF features")


if __name__ == '__main__':
    main()

