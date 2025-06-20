# src/pipeline.py
from pathlib import Path
import pandas as pd
from .vad_processor import process_audio

def run_vad_pipeline(
    audio_path: Path, 
    cache_dir: Path, 
    model, 
    get_speech_timestamps, 
    vad_suffix: str
) -> pd.DataFrame:
    """
    Runs the VAD pipeline for a single audio file with caching.
    
    Checks if the VAD timestamps are already cached. If so, loads them.
    If not, runs the VAD processor, saves the result to the cache,
    and then returns the result.

    Args:
        audio_path (Path): Path to the input audio file.
        cache_dir (Path): Directory where cached results are stored.
        model: The loaded Silero VAD ONNX model.
        get_speech_timestamps: The function from Silero utils to get timestamps.
        vad_suffix (str): The suffix for the VAD output filename.

    Returns:
        pd.DataFrame: A DataFrame with 'start_ms' and 'end_ms' columns.
    """
    # Define the expected path for the cached VAD file
    output_filename = audio_path.stem + vad_suffix
    cache_path = cache_dir / output_filename

    # --- Caching Logic ---
    if cache_path.exists():
        print(f"✅ Found cached VAD file. Loading from: {cache_path}")
        # Load the DataFrame from the cached CSV file
        timestamps_df = pd.read_csv(cache_path)
        return timestamps_df
    
    # --- Processing Logic (if not cached) ---
    print(f"No cache found. Processing VAD for {audio_path.name}...")
    
    # Get the processed data from the VAD module
    timestamps_df = process_audio(
        audio_path=audio_path,
        model=model,
        get_speech_timestamps=get_speech_timestamps
    )

    # Save the new result to the cache for future runs
    if not timestamps_df.empty:
        timestamps_df.to_csv(cache_path, index=False)
        print(f"✅ Success! VAD timestamps saved to cache: {cache_path}")
    else:
        # Still save an empty file to cache the fact that no speech was detected
        timestamps_df.to_csv(cache_path, index=False)
        print("✅ Success! (No speech was detected, result cached).")

    return timestamps_df
