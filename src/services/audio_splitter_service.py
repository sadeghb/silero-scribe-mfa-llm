# src/services/audio_splitter_service.py
from pathlib import Path
import pandas as pd
from pydub import AudioSegment
from typing import List

class AudioSplitterService:
    """A service to split an audio file into chunks based on timestamps."""
    def __init__(self):
        print("AudioSplitterService initialized.")

    # Modified to accept the pydub AudioSegment object directly
    def run(self, audio_segment: AudioSegment, chunk_df: pd.DataFrame, chunks_dir: Path, original_stem: str) -> List[Path]:
        """
        Splits the audio and saves the chunks to a directory.
        """
        print(f"Splitting audio...")
        chunks_dir.mkdir(exist_ok=True, parents=True)
        chunk_paths = []

        for i, row in chunk_df.iterrows():
            start_ms = row['chunk_start_ms']
            end_ms = row['chunk_end_ms']
            chunk = audio_segment[start_ms:end_ms]
            
            chunk_path = chunks_dir / f"{original_stem}_chunk_{i+1}.wav"
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)
        
        print(f"Successfully created {len(chunk_paths)} audio chunks.")
        return chunk_paths
    