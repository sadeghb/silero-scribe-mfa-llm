# src/services/audio_splitter_service.py
from pathlib import Path
import pandas as pd
from pydub import AudioSegment
import logging

class AudioSplitterService:
    """
    A service to split an audio file into chunks based on a DataFrame
    of start and end times.
    """
    def __init__(self):
        logging.info("AudioSplitterService initialized.")

    def run(self, audio: AudioSegment, transcription_chunks_df: pd.DataFrame, chunks_dir: Path, audio_name: str) -> list[Path]:
        """
        Splits the main audio into smaller chunks for transcription.
        """
        chunk_paths = []
        for i, row in transcription_chunks_df.iterrows():
            start_ms = row['start_ms']
            end_ms = row['end_ms']
            
            chunk_path = chunks_dir / f"{audio_name}_scribe_chunk_{i + 1}.wav"
            
            self.split_and_save_chunk(audio, start_ms, end_ms, chunk_path)
            chunk_paths.append(chunk_path)
            
        return chunk_paths

    def split_and_save_chunk(self, audio: AudioSegment, start_ms: float, end_ms: float, output_path: Path):
        """
        Extracts a single audio chunk from the main audio and saves it to a file.
        """
        chunk = audio[start_ms:end_ms]
        chunk.export(output_path, format="wav")
