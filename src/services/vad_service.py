# src/services/vad_service.py
from pathlib import Path
import pandas as pd
from ..vad_processor import process_audio

class VADService:
    """A specialist service for performing Voice Activity Detection."""
    def __init__(self, model, utils):
        print("VADService initialized.")
        self.model = model
        self.get_speech_timestamps = utils[0]

    def run(self, audio_path: Path) -> pd.DataFrame:
        """Runs the VAD processing on a given audio file."""
        return process_audio(
            audio_path=audio_path,
            model=self.model,
            get_speech_timestamps=self.get_speech_timestamps
        )
    