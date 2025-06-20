# src/vad_processor.py
import numpy as np
from pathlib import Path
from pydub import AudioSegment
import pandas as pd

SAMPLE_RATE = 16000

def process_audio(audio_path: Path, model, get_speech_timestamps) -> pd.DataFrame:
    """Processes a single audio file to detect speech segments and returns them as a DataFrame."""
    try:
        audio = AudioSegment.from_file(audio_path)
        if audio.frame_rate != SAMPLE_RATE:
            audio = audio.set_frame_rate(SAMPLE_RATE)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)
        raw_samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        audio_float32 = raw_samples.astype(np.float32) / 32768.0
    except Exception as e:
        raise RuntimeError(f"Error loading or preprocessing audio file {audio_path.name}: {e}")

    try:
        speech_timestamps = get_speech_timestamps(audio_float32, model, sampling_rate=SAMPLE_RATE, return_seconds=False)
    except Exception as e:
        raise RuntimeError(f"Error during VAD processing for {audio_path.name}: {e}")

    if not speech_timestamps:
        return pd.DataFrame(columns=['start_ms', 'end_ms'])
    
    df = pd.DataFrame(speech_timestamps)
    df['start_ms'] = (df['start'] / SAMPLE_RATE * 1000).astype(int)
    df['end_ms'] = (df['end'] / SAMPLE_RATE * 1000).astype(int)
    
    return df[['start_ms', 'end_ms']]
