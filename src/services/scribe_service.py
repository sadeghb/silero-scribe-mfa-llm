# src/services/scribe_service.py
from pathlib import Path
import requests
import logging
from typing import Dict, Any

class ScribeService:
    """A service for transcribing audio using the ElevenLabs Scribe API."""
    def __init__(self, api_key: str):
        print("ScribeService initialized.")
        if not api_key or api_key == "YOUR_ELEVENLABS_API_KEY_HERE":
            raise ValueError("ElevenLabs API key is not configured in config.yaml.")
        self.api_key = api_key
        self.url = 'https://api.elevenlabs.io/v1/speech-to-text'

    def run(self, audio_chunk_path: Path) -> Dict[str, Any]:
        """
        Transcribes a single audio chunk and returns the full JSON response.
        """
        logging.info(f"Requesting Scribe transcription for '{audio_chunk_path.name}'")
        headers = {'xi-api-key': self.api_key}
        
        # --- FIX ---
        # The model_id is now 'scribe_v1' as specified by the API error and documentation.
        # Added 'diarize' to match the cookbook example's best practice.
        data = {
            'model_id': 'scribe_v1',
            'diarize': 'true'
        }

        try:
            with open(audio_chunk_path, 'rb') as audio_file:
                files = {'file': (audio_chunk_path.name, audio_file, 'audio/wav')}
                response = requests.post(self.url, headers=headers, data=data, files=files)
                response.raise_for_status()
            
            logging.info(f"Successfully received transcription for '{audio_chunk_path.name}'.")
            return response.json()
        
        except requests.exceptions.RequestException as e:
            if e.response is not None:
                logging.error(f"Scribe API returned an error. Status: {e.response.status_code}, Body: {e.response.text}")
            
            logging.error(f"Scribe API request failed for '{audio_chunk_path.name}': {e}", exc_info=False)
            raise e
        