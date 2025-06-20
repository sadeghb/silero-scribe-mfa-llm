import logging
import requests
from typing import Dict, Any

# Assuming a similar config loader utility exists.
# If not, this function would need to be created to load configuration
# from a file (e.g., config.yaml).
from .utils.config_loader import load_config 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_scribe_results(audio_path: str) -> Dict[str, Any]:
    """
    Makes a single, comprehensive call to the ElevenLabs Scribe API and returns
    the entire raw JSON response.

    Args:
        audio_path: The path to the audio file (e.g., .mp3, .wav).

    Returns:
        The full, unprocessed JSON response from Scribe as a dictionary.
    """
    logging.info(f"Requesting transcription from ElevenLabs Scribe for '{audio_path}'")
    try:
        # Load configuration, assuming it contains an 'elevenlabs' section
        app_config = load_config()
        scribe_api_key = app_config.get('elevenlabs', {}).get('api_key')
        if not scribe_api_key or scribe_api_key == "YOUR_ELEVENLABS_API_KEY_HERE":
            logging.error("ElevenLabs API key not found or not set in config.yaml.")
            raise ValueError("ElevenLabs API key not configured.")

        # API endpoint for ElevenLabs Scribe
        url = 'https://api.elevenlabs.io/v1/speech-to-text'

        # Set up headers with the required API key format for ElevenLabs
        headers = {
            'xi-api-key': scribe_api_key
        }

        # The data payload for the request, specifying the model
        # and requesting rich details like timestamps and speaker labels (diarization).
        data = {
            'model_id': 'eleven_scribe_v1',
            'timestamps': 'word',  # Request word-level timestamps
            'diarize': 'true'     # Enable speaker detection
        }
        
        with open(audio_path, 'rb') as audio_file:
            # The files payload for the multipart/form-data request
            files = {
                'file': (audio_path, audio_file, 'audio/mpeg') # MIME type can be adjusted
            }

            logging.info("Sending audio to ElevenLabs Scribe...")
            response = requests.post(url, headers=headers, data=data, files=files)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            logging.info("Successfully received full raw response from Scribe.")
            return result

    except Exception as e:
        logging.error(f"An error occurred while calling ElevenLabs Scribe API: {e}", exc_info=True)
        raise
    