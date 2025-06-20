# src/model_loader.py
import torch

def load_silero_model():
    """Loads the Silero VAD model and utils from torch.hub."""
    print("Initializing Silero VAD model... (This should only happen once)")
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=True)
        return model, utils
    except Exception as e:
        print(f"Fatal: Error loading Silero VAD model: {e}")
        raise

# Load the model once when this module is first imported
SILERO_MODEL, SILERO_UTILS = load_silero_model()
