# main.py
from pathlib import Path
from tqdm import tqdm
from src.pipeline_orchestrator import PipelineOrchestrator
from src.utils.config_loader import load_config
from src.model_loader import SILERO_MODEL, SILERO_UTILS
# Import all our services
from src.services.vad_service import VADService
from src.services.split_point_service import SplitPointService
from src.services.transcription_chunker_service import TranscriptionChunkerService
from src.services.audio_splitter_service import AudioSplitterService
from src.services.scribe_service import ScribeService
from src.services.scribe_normalizer_service import ScribeNormalizerService
from src.services.llm_cut_selector_service import LLMCutSelectorService # <-- Import new service

def main():
    """Main function to build the services, orchestrator, and run the pipeline."""
    config = load_config()
    print("Configuration and models loaded.")

    # Build all specialist services
    services = {
        'vad': VADService(model=SILERO_MODEL, utils=SILERO_UTILS),
        'split_point': SplitPointService(),
        'transcription_chunker': TranscriptionChunkerService(),
        'audio_splitter': AudioSplitterService(),
        'scribe': ScribeService(config['api_keys']['elevenlabs']),
        'scribe_normalizer': ScribeNormalizerService(),
        'llm_cut_selector': LLMCutSelectorService(config['api_keys']['llm']), # <-- Add new service
    }
    print("All services initialized.")

    orchestrator = PipelineOrchestrator(services=services, config=config)

    base_dir = Path(__file__).parent
    input_dir = base_dir / 'audio_inputs'
    audio_files = [p for p in input_dir.glob('**/*') if p.suffix.lower() in ['.wav', '.mp3']]

    if not audio_files:
        print(f"\nNo audio files found in '{input_dir}'.")
        return

    print(f"\nFound {len(audio_files)} audio file(s) to process.")
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            orchestrator.run(audio_path=audio_path)
        except Exception as e:
            print(f"\n❌ An unhandled error occurred for {audio_path.name}: {e}")

    print("\n--- All files processed. ---")

if __name__ == '__main__':
    main()
    