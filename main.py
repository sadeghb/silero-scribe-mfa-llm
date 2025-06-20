# main.py
from pathlib import Path
from tqdm import tqdm
import yaml
import logging # <-- 1. Import logging
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
from src.services.llm_cut_selector_service import LLMCutSelectorService
from src.services.mfa_chunker_service import MfaChunkerService
from src.services.mfa_aligner_service import MfaAlignerService
from src.services.mfa_normalizer_service import MfaNormalizerService


def main():
    """Main function to build the services, orchestrator, and run the pipeline."""
    # --- MODIFICATION START ---
    # 2. Configure logging for the entire application
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # --- MODIFICATION END ---

    config = load_config()
    logging.info("Configuration and models loaded.") # <-- 3. Use logging.info

    # Build all specialist services
    services = {
        'vad': VADService(model=SILERO_MODEL, utils=SILERO_UTILS),
        'split_point': SplitPointService(),
        'transcription_chunker': TranscriptionChunkerService(),
        'audio_splitter': AudioSplitterService(),
        'scribe': ScribeService(config['api_keys']['elevenlabs']),
        'scribe_normalizer': ScribeNormalizerService(),
        'llm_cut_selector': LLMCutSelectorService(config['api_keys']['llm']),
        'mfa_chunker': MfaChunkerService(),
        'mfa_aligner': MfaAlignerService(config),
        'mfa_normalizer': MfaNormalizerService(),
    }
    logging.info("All services initialized.") # <-- 3. Use logging.info

    orchestrator = PipelineOrchestrator(services=services, config=config)

    base_dir = Path(__file__).parent
    input_dir = base_dir / 'audio_inputs'
    audio_files = [p for p in input_dir.glob('**/*') if p.suffix.lower() in ['.wav', '.mp3']]

    if not audio_files:
        logging.info(f"\nNo audio files found in '{input_dir}'.") # <-- 3. Use logging.info
        return

    logging.info(f"\nFound {len(audio_files)} audio file(s) to process.") # <-- 3. Use logging.info
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            orchestrator.run(audio_path=audio_path)
        except Exception as e:
            # Use logging.error for errors
            logging.error(f"\n❌ An unhandled error occurred for {audio_path.name}: {e}", exc_info=True)

    logging.info("\n--- All files processed. ---") # <-- 3. Use logging.info

if __name__ == '__main__':
    main()
    