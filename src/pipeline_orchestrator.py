# src/pipeline_orchestrator.py
from pathlib import Path
import pandas as pd
import json
from pydub import AudioSegment
from typing import Dict, Any
import shutil
import logging

from src.utils.mfa_text_normalizer import normalize_text_for_mfa

class PipelineOrchestrator:
    def __init__(self, services: Dict, config: Dict[str, Any]):
        logging.info("PipelineOrchestrator initialized.")
        self.services = services
        self.config = config
        self.use_cache = self.config.get('use_cache', False)
        base_dir = Path(__file__).parent.parent
        self.cache_root = base_dir / 'cache'
        self.cache_root.mkdir(exist_ok=True)

    def _get_cache_path(self, stage_name: str, source_path: Path, suffix_override: str = None) -> Path:
        try:
            stage_cache_dir_str = self.config['cache_paths'][stage_name]
        except KeyError:
            raise KeyError(f"Error: The key '{stage_name}' was not found in the 'cache_paths' section of your config.yaml.")
        
        stage_cache_dir = self.cache_root.parent / stage_cache_dir_str
        stage_cache_dir.mkdir(exist_ok=True, parents=True)

        if suffix_override:
            return stage_cache_dir / (source_path.stem + suffix_override)

        suffix_key_map = {
            'vad': 'vad_timestamps_suffix',
            'split_points': 'split_points_suffix',
            'scribe': 'scribe_timestamps_suffix',
            'llm': 'llm_marked_transcript_suffix',
            'mfa': 'mfa_aligned_suffix'
        }
        if stage_name not in suffix_key_map:
            raise ValueError(f"Unknown suffix key for stage: {stage_name}")
        
        suffix_key = suffix_key_map[stage_name]
        
        try:
            suffix = self.config['output_files'][suffix_key]
        except KeyError:
            raise KeyError(f"Error: The key '{suffix_key}' was not found in the 'output_files' section of your config.yaml.")
        
        return stage_cache_dir / (source_path.stem + suffix)


    def run(self, audio_path: Path):
        logging.info(f"\n--- Starting pipeline for: {audio_path.name} ---")
        audio = AudioSegment.from_file(audio_path)
        total_duration_s = len(audio) / 1000.0
        
        logging.info("Executing VAD stage...")
        vad_cache_path = self._get_cache_path('vad', audio_path)
        if self.use_cache and vad_cache_path.exists():
            vad_df = pd.read_csv(vad_cache_path)
        else:
            vad_df = self.services['vad'].run(audio_path)
            if self.use_cache: vad_df.to_csv(vad_cache_path, index=False)
        logging.info("VAD stage complete.")

        if vad_df.empty: return None

        logging.info("Executing Split Point Generation stage...")
        split_points_cache_path = self._get_cache_path('split_points', audio_path)
        if self.use_cache and split_points_cache_path.exists():
            split_points_df = pd.read_csv(split_points_cache_path)
        else:
            split_points_df = self.services['split_point'].run(vad_df, len(audio))
            if self.use_cache: split_points_df.to_csv(split_points_cache_path, index=False)
        logging.info("Split Point Generation complete.")

        logging.info("Executing Scribe Transcription stage...")
        final_transcript_cache_path = self._get_cache_path('scribe', audio_path)
        if self.use_cache and final_transcript_cache_path.exists():
            logging.info(f"✅ Final scribe transcript found in cache. Loading from {final_transcript_cache_path}")
            with open(final_transcript_cache_path, 'r') as f:
                final_transcript = json.load(f)
        else:
            logging.info("No final scribe transcript cache found. Running full transcription pipeline...")
            
            transcription_chunks_df = self.services['transcription_chunker'].run(split_points_df)
            logging.info("Transcription Chunker stage complete.")
            
            splitter_df = transcription_chunks_df.rename(
                columns={'chunk_start_ms': 'start_ms', 'chunk_end_ms': 'end_ms'}
            )
            
            chunks_dir = self.cache_root.parent / self.config['cache_paths']['audio_chunks']
            # --- MODIFICATION START: Create the audio_chunks directory before use ---
            chunks_dir.mkdir(exist_ok=True, parents=True)
            # --- MODIFICATION END ---
            chunk_paths = self.services['audio_splitter'].run(audio, splitter_df, chunks_dir, audio_path.stem)
            logging.info("Audio Splitter stage complete.")

            raw_scribe_results = []
            scribe_cache_dir = self.cache_root.parent / self.config['cache_paths']['scribe']
            scribe_cache_dir.mkdir(exist_ok=True, parents=True)
            for i, chunk_path in enumerate(chunk_paths):
                scribe_chunk_cache_file = scribe_cache_dir / f"{chunk_path.stem}_scribe_chunk.json"
                if self.use_cache and scribe_chunk_cache_file.exists():
                    with open(scribe_chunk_cache_file, 'r') as f: result = json.load(f)
                else:
                    result = self.services['scribe'].run(chunk_path)
                    if self.use_cache:
                        with open(scribe_chunk_cache_file, 'w') as f: json.dump(result, f, indent=2)
                raw_scribe_results.append(result)
            
            logging.info("Individual chunk transcription complete.")
            
            final_transcript = self.services['scribe_normalizer'].run(raw_scribe_results, transcription_chunks_df)
            logging.info("Normalization stage complete.")

            if self.use_cache:
                with open(final_transcript_cache_path, 'w') as f:
                    json.dump(final_transcript, f, indent=2)
                logging.info(f"✅ Final normalized transcript saved to cache: {final_transcript_cache_path}")
        
        logging.info("Scribe Transcription stage complete.")

        logging.info("Executing MFA Alignment stage...")
        mfa_cache_path = self._get_cache_path('mfa', audio_path)

        if self.use_cache and mfa_cache_path.exists():
            logging.info(f"✅ Final MFA alignment found in cache. Loading from: {mfa_cache_path}")
            with open(mfa_cache_path, 'r') as f:
                final_mfa_data = json.load(f)
        else:
            logging.info("No final MFA cache found. Running full MFA pipeline...")
            
            mfa_chunker_svc = self.services['mfa_chunker']
            mfa_chunks = mfa_chunker_svc.run(split_points_df, final_transcript, total_duration_s=total_duration_s)

            mfa_temp_dir = self.cache_root / "mfa_temp"
            if mfa_temp_dir.exists():
                shutil.rmtree(mfa_temp_dir)
            mfa_temp_dir.mkdir()
            
            audio_splitter_svc = self.services['audio_splitter']
            for chunk in mfa_chunks:
                lab_path = mfa_temp_dir / f"mfa_chunk_{chunk['id']}.lab"
                normalized_text = normalize_text_for_mfa(chunk['transcript'])
                with open(lab_path, 'w') as f:
                    f.write(normalized_text)
                
                audio_splitter_svc.split_and_save_chunk(
                    audio, 
                    chunk['start_s'] * 1000, 
                    chunk['end_s'] * 1000, 
                    mfa_temp_dir / f"mfa_chunk_{chunk['id']}.wav"
                )
            logging.info(f"Prepared {len(mfa_chunks)} .lab and .wav files for MFA.")

            mfa_aligner_svc = self.services['mfa_aligner']
            mfa_output_dir = mfa_aligner_svc.run(mfa_temp_dir, mfa_temp_dir)
            
            mfa_normalizer_svc = self.services['mfa_normalizer']
            final_mfa_data = mfa_normalizer_svc.run(mfa_output_dir, mfa_chunks)

            if self.use_cache:
                with open(mfa_cache_path, 'w') as f:
                    json.dump(final_mfa_data, f, indent=4)
                logging.info(f"✅ Final MFA aligned data saved to cache: {mfa_cache_path}")
            
            shutil.rmtree(mfa_temp_dir)
            logging.info("Cleaned up temporary MFA files.")

        logging.info("MFA Alignment stage complete.")
        
        logging.info("Executing LLM Cut Selection stage...")
        llm_cache_path = self._get_cache_path('llm', audio_path)
        
        if self.use_cache and llm_cache_path.exists():
            logging.info(f"✅ LLM result found in cache. Loading from: {llm_cache_path}")
            with open(llm_cache_path, 'r') as f:
                marked_transcript = f.read()
        else:
            logging.info("No LLM cache found. Running LLMCutSelectorService...")
            llm_service = self.services['llm_cut_selector']
            transcript_text = final_transcript.get('text', '')
            marked_transcript = llm_service.run(transcript_text)
            
            if self.use_cache:
                with open(llm_cache_path, 'w') as f:
                    f.write(marked_transcript)
                logging.info(f"✅ LLM marked transcript saved to cache: {llm_cache_path}")
        
        logging.info("LLM Cut Selection stage complete.")
        
        return final_mfa_data, marked_transcript
    