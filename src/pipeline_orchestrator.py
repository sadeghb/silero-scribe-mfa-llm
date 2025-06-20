# src/pipeline_orchestrator.py
from pathlib import Path
import pandas as pd
import json
from pydub import AudioSegment
from typing import Dict, Any

class PipelineOrchestrator:
    # __init__ and _get_cache_path are unchanged...
    def __init__(self, services: Dict, config: Dict[str, Any]):
        print("PipelineOrchestrator initialized.")
        self.services = services
        self.config = config
        self.use_cache = self.config.get('use_cache', False)
        base_dir = Path(__file__).parent.parent
        self.cache_root = base_dir / 'cache'
        self.cache_root.mkdir(exist_ok=True)

    def _get_cache_path(self, stage_name: str, source_path: Path) -> Path:
        try:
            stage_cache_dir_str = self.config['cache_paths'][stage_name]
        except KeyError:
            raise KeyError(f"Error: The key '{stage_name}' was not found in the 'cache_paths' section of your config.yaml.")
        stage_cache_dir = self.cache_root.parent / stage_cache_dir_str
        stage_cache_dir.mkdir(exist_ok=True, parents=True)
        suffix_key_map = {
            'vad': 'vad_timestamps_suffix',
            'split_points': 'split_points_suffix',
            'scribe': 'scribe_timestamps_suffix',
            'llm': 'llm_marked_transcript_suffix'
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
        # All preceding stages remain the same...
        print(f"\n--- Starting pipeline for: {audio_path.name} ---")
        
        print("Executing VAD stage...")
        vad_cache_path = self._get_cache_path('vad', audio_path)
        if self.use_cache and vad_cache_path.exists():
            vad_df = pd.read_csv(vad_cache_path)
        else:
            vad_df = self.services['vad'].run(audio_path)
            if self.use_cache: vad_df.to_csv(vad_cache_path, index=False)
        print("VAD stage complete.")

        if vad_df.empty: return None

        print("Executing Split Point Generation stage...")
        split_points_cache_path = self._get_cache_path('split_points', audio_path)
        if self.use_cache and split_points_cache_path.exists():
            split_points_df = pd.read_csv(split_points_cache_path)
        else:
            audio = AudioSegment.from_file(audio_path)
            total_duration_ms = len(audio)
            split_points_df = self.services['split_point'].run(vad_df, total_duration_ms)
            if self.use_cache: split_points_df.to_csv(split_points_cache_path, index=False)
        print("Split Point Generation complete.")

        transcription_chunks_df = self.services['transcription_chunker'].run(split_points_df)
        print("Transcription Chunker stage complete.")

        chunks_dir = self.cache_root.parent / self.config['cache_paths']['audio_chunks']
        audio = AudioSegment.from_file(audio_path)
        chunk_paths = self.services['audio_splitter'].run(audio, transcription_chunks_df, chunks_dir, audio_path.stem)
        print("Audio Splitter stage complete.")

        raw_scribe_results = []
        scribe_cache_dir = self.cache_root.parent / self.config['cache_paths']['scribe']
        scribe_cache_dir.mkdir(exist_ok=True, parents=True)
        for chunk_path in chunk_paths:
            scribe_result_cache_file = scribe_cache_dir / f"{chunk_path.stem}.json"
            if self.use_cache and scribe_result_cache_file.exists():
                with open(scribe_result_cache_file, 'r') as f: result = json.load(f)
            else:
                result = self.services['scribe'].run(chunk_path)
                if self.use_cache:
                    with open(scribe_result_cache_file, 'w') as f: json.dump(result, f, indent=2)
            raw_scribe_results.append(result)
        print("Scribe Transcription stage complete.")

        final_transcript = self.services['scribe_normalizer'].run(raw_scribe_results, transcription_chunks_df)
        print("Normalization stage complete.")

        # --- LLM Cut Selection Stage ---
        print("Executing LLM Cut Selection stage...")
        llm_cache_path = self._get_cache_path('llm', audio_path)
        
        if self.use_cache and llm_cache_path.exists():
            print(f"✅ LLM result found in cache. Loading from: {llm_cache_path}")
            with open(llm_cache_path, 'r') as f:
                marked_transcript = f.read()
        else:
            print("No LLM cache found. Running LLMCutSelectorService...")
            llm_service = self.services['llm_cut_selector']
            
            # --- THE FIX ---
            # Correctly access the transcript using the 'text' key instead of 'full_text'.
            transcript_text = final_transcript.get('text', '')
            
            marked_transcript = llm_service.run(transcript_text)
            
            if self.use_cache:
                with open(llm_cache_path, 'w') as f:
                    f.write(marked_transcript)
                print(f"✅ LLM marked transcript saved to cache: {llm_cache_path}")
        
        print("LLM Cut Selection stage complete.")
        print(f"\n*** Final Marked Transcript: ***\n{marked_transcript}\n")

        return marked_transcript
    