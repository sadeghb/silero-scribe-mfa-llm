# src/pipeline_orchestrator.py
from pathlib import Path
import pandas as pd
import json
from pydub import AudioSegment
from typing import Dict, Any
import shutil
import logging
import librosa
import numpy as np

from src.utils.mfa_text_normalizer import normalize_text_for_mfa

class PipelineOrchestrator:
    # __init__ and _get_cache_path are unchanged...
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
        # --- Stages 1-5 (Unchanged) ---
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
            with open(final_transcript_cache_path, 'r') as f:
                final_transcript = json.load(f)
        else:
            transcription_chunks_df = self.services['transcription_chunker'].run(split_points_df)
            splitter_df = transcription_chunks_df.rename(columns={'chunk_start_ms': 'start_ms', 'chunk_end_ms': 'end_ms'})
            chunks_dir = self.cache_root.parent / self.config['cache_paths']['audio_chunks']
            chunks_dir.mkdir(exist_ok=True, parents=True)
            chunk_paths = self.services['audio_splitter'].run(audio, splitter_df, chunks_dir, audio_path.stem)
            raw_scribe_results = []
            scribe_cache_dir = self.cache_root.parent / self.config['cache_paths']['scribe']
            scribe_cache_dir.mkdir(exist_ok=True, parents=True)
            for i, chunk_path in enumerate(chunk_paths):
                scribe_chunk_cache_file = scribe_cache_dir / f"{chunk_path.stem}.json"
                if self.use_cache and scribe_chunk_cache_file.exists():
                    with open(scribe_chunk_cache_file, 'r') as f: result = json.load(f)
                else:
                    result = self.services['scribe'].run(chunk_path)
                    if self.use_cache:
                        with open(scribe_chunk_cache_file, 'w') as f: json.dump(result, f, indent=2)
                raw_scribe_results.append(result)
            final_transcript = self.services['scribe_normalizer'].run(raw_scribe_results, transcription_chunks_df)
            if self.use_cache:
                with open(final_transcript_cache_path, 'w') as f: json.dump(final_transcript, f, indent=2)
        logging.info("Scribe Transcription stage complete.")

        logging.info("Executing MFA Alignment stage...")
        mfa_cache_path = self._get_cache_path('mfa', audio_path)
        if self.use_cache and mfa_cache_path.exists():
            with open(mfa_cache_path, 'r') as f:
                final_mfa_data = json.load(f)
        else:
            mfa_chunker_svc = self.services['mfa_chunker']
            mfa_chunks = mfa_chunker_svc.run(split_points_df, final_transcript, total_duration_s=total_duration_s)
            mfa_temp_dir = self.cache_root / "mfa_temp"
            if mfa_temp_dir.exists(): shutil.rmtree(mfa_temp_dir)
            mfa_temp_dir.mkdir()
            audio_splitter_svc = self.services['audio_splitter']
            for chunk in mfa_chunks:
                lab_path = mfa_temp_dir / f"mfa_chunk_{chunk['id']}.lab"
                normalized_text = normalize_text_for_mfa(chunk['transcript'])
                with open(lab_path, 'w') as f: f.write(normalized_text)
                audio_splitter_svc.split_and_save_chunk(audio, chunk['start_s'] * 1000, chunk['end_s'] * 1000, mfa_temp_dir / f"mfa_chunk_{chunk['id']}.wav")
            mfa_aligner_svc = self.services['mfa_aligner']
            mfa_output_dir = mfa_aligner_svc.run(mfa_temp_dir, mfa_temp_dir)
            mfa_normalizer_svc = self.services['mfa_normalizer']
            final_mfa_data = mfa_normalizer_svc.run(mfa_output_dir, mfa_chunks)
            if self.use_cache:
                with open(mfa_cache_path, 'w') as f: json.dump(final_mfa_data, f, indent=4)
            shutil.rmtree(mfa_temp_dir)
        logging.info("MFA Alignment stage complete.")
        
        logging.info("Executing LLM Cut Selection stage...")
        llm_cache_path = self._get_cache_path('llm', audio_path)
        if self.use_cache and llm_cache_path.exists():
            with open(llm_cache_path, 'r') as f:
                marked_transcript = f.read()
        else:
            llm_service = self.services['llm_cut_selector']
            transcript_text = final_transcript.get('text', '')
            marked_transcript = llm_service.run(transcript_text)
            if self.use_cache:
                with open(llm_cache_path, 'w') as f: f.write(marked_transcript)
        logging.info("LLM Cut Selection stage complete.")
        
        logging.info("Executing Final Editing and Dataset Generation stage...")

        y_full, sr = librosa.load(str(audio_path), sr=None, mono=True)

        cut_parser_svc = self.services['cut_parser']
        audio_editor_svc = self.services['audio_editor']
        dataset_generator_svc = self.services['dataset_generator']

        cut_segments = cut_parser_svc.run(final_transcript['words'], marked_transcript)

        if not cut_segments:
            logging.info("No cut segments identified by the parser. Skipping editing.")
            return

        for i, cut_word_ids in enumerate(cut_segments):
            last_word_id_in_transcript = len(final_mfa_data) - 1
            if not cut_word_ids or cut_word_ids[0] == 0 or cut_word_ids[-1] == last_word_id_in_transcript:
                logging.warning(f"Skipping cut segment {cut_word_ids} because it involves a boundary word.")
                continue

            logging.info(f"Processing cut {i+1}/{len(cut_segments)} (word IDs: {cut_word_ids})...")
            
            edit_results = audio_editor_svc.run(
                cut_word_ids=cut_word_ids,
                full_audio=audio,
                y_full=y_full,
                sr=sr,
                mfa_data=final_mfa_data,
                scribe_data=final_transcript,
                split_points_df=split_points_df
            )
            
            if edit_results is None:
                continue
            
            chunk_start_s = edit_results["metadata"]["chunk_start_s_abs"]
            chunk_end_s = edit_results["metadata"]["chunk_end_s_abs"]
            
            chunk_words = [
                word for word in final_transcript['words'] 
                if word['start'] >= chunk_start_s and word['end'] <= chunk_end_s
            ]

            # --- MODIFICATION START: Determine if the datapoint is usable ---
            is_usable = True
            for word in chunk_words:
                # Find the corresponding word in MFA data to check its reliability
                mfa_word = next((w for w in final_mfa_data if w['id'] == word['id']), None)
                if mfa_word and not mfa_word.get('is_timestamp_reliable', True):
                    is_usable = False
                    logging.warning(f"Cut {i+1} marked as unusable due to unreliable timestamp in word: {word}")
                    break
            # --- MODIFICATION END ---
            
            dataset_generator_svc.run(
                source_audio_name=audio_path.stem,
                cut_id=i + 1,
                cut_word_ids=cut_word_ids,
                chunk_words=chunk_words,
                edit_results=edit_results,
                is_usable=is_usable # Pass the flag to the generator
            )
        
        logging.info("Final Editing and Dataset Generation stage complete.")

# # src/pipeline_orchestrator.py
# from pathlib import Path
# import pandas as pd
# import json
# from pydub import AudioSegment
# from typing import Dict, Any
# import shutil
# import logging
# import librosa
# import numpy as np

# from src.utils.mfa_text_normalizer import normalize_text_for_mfa

# class PipelineOrchestrator:
#     # __init__ and _get_cache_path are unchanged...
#     def __init__(self, services: Dict, config: Dict[str, Any]):
#         logging.info("PipelineOrchestrator initialized.")
#         self.services = services
#         self.config = config
#         self.use_cache = self.config.get('use_cache', False)
#         base_dir = Path(__file__).parent.parent
#         self.cache_root = base_dir / 'cache'
#         self.cache_root.mkdir(exist_ok=True)

#     def _get_cache_path(self, stage_name: str, source_path: Path, suffix_override: str = None) -> Path:
#         try:
#             stage_cache_dir_str = self.config['cache_paths'][stage_name]
#         except KeyError:
#             raise KeyError(f"Error: The key '{stage_name}' was not found in the 'cache_paths' section of your config.yaml.")
        
#         stage_cache_dir = self.cache_root.parent / stage_cache_dir_str
#         stage_cache_dir.mkdir(exist_ok=True, parents=True)

#         if suffix_override:
#             return stage_cache_dir / (source_path.stem + suffix_override)

#         suffix_key_map = {
#             'vad': 'vad_timestamps_suffix',
#             'split_points': 'split_points_suffix',
#             'scribe': 'scribe_timestamps_suffix',
#             'llm': 'llm_marked_transcript_suffix',
#             'mfa': 'mfa_aligned_suffix'
#         }
#         if stage_name not in suffix_key_map:
#             raise ValueError(f"Unknown suffix key for stage: {stage_name}")
        
#         suffix_key = suffix_key_map[stage_name]
        
#         try:
#             suffix = self.config['output_files'][suffix_key]
#         except KeyError:
#             raise KeyError(f"Error: The key '{suffix_key}' was not found in the 'output_files' section of your config.yaml.")
        
#         return stage_cache_dir / (source_path.stem + suffix)


#     def run(self, audio_path: Path):
#         # --- Stages 1-5 (Unchanged) ---
#         logging.info(f"\n--- Starting pipeline for: {audio_path.name} ---")
#         audio = AudioSegment.from_file(audio_path)
#         total_duration_s = len(audio) / 1000.0
        
#         logging.info("Executing VAD stage...")
#         vad_cache_path = self._get_cache_path('vad', audio_path)
#         if self.use_cache and vad_cache_path.exists():
#             vad_df = pd.read_csv(vad_cache_path)
#         else:
#             vad_df = self.services['vad'].run(audio_path)
#             if self.use_cache: vad_df.to_csv(vad_cache_path, index=False)
#         logging.info("VAD stage complete.")

#         if vad_df.empty: return None

#         logging.info("Executing Split Point Generation stage...")
#         split_points_cache_path = self._get_cache_path('split_points', audio_path)
#         if self.use_cache and split_points_cache_path.exists():
#             split_points_df = pd.read_csv(split_points_cache_path)
#         else:
#             split_points_df = self.services['split_point'].run(vad_df, len(audio))
#             if self.use_cache: split_points_df.to_csv(split_points_cache_path, index=False)
#         logging.info("Split Point Generation complete.")

#         logging.info("Executing Scribe Transcription stage...")
#         final_transcript_cache_path = self._get_cache_path('scribe', audio_path)
#         if self.use_cache and final_transcript_cache_path.exists():
#             with open(final_transcript_cache_path, 'r') as f:
#                 final_transcript = json.load(f)
#         else:
#             transcription_chunks_df = self.services['transcription_chunker'].run(split_points_df)
#             splitter_df = transcription_chunks_df.rename(columns={'chunk_start_ms': 'start_ms', 'chunk_end_ms': 'end_ms'})
#             chunks_dir = self.cache_root.parent / self.config['cache_paths']['audio_chunks']
#             chunks_dir.mkdir(exist_ok=True, parents=True)
#             chunk_paths = self.services['audio_splitter'].run(audio, splitter_df, chunks_dir, audio_path.stem)
#             raw_scribe_results = []
#             scribe_cache_dir = self.cache_root.parent / self.config['cache_paths']['scribe']
#             for i, chunk_path in enumerate(chunk_paths):
#                 scribe_chunk_cache_file = scribe_cache_dir / f"{chunk_path.stem}_scribe_chunk.json"
#                 if self.use_cache and scribe_chunk_cache_file.exists():
#                     with open(scribe_chunk_cache_file, 'r') as f: result = json.load(f)
#                 else:
#                     result = self.services['scribe'].run(chunk_path)
#                     if self.use_cache:
#                         with open(scribe_chunk_cache_file, 'w') as f: json.dump(result, f, indent=2)
#                 raw_scribe_results.append(result)
#             final_transcript = self.services['scribe_normalizer'].run(raw_scribe_results, transcription_chunks_df)
#             if self.use_cache:
#                 with open(final_transcript_cache_path, 'w') as f: json.dump(final_transcript, f, indent=2)
#         logging.info("Scribe Transcription stage complete.")

#         logging.info("Executing MFA Alignment stage...")
#         mfa_cache_path = self._get_cache_path('mfa', audio_path)
#         if self.use_cache and mfa_cache_path.exists():
#             with open(mfa_cache_path, 'r') as f:
#                 final_mfa_data = json.load(f)
#         else:
#             mfa_chunker_svc = self.services['mfa_chunker']
#             mfa_chunks = mfa_chunker_svc.run(split_points_df, final_transcript, total_duration_s=total_duration_s)
#             mfa_temp_dir = self.cache_root / "mfa_temp"
#             if mfa_temp_dir.exists(): shutil.rmtree(mfa_temp_dir)
#             mfa_temp_dir.mkdir()
#             audio_splitter_svc = self.services['audio_splitter']
#             for chunk in mfa_chunks:
#                 lab_path = mfa_temp_dir / f"mfa_chunk_{chunk['id']}.lab"
#                 normalized_text = normalize_text_for_mfa(chunk['transcript'])
#                 with open(lab_path, 'w') as f: f.write(normalized_text)
#                 audio_splitter_svc.split_and_save_chunk(audio, chunk['start_s'] * 1000, chunk['end_s'] * 1000, mfa_temp_dir / f"mfa_chunk_{chunk['id']}.wav")
#             mfa_aligner_svc = self.services['mfa_aligner']
#             mfa_output_dir = mfa_aligner_svc.run(mfa_temp_dir, mfa_temp_dir)
#             mfa_normalizer_svc = self.services['mfa_normalizer']
#             final_mfa_data = mfa_normalizer_svc.run(mfa_output_dir, mfa_chunks)
#             if self.use_cache:
#                 with open(mfa_cache_path, 'w') as f: json.dump(final_mfa_data, f, indent=4)
#             shutil.rmtree(mfa_temp_dir)
#         logging.info("MFA Alignment stage complete.")
        
#         logging.info("Executing LLM Cut Selection stage...")
#         llm_cache_path = self._get_cache_path('llm', audio_path)
#         if self.use_cache and llm_cache_path.exists():
#             with open(llm_cache_path, 'r') as f:
#                 marked_transcript = f.read()
#         else:
#             llm_service = self.services['llm_cut_selector']
#             transcript_text = final_transcript.get('text', '')
#             marked_transcript = llm_service.run(transcript_text)
#             if self.use_cache:
#                 with open(llm_cache_path, 'w') as f: f.write(marked_transcript)
#         logging.info("LLM Cut Selection stage complete.")
        
#         # --- Stage 6: Cut Parsing, Editing, and Dataset Generation ---
#         logging.info("Executing Final Editing and Dataset Generation stage...")

#         y_full, sr = librosa.load(str(audio_path), sr=None, mono=True)

#         cut_parser_svc = self.services['cut_parser']
#         audio_editor_svc = self.services['audio_editor']
#         dataset_generator_svc = self.services['dataset_generator']

#         # cut_segments = cut_parser_svc.run(final_mfa_data, marked_transcript)
#         # --- MODIFICATION START: Using Scribe data for parsing ---
#         cut_segments = cut_parser_svc.run(final_transcript['words'], marked_transcript)
#         # --- MODIFICATION END ---

#         if not cut_segments:
#             logging.info("No cut segments identified by the parser. Skipping editing.")
#             return

#         for i, cut_word_ids in enumerate(cut_segments):
#             # --- MODIFICATION START: Skip cuts at the beginning or end of the audio ---
#             last_word_id_in_transcript = len(final_mfa_data) - 1
#             if not cut_word_ids or cut_word_ids[0] == 0 or cut_word_ids[-1] == last_word_id_in_transcript:
#                 logging.warning(f"Skipping cut segment {cut_word_ids} because it involves a boundary word.")
#                 continue
#             # --- MODIFICATION END ---
#             logging.info(f"Processing cut {i+1}/{len(cut_segments)} (word IDs: {cut_word_ids})...")
            
#             edit_results = audio_editor_svc.run(
#                 cut_word_ids=cut_word_ids,
#                 full_audio=audio,
#                 y_full=y_full,
#                 sr=sr,
#                 mfa_data=final_mfa_data,
#                 scribe_data=final_transcript,
#                 split_points_df=split_points_df
#             )
            
#             # --- MODIFICATION START: Filter words for the current chunk ---
#             chunk_start_s = edit_results["metadata"]["chunk_start_s_abs"]
#             chunk_end_s = edit_results["metadata"]["chunk_end_s_abs"]

#             # --- MODIFICATION START: Implement "majority overlap" rule ---
#             chunk_words = []
#             for word in final_transcript['words']:
#                 word_start = word.get('start', 0)
#                 word_end = word.get('end', 0)
#                 word_duration = word_end - word_start
#                 if word_duration <= 0:
#                     continue

#                 overlap_start = max(word_start, chunk_start_s)
#                 overlap_end = min(word_end, chunk_end_s)
                
#                 # Ensure overlap is positive
#                 overlap_duration = max(0, overlap_end - overlap_start)

#                 if overlap_duration > (0.5 * word_duration):
#                     chunk_words.append(word)
#             # --- MODIFICATION END ---

#             # chunk_words = [
#             #     word for word in final_mfa_data 
#             #     if word['start'] >= chunk_start_s and word['end'] <= chunk_end_s
#             # ]
#             # --- MODIFICATION END ---
            
#             dataset_generator_svc.run(
#                 source_audio_name=audio_path.stem,
#                 cut_id=i + 1,
#                 cut_word_ids=cut_word_ids,
#                 chunk_words=chunk_words, # Pass the filtered list of words
#                 edit_results=edit_results
#             )
        
#         logging.info("Final Editing and Dataset Generation stage complete.")
        