# src/services/audio_editor_service.py
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pydub import AudioSegment
import numpy as np
import librosa
import pandas as pd

class AudioEditorService:
    """
    Handles the logic for creating edited audio clips based on cut events.
    """
    def __init__(self, config: Dict[str, Any]):
        logging.info("AudioEditorService initialized.")
        self.config = config.get('editing', {})
        self.backward_invasion = self.config.get('backward_phoneme_invasion_factor', 0.8)
        self.forward_invasion = self.config.get('forward_phoneme_invasion_factor', 0.8)

    def _find_outward_zero_crossing(self, signal: np.ndarray, sample_index: int, direction: str) -> int:
        """Finds the nearest 'outward' zero-crossing from a given sample index."""
        if not (0 <= sample_index < len(signal)):
            return max(0, min(len(signal) - 1, sample_index))

        start_sign = np.sign(signal[sample_index])
        if start_sign == 0: return sample_index

        if direction == 'forward':
            for i in range(sample_index + 1, len(signal)):
                if np.sign(signal[i]) != start_sign: return i - 1
            return len(signal) - 1
        else: # backward
            for i in range(sample_index - 1, -1, -1):
                if np.sign(signal[i]) != start_sign: return i + 1
            return 0

    def _get_cut_boundaries(
        self,
        segment_word_ids: List[int],
        word_id_map: Dict[int, Dict],
        all_words_list: List[Dict],
        backward_invasion: float,
        forward_invasion: float
    ) -> Tuple[float, float]:
        """Calculates the absolute start and end time for a cut segment."""
        first_word_id = segment_word_ids[0]
        last_word_id = segment_word_ids[-1]
        
        first_word_of_segment = word_id_map[first_word_id]
        last_word_of_segment = word_id_map[last_word_id]
        
        first_word_index = next((i for i, item in enumerate(all_words_list) if item["id"] == first_word_id), -1)
        last_word_index = next((i for i, item in enumerate(all_words_list) if item["id"] == last_word_id), -1)

        if backward_invasion > 0:
            prev_word = all_words_list[first_word_index - 1] if first_word_index > 0 else None
            if prev_word and prev_word.get('phonemes'):
                last_phoneme = prev_word['phonemes'][-1]
                duration = last_phoneme['end'] - last_phoneme['start']
                start_time = last_phoneme['end'] - (duration * backward_invasion)
            else:
                start_time = first_word_of_segment['start']
        else:
            prev_word = all_words_list[first_word_index - 1] if first_word_index > 0 else None
            if prev_word:
                start_time = (prev_word['end'] + first_word_of_segment['start']) / 2
            else:
                start_time = 0.0

        if forward_invasion > 0:
            next_word = all_words_list[last_word_index + 1] if last_word_index != -1 and last_word_index < len(all_words_list) - 1 else None
            if next_word and next_word.get('phonemes'):
                first_phoneme = next_word['phonemes'][0]
                duration = first_phoneme['end'] - first_phoneme['start']
                end_time = first_phoneme['start'] + (duration * forward_invasion)
            else:
                end_time = last_word_of_segment['end']
        else:
            next_word = all_words_list[last_word_index + 1] if last_word_index != -1 and last_word_index < len(all_words_list) - 1 else None
            if next_word:
                end_time = (last_word_of_segment['end'] + next_word['start']) / 2
            else:
                end_time = last_word_of_segment['end']
                
        return start_time, end_time

    def _perform_direct_cut(self, audio: AudioSegment, start_s: float, end_s: float) -> AudioSegment:
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        if start_ms < 0 or end_ms > len(audio) or start_ms >= end_ms:
            logging.error(f"Invalid cut timestamps provided: start={start_ms}ms, end={end_ms}ms on a clip of {len(audio)}ms. Returning original clip.")
            return audio
        return audio[:start_ms] + audio[end_ms:]
        
    def _is_scribe_spacing(self, time_s: float, scribe_data: Dict[str, Any]) -> bool:
        for word in scribe_data.get('words', []):
            if word['start'] <= time_s <= word['end']:
                return word['type'] == 'spacing'
        return False

    def run(self,
            cut_word_ids: List[int],
            full_audio: AudioSegment,
            y_full: np.ndarray,
            sr: int,
            mfa_data: List[Dict],
            scribe_data: Dict,
            split_points_df: pd.DataFrame
           ) -> Dict:
        word_id_map = {word['id']: word for word in mfa_data}

        try:
            first_word_to_cut = word_id_map[cut_word_ids[0]]
            last_word_to_cut = word_id_map[cut_word_ids[-1]]
        except KeyError as e:
            logging.error(f"FATAL: Word with ID {e} specified in a cut was not found in the MFA data. Skipping this cut.")
            return None

        # --- MODIFICATION START: Final logic for chunking with context ---
        # 1. Identify the context words
        first_word_index = next((i for i, item in enumerate(mfa_data) if item["id"] == cut_word_ids[0]), -1)
        last_word_index = next((i for i, item in enumerate(mfa_data) if item["id"] == cut_word_ids[-1]), -1)

        context_word_before = mfa_data[first_word_index - 1] if first_word_index > 0 else None
        context_word_after = mfa_data[last_word_index + 1] if last_word_index != -1 and last_word_index < len(mfa_data) - 1 else None
        
        # 2. Define the time range that must be included in the chunk
        context_start_time = context_word_before['start'] if context_word_before else first_word_to_cut['start']
        context_end_time = context_word_after['end'] if context_word_after else last_word_to_cut['end']

        total_duration_s = len(full_audio) / 1000.0
        eligible_points_s = (split_points_df['split_point_ms'] / 1000.0).tolist()

        # --- MODIFICATION START: Remove Scribe 'spacing' check ---
        # Find the nearest eligible point before the context start time.
        start_candidates = [p for p in eligible_points_s if p <= context_start_time]
        chunk_start_s = max(start_candidates) if start_candidates else 0.0

        # Find the nearest eligible point after the context end time.
        end_candidates = [p for p in eligible_points_s if p >= context_end_time]
        chunk_end_s = min(end_candidates) if end_candidates else total_duration_s
        # --- MODIFICATION END ---

        # # 3. Find the nearest valid silent point that *contains* the required context
        # chunk_start_s = 0.0
        # start_candidates = sorted([p for p in eligible_points_s if p <= context_start_time], reverse=True)
        # for p in start_candidates:
        #     if self._is_scribe_spacing(p, scribe_data):
        #         chunk_start_s = p
        #         break
        
        # chunk_end_s = total_duration_s
        # end_candidates = sorted([p for p in eligible_points_s if p >= context_end_time])
        # for p in end_candidates:
        #     if self._is_scribe_spacing(p, scribe_data):
        #         chunk_end_s = p
        #         break
        # --- MODIFICATION END ---
        
        original_chunk = full_audio[int(chunk_start_s * 1000):int(chunk_end_s * 1000)]

        nat_start, nat_end = self._get_cut_boundaries(cut_word_ids, word_id_map, mfa_data, 0.0, 0.0)
        nat_start = self._find_outward_zero_crossing(y_full, int(nat_start * sr), 'backward') / sr
        nat_end = self._find_outward_zero_crossing(y_full, int(nat_end * sr), 'forward') / sr
        
        bwd_start, bwd_end = self._get_cut_boundaries(cut_word_ids, word_id_map, mfa_data, self.backward_invasion, 0.0)
        bwd_start = self._find_outward_zero_crossing(y_full, int(bwd_start * sr), 'backward') / sr
        bwd_end = self._find_outward_zero_crossing(y_full, int(bwd_end * sr), 'forward') / sr

        fwd_start, fwd_end = self._get_cut_boundaries(cut_word_ids, word_id_map, mfa_data, 0.0, self.forward_invasion)
        fwd_start = self._find_outward_zero_crossing(y_full, int(fwd_start * sr), 'backward') / sr
        fwd_end = self._find_outward_zero_crossing(y_full, int(fwd_end * sr), 'forward') / sr

        natural_cut_chunk = self._perform_direct_cut(original_chunk, nat_start - chunk_start_s, nat_end - chunk_start_s)
        backward_invasion_chunk = self._perform_direct_cut(original_chunk, bwd_start - chunk_start_s, bwd_end - chunk_start_s)
        forward_invasion_chunk = self._perform_direct_cut(original_chunk, fwd_start - chunk_start_s, fwd_end - chunk_start_s)

        return {
            "original_audio": original_chunk,
            "natural_cut_audio": natural_cut_chunk,
            "backward_invasion_audio": backward_invasion_chunk,
            "forward_invasion_audio": forward_invasion_chunk,
            "metadata": {
                "chunk_start_s_abs": chunk_start_s,
                "chunk_end_s_abs": chunk_end_s,
                "natural_cut_timestamps_relative": (nat_start - chunk_start_s, nat_end - chunk_start_s),
                "backward_invasion_timestamps_relative": (bwd_start - chunk_start_s, bwd_end - chunk_start_s),
                "forward_invasion_timestamps_relative": (fwd_start - chunk_start_s, fwd_end - chunk_start_s)
            }
        }
    