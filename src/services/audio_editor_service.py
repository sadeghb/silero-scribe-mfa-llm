import logging
from typing import List, Dict, Any, Tuple
from pydub import AudioSegment
import numpy as np
import random

class AudioEditorService:
    """
    Handles the logic for creating edited audio clips based on cut events.
    This version includes the final tweaks for the is_usable flag and metadata text.
    """
    def __init__(self, config: Dict[str, Any]):
        logging.info("AudioEditorService initialized.")
        self.config = config.get('editing', {})
        self.backward_invasion_interval = self.config.get('backward_phoneme_invasion_interval', [0.7, 0.9])
        self.forward_invasion_interval = self.config.get('forward_phoneme_invasion_interval', [0.7, 0.9])
        self.context_duration_ms = self.config.get('context_duration_ms', 3000)

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

    def run(self,
            cut_word_ids: List[int],
            full_audio: AudioSegment,
            y_full: np.ndarray,
            sr: int,
            mfa_data: List[Dict],
            **kwargs # Absorb unused arguments
           ) -> Dict:
        """
        Generates three distinct audio clips and applies final logic for
        the is_usable flag and metadata text.
        """
        word_id_map = {word['id']: word for word in mfa_data}

        try:
            first_word_to_cut = word_id_map[cut_word_ids[0]]
            last_word_to_cut = word_id_map[cut_word_ids[-1]]
        except KeyError as e:
            logging.error(f"FATAL: Word with ID {e} was not found in the MFA data. Skipping this cut.")
            return None

        # --- Tweak #1: Modified is_usable flag logic ---
        words_to_check_for_reliability = []
        first_word_index = next((i for i, item in enumerate(mfa_data) if item["id"] == cut_word_ids[0]), -1)
        last_word_index = next((i for i, item in enumerate(mfa_data) if item["id"] == cut_word_ids[-1]), -1)

        words_to_check_for_reliability.append(first_word_to_cut)
        words_to_check_for_reliability.append(last_word_to_cut)

        if first_word_index > 0:
            words_to_check_for_reliability.append(mfa_data[first_word_index - 1])
        if last_word_index != -1 and last_word_index < len(mfa_data) - 1:
            words_to_check_for_reliability.append(mfa_data[last_word_index + 1])
        
        is_usable = all(w.get('is_reliable_timestamp', True) for w in words_to_check_for_reliability)

        # --- 1. Natural Cut ---
        nat_start_s, nat_end_s = self._get_cut_boundaries(cut_word_ids, word_id_map, mfa_data, 0.0, 0.0)
        nat_splice_before_s = self._find_outward_zero_crossing(y_full, int(nat_start_s * sr), 'backward') / sr
        nat_splice_after_s = self._find_outward_zero_crossing(y_full, int(nat_end_s * sr), 'forward') / sr
        
        nat_before_start_ms = max(0, int(nat_splice_before_s * 1000) - self.context_duration_ms)
        nat_before_segment = full_audio[nat_before_start_ms : int(nat_splice_before_s * 1000)]
        
        nat_after_end_ms = min(len(full_audio), int(nat_splice_after_s * 1000) + self.context_duration_ms)
        nat_after_segment = full_audio[int(nat_splice_after_s * 1000) : nat_after_end_ms]
        
        natural_cut_audio = nat_before_segment + nat_after_segment

        # --- 2. Unnatural Backward Invasion Cut ---
        random_bwd_factor = random.uniform(self.backward_invasion_interval[0], self.backward_invasion_interval[1])
        bwd_start_s, bwd_end_s = self._get_cut_boundaries(cut_word_ids, word_id_map, mfa_data, random_bwd_factor, 0.0)
        bwd_splice_before_s = self._find_outward_zero_crossing(y_full, int(bwd_start_s * sr), 'backward') / sr
        bwd_splice_after_s = self._find_outward_zero_crossing(y_full, int(bwd_end_s * sr), 'forward') / sr
        
        bwd_before_start_ms = max(0, int(bwd_splice_before_s * 1000) - self.context_duration_ms)
        bwd_before_segment = full_audio[bwd_before_start_ms : int(bwd_splice_before_s * 1000)]
        
        bwd_after_end_ms = min(len(full_audio), int(bwd_splice_after_s * 1000) + self.context_duration_ms)
        bwd_after_segment = full_audio[int(bwd_splice_after_s * 1000) : bwd_after_end_ms]
        
        unnatural_backward_audio = bwd_before_segment + bwd_after_segment

        # --- 3. Unnatural Forward Invasion Cut ---
        random_fwd_factor = random.uniform(self.forward_invasion_interval[0], self.forward_invasion_interval[1])
        fwd_start_s, fwd_end_s = self._get_cut_boundaries(cut_word_ids, word_id_map, mfa_data, 0.0, random_fwd_factor)
        fwd_splice_before_s = self._find_outward_zero_crossing(y_full, int(fwd_start_s * sr), 'backward') / sr
        fwd_splice_after_s = self._find_outward_zero_crossing(y_full, int(fwd_end_s * sr), 'forward') / sr

        fwd_before_start_ms = max(0, int(fwd_splice_before_s * 1000) - self.context_duration_ms)
        fwd_before_segment = full_audio[fwd_before_start_ms : int(fwd_splice_before_s * 1000)]
        
        fwd_after_end_ms = min(len(full_audio), int(fwd_splice_after_s * 1000) + self.context_duration_ms)
        fwd_after_segment = full_audio[int(fwd_splice_after_s * 1000) : fwd_after_end_ms]
        
        unnatural_forward_audio = fwd_before_segment + fwd_after_segment

        # --- Tweak #2: Prepare cut_text for metadata ---
        cut_words_list = [word_id_map[wid] for wid in cut_word_ids]
        cut_text = " ".join([w.get('word', '') for w in cut_words_list])

        return {
            "audios": {
                "natural_cut": natural_cut_audio,
                "unnatural_backward": unnatural_backward_audio,
                "unnatural_forward": unnatural_forward_audio,
            },
            "is_usable": is_usable,
            "cut_text": cut_text
        }
    