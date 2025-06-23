# src/services/mfa_normalizer_service.py
from pathlib import Path
import textgrid
import logging
from typing import List, Dict, Any

from src.utils.mfa_text_normalizer import normalize_text_for_mfa

class MfaNormalizerService:
    """
    Parses TextGrid files from an MFA run and normalizes the results into a
    single, structured JSON object.
    """
    def __init__(self):
        logging.info("MfaNormalizerService initialized.")

    def _parse_textgrid(self, tg_path: Path, offset_s: float, original_words: List[Dict], is_reliable: bool) -> List[Dict[str, Any]]:
        """Parses a single TextGrid file and maps aligned words to original scribe words."""
        aligned_words = []
        try:
            tg = textgrid.TextGrid.fromFile(str(tg_path))
            word_tier = tg.getFirst('words')
            phone_tier = tg.getFirst('phones')

            if not word_tier:
                logging.error(f"'words' tier not found in {tg_path.name}")
                return []
            if not phone_tier:
                logging.warning(f"'phones' tier not found in {tg_path.name}. Phonemes will be empty.")

            # Filter original_words to only include those that should be in the TextGrid
            mfa_input_words = [w for w in original_words if w.get('type') == 'word' and w.get('text') != '...']
            
            original_word_idx = 0
            for interval in word_tier:
                if not interval.mark or interval.mark.lower() in ['sp', 'spn', 'sil']:
                    continue
                
                if original_word_idx >= len(mfa_input_words):
                    logging.warning("MFA produced more words than in original transcript, skipping extra.")
                    break
                
                original_word = mfa_input_words[original_word_idx]
                
                # --- MODIFICATION START: Add the reliability flag ---
                word_data = {
                    "id": original_word['id'],
                    "word": original_word['text'],
                    "start": round(interval.minTime + offset_s, 4),
                    "end": round(interval.maxTime + offset_s, 4),
                    "is_timestamp_reliable": is_reliable, # <-- Add the flag here
                    "phonemes": []
                }
                # --- MODIFICATION END ---

                if phone_tier:
                    for p_interval in phone_tier:
                        if p_interval.minTime >= interval.minTime and p_interval.maxTime <= interval.maxTime:
                            if p_interval.mark and p_interval.mark.strip():
                                word_data["phonemes"].append({
                                    "text": p_interval.mark,
                                    "start": round(p_interval.minTime + offset_s, 4),
                                    "end": round(p_interval.maxTime + offset_s, 4)
                                })
                
                aligned_words.append(word_data)
                original_word_idx += 1

            return aligned_words
        except Exception as e:
            logging.error(f"Could not parse TextGrid file: {tg_path}", exc_info=True)
            return []

    def run(self, mfa_output_dir: Path, mfa_chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Processes all TextGrid files from an MFA output directory."""
        logging.info("Normalizing MFA TextGrid results...")
        all_words = []
        
        chunk_map = {chunk['id']: chunk for chunk in mfa_chunks}
        textgrid_files = sorted(mfa_output_dir.glob("*.TextGrid"), key=lambda p: int(p.stem.split('_')[-1]))

        for tg_file in textgrid_files:
            try:
                chunk_id = int(tg_file.stem.split('_')[-1])
            except (ValueError, IndexError):
                continue
                
            chunk_info = chunk_map.get(chunk_id)
            if not chunk_info:
                continue

            offset = chunk_info['start_s']
            original_words = chunk_info['scribe_words']
            # --- MODIFICATION START: Pass reliability info to the parser ---
            is_reliable = not chunk_info.get('contains_audio_event', True) # Default to unreliable if key is missing
            words_from_grid = self._parse_textgrid(tg_file, offset, original_words, is_reliable)
            # --- MODIFICATION END ---
            all_words.extend(words_from_grid)
            
        logging.info(f"Successfully normalized {len(all_words)} words from MFA output.")
        return all_words

# # src/services/mfa_normalizer_service.py
# from pathlib import Path
# import textgrid
# import logging
# from typing import List, Dict, Any

# from src.utils.mfa_text_normalizer import normalize_text_for_mfa

# class MfaNormalizerService:
#     """
#     Parses TextGrid files from an MFA run and normalizes the results into a
#     single, structured JSON object.
#     """
#     def __init__(self):
#         print("MfaNormalizerService initialized.")

#     def _parse_textgrid(self, tg_path: Path, offset_s: float, original_words: List[Dict]) -> List[Dict[str, Any]]:
#         """Parses a single TextGrid file and maps aligned words to original scribe words."""
#         aligned_words = []
#         try:
#             tg = textgrid.TextGrid.fromFile(str(tg_path))
            
#             # --- MODIFICATION: Find tiers by expected name ---
#             word_tier = tg.getFirst('words')
#             phone_tier = tg.getFirst('phones')

#             if not word_tier:
#                 logging.error(f"'words' tier not found in {tg_path.name}")
#                 return []
#             if not phone_tier:
#                 logging.warning(f"'phones' tier not found in {tg_path.name}. Phonemes will be empty.")

#             original_word_idx = 0
#             for interval in word_tier:
#                 # Skip silences or empty intervals from MFA
#                 if not interval.mark or interval.mark.lower() in ['sp', 'spn', 'sil']:
#                     continue
                
#                 if original_word_idx >= len(original_words):
#                     logging.warning("MFA produced more words than in original transcript, skipping extra.")
#                     break
                
#                 original_word = original_words[original_word_idx]
                
#                 # --- MODIFICATION: Preserve original Scribe ID and text ---
#                 word_data = {
#                     "id": original_word['id'],
#                     "word": original_word['text'], # Use original text with punctuation
#                     "start": round(interval.minTime + offset_s, 4),
#                     "end": round(interval.maxTime + offset_s, 4),
#                     "phonemes": []
#                 }

#                 if phone_tier:
#                     for p_interval in phone_tier:
#                         if p_interval.minTime >= interval.minTime and p_interval.maxTime <= interval.maxTime:
#                             if p_interval.mark and p_interval.mark.strip():
#                                 word_data["phonemes"].append({
#                                     "text": p_interval.mark,
#                                     "start": round(p_interval.minTime + offset_s, 4),
#                                     "end": round(p_interval.maxTime + offset_s, 4)
#                                 })
                
#                 aligned_words.append(word_data)
#                 original_word_idx += 1

#             return aligned_words
#         except Exception as e:
#             logging.error(f"Could not parse TextGrid file: {tg_path}", exc_info=True)
#             return []

#     def run(self, mfa_output_dir: Path, mfa_chunks: List[Dict]) -> List[Dict[str, Any]]:
#         """Processes all TextGrid files from an MFA output directory."""
#         logging.info("Normalizing MFA TextGrid results...")
#         all_words = []
        
#         chunk_map = {chunk['id']: chunk for chunk in mfa_chunks}

#         textgrid_files = sorted(mfa_output_dir.glob("*.TextGrid"), key=lambda p: int(p.stem.split('_')[-1]))

#         for tg_file in textgrid_files:
#             try:
#                 chunk_id = int(tg_file.stem.split('_')[-1])
#             except (ValueError, IndexError):
#                 continue
                
#             chunk_info = chunk_map.get(chunk_id)
#             if not chunk_info:
#                 continue

#             offset = chunk_info['start_s']
#             original_words = chunk_info['scribe_words']
            
#             words_from_grid = self._parse_textgrid(tg_file, offset, original_words)
#             all_words.extend(words_from_grid)
            
#         # No longer need to sort or re-index, as IDs are preserved.
#         logging.info(f"Successfully normalized {len(all_words)} words from MFA output.")
#         return all_words
    