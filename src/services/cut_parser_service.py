# src/services/cut_parser_service.py
import logging
from typing import List, Dict, Any

class CutParserService:
    """
    A service to parse a transcript marked with <cut> tags. This version correctly
    syncs against the full Scribe output, including non-word events.
    """
    def __init__(self):
        logging.info("CutParserService initialized.")

    def _normalize_word(self, word: str) -> str:
        """Helper function to lowercase and remove surrounding punctuation."""
        return word.lower().strip(".,;:?!'\"` ")

    def run(self, original_words: List[Dict[str, Any]], marked_transcript: str) -> List[List[int]]:
        """
        Parses the marked transcript to find segments of word IDs to cut.
        """
        logging.info("Parsing marked transcript against full Scribe data.")
        
        processed_text = marked_transcript.replace('<cut>', ' <cut> ').replace('</cut>', ' </cut> ')
        marked_tokens = processed_text.split()

        all_segments = []
        current_segment = []
        
        token_idx = 0
        word_idx = 0
        is_inside_cut_segment = False

        while token_idx < len(marked_tokens) and word_idx < len(original_words):
            token = marked_tokens[token_idx]

            if token == '<cut>':
                is_inside_cut_segment = True
                current_segment = []
                token_idx += 1
                continue

            if token == '</cut>':
                if is_inside_cut_segment:
                    is_inside_cut_segment = False
                    if current_segment:
                        all_segments.append(current_segment)
                    current_segment = []
                token_idx += 1
                continue
            
            normalized_token = self._normalize_word(token)
            if not normalized_token:
                token_idx += 1
                continue

            original_word_obj = original_words[word_idx]

            # --- MODIFICATION START: Correctly handle Scribe event types ---
            # If the Scribe object is 'spacing', it has no token. Skip it.
            if original_word_obj.get('type') == 'spacing':
                word_idx += 1
                continue
            
            # Use the 'text' key, not 'word'
            original_text = original_word_obj.get('text', '')
            normalized_original_word = self._normalize_word(original_text)
            # --- MODIFICATION END ---

            if normalized_token == normalized_original_word:
                if is_inside_cut_segment:
                    # We only want to cut actual words, so we check the type here.
                    if original_word_obj.get('type') == 'word':
                        current_segment.append(original_word_obj['id'])
                token_idx += 1
                word_idx += 1
            else:
                # Mismatch recovery logic
                logging.warning(
                    f"Sync warning at word index {word_idx}: LLM token '{normalized_token}' != Original item '{normalized_original_word}'. Attempting to re-sync..."
                )
                
                found_resync = False
                lookahead_limit = 5 
                for i in range(1, lookahead_limit + 1):
                    if (token_idx + i) < len(marked_tokens):
                        lookahead_token = self._normalize_word(marked_tokens[token_idx + i])
                        if lookahead_token == normalized_original_word:
                            logging.warning(f"Re-synced by skipping {i} LLM token(s): '{' '.join(marked_tokens[token_idx:token_idx+i])}'")
                            token_idx += i
                            found_resync = True
                            break
                
                if found_resync:
                    continue

                logging.warning(f"Could not re-sync. Assuming original item '{normalized_original_word}' was omitted by LLM. Skipping it.")
                word_idx += 1

        if is_inside_cut_segment and current_segment:
            all_segments.append(current_segment)

        logging.info(f"Identified {len(all_segments)} cut segments to process.")
        return all_segments
    