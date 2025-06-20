# src/services/mfa_chunker_service.py
import pandas as pd
from typing import Dict, List, Any

class MfaChunkerService:
    """
    Creates audio chunks for MFA based on eligible split points and Scribe's
    silence detection.
    """
    def __init__(self):
        print("MfaChunkerService initialized.")

    def _find_word_at_time(self, scribe_data: Dict[str, Any], time_s: float) -> Dict[str, Any]:
        """Finds the word/spacing object in scribe data at a specific time."""
        for word in scribe_data.get('words', []):
            if word['start'] <= time_s <= word['end']:
                return word
        return None

    def run(self, 
            split_points_df: pd.DataFrame, 
            scribe_data: Dict[str, Any], 
            min_duration_ms: int = 1000,
            total_duration_s: float = 0.0) -> List[Dict[str, Any]]:
        """
        Generates a list of chunks suitable for MFA alignment.
        """
        print("Executing MFA Chunker Service...")
        mfa_chunks = []
        eligible_split_points_s = (split_points_df['split_point_ms'] / 1000.0).tolist()
        
        if total_duration_s > 0 and total_duration_s not in eligible_split_points_s:
            eligible_split_points_s.append(total_duration_s)
        eligible_split_points_s.sort()

        current_start_s = 0.0
        
        while current_start_s < total_duration_s:
            found_chunk_end = False
            
            for split_point_s in eligible_split_points_s:
                if split_point_s <= current_start_s:
                    continue

                if (split_point_s - current_start_s) * 1000 < min_duration_ms and split_point_s != total_duration_s:
                    continue

                word_at_split = self._find_word_at_time(scribe_data, split_point_s)
                is_last_point = (split_point_s >= total_duration_s)

                if (word_at_split and word_at_split['type'] == 'spacing') or is_last_point:
                    current_end_s = split_point_s
                    
                    # --- MODIFICATION: Gather Scribe words and filter by type ---
                    chunk_scribe_words = []
                    transcript_parts = []
                    for word_obj in scribe_data['words']:
                        # Check if word is within the chunk's time range
                        if word_obj['start'] < current_end_s and word_obj['end'] > current_start_s:
                            # Only include actual words, not spacing or events
                            if word_obj['type'] == 'word':
                                chunk_scribe_words.append(word_obj)
                                transcript_parts.append(word_obj['text'])
                    
                    if not transcript_parts: # Skip chunk if it contains no words
                        current_start_s = current_end_s
                        found_chunk_end = True
                        break

                    mfa_chunks.append({
                        "id": len(mfa_chunks),
                        "start_s": current_start_s,
                        "end_s": current_end_s,
                        "transcript": " ".join(transcript_parts),
                        "scribe_words": chunk_scribe_words # Pass the original word objects
                    })
                    
                    current_start_s = current_end_s
                    found_chunk_end = True
                    break

            if not found_chunk_end:
                break
        
        print(f"Defined {len(mfa_chunks)} chunks for MFA.")
        return mfa_chunks
    