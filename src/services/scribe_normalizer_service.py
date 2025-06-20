# src/services/scribe_normalizer_service.py
import pandas as pd
from typing import Dict, Any, List

class ScribeNormalizerService:
    """
    A service to normalize multiple Scribe JSON results from audio chunks
    into a single JSON object that mirrors the Scribe format.
    """
    def __init__(self):
        print("ScribeNormalizerService initialized.")

    def run(self, scribe_results: List[Dict], chunk_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Normalizes timestamps and combines text from chunked Scribe results.

        Args:
            scribe_results (List[Dict]): A list of raw JSON dicts from Scribe.
            chunk_df (pd.DataFrame): DataFrame with 'chunk_start_ms' of each chunk.

        Returns:
            Dict[str, Any]: A single dictionary mirroring the Scribe output format.
        """
        # Initialize a master dictionary with the Scribe structure
        master_transcript = {
            "text": "",
            "words": [],
            "language_code": scribe_results[0].get('language_code', 'eng') if scribe_results else 'eng'
        }

        full_text_parts = []

        for i, result in enumerate(scribe_results):
            chunk_start_offset_s = chunk_df.iloc[i]['chunk_start_ms'] / 1000.0
            
            # Append the full text from the chunk
            full_text_parts.append(result.get('text', ''))

            # Process each word/event item in the chunk's result
            for item in result.get('words', []):
                # Create a copy to avoid modifying the original dict
                normalized_item = item.copy()
                
                # Adjust timestamps by adding the chunk's start offset
                normalized_item['start'] = round(item['start'] + chunk_start_offset_s, 3)
                normalized_item['end'] = round(item['end'] + chunk_start_offset_s, 3)
                
                master_transcript['words'].append(normalized_item)

        # Join all text parts with a space
        master_transcript['text'] = " ".join(full_text_parts)

        # Add a unique ID to each word object for consistency with the reference project.
        for i, word in enumerate(master_transcript['words']):
            word['id'] = i

        return master_transcript