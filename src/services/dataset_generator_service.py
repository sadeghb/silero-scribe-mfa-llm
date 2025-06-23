# src/services/dataset_generator_service.py
import logging
import json
from pathlib import Path
from pydub import AudioSegment
from typing import Dict, Any, List

class DatasetGeneratorService:
    """
    Saves the generated audio clips and metadata for a single cut event
    to the final dataset directory.
    """
    def __init__(self, config: Dict[str, Any]):
        logging.info("DatasetGeneratorService initialized.")
        self.output_dir = Path(config.get('output_dataset_path', 'output_dataset'))

    def run(self,
            source_audio_name: str,
            cut_id: int,
            cut_word_ids: list[int],
            chunk_words: List[Dict[str, Any]], # <-- MODIFIED: Takes chunk-specific words
            edit_results: Dict):
        """
        Saves the four audio files and the metadata.json file.
        """
        run_output_dir = self.output_dir / source_audio_name / f"cut_{cut_id}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Save audio files
        edit_results["original_audio"].export(run_output_dir / "original.wav", format="wav")
        edit_results["natural_cut_audio"].export(run_output_dir / "natural_cut.wav", format="wav")
        edit_results["backward_invasion_audio"].export(run_output_dir / "unnatural_backward.wav", format="wav")
        edit_results["forward_invasion_audio"].export(run_output_dir / "unnatural_forward.wav", format="wav")

        # --- MODIFICATION START: Filter out 'spacing' events before creating text ---
        # This prevents extra spaces in the final marked_up_text.
        content_words = [w for w in chunk_words if w.get('type') != 'spacing']
        
        word_texts = [w.get('text', '') for w in content_words]
        word_ids_in_chunk = [w.get('id') for w in content_words]
        
        marked_text = " ".join(word_texts) 
        try:
            start_idx_in_chunk = word_ids_in_chunk.index(cut_word_ids[0])
            end_idx_in_chunk = word_ids_in_chunk.index(cut_word_ids[-1])

            # Rebuild the string with <cut> tags
            final_parts = []
            for i, text in enumerate(word_texts):
                if i == start_idx_in_chunk:
                    final_parts.append('<cut>')
                final_parts.append(text)
                if i == end_idx_in_chunk:
                    final_parts.append('</cut>')
            
            marked_text = " ".join(final_parts)
            # Clean up spacing around tags for better readability
            marked_text = marked_text.replace(" <cut> ", " <cut>").replace(" </cut> ", "</cut> ")

        except ValueError:
            logging.error(f"For cut {cut_id}, could not find all cut word IDs within the provided chunk words.")
        # --- MODIFICATION END ---

        # # --- MODIFICATION START: Create marked-up text from the chunk's words ---
        # word_texts = [w['text'] for w in chunk_words]
        # word_ids_in_chunk = [w['id'] for w in chunk_words]
        
        # marked_text = " ".join(word_texts) # Default to full chunk text
        # try:
        #     # Find the index *within the chunk* where the cut starts/ends
        #     start_idx_in_chunk = word_ids_in_chunk.index(cut_word_ids[0])
        #     end_idx_in_chunk = word_ids_in_chunk.index(cut_word_ids[-1])

        #     marked_text_parts = (
        #         word_texts[:start_idx_in_chunk] +
        #         ['<cut>'] +
        #         word_texts[start_idx_in_chunk : end_idx_in_chunk + 1] +
        #         ['</cut>'] +
        #         word_texts[end_idx_in_chunk + 1:]
        #     )
        #     # Join and clean up potential double spaces around tags
        #     marked_text = " ".join(marked_text_parts).replace(" <cut> ", " <cut>").replace(" </cut> ", "</cut> ")

        # except ValueError:
        #     logging.error(f"For cut {cut_id}, could not find all cut word IDs within the provided chunk words.")
        # # --- MODIFICATION END ---


        # Create metadata entry
        metadata = {
            "source_audio": source_audio_name,
            "cut_id": cut_id,
            "cut_word_ids": cut_word_ids,
            "marked_up_text": marked_text, # Now contains chunk-specific text
            "timestamps_in_original_audio": {
                "chunk_start_s": edit_results["metadata"]["chunk_start_s_abs"],
                "chunk_end_s": edit_results["metadata"]["chunk_end_s_abs"]
            },
            "cuts_relative_to_chunk": {
                "natural": {
                    "start_s": edit_results["metadata"]["natural_cut_timestamps_relative"][0],
                    "end_s": edit_results["metadata"]["natural_cut_timestamps_relative"][1]
                },
                "backward_invasion": {
                    "start_s": edit_results["metadata"]["backward_invasion_timestamps_relative"][0],
                    "end_s": edit_results["metadata"]["backward_invasion_timestamps_relative"][1]
                },
                "forward_invasion": {
                    "start_s": edit_results["metadata"]["forward_invasion_timestamps_relative"][0],
                    "end_s": edit_results["metadata"]["forward_invasion_timestamps_relative"][1]
                }
            }
        }
        
        with open(run_output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logging.info(f"Saved datapoint for cut {cut_id} to {run_output_dir}")
        