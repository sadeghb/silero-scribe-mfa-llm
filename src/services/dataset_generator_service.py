import logging
import json
from pathlib import Path
from pydub import AudioSegment
from typing import Dict, Any, List

class DatasetGeneratorService:
    """
    Saves the generated audio clips and metadata for a single cut event
    to the final dataset directory for the evaluation model.
    """
    def __init__(self, config: Dict[str, Any]):
        logging.info("DatasetGeneratorService initialized.")
        # Tweak #4: Use the new output path from the config
        self.output_dir = Path(config.get('output_for_eval', 'output_for_eval'))

    def run(self,
            source_audio_name: str,
            cut_id: int,
            edit_results: Dict,
            # These arguments are no longer needed but kept for signature consistency
            cut_word_ids: list[int] = None,
            chunk_words: List[Dict[str, Any]] = None):
        """
        Saves the three composed audio files and the simplified metadata.json file.
        """
        if not edit_results:
            return

        run_output_dir = self.output_dir / source_audio_name / f"cut_{cut_id:04d}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Save the three composed audio files
        audios_to_save = edit_results.get("audios", {})
        audios_to_save["natural_cut"].export(run_output_dir / "natural_cut.wav", format="wav")
        audios_to_save["unnatural_backward"].export(run_output_dir / "unnatural_backward.wav", format="wav")
        audios_to_save["unnatural_forward"].export(run_output_dir / "unnatural_forward.wav", format="wav")

        # Tweak #2: Create new, simplified metadata
        cut_text_from_editor = edit_results.get("cut_text", "")
        metadata = {
            "source_audio": source_audio_name,
            "cut_id": cut_id,
            "is_usable": edit_results.get("is_usable", False),
            "cut_text": f"... {cut_text_from_editor} ..."
        }

        with open(run_output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        logging.info(f"Saved datapoint for cut {cut_id} to {run_output_dir}")

# # src/services/dataset_generator_service.py
# import logging
# import json
# from pathlib import Path
# from pydub import AudioSegment
# from typing import Dict, Any, List

# class DatasetGeneratorService:
#     """
#     Saves the generated audio clips and metadata for a single cut event
#     to the final dataset directory.
#     """
#     def __init__(self, config: Dict[str, Any]):
#         logging.info("DatasetGeneratorService initialized.")
#         self.output_dir = Path(config.get('output_dataset_path', 'output_dataset'))

#     def run(self,
#             source_audio_name: str,
#             cut_id: int,
#             cut_word_ids: list[int],
#             chunk_words: List[Dict[str, Any]],
#             edit_results: Dict,
#             is_usable: bool): # <-- Accept the new flag
#         """
#         Saves the four audio files and the metadata.json file.
#         """
#         run_output_dir = self.output_dir / source_audio_name / f"cut_{cut_id}"
#         run_output_dir.mkdir(parents=True, exist_ok=True)

#         # Save audio files
#         edit_results["original_audio"].export(run_output_dir / "original.wav", format="wav")
#         edit_results["natural_cut_audio"].export(run_output_dir / "natural_cut.wav", format="wav")
#         edit_results["backward_invasion_audio"].export(run_output_dir / "unnatural_backward.wav", format="wav")
#         edit_results["forward_invasion_audio"].export(run_output_dir / "unnatural_forward.wav", format="wav")
        
#         content_words = [w for w in chunk_words if w.get('type') != 'spacing']
#         word_texts = [w.get('text', '') for w in content_words]
#         word_ids_in_chunk = [w.get('id') for w in content_words]
        
#         marked_text = " ".join(word_texts) 
#         try:
#             start_idx_in_chunk = word_ids_in_chunk.index(cut_word_ids[0])
#             end_idx_in_chunk = word_ids_in_chunk.index(cut_word_ids[-1])
#             final_parts = []
#             for i, text in enumerate(word_texts):
#                 if i == start_idx_in_chunk:
#                     final_parts.append('<cut>')
#                 final_parts.append(text)
#                 if i == end_idx_in_chunk:
#                     final_parts.append('</cut>')
#             marked_text = " ".join(final_parts)
#             marked_text = marked_text.replace(" <cut> ", " <cut>").replace(" </cut> ", "</cut> ")
#         except ValueError:
#             logging.error(f"For cut {cut_id}, could not find all cut word IDs within the provided chunk words.")

#         metadata = {
#             "source_audio": source_audio_name,
#             "cut_id": cut_id,
#             "is_usable": is_usable, # <-- Add the new flag to the metadata
#             "cut_word_ids": cut_word_ids,
#             "marked_up_text": marked_text,
#             "timestamps_in_original_audio": {
#                 "chunk_start_s": edit_results["metadata"]["chunk_start_s_abs"],
#                 "chunk_end_s": edit_results["metadata"]["chunk_end_s_abs"]
#             },
#             "cuts_relative_to_chunk": {
#                 "natural": {
#                     "start_s": edit_results["metadata"]["natural_cut_timestamps_relative"][0],
#                     "end_s": edit_results["metadata"]["natural_cut_timestamps_relative"][1]
#                 },
#                 "backward_invasion": {
#                     "start_s": edit_results["metadata"]["backward_invasion_timestamps_relative"][0],
#                     "end_s": edit_results["metadata"]["backward_invasion_timestamps_relative"][1],
#                     "invasion_factor": round(edit_results["metadata"]["backward_invasion_factor_used"], 4)
#                 },
#                 "forward_invasion": {
#                     "start_s": edit_results["metadata"]["forward_invasion_timestamps_relative"][0],
#                     "end_s": edit_results["metadata"]["forward_invasion_timestamps_relative"][1],
#                     "invasion_factor": round(edit_results["metadata"]["forward_invasion_factor_used"], 4)
#                 }
#             }
#         }
        
#         with open(run_output_dir / "metadata.json", 'w') as f:
#             json.dump(metadata, f, indent=4)
        
#         logging.info(f"Saved datapoint for cut {cut_id} to {run_output_dir}")
