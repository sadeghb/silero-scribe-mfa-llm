# src/services/mfa_aligner_service.py
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any
import shutil

from src.utils.mfa_text_normalizer import normalize_text_for_mfa

class MfaAlignerService:
    """
    A service to run the Montreal Forced Aligner on a given audio chunk and
    its transcript. It manages the temporary files required for MFA.
    """
    def __init__(self, config: Dict[str, Any]):
        print("MfaAlignerService initialized.")
        self.mfa_config = config.get('mfa', {})
        self.num_jobs = self.mfa_config.get('num_jobs', 1)
        self.dictionary_name = self.mfa_config.get('dictionary_name')
        self.acoustic_model_name = self.mfa_config.get('acoustic_model_name')

    def run(self, mfa_chunks_dir: Path, audio_chunks_dir: Path) -> Path:
        """
        Runs the MFA alignment process on a directory of prepared chunks.

        Args:
            mfa_chunks_dir: The directory containing the .lab transcript files.
            audio_chunks_dir: The directory containing the .wav audio chunk files.

        Returns:
            The path to the directory containing the output TextGrid files.
        """
        logging.info(f"Starting MFA alignment for files in {mfa_chunks_dir}...")
        
        output_dir = mfa_chunks_dir / "mfa_output"
        if output_dir.exists():
            shutil.rmtree(output_dir) # Clean up previous runs
        
        # The `mfa align` command
        # Using --clean flag to ensure a fresh run
        mfa_command = [
            "mfa", "align", str(mfa_chunks_dir),
            self.dictionary_name,
            self.acoustic_model_name,
            str(output_dir),
            "--clean",
            "--overwrite",
            "--num_jobs", str(self.num_jobs)
        ]

        logging.info(f"Executing MFA command: {' '.join(mfa_command)}")

        try:
            # We use capture_output=True to get stdout/stderr
            process = subprocess.run(
                mfa_command,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            logging.info("MFA process stdout:\n" + process.stdout)
            logging.info("MFA alignment completed successfully.")
            return output_dir
            
        except FileNotFoundError:
            logging.error("MFA command not found. Is MFA installed and in your system's PATH?")
            raise
        except subprocess.CalledProcessError as e:
            logging.error(f"MFA process failed with exit code {e.returncode}.")
            logging.error("MFA Stderr:\n" + e.stderr)
            logging.error("MFA Stdout:\n" + e.stdout)
            raise e
        